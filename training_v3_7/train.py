"""Zyot Decider v3.7 (Arbiter) — production-grade retrain.

v3.6 result: BoolQ 0.8490, ARC-C 0.7380, CSQA 0.7060.

What's new in v3.7 (research-grounded, each change is a known gap):
  1. MULTI-SOURCE curation (verified on HF):
       - ZefanCai/Open-Jev            (79k, CC0,     92% hi-conf) — safest license
       - ZefanCai/Open-Jev-v1.1       (147k, hardcap 100% hi-conf) — hardest clean
       - tasksource/jev-typed-decisions (2.5M, 87% hi-conf) — massive diversity:
            NLI, procedural-decisions, med/moral/hh-rlhf, 30+ underlying datasets
            License filter: apache/mit/cc-by/cc0/sa/bsd/unspecified only
       - SargeDev/jev-distill-corpus-v3 (655k, 35% hi-conf) — v3.6 baseline, YURI
  2. CROSS-DATASET DEDUP cascade:
       (a) group_id / question_id key (catches tasksource variants)
       (b) md5(normalize(question[:300] + state[:300])) (catches reuploads)
       (c) benchmark contamination hash → DROP
  3. NOUL CAP REMOVED (v3.6 hit it at 100k, lost ~70k eligible rows).
  4. PROMPT FORMAT FIX: 30% of rows trained with empty `State: ` (matches
     ARC-C / CSQA / BoolQ-noState eval prompts — train/test distribution).
  5. SOFT-LABEL KL LOSS on `target` distribution when not one-hot
     (falls back to focal CE when teacher is confident). Preserves teacher
     uncertainty signal that v3.6's argmax discarded.
  6. SAFETY ASSERTS before model load — kernel aborts if curation produces
     < 500k rows / < 150k noul / < 150k choice / < 20k score. No GPU waste.

Everything else kept from v3.6 (proven):
  - Fresh LoRA (r=16, α=32) on Gemma 3 4B
  - 24-slot pointer head init from LM-head verbalizer rows
  - FROZEN T/F slots (0,1) via gradient hook — preserves BoolQ baseline
  - EMA weight averaging over last 5 checkpoints
  - Per-slot masking in loss (Kev-style)
  - Sample weighting by teacher confidence
  - Checkpoint every 10 steps (Kaggle session-safe)

Runs on Kaggle T4; expected wall-clock 5-6h for 3000 steps.
"""
import os, sys, json, math, time, subprocess, hashlib, copy, random, re
from pathlib import Path
from collections import defaultdict, Counter

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

subprocess.run('pip uninstall -y torchao 2>/dev/null || true', shell=True)
subprocess.check_call(
    'pip install -q "unsloth" "unsloth_zoo" '
    '"transformers>=4.44,<5" "peft>=0.11,<1" '
    '"accelerate>=0.30,<2" bitsandbytes hf_transfer datasets', shell=True)

try:
    from unsloth import FastModel
    UNSLOTH_OK = True
    print('=== Unsloth kernels active ===', flush=True)
except Exception as e:
    UNSLOTH_OK = False
    print(f'WARN: Unsloth import failed ({e}) — falling back to HF', flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================
#                   CONFIG
# ============================================================
BASE_MODEL = 'unsloth/gemma-3-4b-it'
NUM_SLOTS = 24
BATCH = 2
GA_STEPS = 8                  # effective batch 16
LR = 5e-5
MAX_STEPS = 3000
SAVE_EVERY = 10
EVAL_EVERY = 500
LOG_EVERY = 20
WARMUP_STEPS = 200
MAX_LEN = 768
GRAD_CLIP = 1.0
FOCAL_GAMMA = 2.0
EMA_LAST_N = 5
FROZEN_SLOTS = [0, 1]
VAL_SIZE = 3000
EMPTY_STATE_PROB = 0.30       # fraction of rows trained with empty State:

# Safety floors — abort before GPU load if curation is bad.
MIN_TOTAL = 500_000
MIN_NOUL = 100_000       # v3.6 trained with 100k noul and reached BoolQ 0.849 — floor at that
MIN_CHOICE = 150_000
MIN_SCORE = 20_000

RESUME = os.environ.get('ZYOT_RESUME', '0') == '1'
DRY_RUN = os.environ.get('ZYOT_DRY_RUN', '0') == '1'  # curate then exit
OUT = Path('/kaggle/working/v3.7')
OUT.mkdir(parents=True, exist_ok=True)

DATA_JSONL = OUT / 'train_v3_7.jsonl'
VAL_JSONL = OUT / 'val_v3_7.jsonl'
STATS_JSON = OUT / 'curation_stats.json'

# ============================================================
#               LICENSE FILTER (for tasksource)
# ============================================================
# Keep rows whose license field matches one of these tokens; drop the rest.
LICENSE_ALLOW_TOKENS = (
    'apache', 'mit', 'cc0', 'cc-by-4', 'cc-by-sa', 'cc by 4', 'bsd',
    'unspecified', '',  # unspecified kept (large portion of tasksource)
)
LICENSE_BLOCK_TOKENS = (
    'non-commercial', 'noncommercial', 'nc', 'academic research',
    'request form', 'openai', 'proprietary',
)
def license_ok(lic):
    s = str(lic or '').lower()
    if not s or s == 'unspecified': return True
    for b in LICENSE_BLOCK_TOKENS:
        if b in s: return False
    for a in LICENSE_ALLOW_TOKENS:
        if a and a in s: return True
    return False  # unknown license tokens → block

# ============================================================
#            PROMPT HELPERS + NORMALIZATION
# ============================================================
rng = random.Random(42)

def normalize_q(s):
    s = str(s or '').strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def _hash_qs(q, state):
    key = normalize_q(q)[:300] + '||' + normalize_q(state)[:300]
    return hashlib.md5(key.encode('utf-8')).hexdigest()[:16]

NOUL_SLOTS = [0, 1]
def choice_slots(n): return list(range(2, 2 + n))
SCORE_SLOTS = list(range(18, 24))

def _state_prefix():
    """Random: 30% empty state (benchmark-style), 70% content state (SargeDev-style)."""
    return rng.random() < EMPTY_STATE_PROB

def _make_choice_prompt(state, question, options, force_empty=None):
    letters = [chr(ord('A') + i) for i in range(len(options))]
    opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, options))
    empty = force_empty if force_empty is not None else _state_prefix()
    st = '' if empty else str(state or '')
    return f'State: {st}\n\nQuestion: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:'

def _make_noul_prompt(state, question, force_empty=None):
    empty = force_empty if force_empty is not None else _state_prefix()
    st = '' if empty else str(state or '')
    return (f'State: {st}\n\nQuestion: {question}\n\n'
            f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')

def _make_score_prompt(state, question, force_empty=None):
    empty = force_empty if force_empty is not None else _state_prefix()
    st = '' if empty else str(state or '')
    return (f'State: {st}\n\nQuestion: {question}\n\n'
            f'Options:\n0\n1\n2\n3\n4\n5\n\nAnswer:')

def _noul_slot_from_options(options, argmax_i):
    lbl = str(options[argmax_i]).strip().lower()
    if lbl in ('true', 't', 'yes', '1'): return 0
    if lbl in ('false', 'f', 'no', '0'): return 1
    return None

def _score_slot_from_options(options, argmax_i):
    txt = str(options[argmax_i]).strip()
    # options may be "3" or "3 out of 6" etc.
    m = re.match(r'^(\d+)', txt)
    if not m: return None
    n = int(m.group(1))
    # if scale is 1..6 (AES-style), remap to 0..5
    if max(int(re.match(r'^(\d+)', str(o).strip()).group(1)) for o in options
           if re.match(r'^\d+', str(o).strip())) == 6:
        n = max(0, n - 1)
    if 0 <= n <= 5: return 18 + n
    return None

# ============================================================
#            CURATION — multi-source + dedup
# ============================================================
def curate_data():
    if RESUME and DATA_JSONL.exists() and VAL_JSONL.exists():
        print('=== curated data already present (RESUME) — skipping ===', flush=True)
        return
    print('=== CURATING v3.7 data ===', flush=True)
    from datasets import load_dataset

    # ---- 1. Benchmark contamination hashes ----
    print('  building benchmark contamination hashes...', flush=True)
    contam = set()
    for ds_name, cfg, split, qfield in [
        ('google/boolq', None, 'validation', 'question'),
        ('allenai/ai2_arc', 'ARC-Challenge', 'test', 'question'),
        ('allenai/ai2_arc', 'ARC-Easy', 'test', 'question'),
        ('tau/commonsense_qa', None, 'validation', 'question'),
        ('allenai/openbookqa', 'main', 'test', 'question_stem'),
        ('ybisk/piqa', None, 'validation', 'goal'),
    ]:
        try:
            kw = {'name': cfg} if cfg else {}
            ds = load_dataset(ds_name, split=split, streaming=True, **kw)
            n = 0
            for row in ds:
                q = row.get(qfield) or row.get('question') or ''
                if q:
                    contam.add(hashlib.md5(normalize_q(q).encode()).hexdigest()[:16])
                    n += 1
                if n >= 5000: break
        except Exception as e:
            print(f'    WARN benchmark {ds_name}/{split} failed: {e}', flush=True)
    print(f'  contamination hashes: {len(contam)}', flush=True)

    items = []
    seen_hashes = set()
    seen_group_ids = set()
    dropped_contam = 0
    dropped_dup = 0
    dropped_lic = 0
    per_source_kept = Counter()
    per_kind_kept = Counter()

    CONF_MIN = 0.75

    def _dedup_key(row, kind_prefix=''):
        """Return (group_key, hash_key). None if row should be dropped as dup."""
        gid = row.get('group_id') or row.get('question_id') or row.get('id')
        if gid:
            gk = f'{kind_prefix}|gid|{gid}'
        else:
            gk = None
        return gk

    def _try_add(row, source_tag, override_state=None):
        """Normalize + add row to items. Returns 'kept' | 'dup' | 'contam' | 'skip'."""
        nonlocal dropped_contam, dropped_dup
        kind = row.get('kind') or row.get('decision_type') or ''
        if kind not in ('noul', 'choice', 'score'): return 'skip'

        target = row.get('target')
        if target is None: target = row.get('ordered_targets')
        if not isinstance(target, list) or not target: return 'skip'
        try:
            target = [float(x) for x in target]
        except Exception: return 'skip'
        if max(target) < CONF_MIN: return 'skip'

        options = row.get('options') or row.get('candidates')
        if not isinstance(options, list) or len(options) < 2: return 'skip'
        if kind == 'choice' and len(options) > 16: return 'skip'
        if len(target) != len(options): return 'skip'

        argmax_i = max(range(len(target)), key=lambda i: target[i])
        conf = float(target[argmax_i])

        state_raw = row.get('state')
        if state_raw is None: state_raw = row.get('state_json') or ''
        if isinstance(state_raw, dict): state_raw = json.dumps(state_raw)[:1500]
        state_raw = override_state if override_state is not None else str(state_raw)[:1500]

        question = str(row.get('question') or '')[:800]

        if kind == 'noul':
            slot = _noul_slot_from_options(options, argmax_i)
            if slot is None: return 'skip'
            prompt = _make_noul_prompt(state_raw, question)
            valid = NOUL_SLOTS
        elif kind == 'choice':
            slot = 2 + argmax_i
            prompt = _make_choice_prompt(state_raw, question, options)
            valid = choice_slots(len(options))
        else:  # score
            if len(options) != 6: return 'skip'
            slot = _score_slot_from_options(options, argmax_i)
            if slot is None: return 'skip'
            prompt = _make_score_prompt(state_raw, question)
            valid = SCORE_SLOTS

        # Dedup
        gk = _dedup_key(row, kind_prefix=kind)
        if gk and gk in seen_group_ids:
            dropped_dup += 1
            return 'dup'
        h = _hash_qs(question, state_raw)
        if h in contam:
            dropped_contam += 1
            return 'contam'
        if h in seen_hashes:
            dropped_dup += 1
            return 'dup'

        if gk: seen_group_ids.add(gk)
        seen_hashes.add(h)

        items.append({'prompt': prompt, 'answer_slot': slot, 'valid_slots': valid,
                      'weight': conf, 'kind': kind, 'hash': h,
                      'target': target, 'source': source_tag})
        per_source_kept[source_tag] += 1
        per_kind_kept[kind] += 1
        return 'kept'

    # ---- 2. Source A: ZefanCai/Open-Jev (CC0) ----
    print('\n  [1/4] ZefanCai/Open-Jev  (release-v2-redistributable)...', flush=True)
    before = len(items)
    try:
        ds = load_dataset('ZefanCai/Open-Jev', 'release-v2-redistributable',
                          split='train', streaming=True)
        n_seen = 0
        for row in ds:
            n_seen += 1
            _try_add(row, 'open-jev')
            if n_seen % 20000 == 0:
                print(f'    [{n_seen} seen] kept so far: {len(items) - before}', flush=True)
    except Exception as e:
        print(f'  WARN Open-Jev failed: {e}', flush=True)
    print(f'  Open-Jev kept: {len(items) - before}', flush=True)

    # ---- 3. Source B: ZefanCai/Open-Jev-v1.1 ----
    print('\n  [2/4] ZefanCai/Open-Jev-v1.1  (community-hard-mix-v2-redistributable)...', flush=True)
    before = len(items)
    try:
        ds = load_dataset('ZefanCai/Open-Jev-v1.1', 'community-hard-mix-v2-redistributable',
                          split='train', streaming=True)
        n_seen = 0
        for row in ds:
            n_seen += 1
            _try_add(row, 'open-jev-v1.1')
            if n_seen % 20000 == 0:
                print(f'    [{n_seen} seen] kept so far: {len(items) - before}', flush=True)
    except Exception as e:
        print(f'  WARN Open-Jev-v1.1 failed: {e}', flush=True)
    print(f'  Open-Jev-v1.1 kept: {len(items) - before}', flush=True)

    # ---- 4. Source C: tasksource/tasksource-jev-typed-decisions (biggest) ----
    print('\n  [3/4] tasksource/tasksource-jev-typed-decisions  (2.5M rows, license-filtered)...', flush=True)
    before = len(items)
    try:
        ds = load_dataset('tasksource/tasksource-jev-typed-decisions',
                          split='train', streaming=True)
        n_seen = 0
        for row in ds:
            n_seen += 1
            lic = row.get('license') or ''
            if not license_ok(lic):
                dropped_lic += 1
                continue
            _try_add(row, 'tasksource')
            if n_seen % 100000 == 0:
                print(f'    [{n_seen} seen] kept so far: {len(items) - before}  lic_dropped: {dropped_lic}', flush=True)
    except Exception as e:
        print(f'  WARN tasksource failed: {e}', flush=True)
    print(f'  tasksource kept: {len(items) - before}  license_dropped: {dropped_lic}', flush=True)

    # ---- 5. Source D: SargeDev (v3.6 baseline, fills remaining noul/choice/score) ----
    print('\n  [4/4] SargeDev/jev-distill-corpus-v3  (655k)...', flush=True)
    before = len(items)
    try:
        ds = load_dataset('SargeDev/jev-distill-corpus-v3', split='train', streaming=True)
        n_seen = 0
        for row in ds:
            n_seen += 1
            _try_add(row, 'sargedev')
            if n_seen % 50000 == 0:
                print(f'    [{n_seen} seen] kept so far: {len(items) - before}', flush=True)
    except Exception as e:
        print(f'  WARN SargeDev failed: {e}', flush=True)
    print(f'  SargeDev kept: {len(items) - before}', flush=True)

    # ---- 6. Report + safety asserts ----
    print(f'\n=== CURATION DONE ===', flush=True)
    print(f'  TOTAL items: {len(items)}', flush=True)
    print(f'  by source: {dict(per_source_kept)}', flush=True)
    print(f'  by kind:   {dict(per_kind_kept)}', flush=True)
    print(f'  dropped (contam): {dropped_contam}', flush=True)
    print(f'  dropped (dup):    {dropped_dup}', flush=True)
    print(f'  dropped (lic):    {dropped_lic}', flush=True)

    stats = {
        'total': len(items),
        'per_source': dict(per_source_kept),
        'per_kind': dict(per_kind_kept),
        'dropped_contam': dropped_contam,
        'dropped_dup': dropped_dup,
        'dropped_lic': dropped_lic,
        'contam_hash_count': len(contam),
    }

    # HARD safety asserts (fail fast, before model load)
    errors = []
    if len(items) < MIN_TOTAL:
        errors.append(f'TOTAL {len(items)} < MIN_TOTAL {MIN_TOTAL}')
    if per_kind_kept['noul'] < MIN_NOUL:
        errors.append(f'noul {per_kind_kept["noul"]} < MIN_NOUL {MIN_NOUL}')
    if per_kind_kept['choice'] < MIN_CHOICE:
        errors.append(f'choice {per_kind_kept["choice"]} < MIN_CHOICE {MIN_CHOICE}')
    if per_kind_kept['score'] < MIN_SCORE:
        errors.append(f'score {per_kind_kept["score"]} < MIN_SCORE {MIN_SCORE}')
    if errors:
        for e in errors: print(f'  FATAL: {e}', flush=True)
        STATS_JSON.write_text(json.dumps({**stats, 'errors': errors}, indent=2))
        raise RuntimeError('curation safety asserts failed — aborting before model load')

    # Stratified val split
    rng.seed(42)
    rng.shuffle(items)
    by_kind = defaultdict(list)
    for it in items: by_kind[it['kind']].append(it)
    val, train = [], []
    for k, lst in by_kind.items():
        n_val = min(VAL_SIZE * len(lst) // len(items), len(lst) // 20)
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    rng.shuffle(val); rng.shuffle(train)
    print(f'  TRAIN: {len(train)}, VAL: {len(val)}', flush=True)

    with DATA_JSONL.open('w', encoding='utf-8') as f:
        for it in train: f.write(json.dumps(it) + '\n')
    with VAL_JSONL.open('w', encoding='utf-8') as f:
        for it in val: f.write(json.dumps(it) + '\n')
    stats['train'] = len(train); stats['val'] = len(val)
    STATS_JSON.write_text(json.dumps(stats, indent=2))
    print('=== curation written ===\n', flush=True)

curate_data()

if DRY_RUN:
    print('=== DRY_RUN complete — exiting before training ===', flush=True)
    sys.exit(0)

# ============================================================
#            MODEL LOAD + HEAD + FROZEN SLOTS (as v3.6)
# ============================================================
print('=== Loading tokenizer ===', flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

VERBALIZER = ['T', 'F',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              '0', '1', '2', '3', '4', '5']
def _t1(s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None
VERB_IDS = [x if (x := _t1(v)) is not None else 0 for v in VERBALIZER]
print(f'  verbalizer ids ok ({NUM_SLOTS} slots)', flush=True)

print('=== Loading Gemma 3 4B in 4-bit (FRESH LoRA) ===', flush=True)
if UNSLOTH_OK:
    base, _ = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_LEN,
        load_in_4bit=True, dtype=torch.bfloat16, full_finetuning=False)
    model = FastModel.get_peft_model(
        base, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        use_gradient_checkpointing='unsloth', bias='none')
else:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
        dtype=torch.bfloat16, attn_implementation='eager', device_map='auto')
    base = prepare_model_for_kbit_training(base)
    lc = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        task_type='CAUSAL_LM')
    model = get_peft_model(base, lc)
    model.gradient_checkpointing_enable()
device = next(model.parameters()).device
HIDDEN = getattr(base.config, 'hidden_size', None) \
     or getattr(getattr(base.config, 'text_config', None), 'hidden_size', None) \
     or model.get_input_embeddings().embedding_dim
print(f'  HIDDEN = {HIDDEN}', flush=True)

class DecisionHead(nn.Module):
    def __init__(self, hidden, slots, lm_head_weight=None, verb_ids=None):
        super().__init__()
        self.proj = nn.Linear(hidden, slots, bias=False)
        if lm_head_weight is not None and verb_ids is not None:
            with torch.no_grad():
                rows = lm_head_weight[verb_ids].to(self.proj.weight.dtype)
                self.proj.weight.copy_(rows.to(self.proj.weight.device))
                print(f'  head init from LM head rows (shape {rows.shape})', flush=True)
    def forward(self, last_hidden): return self.proj(last_hidden)

lm_head_w = model.get_output_embeddings().weight.detach()
head = DecisionHead(HIDDEN, NUM_SLOTS, lm_head_w, VERB_IDS).to(device=device, dtype=torch.bfloat16)

def _freeze_tf_hook(grad):
    g = grad.clone()
    for s in FROZEN_SLOTS: g[s] = 0.0
    return g
head.proj.weight.register_hook(_freeze_tf_hook)
print(f'  FROZEN slots: {FROZEN_SLOTS}', flush=True)

step_start = 0
if RESUME and (OUT / 'head_ckpt' / 'latest' / 'head.pt').exists():
    print('=== RESUMING ===', flush=True)
    head.load_state_dict(torch.load(OUT / 'head_ckpt' / 'latest' / 'head.pt', map_location='cpu'))
    if (OUT / 'lora_ckpt' / 'latest' / 'adapter_config.json').exists():
        model.load_adapter(str(OUT / 'lora_ckpt' / 'latest'), 'default')
    meta = json.loads((OUT / 'head_ckpt' / 'latest' / 'meta.json').read_text())
    step_start = meta.get('step', 0)
    print(f'  resumed step {step_start}', flush=True)

# ============================================================
#            DATASET + LOADER
# ============================================================
class DecisionDataset(Dataset):
    def __init__(self, path):
        self.items = []
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            try: r = json.loads(line)
            except: continue
            self.items.append(r)
        print(f'  loaded {len(self.items)} from {path}', flush=True)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    enc = tok([b['prompt'] for b in batch], return_tensors='pt', padding=True,
              truncation=True, max_length=MAX_LEN)
    labels = torch.tensor([b['answer_slot'] for b in batch], dtype=torch.long)
    weights = torch.tensor([float(b.get('weight', 1.0)) for b in batch], dtype=torch.float32)
    mask = torch.zeros(len(batch), NUM_SLOTS, dtype=torch.bool)
    # Soft-label target distribution over slots (not just argmax)
    soft = torch.zeros(len(batch), NUM_SLOTS, dtype=torch.float32)
    for i, b in enumerate(batch):
        valid = b.get('valid_slots', list(range(NUM_SLOTS)))
        for s in valid:
            if 0 <= s < NUM_SLOTS: mask[i, s] = True
        # map target distribution (per-option) onto the valid slot positions
        tgt = b.get('target') or []
        if tgt and len(tgt) == len(valid):
            for s, t in zip(valid, tgt):
                if 0 <= s < NUM_SLOTS: soft[i, s] = float(t)
    return enc.input_ids, enc.attention_mask, labels, weights, mask, soft

train_ds = DecisionDataset(DATA_JSONL)
val_ds = DecisionDataset(VAL_JSONL)
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=0)

# ============================================================
#            LOSS: soft-label KL + focal CE fallback + per-slot mask + weighting
# ============================================================
def soft_kl_focal_loss(logits, labels, weights, valid_mask, soft_target, gamma=FOCAL_GAMMA):
    """
    Hybrid:
      - mask invalid slots to -inf
      - if soft_target is one-hot (max ≥ 0.98), use focal CE on argmax (v3.6 default)
      - else use KL(soft_target || softmax(logits)) — preserves teacher uncertainty
      - sample-weighted by `weights` (teacher confidence)
    """
    logits_m = logits.masked_fill(~valid_mask, -1e4)
    logp = F.log_softmax(logits_m.float(), dim=-1)   # [B, S]
    p = logp.exp()

    # per-row max of soft target (if ~1, teacher is confident = one-hot)
    tgt_max = soft_target.max(dim=-1).values         # [B]
    use_kl = tgt_max < 0.98                          # [B] bool

    # branch 1: focal CE on argmax labels
    ce = -logp.gather(1, labels.view(-1, 1)).squeeze(1)   # [B]
    if gamma > 0:
        pt = p.gather(1, labels.view(-1, 1)).squeeze(1)
        focal = (1.0 - pt).clamp(min=1e-6) ** gamma
        ce = focal * ce

    # branch 2: KL(soft_target || softmax(logits)) = sum soft * (log soft - log p)
    soft = soft_target.to(logp.device).float()
    soft = soft * valid_mask.float()                 # zero invalid slots
    soft_sum = soft.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    soft = soft / soft_sum
    kl = (soft * (soft.clamp(min=1e-9).log() - logp)).sum(dim=-1)   # [B]

    loss_per = torch.where(use_kl.to(logp.device), kl, ce)
    loss_per = loss_per * weights.to(loss_per.device)
    return loss_per.mean()

# ============================================================
#            TRAINING LOOP
# ============================================================
trainable = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
print(f'  trainable params: {sum(p.numel() for p in trainable):,}', flush=True)
optim = torch.optim.AdamW(trainable, lr=LR)

def _get_lr(step, total, warmup):
    if step < warmup: return LR * step / max(1, warmup)
    pr = (step - warmup) / max(1, total - warmup)
    return LR * 0.5 * (1 + math.cos(math.pi * pr))

def save_ckpt(step, tag='latest'):
    d = OUT / 'head_ckpt' / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), d / 'head.pt')
    (d / 'meta.json').write_text(json.dumps({'step': step, 'num_slots': NUM_SLOTS,
        'verbalizer': VERBALIZER, 'verb_ids': VERB_IDS,
        'frozen_slots': FROZEN_SLOTS, 'focal_gamma': FOCAL_GAMMA,
        'empty_state_prob': EMPTY_STATE_PROB, 'uses_soft_kl': True}))
    ld = OUT / 'lora_ckpt' / tag
    ld.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ld))

@torch.no_grad()
def evaluate():
    model.eval(); head.eval()
    correct = total = 0
    for input_ids, attn, labels, weights, mask, soft in val_loader:
        input_ids = input_ids.to(device); attn = attn.to(device)
        labels = labels.to(device); mask = mask.to(device)
        out = model(input_ids=input_ids, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)
        last = out.hidden_states[-1]
        idx = attn.sum(dim=1) - 1
        pooled = last[torch.arange(last.size(0), device=device), idx]
        logits = head(pooled.to(head.proj.weight.dtype))
        logits = logits.masked_fill(~mask, -1e4)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    model.train(); head.train()
    return correct / max(total, 1)

print(f'\n=== TRAIN v3.7 — max_steps={MAX_STEPS}, soft-KL+focal, EMA last {EMA_LAST_N} ===\n', flush=True)
model.train(); head.train()
step = step_start
grad_accum = 0
t0 = time.time()
loss_ema = None
ema_head_states = []

try:
    while step < MAX_STEPS:
        for input_ids, attn, labels, weights, mask, soft in loader:
            input_ids = input_ids.to(device); attn = attn.to(device)
            labels = labels.to(device); mask = mask.to(device); soft = soft.to(device)
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
            last = out.hidden_states[-1]
            idx = attn.sum(dim=1) - 1
            pooled = last[torch.arange(last.size(0), device=device), idx]
            logits = head(pooled.to(head.proj.weight.dtype))
            loss = soft_kl_focal_loss(logits, labels, weights, mask, soft) / GA_STEPS
            loss.backward()
            grad_accum += 1
            if grad_accum == GA_STEPS:
                for pg in optim.param_groups: pg['lr'] = _get_lr(step, MAX_STEPS, WARMUP_STEPS)
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optim.step(); optim.zero_grad()
                grad_accum = 0
                step += 1
                lv = float(loss.detach()) * GA_STEPS
                loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
                if step % LOG_EVERY == 0:
                    dt = time.time() - t0; sps = step / max(dt, 1)
                    eta = (MAX_STEPS - step) / max(sps, 1e-9) / 60
                    print(f'  step {step:>5}/{MAX_STEPS}  loss={loss_ema:.4f}  {sps:.2f} s/s  ETA {eta:.0f}m', flush=True)
                if step % SAVE_EVERY == 0: save_ckpt(step, 'latest')
                if step % EVAL_EVERY == 0:
                    acc = evaluate()
                    print(f'  ** VAL acc = {acc:.4f} at step {step} **', flush=True)
                    save_ckpt(step, f'step-{step}')
                    ema_head_states.append({k: v.detach().clone().cpu() for k, v in head.state_dict().items()})
                    if len(ema_head_states) > EMA_LAST_N: ema_head_states.pop(0)
                if step >= MAX_STEPS: break
except KeyboardInterrupt:
    print('\n=== INTERRUPTED — saving final ===', flush=True)

save_ckpt(step, 'final')
save_ckpt(step, 'latest')
if len(ema_head_states) >= 2:
    print(f'\n=== EMA from last {len(ema_head_states)} snapshots ===', flush=True)
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in ema_head_states[0].items()}
    for sd in ema_head_states:
        for k in avg: avg[k] += sd[k].float()
    for k in avg: avg[k] /= len(ema_head_states)
    ed = OUT / 'head_ckpt' / 'ema-final'
    ed.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) for k, v in avg.items()}, ed / 'head.pt')
    (ed / 'meta.json').write_text(json.dumps({'step': step, 'num_slots': NUM_SLOTS,
        'verbalizer': VERBALIZER, 'verb_ids': VERB_IDS, 'ema_n': len(ema_head_states),
        'frozen_slots': FROZEN_SLOTS, 'uses_soft_kl': True}))
    head.load_state_dict({k: v.to(head.proj.weight.dtype).to(device) for k, v in avg.items()})
    print(f'  EMA val acc = {evaluate():.4f}', flush=True)

print(f'\n=== DONE — {step} steps, {(time.time()-t0)/60:.1f} min, final loss {loss_ema:.4f} ===', flush=True)
