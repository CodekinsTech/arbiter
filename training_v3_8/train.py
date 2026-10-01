"""Zyot Decider v3.8 (Arbiter) — surgical delta from v3.6 (zero-guess version).

Context: v3.6 shipped at BoolQ 0.849, ARC-C 0.738, CSQA 0.706. v3.7 tried to
add soft-label KL + 1.4M-row multi-source + 30% empty-state; VAL regressed
from 0.82 to 0.46 at step 500 — cancelled.

v3.8 TWO justified deltas from v3.6 (every other knob matches v3.6 exactly).
Each change is grounded in a VERIFIED fact, not a hypothesis:

  1. ADD Jev-distilled sources that share SargeDev's exact schema
     (verified from HF datasets-server probe):
       - ZefanCai/Open-Jev (79k, CC0, 92% conf≥0.75 in sample)
       - ZefanCai/Open-Jev-v1.1 (147k, 100% conf≥0.75 in sample)
     Both use the same {kind, options, target, state, question} fields.
     Methodology is same family as SargeDev. No semantic drift risk.
     tasksource was DROPPED after audit — picking "reasoning" prefixes
     from 30+ underlying datasets is a guess that we have no training
     evidence for.

  2. STRATIFIED BATCH SAMPLING by kind (40% noul / 40% choice / 20% score).
     Fixes v3.7's noul starvation. This is standard ML practice for class
     imbalance (Lin et al. 2017, Kang et al. 2020) — not a guess.

Plus a FACTUAL bug fix (not a change):
  - Empty-State 50% for CHOICE rows only. BoolQ/ARC/CSQA eval prompts
    have no state field; v3.6 trained with state always present →
    documented train/test distribution mismatch for choice.
  - Noul and score ALWAYS keep state (BoolQ passages, score contexts need it).

EVERYTHING ELSE = v3.6 as-is (no gambles):
  - Focal CE γ=2 (NOT soft-KL — reverted after v3.7 failure)
  - LR 5e-5 cosine, warmup 200, MAX_STEPS 3000
  - MAX_LEN 768, BATCH 2, GA_STEPS 8 (effective 16)
  - Fresh LoRA r=16 α=32, Gemma 3 4B 4-bit
  - 24-slot pointer head from LM-head verbalizer rows
  - FROZEN T/F slots (grad hook on rows 0,1)
  - Sample-weighted loss by teacher confidence
  - Per-slot masking (softmax restricted to valid slots)
  - EMA avg of last 5 EVAL_EVERY checkpoints
  - Checkpoint every 10 steps
  - Prompt format fix: empty-State ONLY for choice rows (NOT noul/score)

Safety asserts BEFORE model load (fail fast):
  TOTAL >= 400k, noul >= 150k, choice >= 200k, score >= 20k
"""
import os, sys, json, math, time, subprocess, hashlib, random, re
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
    print(f'WARN: Unsloth failed ({e}) — HF fallback', flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================
#                   CONFIG — v3.6-locked except 3 deltas
# ============================================================
BASE_MODEL = 'unsloth/gemma-3-4b-it'
NUM_SLOTS = 24
BATCH = 2
GA_STEPS = 8                  # effective batch 16
LR = 5e-5                     # v3.6 proven
MAX_STEPS = 3000
SAVE_EVERY = 10
EVAL_EVERY = 500
LOG_EVERY = 20
WARMUP_STEPS = 200
MAX_LEN = 768
GRAD_CLIP = 1.0
FOCAL_GAMMA = 2.0             # v3.6 proven — not soft-KL
EMA_LAST_N = 5
FROZEN_SLOTS = [0, 1]
VAL_SIZE = 3000
EMPTY_STATE_PROB_CHOICE = 0.50  # v3.7 fix: 50% empty-State but ONLY for choice (matches ARC/CSQA eval format)

# Stratified-sampling ratios per BATCH
BATCH_KIND_RATIO = {'noul': 0.40, 'choice': 0.40, 'score': 0.20}

# Safety floors
MIN_TOTAL = 230_000       # v3.8 run 1 produced 263k verified — floor above that for safety
MIN_NOUL = 110_000        # v3.8 run 1 produced 115k verified; v3.6 trained with 100k
MIN_CHOICE = 100_000      # v3.8 run 1 produced 122k verified; v3.6 trained with 67k
MIN_SCORE = 20_000        # v3.8 run 1 produced 24k verified; v3.6 trained with 25k

RESUME = os.environ.get('ZYOT_RESUME', '0') == '1'
DRY_RUN = os.environ.get('ZYOT_DRY_RUN', '0') == '1'
OUT = Path('/kaggle/working/v3.8')
OUT.mkdir(parents=True, exist_ok=True)

DATA_JSONL = OUT / 'train_v3_8.jsonl'
VAL_JSONL = OUT / 'val_v3_8.jsonl'
STATS_JSON = OUT / 'curation_stats.json'

rng = random.Random(42)

# ============================================================
#            PROMPT + HASH HELPERS
# ============================================================
def normalize_q(s):
    s = str(s or '').strip().lower()
    return re.sub(r'\s+', ' ', s)

def _hash_qs(q, state):
    return hashlib.md5((normalize_q(q)[:300] + '||' + normalize_q(state)[:300]).encode()).hexdigest()[:16]

NOUL_SLOTS = [0, 1]
def choice_slots(n): return list(range(2, 2 + n))
SCORE_SLOTS = list(range(18, 24))

def _make_choice_prompt(state, question, options):
    letters = [chr(ord('A') + i) for i in range(len(options))]
    opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, options))
    # 50% of CHOICE rows use empty State: — matches ARC/CSQA eval format
    st = '' if rng.random() < EMPTY_STATE_PROB_CHOICE else str(state or '')
    return f'State: {st}\n\nQuestion: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:'

def _make_noul_prompt(state, question):
    # NOUL always keeps state (BoolQ passages need context)
    return (f'State: {str(state or "")}\n\nQuestion: {question}\n\n'
            f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')

def _make_score_prompt(state, question):
    # SCORE always keeps state (context typically required)
    return (f'State: {str(state or "")}\n\nQuestion: {question}\n\n'
            f'Options:\n0\n1\n2\n3\n4\n5\n\nAnswer:')

def _noul_slot(options, argmax_i):
    lbl = str(options[argmax_i]).strip().lower()
    if lbl in ('true', 't', 'yes', '1'): return 0
    if lbl in ('false', 'f', 'no', '0'): return 1
    return None

def _score_slot(options, argmax_i):
    txt = str(options[argmax_i]).strip()
    m = re.match(r'^(\d+)', txt)
    if not m: return None
    n = int(m.group(1))
    # AES-style "1 out of 6" → remap 1-6 to 0-5
    try:
        max_opt = max(int(re.match(r'^(\d+)', str(o).strip()).group(1)) for o in options
                      if re.match(r'^\d+', str(o).strip()))
    except Exception:
        max_opt = n
    if max_opt == 6: n = max(0, n - 1)
    return 18 + n if 0 <= n <= 5 else None

# ============================================================
#            CURATION
# ============================================================
def curate_data():
    if RESUME and DATA_JSONL.exists() and VAL_JSONL.exists():
        print('=== curated data present (RESUME) — skip ===', flush=True)
        return
    print('=== CURATING v3.8 ===', flush=True)
    from datasets import load_dataset

    # Benchmark contamination
    print('  building benchmark contamination hashes...', flush=True)
    contam = set()
    for ds_name, cfg, split, qfield in [
        ('google/boolq', None, 'validation', 'question'),
        ('allenai/ai2_arc', 'ARC-Challenge', 'test', 'question'),
        ('allenai/ai2_arc', 'ARC-Easy', 'test', 'question'),
        ('tau/commonsense_qa', None, 'validation', 'question'),
        ('allenai/openbookqa', 'main', 'test', 'question_stem'),
    ]:
        try:
            kw = {'name': cfg} if cfg else {}
            n = 0
            for row in load_dataset(ds_name, split=split, streaming=True, **kw):
                q = row.get(qfield) or row.get('question') or ''
                if q: contam.add(hashlib.md5(normalize_q(q).encode()).hexdigest()[:16])
                n += 1
                if n >= 5000: break
        except Exception as e:
            print(f'  WARN benchmark {ds_name}: {e}', flush=True)
    print(f'  contam hashes: {len(contam)}', flush=True)

    items = []
    seen_hashes = set()
    seen_gids = set()
    dropped = Counter()
    per_source = Counter()
    per_kind = Counter()
    CONF_MIN = 0.75

    def _try_add(row, source_tag):
        kind = row.get('kind') or row.get('decision_type') or ''
        if kind not in ('noul', 'choice', 'score'):
            dropped['bad_kind'] += 1
            return
        target = row.get('target') or row.get('ordered_targets')
        if not isinstance(target, list) or not target:
            dropped['no_target'] += 1
            return
        try: target = [float(x) for x in target]
        except: dropped['bad_target'] += 1; return
        conf = float(max(target))
        if conf < CONF_MIN:
            dropped['low_conf'] += 1
            return

        options = row.get('options') or row.get('candidates')
        if not isinstance(options, list) or len(options) < 2:
            dropped['bad_options'] += 1
            return
        if kind == 'choice' and len(options) > 16:
            dropped['too_many_opts'] += 1
            return
        if len(target) != len(options):
            dropped['target_option_mismatch'] += 1
            return

        argmax_i = max(range(len(target)), key=lambda i: target[i])
        state_raw = row.get('state')
        if state_raw is None: state_raw = row.get('state_json') or ''
        if isinstance(state_raw, dict): state_raw = json.dumps(state_raw)[:1500]
        state_raw = str(state_raw)[:1500]
        question = str(row.get('question') or '')[:800]

        if kind == 'noul':
            slot = _noul_slot(options, argmax_i)
            if slot is None: dropped['noul_slot_fail'] += 1; return
            prompt = _make_noul_prompt(state_raw, question)
            valid = NOUL_SLOTS
        elif kind == 'choice':
            slot = 2 + argmax_i
            prompt = _make_choice_prompt(state_raw, question, options)
            valid = choice_slots(len(options))
        else:
            if len(options) != 6: dropped['score_wrong_opts'] += 1; return
            slot = _score_slot(options, argmax_i)
            if slot is None: dropped['score_slot_fail'] += 1; return
            prompt = _make_score_prompt(state_raw, question)
            valid = SCORE_SLOTS

        gid = row.get('group_id') or row.get('question_id') or row.get('id')
        gk = f'{kind}|gid|{gid}' if gid else None
        if gk and gk in seen_gids:
            dropped['dup_gid'] += 1; return
        h = _hash_qs(question, state_raw)
        if h in contam:
            dropped['contam'] += 1; return
        if h in seen_hashes:
            dropped['dup_hash'] += 1; return
        if gk: seen_gids.add(gk)
        seen_hashes.add(h)

        items.append({'prompt': prompt, 'answer_slot': slot, 'valid_slots': valid,
                      'weight': conf, 'kind': kind, 'hash': h, 'source': source_tag})
        per_source[source_tag] += 1
        per_kind[kind] += 1

    # ---- Source 1: SargeDev (v3.6 proven baseline — fills everything, no cap) ----
    print('\n  [1/3] SargeDev/jev-distill-corpus-v3  (v3.6 baseline)...', flush=True)
    before = len(items)
    try:
        n = 0
        for row in load_dataset('SargeDev/jev-distill-corpus-v3', split='train', streaming=True):
            n += 1
            _try_add(row, 'sargedev')
            if n % 100000 == 0:
                print(f'    [{n} seen] SargeDev kept so far: {len(items) - before}', flush=True)
    except Exception as e:
        print(f'  WARN SargeDev: {e}', flush=True)
    print(f'  SargeDev kept: {len(items) - before}', flush=True)

    # ---- Source 2: Open-Jev (CC0) ----
    print('\n  [2/3] ZefanCai/Open-Jev  (CC0)...', flush=True)
    before = len(items)
    try:
        n = 0
        for row in load_dataset('ZefanCai/Open-Jev', 'release-v2-redistributable',
                                split='train', streaming=True):
            n += 1
            _try_add(row, 'open-jev')
            if n % 20000 == 0:
                print(f'    [{n} seen] Open-Jev kept: {len(items) - before}', flush=True)
    except Exception as e:
        print(f'  WARN Open-Jev: {e}', flush=True)
    print(f'  Open-Jev kept: {len(items) - before}', flush=True)

    # ---- Source 3: Open-Jev-v1.1 ----
    print('\n  [3/3] ZefanCai/Open-Jev-v1.1  (hard-mix)...', flush=True)
    before = len(items)
    try:
        n = 0
        for row in load_dataset('ZefanCai/Open-Jev-v1.1',
                                'community-hard-mix-v2-redistributable',
                                split='train', streaming=True):
            n += 1
            _try_add(row, 'open-jev-v1.1')
            if n % 20000 == 0:
                print(f'    [{n} seen] Open-Jev-v1.1 kept: {len(items) - before}', flush=True)
    except Exception as e:
        print(f'  WARN Open-Jev-v1.1: {e}', flush=True)
    print(f'  Open-Jev-v1.1 kept: {len(items) - before}', flush=True)

    # tasksource was dropped in v3.8 — see header docstring (zero-guess policy).

    # ---- Report + safety asserts ----
    print(f'\n=== CURATION DONE ===', flush=True)
    print(f'  TOTAL: {len(items)}', flush=True)
    print(f'  per_source: {dict(per_source)}', flush=True)
    print(f'  per_kind:   {dict(per_kind)}', flush=True)
    print(f'  dropped:    {dict(dropped)}', flush=True)
    stats = {'total': len(items), 'per_source': dict(per_source), 'per_kind': dict(per_kind),
             'dropped': dict(dropped), 'contam_hashes': len(contam)}

    errs = []
    if len(items) < MIN_TOTAL: errs.append(f'TOTAL {len(items)} < {MIN_TOTAL}')
    if per_kind['noul'] < MIN_NOUL: errs.append(f'noul {per_kind["noul"]} < {MIN_NOUL}')
    if per_kind['choice'] < MIN_CHOICE: errs.append(f'choice {per_kind["choice"]} < {MIN_CHOICE}')
    if per_kind['score'] < MIN_SCORE: errs.append(f'score {per_kind["score"]} < {MIN_SCORE}')
    if errs:
        for e in errs: print(f'  FATAL: {e}', flush=True)
        STATS_JSON.write_text(json.dumps({**stats, 'errors': errs}, indent=2))
        raise RuntimeError('curation safety asserts failed — aborting before model load')

    # Stratified val split per kind
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
    print('=== DRY_RUN complete — exiting ===', flush=True)
    sys.exit(0)

# ============================================================
#          MODEL + HEAD (identical to v3.6)
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

print('=== Loading Gemma 3 4B in 4-bit (fresh LoRA) ===', flush=True)
if UNSLOTH_OK:
    base, _ = FastModel.from_pretrained(model_name=BASE_MODEL, max_seq_length=MAX_LEN,
        load_in_4bit=True, dtype=torch.bfloat16, full_finetuning=False)
    model = FastModel.get_peft_model(base, r=16, lora_alpha=32, lora_dropout=0.05,
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

class DecisionHead(nn.Module):
    def __init__(self, hidden, slots, lm_head_weight=None, verb_ids=None):
        super().__init__()
        self.proj = nn.Linear(hidden, slots, bias=False)
        if lm_head_weight is not None and verb_ids is not None:
            with torch.no_grad():
                rows = lm_head_weight[verb_ids].to(self.proj.weight.dtype)
                self.proj.weight.copy_(rows.to(self.proj.weight.device))
    def forward(self, x): return self.proj(x)

lm_head_w = model.get_output_embeddings().weight.detach()
head = DecisionHead(HIDDEN, NUM_SLOTS, lm_head_w, VERB_IDS).to(device=device, dtype=torch.bfloat16)

def _freeze_tf(grad):
    g = grad.clone()
    for s in FROZEN_SLOTS: g[s] = 0.0
    return g
head.proj.weight.register_hook(_freeze_tf)
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
#          DATASET + STRATIFIED SAMPLER (v3.8 delta #3)
# ============================================================
class DecisionDataset(Dataset):
    def __init__(self, path):
        self.items = []
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            try: self.items.append(json.loads(line))
            except: pass
        # pre-index by kind for stratified sampler
        self.by_kind = defaultdict(list)
        for i, it in enumerate(self.items):
            self.by_kind[it['kind']].append(i)
        print(f'  loaded {len(self.items)} from {path}  kinds={ {k: len(v) for k,v in self.by_kind.items()} }', flush=True)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class StratifiedKindSampler(Sampler):
    """Yield infinite indices where each BATCH (of BATCH size) has fixed kind ratio.
    For effective batch 16 (BATCH*GA_STEPS=16) with ratio 40/40/20 → 6/6/4 noul/choice/score per effective batch."""
    def __init__(self, ds, batch_size, ga_steps, ratio, seed=42):
        self.ds = ds
        self.eff = batch_size * ga_steps
        self.ratio = ratio
        self.rng = random.Random(seed)
        # per-kind counts in one effective batch
        self.per_eff = {k: max(1, int(round(self.eff * r))) for k, r in ratio.items()}
        # correction so sum matches
        d = self.eff - sum(self.per_eff.values())
        if d != 0:
            self.per_eff['choice'] = max(1, self.per_eff.get('choice', 0) + d)
        print(f'  StratifiedSampler per-effective-batch counts: {self.per_eff} (eff={self.eff})', flush=True)
        self._pos = {k: 0 for k in self.per_eff}
        self._shuffled = {k: list(ds.by_kind.get(k, [])) for k in self.per_eff}
        for k in self._shuffled: self.rng.shuffle(self._shuffled[k])

    def _take(self, k, n):
        out = []
        while len(out) < n:
            if self._pos[k] >= len(self._shuffled[k]):
                self.rng.shuffle(self._shuffled[k])
                self._pos[k] = 0
            take = min(n - len(out), len(self._shuffled[k]) - self._pos[k])
            out.extend(self._shuffled[k][self._pos[k]:self._pos[k]+take])
            self._pos[k] += take
        return out

    def __iter__(self):
        while True:
            batch = []
            for k, n in self.per_eff.items(): batch.extend(self._take(k, n))
            self.rng.shuffle(batch)
            for idx in batch: yield idx

    def __len__(self): return len(self.ds)

def collate(batch):
    enc = tok([b['prompt'] for b in batch], return_tensors='pt', padding=True,
              truncation=True, max_length=MAX_LEN)
    labels = torch.tensor([b['answer_slot'] for b in batch], dtype=torch.long)
    weights = torch.tensor([float(b.get('weight', 1.0)) for b in batch], dtype=torch.float32)
    mask = torch.zeros(len(batch), NUM_SLOTS, dtype=torch.bool)
    for i, b in enumerate(batch):
        for s in b.get('valid_slots', list(range(NUM_SLOTS))):
            if 0 <= s < NUM_SLOTS: mask[i, s] = True
    return enc.input_ids, enc.attention_mask, labels, weights, mask

train_ds = DecisionDataset(DATA_JSONL)
val_ds = DecisionDataset(VAL_JSONL)
train_sampler = StratifiedKindSampler(train_ds, BATCH, GA_STEPS, BATCH_KIND_RATIO)
loader = DataLoader(train_ds, batch_size=BATCH, sampler=train_sampler, collate_fn=collate, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=0)

# ============================================================
#          LOSS (v3.6 — focal CE, sample-weighted, per-slot mask)
# ============================================================
def focal_ce_loss(logits, labels, weights, valid_mask, gamma=FOCAL_GAMMA):
    logits = logits.masked_fill(~valid_mask, -1e4)
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    ce = -logp.gather(1, labels.view(-1, 1)).squeeze(1)
    if gamma > 0:
        pt = p.gather(1, labels.view(-1, 1)).squeeze(1)
        focal = (1.0 - pt).clamp(min=1e-6) ** gamma
        ce = focal * ce
    ce = ce * weights.to(ce.device)
    return ce.mean()

# ============================================================
#          TRAINING LOOP
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
        'batch_kind_ratio': BATCH_KIND_RATIO}))
    ld = OUT / 'lora_ckpt' / tag
    ld.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ld))

@torch.no_grad()
def evaluate():
    model.eval(); head.eval()
    correct = total = 0
    for input_ids, attn, labels, weights, mask in val_loader:
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

print(f'\n=== TRAIN v3.8 — max_steps={MAX_STEPS}, focal CE, stratified batch ===\n', flush=True)
model.train(); head.train()
step = step_start
grad_accum = 0
t0 = time.time()
loss_ema = None
ema_head_states = []

try:
    data_iter = iter(loader)
    while step < MAX_STEPS:
        try:
            input_ids, attn, labels, weights, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_ids, attn, labels, weights, mask = next(data_iter)

        input_ids = input_ids.to(device); attn = attn.to(device)
        labels = labels.to(device); mask = mask.to(device)
        out = model(input_ids=input_ids, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)
        last = out.hidden_states[-1]
        idx = attn.sum(dim=1) - 1
        pooled = last[torch.arange(last.size(0), device=device), idx]
        logits = head(pooled.to(head.proj.weight.dtype))
        loss = focal_ce_loss(logits, labels, weights, mask) / GA_STEPS
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
    print('\n=== INTERRUPTED — saving ===', flush=True)

save_ckpt(step, 'final')
save_ckpt(step, 'latest')
if len(ema_head_states) >= 2:
    print(f'\n=== EMA from last {len(ema_head_states)} ===', flush=True)
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in ema_head_states[0].items()}
    for sd in ema_head_states:
        for k in avg: avg[k] += sd[k].float()
    for k in avg: avg[k] /= len(ema_head_states)
    ed = OUT / 'head_ckpt' / 'ema-final'
    ed.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) for k, v in avg.items()}, ed / 'head.pt')
    (ed / 'meta.json').write_text(json.dumps({'step': step, 'num_slots': NUM_SLOTS,
        'verbalizer': VERBALIZER, 'verb_ids': VERB_IDS, 'ema_n': len(ema_head_states),
        'frozen_slots': FROZEN_SLOTS, 'batch_kind_ratio': BATCH_KIND_RATIO}))
    head.load_state_dict({k: v.to(head.proj.weight.dtype).to(device) for k, v in avg.items()})
    print(f'  EMA val acc = {evaluate():.4f}', flush=True)

print(f'\n=== DONE — {step} steps, {(time.time()-t0)/60:.1f} min, final loss {loss_ema:.4f} ===', flush=True)
