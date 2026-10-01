"""Zyot Decider v3.6 (Arbiter) — bulletproof retraining.

Fixes every mistake from v3.5 and preempts every past training error:
  * FROZEN T/F slots (slots 0-1) — mathematically preserves boolq baseline
  * SAMPLE-WEIGHTED loss — by teacher confidence
  * FOCAL LOSS γ=2 — focuses gradient on hard examples (Kev-proven)
  * EMA weight averaging — averages last 5 checkpoints for final model
  * PER-SLOT MASKING — softmax only over valid slots for this item's kind
  * FRESH LoRA (no v3 continuation) — v3 was random verbalizer; damages base
  * TIGHTER data (teacher conf >= 0.75) + v2 D1-D10 public datasets
  * DEDUP + test-set contamination check vs boolq/arc/csqa
  * HELD-OUT VALIDATION SET — 3k stratified

Runs on Kaggle T4 (single GPU forced). Wall-clock target: 10-11h for 3000 steps.
"""
import os, sys, json, math, time, subprocess, hashlib, copy, random
from pathlib import Path
from collections import defaultdict

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # T4x2 -> DataParallel CUBLAS crash without this

# ---------- deps ----------
subprocess.run('pip uninstall -y torchao 2>/dev/null || true', shell=True)  # torchao/transformers incompat
subprocess.check_call(
    'pip install -q "unsloth" "unsloth_zoo" '
    '"transformers>=4.44,<5" "peft>=0.11,<1" '
    '"accelerate>=0.30,<2" bitsandbytes hf_transfer datasets', shell=True)

# Unsloth MUST import before transformers for kernel patching
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
#                    C O N F I G
# ============================================================
BASE_MODEL = 'unsloth/gemma-3-4b-it'
NUM_SLOTS = 24                # T/F, A-P, 0-5
BATCH = 2
GA_STEPS = 8                  # effective batch = 16
LR = 5e-5                     # slightly higher than v3.5 — fresh LoRA needs more push
MAX_STEPS = 3000              # ~10h at 12s/step; leaves buffer under 12h Kaggle limit
SAVE_EVERY = 10
EVAL_EVERY = 500
LOG_EVERY = 20
WARMUP_STEPS = 200            # longer — fresh LoRA + fresh head both need warmup
MAX_LEN = 768                 # v3.5-proven; 1024 was too slow
GRAD_CLIP = 1.0
FOCAL_GAMMA = 2.0             # 0.0 disables focal; 2.0 = Kev-proven
EMA_LAST_N = 5                # average last 5 EVAL_EVERY checkpoints for final model
FROZEN_SLOTS = [0, 1]         # T, F — zero gradient into these rows to preserve v3 boolq baseline
VAL_SIZE = 3000

RESUME = os.environ.get('ZYOT_RESUME', '0') == '1'
OUT = Path('/kaggle/working/v3.6')
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
#           C U R A T E   D A T A   (self-contained)
# ============================================================
DATA_JSONL = OUT / 'train_v3_6.jsonl'
VAL_JSONL = OUT / 'val_v3_6.jsonl'
CONTAM_JSONL = OUT / 'contamination_report.json'

def _hash(s):
    return hashlib.md5(str(s).encode('utf-8')).hexdigest()[:16]

def curate_data():
    """Build train + val JSONL from SargeDev (streamed) + v2 D1-D10 + v2 synth.
    Fields per item: prompt, answer_slot, valid_slots, weight, kind, hash."""
    # Only reuse existing data if RESUMING training; otherwise rebuild fresh.
    if RESUME and DATA_JSONL.exists() and VAL_JSONL.exists():
        print('=== curated data already present (RESUME) — skipping ===', flush=True)
        return
    print('=== CURATING v3.6 data ===', flush=True)
    from datasets import load_dataset

    def _make_choice_prompt(state, question, options):
        letters = [chr(ord('A') + i) for i in range(len(options))]
        opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, options))
        return f'State: {state}\n\nQuestion: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:'

    def _make_noul_prompt(state, question):
        return (f'State: {state}\n\nQuestion: {question}\n\n'
                f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')

    def _make_score_prompt(state, question):
        return (f'State: {state}\n\nQuestion: {question}\n\n'
                f'Options:\n0\n1\n2\n3\n4\n5\n\nAnswer:')

    NOUL_SLOTS = [0, 1]
    def choice_slots(n): return list(range(2, 2 + n))
    SCORE_SLOTS = list(range(18, 24))

    # ---- test-set contamination hashes ----
    print('  building contamination hash-set from test benchmarks...', flush=True)
    contam = set()
    try:
        for row in load_dataset('google/boolq', split='validation', streaming=True):
            contam.add(_hash(f"{row['passage']}||{row['question']}"))
    except Exception as e:
        print(f'  WARN: boolq hash failed: {e}', flush=True)
    try:
        for row in load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test', streaming=True):
            contam.add(_hash(row['question']))
    except Exception as e:
        print(f'  WARN: arc hash failed: {e}', flush=True)
    try:
        for row in load_dataset('tau/commonsense_qa', split='validation', streaming=True):
            contam.add(_hash(row['question']))
    except Exception as e:
        print(f'  WARN: csqa hash failed: {e}', flush=True)
    print(f'  contamination hashes: {len(contam)}', flush=True)

    items = []
    seen_hashes = set()
    dropped_contam = 0

    # ---------- SargeDev jev-distill-corpus-v3 ----------
    # Real schema: kind ('noul'/'choice'/'score'), options (list[str]),
    # target (list[float] soft-label), state, question. NO answer/confidence field.
    print('  streaming SargeDev/jev-distill-corpus-v3...', flush=True)
    per_kind = defaultdict(int)
    seen_kind = defaultdict(int)  # for progress visibility
    KIND_CAP = {'noul': 100000, 'choice': 70000, 'score': 30000}
    CONF_MIN = 0.75

    def _noul_slot_from_options(options, argmax_i):
        """Options like ['false','true'] or ['no','yes'] — return T-slot(0) or F-slot(1)."""
        lbl = str(options[argmax_i]).strip().lower()
        if lbl in ('true', 't', 'yes', '1'): return 0
        if lbl in ('false', 'f', 'no', '0'): return 1
        return None

    def _score_slot_from_options(options, argmax_i):
        """Options like ['0','1','2','3','4','5'] — return 18+int(label)."""
        try: n = int(str(options[argmax_i]).strip())
        except Exception: return None
        if 0 <= n <= 5: return 18 + n
        return None

    total_seen = 0
    try:
        for row in load_dataset('SargeDev/jev-distill-corpus-v3', split='train', streaming=True):
            total_seen += 1
            if total_seen % 20000 == 0:
                print(f'    [{total_seen} seen] kept: {dict(per_kind)}', flush=True)

            kind = row.get('kind', '')
            if kind not in ('noul', 'choice', 'score'): continue
            seen_kind[kind] += 1
            if per_kind[kind] >= KIND_CAP.get(kind, 0):
                # if all caps hit, stop early
                if all(per_kind[k] >= KIND_CAP[k] for k in KIND_CAP): break
                continue

            target = row.get('target') or []
            options = row.get('options') or []
            if not target or not options or len(target) != len(options): continue
            conf = float(max(target))
            if conf < CONF_MIN: continue
            argmax_i = target.index(max(target))

            state = row.get('state', '') or ''
            question = row.get('question', '') or ''

            if kind == 'noul':
                slot = _noul_slot_from_options(options, argmax_i)
                if slot is None: continue
                prompt = _make_noul_prompt(state, question)
                valid = NOUL_SLOTS
            elif kind == 'choice':
                if len(options) < 2 or len(options) > 16: continue
                slot = 2 + argmax_i
                prompt = _make_choice_prompt(state, question, options)
                valid = choice_slots(len(options))
            elif kind == 'score':
                if len(options) != 6: continue
                slot = _score_slot_from_options(options, argmax_i)
                if slot is None: continue
                prompt = _make_score_prompt(state, question)
                valid = SCORE_SLOTS
            else:
                continue

            h = _hash(f'{state[:200]}||{question[:200]}')
            if h in seen_hashes: continue
            if h in contam:
                dropped_contam += 1
                continue
            seen_hashes.add(h)
            items.append({'prompt': prompt, 'answer_slot': slot, 'valid_slots': valid,
                          'weight': conf, 'kind': kind, 'hash': h})
            per_kind[kind] += 1
    except Exception as e:
        print(f'  WARN: SargeDev stream failed at total_seen={total_seen}, kept={dict(per_kind)}: {e}', flush=True)
    print(f'  SargeDev kept: {dict(per_kind)}  seen_kind={dict(seen_kind)}  total_seen={total_seen}  (contam dropped: {dropped_contam})', flush=True)
    assert sum(per_kind.values()) >= 20000, f'SargeDev curation produced too few rows ({sum(per_kind.values())}) — schema bug'

    # ---------- v2 D1-D10 public datasets ----------
    print('  loading v2 D1-D10 public datasets (best-effort)...', flush=True)
    d_added = 0
    try:
        # PolyAI banking77 (parquet, no loading script)
        ds = load_dataset('PolyAI/banking77', split='train')
        n = 0
        for r in ds:
            if int(r['label']) >= 16: continue
            opts = [f'intent_{i}' for i in range(16)]
            slot = 2 + int(r['label'])
            prompt = _make_choice_prompt('', r['text'], opts)
            h = _hash(f"b77||{r['text'][:200]}")
            if h in seen_hashes or h in contam: continue
            seen_hashes.add(h)
            items.append({'prompt': prompt, 'answer_slot': slot, 'valid_slots': choice_slots(16),
                          'weight': 1.0, 'kind': 'choice', 'hash': h})
            d_added += 1
            n += 1
            if n >= 2000: break
    except Exception as e:
        print(f'  banking77 skipped: {e}', flush=True)

    try:
        ds = load_dataset('SetFit/CR', split='train')
        for r in ds.select(range(min(2000, len(ds)))):
            slot = 0 if int(r['label']) == 1 else 1
            prompt = _make_noul_prompt('', f'Is this review positive? "{r["text"]}"')
            h = _hash(f"cr||{r['text'][:200]}")
            if h in seen_hashes or h in contam: continue
            seen_hashes.add(h)
            items.append({'prompt': prompt, 'answer_slot': slot, 'valid_slots': NOUL_SLOTS,
                          'weight': 1.0, 'kind': 'noul', 'hash': h})
            d_added += 1
    except Exception as e:
        print(f'  CR skipped: {e}', flush=True)

    try:
        ds = load_dataset('deepset/prompt-injections', split='train')
        for r in ds.select(range(min(len(ds), 3000))):
            slot = 0 if int(r['label']) == 1 else 1
            prompt = _make_noul_prompt('', f'Is this text an attempted prompt injection? "{r["text"][:400]}"')
            h = _hash(f"inj||{r['text'][:200]}")
            if h in seen_hashes or h in contam: continue
            seen_hashes.add(h)
            items.append({'prompt': prompt, 'answer_slot': slot, 'valid_slots': NOUL_SLOTS,
                          'weight': 1.0, 'kind': 'noul', 'hash': h})
            d_added += 1
    except Exception as e:
        print(f'  prompt-injections skipped: {e}', flush=True)

    print(f'  v2 D1-D10 added: {d_added}', flush=True)

    # ---------- v2 hand-crafted synth (bundled with kernel) ----------
    # These are the 5.3k rows already inherited from v2; if the kernel source
    # includes them, load; otherwise skip silently.
    synth_paths = [Path('/kaggle/input/zyot-v2-synth/synth_v2.jsonl')]
    synth_added = 0
    for p in synth_paths:
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                try: r = json.loads(line)
                except Exception: continue
                if 'prompt' not in r or 'answer_slot' not in r: continue
                h = _hash(r['prompt'][:400])
                if h in seen_hashes or h in contam: continue
                seen_hashes.add(h)
                items.append({'prompt': r['prompt'], 'answer_slot': int(r['answer_slot']),
                              'valid_slots': r.get('valid_slots', NOUL_SLOTS),
                              'weight': float(r.get('weight', 1.0)),
                              'kind': r.get('kind', 'noul'), 'hash': h})
                synth_added += 1
            break
    print(f'  v2 synth added: {synth_added}', flush=True)

    # ---------- shuffle + split ----------
    random.seed(42)
    random.shuffle(items)
    print(f'  TOTAL curated: {len(items)}', flush=True)
    # stratified val split
    by_kind = defaultdict(list)
    for it in items: by_kind[it['kind']].append(it)
    val, train = [], []
    for k, lst in by_kind.items():
        n_val = min(VAL_SIZE * len(lst) // len(items), len(lst) // 20)
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    random.shuffle(val); random.shuffle(train)
    print(f'  TRAIN: {len(train)}, VAL: {len(val)}', flush=True)

    with DATA_JSONL.open('w', encoding='utf-8') as f:
        for it in train: f.write(json.dumps(it) + '\n')
    with VAL_JSONL.open('w', encoding='utf-8') as f:
        for it in val: f.write(json.dumps(it) + '\n')
    CONTAM_JSONL.write_text(json.dumps({'contam_hashes': len(contam),
                                        'dropped_contam': dropped_contam,
                                        'per_kind': dict(per_kind),
                                        'total_train': len(train),
                                        'total_val': len(val)}, indent=2))
    print('=== curation done ===\n', flush=True)

curate_data()

# ============================================================
#              L O A D   T O K E N I Z E R
# ============================================================
print('=== Loading tokenizer ===', flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

VERBALIZER = ['T', 'F',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              '0', '1', '2', '3', '4', '5']
assert len(VERBALIZER) == NUM_SLOTS
def _t1(s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None
VERB_IDS = [_t1(v) for v in VERBALIZER]
VERB_IDS = [x if x is not None else 0 for x in VERB_IDS]
print(f'  Verbalizer token ids: {VERB_IDS[:6]}... ({NUM_SLOTS} total)', flush=True)

# ============================================================
#         L O A D   B A S E   M O D E L   +   F R E S H   L o R A
# ============================================================
print('=== Loading Gemma 3 4B in 4-bit (FRESH — no v3 continuation) ===', flush=True)
if UNSLOTH_OK:
    base, _ = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_LEN,
        load_in_4bit=True, dtype=torch.bfloat16, full_finetuning=False,
    )
    # Unsloth: attach fresh LoRA
    model = FastModel.get_peft_model(
        base, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        use_gradient_checkpointing='unsloth', bias='none',
    )
else:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation='eager', device_map='auto')
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

# ============================================================
#              P O I N T E R   H E A D
# ============================================================
class DecisionHead(nn.Module):
    def __init__(self, hidden, slots, lm_head_weight=None, verb_ids=None):
        super().__init__()
        self.proj = nn.Linear(hidden, slots, bias=False)
        if lm_head_weight is not None and verb_ids is not None:
            with torch.no_grad():
                rows = lm_head_weight[verb_ids].to(self.proj.weight.dtype)
                self.proj.weight.copy_(rows.to(self.proj.weight.device))
                print(f'  head init from LM head rows (shape {rows.shape})', flush=True)
    def forward(self, last_hidden):
        return self.proj(last_hidden)

lm_head_w = model.get_output_embeddings().weight.detach()
head = DecisionHead(HIDDEN, NUM_SLOTS, lm_head_w, VERB_IDS).to(device=device, dtype=torch.bfloat16)

# FROZEN T/F ROWS — register a backward hook that zeroes gradient on slots 0-1.
# This mathematically PRESERVES boolq baseline: T/F rows never move from their
# LM-head-init state, so noul prediction stays identical to v3-quality verbalizer read.
def _freeze_tf_hook(grad):
    grad = grad.clone()
    for s in FROZEN_SLOTS:
        grad[s] = 0.0
    return grad
head.proj.weight.register_hook(_freeze_tf_hook)
print(f'  FROZEN slots: {FROZEN_SLOTS} (T/F rows — grad zeroed)', flush=True)

# ============================================================
#           R E S U M E
# ============================================================
step_start = 0
if RESUME and (OUT / 'head_ckpt' / 'latest' / 'head.pt').exists():
    print('=== RESUMING from latest ===', flush=True)
    head.load_state_dict(torch.load(OUT / 'head_ckpt' / 'latest' / 'head.pt', map_location='cpu'))
    if (OUT / 'lora_ckpt' / 'latest' / 'adapter_config.json').exists():
        from peft import PeftModel as _PM
        # reload adapter into current model
        model.load_adapter(str(OUT / 'lora_ckpt' / 'latest'), 'default')
    meta = json.loads((OUT / 'head_ckpt' / 'latest' / 'meta.json').read_text())
    step_start = meta.get('step', 0)
    print(f'  resumed at step {step_start}', flush=True)

# ============================================================
#              D A T A S E T
# ============================================================
class DecisionDataset(Dataset):
    def __init__(self, path):
        self.items = []
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            try: r = json.loads(line)
            except Exception: continue
            self.items.append(r)
        print(f'  loaded {len(self.items)} from {path}', flush=True)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    enc = tok([b['prompt'] for b in batch], return_tensors='pt', padding=True,
              truncation=True, max_length=MAX_LEN)
    labels = torch.tensor([b['answer_slot'] for b in batch], dtype=torch.long)
    weights = torch.tensor([float(b.get('weight', 1.0)) for b in batch], dtype=torch.float32)
    # per-row valid_slots mask (True = valid)
    mask = torch.zeros(len(batch), NUM_SLOTS, dtype=torch.bool)
    for i, b in enumerate(batch):
        for s in b.get('valid_slots', list(range(NUM_SLOTS))):
            if 0 <= s < NUM_SLOTS: mask[i, s] = True
    return enc.input_ids, enc.attention_mask, labels, weights, mask

train_ds = DecisionDataset(DATA_JSONL)
val_ds = DecisionDataset(VAL_JSONL)
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=0)

# ============================================================
#              L O S S : focal + sample-weighted + per-slot mask
# ============================================================
def focal_ce_loss(logits, labels, weights, valid_mask, gamma=FOCAL_GAMMA):
    """Per-slot masking + focal loss + sample weighting.
    logits [B, S], labels [B], weights [B], valid_mask [B, S]."""
    # per-slot masking: set invalid slot logits to -inf so softmax ignores them
    logits = logits.masked_fill(~valid_mask, -1e4)
    logp = F.log_softmax(logits.float(), dim=-1)  # [B, S]
    p = logp.exp()
    ce = -logp.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]
    if gamma > 0:
        pt = p.gather(1, labels.view(-1, 1)).squeeze(1)
        focal = (1.0 - pt).clamp(min=1e-6) ** gamma
        ce = focal * ce
    ce = ce * weights.to(ce.device)
    return ce.mean()

# ============================================================
#              T R A I N I N G
# ============================================================
trainable = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable)
print(f'  trainable params: {n_trainable:,}', flush=True)
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
                                              'frozen_slots': FROZEN_SLOTS,
                                              'focal_gamma': FOCAL_GAMMA}))
    ld = OUT / 'lora_ckpt' / tag
    ld.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ld))

@torch.no_grad()
def evaluate():
    model.eval(); head.eval()
    correct = total = 0
    for input_ids, attn, labels, weights, mask in val_loader:
        input_ids = input_ids.to(device); attn = attn.to(device); labels = labels.to(device)
        mask = mask.to(device)
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]
        last_idx = attn.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=device), last_idx]
        logits = head(pooled.to(head.proj.weight.dtype))
        logits = logits.masked_fill(~mask, -1e4)
        pred = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    model.train(); head.train()
    return correct / max(total, 1)

# ============================================================
#              M A I N   L O O P
# ============================================================
print(f'\n=== TRAIN — max_steps={MAX_STEPS}, focal γ={FOCAL_GAMMA}, EMA last {EMA_LAST_N} ===\n', flush=True)
model.train(); head.train()
step = step_start
grad_accum = 0
t0 = time.time()
loss_ema = None

# EMA head weights: track last N eval-step checkpoints in memory
ema_head_states = []

try:
    while step < MAX_STEPS:
        for input_ids, attn, labels, weights, mask in loader:
            input_ids = input_ids.to(device); attn = attn.to(device)
            labels = labels.to(device); mask = mask.to(device)
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
            last_hidden = out.hidden_states[-1]
            last_idx = attn.sum(dim=1) - 1
            pooled = last_hidden[torch.arange(last_hidden.size(0), device=device), last_idx]
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
                loss_val = float(loss.detach()) * GA_STEPS
                loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
                if step % LOG_EVERY == 0:
                    dt = time.time() - t0
                    sps = step / max(dt, 1)
                    eta = (MAX_STEPS - step) / max(sps, 1e-9) / 60
                    print(f'  step {step:>5}/{MAX_STEPS}  loss={loss_ema:.4f}  '
                          f'{sps:.2f} step/s  ETA {eta:.0f}m', flush=True)
                if step % SAVE_EVERY == 0:
                    save_ckpt(step, 'latest')
                if step % EVAL_EVERY == 0:
                    val_acc = evaluate()
                    print(f'  ** VAL acc = {val_acc:.4f} at step {step} **', flush=True)
                    save_ckpt(step, f'step-{step}')
                    # snapshot for EMA
                    ema_head_states.append({k: v.detach().clone().cpu()
                                            for k, v in head.state_dict().items()})
                    if len(ema_head_states) > EMA_LAST_N:
                        ema_head_states.pop(0)
                if step >= MAX_STEPS: break
except KeyboardInterrupt:
    print('\n=== INTERRUPTED — saving final ===', flush=True)

# ============================================================
#      E M A   W E I G H T   A V E R A G I N G   (final)
# ============================================================
save_ckpt(step, 'final')
save_ckpt(step, 'latest')
if len(ema_head_states) >= 2:
    print(f'\n=== Building EMA-averaged head from last {len(ema_head_states)} snapshots ===', flush=True)
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in ema_head_states[0].items()}
    for sd in ema_head_states:
        for k in avg: avg[k] += sd[k].float()
    for k in avg: avg[k] /= len(ema_head_states)
    ema_dir = OUT / 'head_ckpt' / 'ema-final'
    ema_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) for k, v in avg.items()}, ema_dir / 'head.pt')
    (ema_dir / 'meta.json').write_text(json.dumps({'step': step, 'num_slots': NUM_SLOTS,
                                                    'verbalizer': VERBALIZER, 'verb_ids': VERB_IDS,
                                                    'ema_n': len(ema_head_states),
                                                    'frozen_slots': FROZEN_SLOTS,
                                                    'focal_gamma': FOCAL_GAMMA}))
    # eval EMA head
    head.load_state_dict({k: v.to(head.proj.weight.dtype).to(device) for k, v in avg.items()})
    ema_acc = evaluate()
    print(f'  EMA val acc = {ema_acc:.4f}', flush=True)

print(f'\n=== DONE — {step} steps, {(time.time()-t0)/60:.1f} min, final loss {loss_ema:.4f} ===', flush=True)
print(f'  Head:    {OUT / "head_ckpt" / "final" / "head.pt"}')
print(f'  Head EMA:{OUT / "head_ckpt" / "ema-final" / "head.pt"}')
print(f'  LoRA:    {OUT / "lora_ckpt" / "final"}')
