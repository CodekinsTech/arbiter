"""
Zyot Decider v3 — Minimal custom trainer for Kaggle T4x2.

Learned from Kev + simple-jev (avoids both's specific dependencies):
  - No PromptCompiler (simple-jev) → no chat_template requirement
  - No pointer head (Kev) → no Qwen-specific block-causal mask
  - Direct MMLU-style prompt: state + question + labeled options + "Answer:"
  - Loss: KL(soft_target || softmax(logits at answer position, restricted to option tokens))

This IS the AutoTrust JEV-27B recipe, minus the 24-slot verbalizer head:
  - Frozen base + LoRA r=16
  - Soft-label distillation loss
  - Restricted logit scoring

Reference: SargeDev/jev-distill-corpus-v3 target_dist is Jev-teacher soft labels.
Expected: pilot 20 steps < 5 min, full 1 epoch on 154k rows < 8h on T4x2.

Kaggle: New notebook, GPU T4x2, Internet ON, attach nishanml/zyot-decider-v3-curated.
"""
import os, json, sys, subprocess
from pathlib import Path

# ---- config ---------------------------------------------------------------
BASE_MODEL = 'unsloth/gemma-3-4b-it'
_CANDIDATES = [
    Path('/kaggle/input/zyot-decider-v3-curated'),
    Path('/kaggle/input/datasets/nishanml/zyot-decider-v3-curated'),
]
DATA_DIR = next((p for p in _CANDIDATES if (p / 'train.jsonl').exists()), _CANDIDATES[-1])
OUT_DIR = Path('/kaggle/working/zyot-v3-minimal')

# LoRA (AutoTrust config)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training — v11: bf16 (not 4-bit), shorter seq, smaller data for feasibility on T4
LR = 5e-5
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4               # effective batch = 16
EPOCHS = 1
MAX_SEQ = 256                # SargeDev states are short; 256 covers ~90%, huge compute savings
PILOT_STEPS = 10
TRAIN_LIMIT = 10_000         # subset for single-session feasibility on T4 (~2-3h total)
SAVE_STEPS = 200
LOG_STEPS = 10

os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# QLoRA + HF Trainer DataParallel is known-flaky (CUBLAS crashes). Single T4 is fine.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def sh(cmd, check=True):
    print(f'\n$ {cmd}', flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0: sys.exit(r.returncode)


# =========== 1. Preflight ===========
print('=== Zyot Decider v3 — Minimal Trainer ===\n', flush=True)
sh('nvidia-smi | head -12')
if not DATA_DIR.exists() or not (DATA_DIR / 'train.jsonl').exists():
    print(f'ERROR: data not found. Tried: {_CANDIDATES}', file=sys.stderr); sys.exit(1)
sh(f'ls -la {DATA_DIR}')

# =========== 2. Install ===========
# CRITICAL: uninstall Kaggle's baked-in torchao 0.10 — incompatible with peft
sh('pip uninstall -y torchao 2>/dev/null || true', check=False)
# Pin to stable 4.x — transformers 5.x is dev/alpha and renamed some TrainingArguments
sh('pip install -q "transformers>=4.44,<5" "peft>=0.11,<1" "accelerate>=0.30,<2" '
   'datasets safetensors hf_transfer')

import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

print(f'\ntorch {torch.__version__}  |  CUDA {torch.cuda.is_available()}  |  '
      f'GPUs {torch.cuda.device_count()}', flush=True)

# =========== 3. Load base 4-bit + LoRA ===========
print(f'\nLoading tokenizer + model: {BASE_MODEL}', flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# v11: bf16 (not 4-bit) — 4B in bf16 = 8GB, fits T4 16GB. Removes 2x QLoRA overhead.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, attn_implementation='eager',
)
model.gradient_checkpointing_enable()
model = get_peft_model(model, LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias='none', task_type='CAUSAL_LM',
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
))
model.print_trainable_parameters()

# =========== 4. Encode dataset ===========
# Prompt format (MMLU-style, single-turn, no chat template dependency):
#   State: {state}
#   Question: {question}
#   Options:
#   A. opt0
#   B. opt1
#   ...
#   Answer:
# The answer position gets a single-token label (A/B/C.../0/1/2...). Loss is KL over
# the specific option-token IDs vs the soft target distribution.

LETTERS_CHOICE = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']  # 16 max
DIGITS_SCORE = ['0','1','2','3','4','5']
NOUL_LETTERS = ['T','F']  # true/false → T/F

# Precompute single-token IDs for verbalizers
def _tok1(s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None

_letter_ids = {L: _tok1(L) for L in LETTERS_CHOICE}
_digit_ids = {d: _tok1(d) for d in DIGITS_SCORE}
_bool_ids = {b: _tok1(b) for b in NOUL_LETTERS}
for m, missing in [
    ('choice-letters', {L for L,i in _letter_ids.items() if i is None}),
    ('score-digits', {d for d,i in _digit_ids.items() if i is None}),
    ('bool-letters', {b for b,i in _bool_ids.items() if i is None}),
]:
    if missing:
        raise RuntimeError(f'{m} not single-token in this tokenizer: {missing}')

def encode_record(rec):
    """Return (input_ids, allowed_ids, target_probs) for one record."""
    r = rec['request']
    qid, q = next(iter(r['questions'].items()))
    kind = q['type']
    state = r.get('state', '')
    instr = q.get('instructions', '')

    if kind == 'choice':
        criteria = q['criteria']  # dict opt_key -> description
        opt_keys = list(criteria.keys())
        letters = LETTERS_CHOICE[:len(opt_keys)]
        opts_block = '\n'.join(f'{L}. {desc}' for L, desc in zip(letters, criteria.values()))
        allowed = [_letter_ids[L] for L in letters]
        # target: key -> prob → letter -> prob
        probs = rec['targets'][qid]['probabilities']
        target = [float(probs.get(k, 0.0)) for k in opt_keys]
    elif kind == 'score':
        levels = q['criteria']  # list of level descriptions
        digits = DIGITS_SCORE[:len(levels)]
        opts_block = '\n'.join(f'{d}. {desc}' for d, desc in zip(digits, levels))
        allowed = [_digit_ids[d] for d in digits]
        probs = rec['targets'][qid]['probabilities']
        target = [float(probs.get(str(i), 0.0)) for i in range(len(digits))]
    elif kind == 'noul':
        opts_block = 'T. Yes / True\nF. No / False'
        allowed = [_bool_ids['T'], _bool_ids['F']]
        probs = rec['targets'][qid]['probabilities']
        # SargeDev noul stores as 'true'/'false' keys
        target = [float(probs.get('true', 0.0)), float(probs.get('false', 0.0))]
    else:
        return None

    # Normalize target
    s = sum(target)
    if s <= 0: return None
    target = [t / s for t in target]

    prompt = f'State: {state}\n\nQuestion: {instr}\n\nOptions:\n{opts_block}\n\nAnswer:'
    ids = tok(prompt, add_special_tokens=True, truncation=True, max_length=MAX_SEQ - 1).input_ids
    return {'input_ids': ids, 'allowed_ids': allowed, 'target': target}


print('\nEncoding datasets...', flush=True)

def load_all_encoded(path):
    """Load, encode, and tag each row with its (kind, family) stratum."""
    from collections import defaultdict
    rows_by_stratum = defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            enc = encode_record(r)
            if enc is None: continue
            meta = r.get('_meta', {})
            key = (meta.get('kind','?'), meta.get('family','?'))
            rows_by_stratum[key].append(enc)
    return rows_by_stratum

def stratified_sample(by_stratum, target_n, seed=42):
    """Proportional stratified sample — matches the pool's distribution."""
    import random as _rng
    _rng.seed(seed)
    total = sum(len(v) for v in by_stratum.values())
    if target_n >= total:
        out = [r for v in by_stratum.values() for r in v]
        _rng.shuffle(out); return out
    # Proportional allocation, at least 1 per stratum with data
    allocations = {}
    for k, v in by_stratum.items():
        allocations[k] = max(1, round(len(v) * target_n / total))
    # Adjust so total matches target_n
    diff = target_n - sum(allocations.values())
    keys_sorted = sorted(allocations, key=lambda k: -len(by_stratum[k]))
    i = 0
    while diff != 0 and i < len(keys_sorted) * 3:
        k = keys_sorted[i % len(keys_sorted)]
        if diff > 0:
            if allocations[k] < len(by_stratum[k]):
                allocations[k] += 1; diff -= 1
        else:
            if allocations[k] > 1:
                allocations[k] -= 1; diff += 1
        i += 1
    # Sample
    out = []
    for k, n in allocations.items():
        pool = by_stratum[k]
        out.extend(_rng.sample(pool, min(n, len(pool))))
    _rng.shuffle(out)
    return out

print('  loading train pool...', flush=True)
train_pool = load_all_encoded(DATA_DIR / 'train.jsonl')
pool_total = sum(len(v) for v in train_pool.values())
print(f'  train pool: {pool_total:,} rows across {len(train_pool)} (kind, family) strata', flush=True)
train_rows = stratified_sample(train_pool, TRAIN_LIMIT)
print(f'  stratified sample: {len(train_rows):,} rows', flush=True)

# validation: keep small random slice
val_pool = load_all_encoded(DATA_DIR / 'validation.jsonl')
val_rows = stratified_sample(val_pool, 300)
print(f'  validation: {len(val_rows):,} rows', flush=True)

# =========== 5. Collator + custom Trainer ===========
class DecisionCollator:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, rows):
        n = len(rows)
        width = max(len(r['input_ids']) for r in rows)
        n_opts = max(len(r['allowed_ids']) for r in rows)
        ids = torch.full((n, width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        allowed = torch.zeros((n, n_opts), dtype=torch.long)
        target = torch.zeros((n, n_opts), dtype=torch.float32)
        allowed_mask = torch.zeros((n, n_opts), dtype=torch.bool)
        for i, r in enumerate(rows):
            L = len(r['input_ids']); K = len(r['allowed_ids'])
            ids[i, :L] = torch.tensor(r['input_ids'])
            mask[i, :L] = 1
            allowed[i, :K] = torch.tensor(r['allowed_ids'])
            target[i, :K] = torch.tensor(r['target'])
            allowed_mask[i, :K] = True
        return {'input_ids': ids, 'attention_mask': mask,
                'allowed_ids': allowed, 'target': target, 'allowed_mask': allowed_mask}


class DecisionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], use_cache=False)
        # last real position per row
        pos = inputs['attention_mask'].sum(dim=1) - 1
        final = out.logits[torch.arange(len(pos), device=pos.device), pos]  # (B, V)
        # gather option-token logits
        sel = final.gather(1, inputs['allowed_ids']).float()  # (B, K)
        sel = sel.masked_fill(~inputs['allowed_mask'], -1e9)
        logp = F.log_softmax(sel, dim=-1)
        loss = -(inputs['target'] * logp).sum(dim=-1).mean()
        return (loss, {'logits': sel}) if return_outputs else loss


# =========== 6. Train ===========
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- PILOT: 20 steps on 200 rows first ---
print('\n=== PILOT: 20 steps on 200 rows ===', flush=True)
pilot_train = train_rows[:200]
args = TrainingArguments(
    output_dir=str(OUT_DIR / 'pilot'),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LR, max_steps=PILOT_STEPS,
    logging_steps=5, save_strategy='no', bf16=True,
    remove_unused_columns=False,
    report_to='none', seed=42,
)
tr = DecisionTrainer(model=model, args=args,
                     train_dataset=Dataset.from_list(pilot_train),
                     data_collator=DecisionCollator(tok.pad_token_id))
tr.train()
print('\n=== PILOT SUCCESS. Loss curve above should be decreasing. ===', flush=True)

# --- FULL: 1 epoch ---
print('\n=== FULL: 1 epoch on all data ===', flush=True)
full_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LR, num_train_epochs=EPOCHS,
    logging_steps=LOG_STEPS,
    save_strategy='steps', save_steps=SAVE_STEPS, save_total_limit=3,
    bf16=True, remove_unused_columns=False,
    report_to='none', seed=42, lr_scheduler_type='cosine',
)
full_tr = DecisionTrainer(model=model, args=full_args,
                          train_dataset=Dataset.from_list(train_rows),
                          data_collator=DecisionCollator(tok.pad_token_id))
full_tr.train()

# =========== 7. Save adapter ===========
model.save_pretrained(OUT_DIR / 'final')
tok.save_pretrained(OUT_DIR / 'final')
print(f'\nAdapter + tokenizer saved to {OUT_DIR / "final"}')
sh(f'du -sh {OUT_DIR / "final"}')
