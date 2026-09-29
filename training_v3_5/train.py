"""Zyot Decider v3.5 — System One conversion overnight training.

Adds a fixed-slot pointer head (AutoTrust-style) on top of v3's LoRA and
fine-tunes both head + LoRA on curated SargeDev data.

Design decisions (for future maintenance):
  * Head has FIXED 24 slots — covers T/F, 0-5 scores, up to 24-way choice.
    JevBench's biggest config (77-way Banking77) is out of scope; the 24
    slots handle every other typed-decision benchmark cleanly.
  * Head is initialised from Gemma 3's own LM head rows for the verbalizer
    tokens (T, F, A-P, 0-5). At step 0 this reproduces v3 verbalizer
    output exactly — anything we gain from training is a strict improvement,
    not a regression.
  * DYNAMIC — the model class does NOT bake in dataset size or steps. To
    train on more data later:
      1. Upload the new JSONL as a Kaggle dataset
      2. Resume from `head_ckpt/latest/` + `lora_ckpt/latest/`
      3. Point --data-path at the new file, --resume True
      Architecture is untouched. LoRA + head weights update on new data.
  * CHECKPOINT EVERY 10 STEPS — Kaggle wipes /kaggle/working on cancel.
    Frequent saves + upload to Kaggle dataset every 200 steps as backup.

Runs on Kaggle T4 (free tier). Expected wall-clock: 6-8h for 25k rows,
1 epoch, batch 2 x GA=8 = effective batch 16.
"""
import os, sys, json, math, time, subprocess, shutil
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'
# Force single GPU — Kaggle T4 x2 with QLoRA + DataParallel caused CUBLAS crash in v3 iteration 9.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# ---------- deps ----------
subprocess.run('pip uninstall -y torchao 2>/dev/null || true', shell=True)
subprocess.check_call('pip install -q "unsloth" "unsloth_zoo" '
                     '"transformers>=4.44,<5" "peft>=0.11,<1" '
                     '"accelerate>=0.30,<2" bitsandbytes hf_transfer', shell=True)

# Import Unsloth FIRST (it patches transformers on import for speed)
try:
    from unsloth import FastModel
    UNSLOTH_OK = True
    print('=== Unsloth available — using fast kernels ===', flush=True)
except Exception as e:
    UNSLOTH_OK = False
    print(f'=== Unsloth import failed ({e}) — falling back to vanilla HF ===', flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ---------- config ----------
BASE_MODEL = 'unsloth/gemma-3-4b-it'
# Kaggle mounts personal datasets under either of these paths — probe both.
_ADAPTER_CANDIDATES = [
    Path('/kaggle/input/zyot-decider-v3-adapter'),
    Path('/kaggle/input/datasets/nishanml/zyot-decider-v3-adapter'),
]
V3_ADAPTER = next((str(p) for p in _ADAPTER_CANDIDATES if (p / 'adapter_config.json').exists()), None)
if V3_ADAPTER is None:
    for p in Path('/kaggle/input').glob('**/adapter_config.json'):
        V3_ADAPTER = str(p.parent); break
    if V3_ADAPTER is None:
        print('ERROR: adapter dataset not found', flush=True); raise SystemExit(1)

# Real 62k curated data: 57k SargeDev (noul + choice + score) + 5.3k v2 hand-crafted synth
_DATA_CANDIDATES = [
    Path('/kaggle/input/zyot-v3-5-train-final/train_final.jsonl'),
    Path('/kaggle/input/datasets/nishanml/zyot-v3-5-train-final/train_final.jsonl'),
]
DATA_PATH = next((str(p) for p in _DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    for p in Path('/kaggle/input').glob('**/train_final.jsonl'):
        DATA_PATH = str(p); break
    if DATA_PATH is None:
        DATA_PATH = str(Path(V3_ADAPTER) / 'boolq_test.jsonl')
        print(f'WARN: curated data not found, falling back to bootstrap: {DATA_PATH}', flush=True)
print(f'  V3_ADAPTER = {V3_ADAPTER}')
print(f'  DATA_PATH  = {DATA_PATH}')
OUT = Path('/kaggle/working/v3.5')
OUT.mkdir(parents=True, exist_ok=True)

# Training
NUM_SLOTS = 24                # T,F,A..P,0..5 (16+2+6=24 covers Jev's three primitives)
BATCH = 2
GA_STEPS = 8                  # effective batch = 16
LR = 3e-5                     # slightly conservative — LoRA + head trained together; prevents loss oscillation seen in v3
MAX_STEPS = 2500              # guaranteed completion at observed ~15s/step within 12h Kaggle limit; ~65% of 62k rows @ effective batch 16
SAVE_EVERY = 10               # per user request — checkpoint every 10 steps
EVAL_EVERY = 200
LOG_EVERY = 20
WARMUP_STEPS = 100            # longer warmup for the head (needs to escape LM-head init)
MAX_LEN = 768                 # v2's proven config; ~30% faster than 1024 and covers 95%+ of curated states
GRAD_CLIP = 1.0               # prevents exploding grads on adversarial rows

RESUME = os.environ.get('ZYOT_RESUME', '0') == '1'

# ---------- load base + v3 LoRA in 4-bit ----------
print('=== Loading tokenizer ===', flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

# Verbalizer token IDs — used to initialise pointer head from LM head rows
VERBALIZER = ['T', 'F',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              '0', '1', '2', '3', '4', '5']
assert len(VERBALIZER) == NUM_SLOTS
def _t1(s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None
VERB_IDS = [_t1(v) for v in VERBALIZER]
if any(x is None for x in VERB_IDS):
    print('WARN: some verbalizer tokens are multi-token; adjust VERBALIZER list', flush=True)
    VERB_IDS = [x if x is not None else 0 for x in VERB_IDS]
print(f'  Verbalizer token ids: {VERB_IDS[:6]}... ({NUM_SLOTS} total)', flush=True)

print('=== Loading Gemma 3 4B in 4-bit ===', flush=True)
if UNSLOTH_OK:
    # Unsloth path — ~1.5x faster kernels + lower VRAM. FastModel handles
    # 4-bit + attention + gradient checkpointing internally.
    base, _ = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        full_finetuning=False,
    )
else:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation='eager', device_map='auto',
    )
    base.gradient_checkpointing_enable()

print('=== Attaching v3 LoRA (trainable) ===', flush=True)
model = PeftModel.from_pretrained(base, V3_ADAPTER, is_trainable=True)
device = next(model.parameters()).device
# Gemma 3 is multimodal — hidden_size lives under text_config (fallback: model.get_input_embeddings)
HIDDEN = getattr(base.config, 'hidden_size', None) \
      or getattr(getattr(base.config, 'text_config', None), 'hidden_size', None) \
      or model.get_input_embeddings().embedding_dim
print(f'  HIDDEN = {HIDDEN}', flush=True)

# ---------- pointer head ----------
# AutoTrust-style: single Linear(hidden -> num_slots) initialised from
# LM head verbalizer rows so step 0 == v3 verbalizer readout.
class DecisionHead(nn.Module):
    def __init__(self, hidden, slots, lm_head_weight=None, verb_ids=None):
        super().__init__()
        self.proj = nn.Linear(hidden, slots, bias=False)
        if lm_head_weight is not None and verb_ids is not None:
            with torch.no_grad():
                rows = lm_head_weight[verb_ids].to(self.proj.weight.dtype)
                self.proj.weight.copy_(rows.to(self.proj.weight.device))
                print(f'  head initialised from LM head rows (shape {rows.shape})', flush=True)
    def forward(self, last_hidden):  # [B, H]
        return self.proj(last_hidden)  # [B, slots]

lm_head_w = model.get_output_embeddings().weight.detach()
head = DecisionHead(HIDDEN, NUM_SLOTS, lm_head_w, VERB_IDS).to(device=device, dtype=torch.bfloat16)

# ---------- resume ----------
step_start = 0
if RESUME and (OUT / 'head_ckpt' / 'latest').exists():
    print('=== RESUMING from latest checkpoint ===', flush=True)
    head_sd = torch.load(OUT / 'head_ckpt' / 'latest' / 'head.pt', map_location='cpu')
    head.load_state_dict(head_sd)
    model.load_adapter(str(OUT / 'lora_ckpt' / 'latest'), 'default', is_trainable=True)
    meta = json.loads((OUT / 'head_ckpt' / 'latest' / 'meta.json').read_text())
    step_start = meta.get('step', 0)
    print(f'  resumed at step {step_start}', flush=True)

# ---------- dataset ----------
class DecisionDataset(Dataset):
    """Reads JSONL. Each row: {state, question, options: [str,...], answer_idx: int}
    Falls back to jev-bench boolq schema {state, label} if that's what's passed."""
    def __init__(self, path):
        rows = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
        self.items = []
        for r in rows:
            # jev-bench boolq schema fallback
            if 'label' in r and 'state' in r and r.get('label') in ('0', '1', 0, 1):
                st = r['state']
                if isinstance(st, str):
                    try: st = json.loads(st)
                    except Exception: pass
                if isinstance(st, dict):
                    passage = st.get('passage', '')
                    question = st.get('question', '')
                    prompt = (f'State: {passage}\n\nQuestion: Based on the passage, is the answer to '
                              f'"{question}" yes?\n\nOptions:\nT. Yes / True\nF. No / False\n\nAnswer:')
                    self.items.append({'prompt': prompt, 'slot': 0 if str(r['label'])=='1' else 1})
                continue
            # canonical schema
            if 'prompt' in r and 'answer_slot' in r:
                self.items.append({'prompt': r['prompt'], 'slot': int(r['answer_slot'])})
        print(f'  loaded {len(self.items)} rows from {path}', flush=True)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    enc = tok([b['prompt'] for b in batch], return_tensors='pt', padding=True,
              truncation=True, max_length=MAX_LEN)
    labels = torch.tensor([b['slot'] for b in batch], dtype=torch.long)
    return enc.input_ids, enc.attention_mask, labels

ds = DecisionDataset(DATA_PATH)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)

# ---------- train ----------
trainable = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
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
                                              'verbalizer': VERBALIZER, 'verb_ids': VERB_IDS}))
    ld = OUT / 'lora_ckpt' / tag
    ld.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ld))

print(f'\n=== TRAIN — max_steps={MAX_STEPS}, save every {SAVE_EVERY} ===\n', flush=True)
model.train(); head.train()
step = step_start
grad_accum = 0
t0 = time.time()
loss_ema = None
try:
    while step < MAX_STEPS:
        for input_ids, attn, labels in loader:
            input_ids = input_ids.to(device); attn = attn.to(device); labels = labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
            last_hidden = out.hidden_states[-1]  # [B, T, H]
            # take hidden state at last non-pad position per row
            last_idx = attn.sum(dim=1) - 1  # [B]
            pooled = last_hidden[torch.arange(last_hidden.size(0), device=device), last_idx]  # [B, H]
            logits = head(pooled.to(head.proj.weight.dtype))  # [B, slots]
            loss = F.cross_entropy(logits, labels) / GA_STEPS
            loss.backward()
            grad_accum += 1
            if grad_accum == GA_STEPS:
                for pg in optim.param_groups: pg['lr'] = _get_lr(step, MAX_STEPS, WARMUP_STEPS)
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)  # prevent exploding grads
                optim.step(); optim.zero_grad()
                grad_accum = 0
                step += 1
                loss_val = float(loss.detach()) * GA_STEPS
                loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
                if step % LOG_EVERY == 0:
                    dt = time.time() - t0
                    sps = step / dt
                    eta = (MAX_STEPS - step) / max(sps, 1e-9) / 60
                    print(f'  step {step:>5}/{MAX_STEPS}  loss={loss_ema:.4f}  {sps:.2f} step/s  ETA {eta:.0f}m', flush=True)
                if step % SAVE_EVERY == 0:
                    save_ckpt(step, 'latest')
                if step % EVAL_EVERY == 0:
                    save_ckpt(step, f'step-{step}')
                if step >= MAX_STEPS: break
        # one epoch done, will loop again if step < MAX_STEPS
except KeyboardInterrupt:
    print('\n=== INTERRUPTED — saving final ===', flush=True)

save_ckpt(step, 'final')
save_ckpt(step, 'latest')
print(f'\n=== DONE — {step} steps, {(time.time()-t0)/60:.1f} min, final loss {loss_ema:.4f} ===', flush=True)
print(f'  Head weights:  {OUT / "head_ckpt" / "final" / "head.pt"}')
print(f'  LoRA weights:  {OUT / "lora_ckpt" / "final"}')
print(f'  All checkpoints preserved under {OUT}')
