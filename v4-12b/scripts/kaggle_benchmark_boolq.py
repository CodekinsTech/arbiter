# %% [code]
"""Arbiter v4 12B - BoolQ head-based benchmark (boolean -> noul T/F slots).

Benchmarks the just-trained Arbiter v4 12B decision model on the
LocalLLaMA/typed-decisions TEST set, reporting an "Ollaya score"
comparable to the prior self-consistency number (0.584).

Two evaluations over the full test set:
  (a) HEAD-BASED  â€” forward pass -> last-token hidden state -> DecisionHead
                    -> mask to the question's valid slots -> argmax -> answer,
                    with the training calibration_temp (3.4) divide on logits.
  (b) TEXT-GEN    â€” greedy-generate a few tokens, parse the emitted answer
                    token (letter / digit / T,F), map to the answer.

Everything that touches the model MIRRORS training_v3_9_12b/kaggle_current.py:
  BASE_MODEL, NUM_SLOTS=28, VERBALIZER/VERB_IDS, the slot layout
  (noul=0/1, choice=2..17, score=18..27), the DecisionHead
  (nn.Linear(hidden,28,bias=False) seeded from lm_head rows at verb_ids),
  the 4-bit Unsloth load (+ UNSLOTH_OK fallback), and the prompt builders
  _make_choice_prompt / _make_noul_prompt / _make_score_prompt. The model was
  trained on exactly those prompts, so the benchmark builds them identically.

Dataset loading + the Ollaya scoring convention are copied from
self_consistency/benchmark_td.py so the number is comparable.

Kaggle: GPU T4x2, Internet ON. Does NOT train; load + evaluate only.
"""
import os, sys, json, time, subprocess, re
from pathlib import Path
from collections import Counter

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ.setdefault('HF_TOKEN', '')  # set your own Hugging Face token in the environment
os.environ['ZYOT_SKIP_TEXTGEN'] = '1'  # EMA-head run: head-based only (fast); text-gen is head-independent

# ------------------------------------------------------------------
# pip â€” same stack as the training script
# ------------------------------------------------------------------
subprocess.run('pip uninstall -y torchao 2>/dev/null || true', shell=True)
subprocess.check_call(
    'pip install -q "unsloth" "unsloth_zoo" '
    '"transformers>=4.58" "peft>=0.11,<1" '
    '"accelerate>=0.30,<2" bitsandbytes hf_transfer datasets huggingface_hub '
    'requests pandas pyarrow', shell=True)
subprocess.run('pip install -q --upgrade transformers', shell=True)

try:
    from unsloth import FastModel
    UNSLOTH_OK = True
    print('=== Unsloth kernels active ===', flush=True)
except Exception as e:
    UNSLOTH_OK = False
    print(f'WARN: Unsloth failed ({e}) â€” HF fallback', flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================
#   CONFIG â€” must match training_v3_9_12b/kaggle_current.py
# ============================================================
BASE_MODEL = 'google/gemma-4-12b-it'
NUM_SLOTS = 28          # 0-1=noul, 2-17=choice(16), 18-27=score(10 levels max)
MAX_LEN = 1024
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
CALIBRATION_TEMP = 3.4  # training's evaluate() divides head logits by this
GEN_MAX_NEW = 5         # greedy tokens for the text-gen cross-check

HF_CKPT_REPO = os.environ.get('HF_CKPT_REPO', 'hiteshluke/arbiter-v4-12b-ckpts')
HF_TOKEN = os.environ.get('HF_TOKEN')
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN') or HF_TOKEN
    print('=== HF_TOKEN loaded from Kaggle Secrets ===', flush=True)
except Exception:
    print('=== HF_TOKEN from env var ===', flush=True)

# typed-decisions test set (mirror benchmark_td.py)
TD_REPO = 'LocalLLaMA/typed-decisions'
TD_REV = 'd0e2f0c42fef86cc15d1688d25a19f5ba7c85b18'
TD_PARQUET = 'all/test-00000-of-00001.parquet'

OUT = Path('/kaggle/working/v4-12b-bench')
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
#   PROMPT HELPERS  (verbatim from kaggle_current.py lines 173-203)
# ============================================================
NOUL_SLOTS = [0, 1]
def choice_slots(n): return list(range(2, 2 + min(n, 16)))
def score_slots(n_levels): return list(range(18, 18 + min(n_levels, 10)))

def _make_choice_prompt(state, question, options):
    letters = [chr(ord('A') + i) for i in range(len(options))]
    opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, options))
    return f'State: {str(state or "")}\n\nQuestion: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:'

def _make_noul_prompt(state, question):
    return (f'State: {str(state or "")}\n\nQuestion: {question}\n\n'
            f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')

def _make_score_prompt(state, question, n_levels):
    """Flexible score prompt: supports 2-10 levels."""
    level_lines = '\n'.join(str(i) for i in range(n_levels))
    return (f'State: {str(state or "")}\n\nQuestion: {question}\n\n'
            f'Options:\n{level_lines}\n\nAnswer:')

def _score_slot_flex(argmax_i, n_levels):
    """Flexible score slot: maps answer index to slot 18..18+n_levels-1."""
    if argmax_i < 0 or argmax_i >= n_levels: return None
    slot = 18 + argmax_i
    if slot >= NUM_SLOTS: return None
    return slot

# ============================================================
#   MODEL + HEAD  (mirror kaggle_current.py lines 589-644)
# ============================================================
print('=== Loading tokenizer ===', flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

# 28-slot verbalizer: T,F, A-P(16), 0-9(10) â€” identical to training
VERBALIZER = ['T', 'F',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def _t1(s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None
VERB_IDS = [x if (x := _t1(v)) is not None else 0 for v in VERBALIZER]

print(f'=== Loading {BASE_MODEL} in 4-bit ===', flush=True)
if UNSLOTH_OK:
    base, _ = FastModel.from_pretrained(model_name=BASE_MODEL, max_seq_length=MAX_LEN,
        load_in_4bit=True, dtype=torch.bfloat16, full_finetuning=False)
    model = FastModel.get_peft_model(base, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        use_gradient_checkpointing='unsloth', bias='none')
else:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
        dtype=torch.bfloat16, attn_implementation='eager', device_map='auto', token=HF_TOKEN)
    base = prepare_model_for_kbit_training(base)
    lc = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        task_type='CAUSAL_LM')
    model = get_peft_model(base, lc)

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
print(f'  hidden_size: {HIDDEN}, num_slots: {NUM_SLOTS}', flush=True)

# ============================================================
#   CHECKPOINT DOWNLOAD  (mirror the HF layout written by _hf_push_ckpt)
#     head/meta -> step-{N}/head/{head.pt,meta.json}  and  ema-final/head/*
#     lora       -> step-{N}/lora/{adapter_config.json, adapter_model.safetensors}
#     ema-final has HEAD ONLY (no lora) â€” adapter comes from highest step-N.
# ============================================================
from huggingface_hub import hf_hub_download

def _dl(fname):
    return hf_hub_download(HF_CKPT_REPO, fname, token=HF_TOKEN, repo_type='model',
                           local_dir=str(OUT / '.hf'), local_dir_use_symlinks=False)

latest_step = None
try:
    sp = _dl('LATEST_STEP')
    latest_step = int(open(sp).read().strip())
    print(f'  LATEST_STEP = {latest_step}', flush=True)
except Exception as e:
    print(f'  WARN: LATEST_STEP unavailable ({e})', flush=True)

# --- head.pt + meta.json: prefer ema-final, else highest step-N ---
head_pt = meta_json = None
head_tag = None
try:
    # EMA head is pushed under step-ema-final/ (push used path step-<step> with step='ema-final')
    try:
        head_pt = _dl('step-ema-final/head/head.pt'); _ema_prefix = 'step-ema-final'
    except Exception:
        head_pt = _dl('ema-final/head/head.pt'); _ema_prefix = 'ema-final'
    try: meta_json = _dl(f'{_ema_prefix}/head/meta.json')
    except Exception: meta_json = None
    head_tag = 'ema-final'
    print(f'  head: using {_ema_prefix}', flush=True)
except Exception as e:
    print(f'  ema-final head not present ({e}) â€” falling back to step-N', flush=True)
    if latest_step is None:
        raise RuntimeError('no ema-final head and no LATEST_STEP â€” cannot locate checkpoint')
    head_pt = _dl(f'step-{latest_step}/head/head.pt')
    try: meta_json = _dl(f'step-{latest_step}/head/meta.json')
    except Exception: meta_json = None
    head_tag = f'step-{latest_step}'
    print(f'  head: using step-{latest_step}', flush=True)

# --- rebuild VERB_IDS from meta.json if available (matches training exactly) ---
if meta_json:
    try:
        meta = json.loads(open(meta_json).read())
        if meta.get('verb_ids') and len(meta['verb_ids']) == NUM_SLOTS:
            VERB_IDS = list(meta['verb_ids'])
            print('  VERB_IDS taken from checkpoint meta.json', flush=True)
        if meta.get('calibration_temp'):
            CALIBRATION_TEMP = float(meta['calibration_temp'])
            print(f'  calibration_temp from meta = {CALIBRATION_TEMP}', flush=True)
    except Exception as e:
        print(f'  WARN reading meta.json: {e}', flush=True)

# --- load head weights ---
head.load_state_dict(torch.load(head_pt, map_location='cpu'))
head = head.to(device=device, dtype=torch.bfloat16)
print(f'  head loaded from {head_tag}', flush=True)

# --- LoRA adapter: always from a step-N dir (ema-final carries no lora) ---
if latest_step is None:
    raise RuntimeError('LATEST_STEP missing â€” cannot locate LoRA adapter')
adapter_dir = OUT / 'lora_adapter'
adapter_dir.mkdir(parents=True, exist_ok=True)
for f in ('adapter_config.json', 'adapter_model.safetensors'):
    src = _dl(f'step-{latest_step}/lora/{f}')
    import shutil as _sh
    _sh.copy(src, str(adapter_dir / f))
# load the trained adapter into the 'default' slot (same call training's resume uses)
model.load_adapter(str(adapter_dir), 'default')
print(f'  LoRA adapter loaded from step-{latest_step}', flush=True)

model.eval(); head.eval()

# ============================================================
#   BoolQ  (head-based, boolean -> noul T/F slots 0/1)
#   Passage -> state, question -> question; T=slot0=True, F=slot1=False.
#   Standard reported metric: accuracy on the BoolQ validation split (3270).
# ============================================================
from datasets import load_dataset

MAX_LEN_BQ = int(os.environ.get('ZYOT_BOOLQ_MAXLEN', '1024'))
N_LIMIT = int(os.environ.get('ZYOT_BOOLQ_N', '0'))  # 0 = full validation

@torch.no_grad()
def eval_head_bool(prompt):
    enc = tok([prompt], return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN_BQ)
    input_ids = enc.input_ids.to(device); attn = enc.attention_mask.to(device)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    last = out.hidden_states[-1]
    idx = attn.sum(dim=1) - 1
    pooled = last[torch.arange(last.size(0), device=device), idx]
    logits = head(pooled.to(head.proj.weight.dtype)) / CALIBRATION_TEMP
    mask = torch.zeros(1, NUM_SLOTS, dtype=torch.bool, device=device)
    mask[0, 0] = True; mask[0, 1] = True            # only T(0)/F(1)
    logits = logits.masked_fill(~mask, -1e4)
    slot = int(logits.argmax(-1).item())
    return slot == 0                                # slot0=T=True

def main():
    print('=== Loading BoolQ validation ===', flush=True)
    ds = load_dataset('google/boolq', split='validation')
    n = len(ds) if N_LIMIT <= 0 else min(N_LIMIT, len(ds))
    print(f'  BoolQ validation: evaluating {n}/{len(ds)}', flush=True)

    correct = 0; total = 0
    tp = tn = fp = fn = 0                            # gold-True vs pred
    gold_true = 0
    t0 = time.time()
    for k in range(n):
        ex = ds[k]
        passage = ex['passage']; question = ex['question']; ans = bool(ex['answer'])
        prompt = _make_noul_prompt(passage, question)
        pred = eval_head_bool(prompt)
        total += 1
        gold_true += int(ans)
        if pred == ans: correct += 1
        if ans and pred: tp += 1
        elif ans and not pred: fn += 1
        elif (not ans) and pred: fp += 1
        else: tn += 1
        if (k + 1) % 200 == 0:
            print(f'  [{k+1}/{n}] acc={correct/total:.4f}  [{time.time()-t0:.0f}s]', flush=True)

    acc = correct / total
    # majority-class baseline for context
    maj = max(gold_true, total - gold_true) / total
    print('\n=== RESULT BoolQ (head-based, noul slots) ===', flush=True)
    print(f'  accuracy = {acc:.4f}  ({correct}/{total})', flush=True)
    print(f'  gold balance: True={gold_true} False={total-gold_true}  majority-class={maj:.4f}', flush=True)
    print(f'  confusion: TP={tp} TN={tn} FP={fp} FN={fn}', flush=True)
    res = {'benchmark': 'boolq-validation', 'n': total, 'accuracy': round(acc, 4),
           'correct': correct, 'gold_true': gold_true, 'majority_class': round(maj, 4),
           'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
           'checkpoint': {'head': head_tag, 'lora_step': latest_step, 'calibration_temp': CALIBRATION_TEMP}}
    (OUT / 'boolq_results.json').write_text(json.dumps(res, indent=2))
    print(f'\nSaved {OUT / "boolq_results.json"}', flush=True)

if __name__ == '__main__':
    main()
