"""Zyot Decider v3.6 (Arbiter) benchmark: BoolQ / ARC-Challenge / CommonsenseQA.
Uses the EMA-averaged head + v3.6 LoRA on Gemma 3 4B."""
import os, json, math, time, subprocess
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

subprocess.run('pip uninstall -y torchao 2>/dev/null || true', shell=True)
subprocess.check_call('pip install -q "transformers>=4.44,<5" "peft>=0.11,<1" '
                     '"accelerate>=0.30,<2" bitsandbytes datasets', shell=True)

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

BASE = 'unsloth/gemma-3-4b-it'

# locate v3.6 artifacts (dataset sources mount under /kaggle/input/<slug>/)
V36 = None
for cand in [Path('/kaggle/input/zyot-v3-6-train/v3.6'),
             Path('/kaggle/input/datasets/rayntracks/zyot-v3-6-train/v3.6')]:
    if (cand / 'lora_ckpt' / 'final').exists():
        V36 = cand; break
if V36 is None:
    for p in Path('/kaggle/input').glob('**/lora_ckpt/final/adapter_config.json'):
        V36 = p.parent.parent.parent; break
assert V36 is not None, 'v3.6 checkpoint not found under /kaggle/input'
print(f'V3.6 root = {V36}', flush=True)

LORA = V36 / 'lora_ckpt' / 'final'
# prefer EMA head; fallback to final
HEAD = V36 / 'head_ckpt' / 'ema-final' / 'head.pt'
META = V36 / 'head_ckpt' / 'ema-final' / 'meta.json'
if not HEAD.exists():
    HEAD = V36 / 'head_ckpt' / 'final' / 'head.pt'
    META = V36 / 'head_ckpt' / 'final' / 'meta.json'
m = json.loads(META.read_text())
NUM_SLOTS = m['num_slots']
print(f'HEAD = {HEAD}  slots={NUM_SLOTS}', flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
    dtype=torch.bfloat16, attn_implementation='eager', device_map='auto')
model = PeftModel.from_pretrained(base, str(LORA)); model.eval()
device = next(model.parameters()).device
HIDDEN = getattr(base.config, 'hidden_size', None) \
     or getattr(getattr(base.config, 'text_config', None), 'hidden_size', None) \
     or model.get_input_embeddings().embedding_dim
print(f'HIDDEN={HIDDEN}', flush=True)

class Head(nn.Module):
    def __init__(self, h, s): super().__init__(); self.proj = nn.Linear(h, s, bias=False)
    def forward(self, x): return self.proj(x)
head = Head(HIDDEN, NUM_SLOTS).to(device=device, dtype=torch.bfloat16)
head.load_state_dict(torch.load(str(HEAD), map_location=device)); head.eval()
print('model + head loaded\n', flush=True)

def wilson(p, n, z=1.96):
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return center-half, center+half

@torch.no_grad()
def predict(prompt, valid):
    ids = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).input_ids.to(device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    last = out.hidden_states[-1][0, -1]
    logits = head(last.to(head.proj.weight.dtype)).float().cpu().numpy()
    vals = [logits[s] for s in valid]
    return valid[int(np.argmax(vals))]

RESULTS = {}

# ============================================================
# 1) BoolQ — noul (T=slot 0, F=slot 1)
# ============================================================
print('=== BoolQ (validation, n=1000) ===', flush=True)
ds = list(load_dataset('google/boolq', split='validation'))[:1000]
correct = total = 0; t0 = time.time()
for i, r in enumerate(ds, 1):
    prompt = (f'State: {r["passage"]}\n\nQuestion: {r["question"]}\n\n'
              f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')
    truth = 0 if r['answer'] else 1
    pred = predict(prompt, [0, 1])
    if pred == truth: correct += 1
    total += 1
    if i % 200 == 0:
        print(f'  [{i}/{len(ds)}] acc={correct/i:.4f}  {time.time()-t0:.0f}s', flush=True)
acc = correct / total
lo, hi = wilson(acc, total)
print(f'  BoolQ: {acc:.4f} ({correct}/{total})  CI[{lo:.4f},{hi:.4f}]  {time.time()-t0:.0f}s\n', flush=True)
RESULTS['boolq'] = {'n': total, 'correct': correct, 'accuracy': acc, 'ci': [lo, hi]}
Path('/kaggle/working/v3_6_bench.json').write_text(json.dumps(RESULTS, indent=2))

# ============================================================
# 2) ARC-Challenge — 4-choice (slot 2..5)
# ============================================================
print('=== ARC-Challenge (test, n=500) ===', flush=True)
ds = list(load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test'))[:500]
correct = total = 0; t0 = time.time()
for i, r in enumerate(ds, 1):
    opts = r['choices']['text']; lbls = r['choices']['label']
    truth_i = next((j for j, L in enumerate(lbls) if L == r['answerKey']), None)
    if truth_i is None or len(opts) < 2 or len(opts) > 16: continue
    letters = [chr(ord('A')+j) for j in range(len(opts))]
    opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, opts))
    prompt = f'State: \n\nQuestion: {r["question"]}\n\nOptions:\n{opt_lines}\n\nAnswer:'
    valid = list(range(2, 2+len(opts)))
    truth = 2 + truth_i
    pred = predict(prompt, valid)
    if pred == truth: correct += 1
    total += 1
    if i % 100 == 0:
        print(f'  [{i}/{len(ds)}] acc={correct/i:.4f}  {time.time()-t0:.0f}s', flush=True)
acc = correct / max(total, 1)
lo, hi = wilson(acc, total) if total else (0,0)
print(f'  ARC-Challenge: {acc:.4f} ({correct}/{total})  CI[{lo:.4f},{hi:.4f}]  {time.time()-t0:.0f}s\n', flush=True)
RESULTS['arc_challenge'] = {'n': total, 'correct': correct, 'accuracy': acc, 'ci': [lo, hi]}
Path('/kaggle/working/v3_6_bench.json').write_text(json.dumps(RESULTS, indent=2))

# ============================================================
# 3) CommonsenseQA — 5-choice
# ============================================================
print('=== CommonsenseQA (validation, n=500) ===', flush=True)
ds = list(load_dataset('tau/commonsense_qa', split='validation'))[:500]
correct = total = 0; t0 = time.time()
for i, r in enumerate(ds, 1):
    opts = r['choices']['text']; lbls = r['choices']['label']
    truth_i = next((j for j, L in enumerate(lbls) if L == r['answerKey']), None)
    if truth_i is None or len(opts) < 2 or len(opts) > 16: continue
    letters = [chr(ord('A')+j) for j in range(len(opts))]
    opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, opts))
    prompt = f'State: \n\nQuestion: {r["question"]}\n\nOptions:\n{opt_lines}\n\nAnswer:'
    valid = list(range(2, 2+len(opts)))
    truth = 2 + truth_i
    pred = predict(prompt, valid)
    if pred == truth: correct += 1
    total += 1
    if i % 100 == 0:
        print(f'  [{i}/{len(ds)}] acc={correct/i:.4f}  {time.time()-t0:.0f}s', flush=True)
acc = correct / max(total, 1)
lo, hi = wilson(acc, total) if total else (0,0)
print(f'  CommonsenseQA: {acc:.4f} ({correct}/{total})  CI[{lo:.4f},{hi:.4f}]  {time.time()-t0:.0f}s\n', flush=True)
RESULTS['csqa'] = {'n': total, 'correct': correct, 'accuracy': acc, 'ci': [lo, hi]}

Path('/kaggle/working/v3_6_bench.json').write_text(json.dumps(RESULTS, indent=2))

# ============================================================
# SUMMARY
# ============================================================
print('\n' + '='*60)
print(f"{'Benchmark':<20} {'n':>6} {'acc':>8} {'CI':<25}")
print('='*60)
for k, v in RESULTS.items():
    print(f"{k:<20} {v['n']:>6} {v['accuracy']:>8.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
print('\n--- Reference baselines ---')
print('v3   boolq (hosted):     0.857')
print('v3.5 boolq (frozen=no):  0.836')
print('base Gemma 3 4B boolq:   ~0.86')
print('base Gemma 3 4B ARC-C:   ~0.60')
print('base Gemma 3 4B CSQA:    ~0.70')
print(f'\nSaved: /kaggle/working/v3_6_bench.json')
