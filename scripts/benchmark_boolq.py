"""
Zyot Decider v3 — JevBench boolq benchmark (200 items).

Loads Gemma 3 4B base + our v3 LoRA adapter, runs restricted-logit prediction
on boolq test set, reports accuracy vs published Jev/Gemma numbers.
"""
import json, time, os
from pathlib import Path

os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

# ---- config ----
BASE_MODEL = 'unsloth/gemma-3-4b-it'
ADAPTER = Path(r'C:\Users\Nishan\Desktop\projects\klexdecide\zyot_decider_v3\adapter_v3\zyot-v3-minimal\final')
BOOLQ_FILE = Path(r'C:\Users\Nishan\Desktop\projects\klexdecide\zyot_decider_v2\reports\jevbench_boolq_test.jsonl')
N_SAMPLES = 200
REPORTS = Path(r'C:\Users\Nishan\Desktop\projects\klexdecide\zyot_decider_v3\reports')
REPORTS.mkdir(exist_ok=True)

# ---- load ----
print('Loading base + adapter...', flush=True)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

# 4-bit for GTX 1650 fit
# CPU-only bf16 — torch install is CPU-only, GPU path broken. Slower but works.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16,
    attn_implementation='eager', device_map={'': 'cpu'},
)
model = PeftModel.from_pretrained(model, str(ADAPTER))
model.eval()

# Match training's prompt format
def _tok1(s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None
T_ID = _tok1('T'); F_ID = _tok1('F')
assert T_ID is not None and F_ID is not None, 'Verbalizer tokens must be single-token'
device = next(model.parameters()).device
print(f'Model loaded. Device: {device}  T={T_ID} F={F_ID}', flush=True)

# ---- load boolq ----
items_raw = [json.loads(l) for l in BOOLQ_FILE.read_text(encoding='utf-8').splitlines()[:N_SAMPLES]]
print(f'\nLoaded {len(items_raw)} boolq items', flush=True)

# JevBench boolq: state = JSON string with passage+question, label = "1"/"0" (1=true/yes)
def parse_boolq(item):
    state = json.loads(item.get('state', '{}')) if isinstance(item.get('state'), str) else item.get('state', {})
    passage = state.get('passage', '')
    question = state.get('question', '')
    label = str(item.get('label', ''))
    if label not in ('0', '1'):
        return None
    truth = 'T' if label == '1' else 'F'
    return {'passage': passage, 'question': question, 'truth': truth}

items = [x for x in (parse_boolq(i) for i in items_raw) if x is not None]
print(f'Parsed valid: {len(items)}', flush=True)

# ---- predict ----
def predict(passage, question):
    prompt = (f'State: {passage}\n\nQuestion: Based on the passage, is the answer to "{question}" yes?\n\n'
              f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')
    ids = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0, -1]  # last position
    t_logit = logits[T_ID].item()
    f_logit = logits[F_ID].item()
    pred = 'T' if t_logit > f_logit else 'F'
    # confidence
    import math
    m = max(t_logit, f_logit)
    p_t = math.exp(t_logit - m) / (math.exp(t_logit - m) + math.exp(f_logit - m))
    return pred, p_t

correct = 0
results = []
print('\nRunning benchmark...', flush=True)
t0 = time.time()
for i, item in enumerate(items, 1):
    pred, p_t = predict(item['passage'], item['question'])
    ok = pred == item['truth']
    if ok: correct += 1
    results.append({'i': i, 'truth': item['truth'], 'pred': pred, 'p_true': round(p_t, 3), 'ok': ok})
    if i % 20 == 0:
        acc = correct / i
        elapsed = time.time() - t0
        eta = (len(items) - i) * (elapsed / i)
        print(f'  [{i:3}/{len(items)}] acc={acc:.3f}  {elapsed:.0f}s  ETA {eta:.0f}s', flush=True)

acc = correct / len(items)
elapsed = time.time() - t0
print(f'\n=== RESULT ===')
print(f'Zyot v3 4B (200 boolq): accuracy = {acc:.3f} ({correct}/{len(items)})  time {elapsed:.0f}s')
print(f'\nComparison (JevBench boolq public):')
print(f'  Jev 1.13:       0.917')
print(f'  Gemma 4 12B:    0.880')
print(f'  Our v2 (v4 LoRA): 0.885 (from prior head-to-head)')
print(f'  Qwen 3.5 9B:    0.861')
print(f'  Qwen 3.5 4B:    0.856')
print(f'  Laya (measured): 0.860 (from prior head-to-head)')
print(f'  Zyot v3 4B: {acc:.3f}  <-- ours')

# Save
ts = time.strftime('%Y%m%d_%H%M')
report = REPORTS / f'boolq200_{ts}.json'
report.write_text(json.dumps({
    'n': len(items), 'accuracy': acc, 'correct': correct,
    'model': 'zyot-decider-v3-4b (unsloth/gemma-3-4b-it + LoRA)',
    'benchmark': 'JevBench boolq (Praveenrajus/jev-bench)',
    'time_seconds': round(elapsed, 1),
    'per_item_seconds': round(elapsed / len(items), 3),
    'results': results,
}, indent=2))
print(f'\nSaved: {report}')
