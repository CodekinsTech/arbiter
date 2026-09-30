# Zyot Decider v3.5 — 4B

First open **Gemma-3-based** decision model with a **real System One pointer head**. Trained on 62 k rows across all three Jev decision primitives — `noul` (T/F), `choice` (2–16 options), `score` (0–5) — in a single forward pass.

- **Weights:** [hiteshluke/zyot-decider-v3-4b-lora](https://huggingface.co/hiteshluke/zyot-decider-v3-4b-lora)
- **Site:** [Zyot Lab](https://github.com/CodekinsTech/) · a Codekins Pvt Ltd model family
- **License:** Apache-2.0 (adapter + head); base is Gemma-licensed

## v3.5 vs v3 (superseded)

| Property | v3 (verbalizer, deprecated) | v3.5 (current) |
|---|---|---|
| Architecture | LoRA + LM head restricted-logit read | LoRA + **trained 24-slot pointer head** |
| Multi-choice? | ❌ T/F only | ✅ A–P letters |
| Score/rating? | ❌ | ✅ 0–5 scale |
| Reproducible without fine-tuning? | Yes (any LLM can do verbalizer read) | **No** — real trained classifier |

v3 was a fine-tuned LLM with verbalizer readout — any base LLM can replicate its T/F behavior. v3.5 is a real System One model: pointer head initialized from LM-head verbalizer rows (T, F, A–P, 0–5), then trained jointly with LoRA on 62 k rows.

## Headline numbers

| Benchmark | Primitive | v3.5 | Random | Base Gemma 3 4B (~) |
|---|---|---|---|---|
| **ARC-Challenge** (n=500) | 4-choice | **0.756** | 0.25 | ~0.60 |
| **CommonsenseQA** (n=500) | 5-choice | **0.712** | 0.20 | ~0.70 |
| **JevBench boolq** (calibrated, n=800) | noul | **0.836** | 0.50 | ~0.86 |
| Yelp 5-star exact (n=500) | score | 0.292 | 0.20 | ~0.35 |
| Yelp 5-star ±1 (n=500) | score tolerance | 0.584 | 0.60 | — |

**Headline: +15 pp above base Gemma 3 4B on ARC-Challenge.** That's the "real System One" signal — verbalizer readout on a base LLM cannot produce this.

**Honest weakness:** Yelp 5-star exact is only 0.292. The score head was trained on 7 k rows vs 32 k noul + 17 k choice. v3.6 will rebalance.

## Slot layout

The pointer head has 24 output slots, initialized from Gemma 3's LM head verbalizer rows:

- Slots 0–1 = `T` / `F` (noul)
- Slots 2–17 = `A`–`P` (choice, up to 16 options)
- Slots 18–23 = `0`–`5` (score)

For each question type, softmax over only the valid slots.

## vs open decision-model tier (JevBench boolq for continuity)

| Model | Params | boolq | Notes |
|---|---|---|---|
| Jev 1.13 (hosted) | – | 0.917 | published |
| Gemma 4 12B (base) | 12B | 0.880 | published |
| Qwen 3.5 9B (base) | 9B | 0.861 | published |
| Qwen 3.5 4B (base) | 4B | 0.856 | published |
| **Zyot Decider v3.5 4B** | 4B | **0.836** | real System One head — not verbalizer |
| Kev-4B | 4B | 0.595 | +24 pp over Kev |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, torch.nn as nn, json
from huggingface_hub import hf_hub_download

BASE = 'unsloth/gemma-3-4b-it'
REPO = 'hiteshluke/zyot-decider-v3-4b-lora'  # v3.5 contents

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(model, REPO).eval()

# Load the trained pointer head
head_path = hf_hub_download(REPO, 'head.pt')
meta = json.loads(open(hf_hub_download(REPO, 'head_meta.json')).read())
class Head(nn.Module):
    def __init__(self, h, s): super().__init__(); self.proj = nn.Linear(h, s, bias=False)
    def forward(self, x): return self.proj(x)
head = Head(model.config.text_config.hidden_size, meta['num_slots'])
head.load_state_dict(torch.load(head_path, map_location=model.device))
head.to(device=model.device, dtype=torch.bfloat16).eval()

def decide(prompt, valid_slots):
    ids = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        last = out.hidden_states[-1][0, -1]
        logits = head(last.to(head.proj.weight.dtype)).float().cpu().numpy()
    valid_logits = [logits[s] for s in valid_slots]
    return valid_slots[max(range(len(valid_logits)), key=lambda i: valid_logits[i])]
```

## Training

- **Base:** `unsloth/gemma-3-4b-it` (Gemma 3 4B, multimodal — text-only path used)
- **LoRA:** r=16, α=32, dropout=0.05, targets q/k/v/o + gate/up/down
- **Head:** `nn.Linear(2560, 24)` in bf16, init from LM head verbalizer rows
- **Data:** 62 448 rows — 32 k noul + 17 k choice + 7 k score from SargeDev + 5.3 k hand-crafted synthetic
- **Loss:** cross-entropy on 24-slot pointer head + LoRA joint update
- **Optimizer:** AdamW, LR 3e-5 cosine, warmup 100
- **Effective batch 16** (batch 2 × grad accum 8), max seq 768, fp16 on Kaggle T4
- **Wall clock:** 2 500 steps in 4h 22min = ~1 epoch
- **Final loss:** 0.359

Full training scripts + notes: [`training_v3_5/`](training_v3_5/)

## Positioning

- **First** Gemma-3-based open decision model with real System One head
- **Only** open 4B decider that natively handles all three Jev primitives (noul + choice + score) in one head
- Deploys via standard `transformers` + `peft` + one small `head.pt` load — no custom Python server required

## Reproducing the numbers

- **Training**: [`training_v3_5/train.py`](training_v3_5/train.py) on Kaggle T4 with our curated 62 k dataset
- **BoolQ eval + calibration**: same script, held-out 800 items from Praveenrajus/jev-bench boolq test split
- **ARC-Challenge**: run against 500 items of `allenai/ai2_arc` (test)
- **CommonsenseQA**: 500 items of `tau/commonsense_qa` (validation)
- **Yelp 5-star**: 500 items of `yelp_review_full` (test)

## Limitations

- Score head undertrained (only 7 k score rows) — Yelp 5-star near random on off-by-1
- Boolq accuracy 0.836 is ~2 pp below what a verbalizer read on the same base would achieve — cost of generalizing across primitives
- No conformal calibration yet — v3.6 will add
- Long-context (>768 tokens) truncated at training time; may affect long-document decisions
- Multi-question single-pass not implemented yet (planned for v3.6)

## Data provenance

Trained on [`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) — Jev-distilled soft labels via OpenRouter — plus 5.3 k hand-crafted synthetic rows covering math, dates, business routing, adversarial prompt injection, abstention, severity scoring, and finance routing.

## License

Apache-2.0 for the LoRA + head + code in this repo. Base `unsloth/gemma-3-4b-it` and `google/gemma-3-4b-it` governed by the Gemma license.
