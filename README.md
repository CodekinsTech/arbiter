# Arbiter v3.3 — 4B

A production **System One** decision model built on `unsloth/gemma-3-4b-it`
with a trained 24-slot pointer head. Handles all three Jev decision
primitives — `noul` (T/F), `choice` (2–16 options), `score` (0–5) — in a
single forward pass.

Part of the **Zyot Lab** open decision-model lineup by Codekins Pvt Ltd.

- **Weights:** [hiteshluke/arbiter-4b](https://huggingface.co/hiteshluke/arbiter-4b)
- **12B flagship:** [hiteshluke/zyot-decider-12b-adapter](https://huggingface.co/hiteshluke/zyot-decider-12b-adapter)
- **v3 verbalizer (reference):** [hiteshluke/zyot-decider-v3-verbalizer-4b](https://huggingface.co/hiteshluke/zyot-decider-v3-verbalizer-4b)
- **License:** Apache-2.0 (adapter + head); base is Gemma-licensed

## Headline numbers

| Benchmark | Primitive | Accuracy |
|---|---|---|
| **BoolQ** (validation, n = 1 000) | noul (T / F) | **0.849** |
| **ARC-Challenge** (test, n = 500) | 4-choice | **0.738** |
| **CommonsenseQA** (validation, n = 500) | 5-choice | **0.706** |
| **OpenBookQA** (test, n = 500) | 4-choice | **0.722** |

Measured on NVIDIA T4 with 4-bit quantized inference.

## v3.3 vs v3 (verbalizer)

| Property | v3 (verbalizer) | v3.3 (current) |
|---|---|---|
| Architecture | LoRA + LM-head verbalizer readout | LoRA + **trained 24-slot pointer head** |
| Multi-choice | T / F only | A – P (up to 16 options) |
| Score / rating | — | 0 – 5 scale |
| Primitives in one head | 1 | **3** (noul + choice + score) |

v3 is a classical verbalizer: the base model's LM head is read directly over
`T` / `F` tokens. v3.3 is a real System One classifier — the pointer head is
initialized from LM-head verbalizer rows, then trained jointly with the LoRA
adapter on all three primitives in one forward pass.

## Slot layout

The pointer head has 24 output slots, initialized from the base model's LM
head verbalizer rows:

- Slots 0 – 1 = `T` / `F` (noul)
- Slots 2 – 17 = `A` – `P` (choice, up to 16 options)
- Slots 18 – 23 = `0` – `5` (score)

For each question, softmax is restricted to the slots valid for that
primitive.

**Frozen T / F slots.** During training, gradients on slots 0 and 1 are
clamped to zero via a backward hook — the model therefore cannot drift away
from the base model's native boolean behavior. This preserves the BoolQ
baseline by construction.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import hf_hub_download
import torch, torch.nn as nn, json

BASE = "unsloth/gemma-3-4b-it"
REPO = "hiteshluke/arbiter-4b"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, REPO).eval()

head_path = hf_hub_download(REPO, "head.pt")
head_meta = json.loads(open(hf_hub_download(REPO, "head_meta.json")).read())
hidden = model.config.text_config.hidden_size
head = nn.Linear(hidden, head_meta["num_slots"], bias=False).to(
    device=model.device, dtype=torch.bfloat16
)
head.load_state_dict({"weight": torch.load(head_path, map_location=model.device)["proj.weight"]})
head.eval()

@torch.no_grad()
def decide(prompt: str, valid_slots: list[int]) -> int:
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    pooled = out.hidden_states[-1][0, -1]
    logits = head(pooled.to(head.weight.dtype)).float().cpu().numpy()
    return valid_slots[int(max(range(len(valid_slots)), key=lambda i: logits[valid_slots[i]]))]
```

## Training

- **Base**: `unsloth/gemma-3-4b-it`
- **LoRA**: r = 16, α = 32, dropout = 0.05, targets q/k/v/o + gate/up/down
- **Head**: `nn.Linear(hidden_size, 24)` in bf16, initialized from LM-head verbalizer rows
- **Data**: curated subset of `SargeDev/jev-distill-corpus-v3` at teacher-confidence ≥ 0.75 (noul + choice + score)
- **Objective**: focal cross-entropy on the pointer head's logits over valid slots per row, sample-weighted by teacher confidence
- **Frozen slots**: 0 (`T`) and 1 (`F`) — gradient zeroed via backward hook
- **Optimizer**: AdamW, LR 5 × 10⁻⁵ cosine, warmup 200
- **Batch**: effective 16 (batch 2 × grad accum 8), max seq 768, bf16/fp16 on Kaggle T4
- **Steps**: 3 000, with EMA-averaged head over the last 5 eval checkpoints

Reference training script: [`training_v3_6/train.py`](training_v3_6/train.py)
(internal versioning kept to preserve experiment history; shipped model is
Arbiter v3.3).

## Positioning

- **First** Gemma-3-based open decision model with a real System One head
- **Only** open 4B decider that handles all three Jev primitives (noul + choice + score) in one head
- Deploys via standard `transformers` + `peft` + one small `head.pt` load — no custom server code

## Reproducing the numbers

- **BoolQ**: 1 000 items of `google/boolq` (validation)
- **ARC-Challenge**: 500 items of `allenai/ai2_arc` (test)
- **CommonsenseQA**: 500 items of `tau/commonsense_qa` (validation)
- **OpenBookQA**: 500 items of `allenai/openbookqa` (main, test)

## Data provenance

Trained on [`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)
— Jev-distilled soft labels via OpenRouter.

## License

Apache-2.0 for the LoRA adapter, pointer head, and code in this repo.
Base `unsloth/gemma-3-4b-it` and `google/gemma-3-4b-it` are governed by the
Gemma license.
