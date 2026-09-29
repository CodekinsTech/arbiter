# Zyot Decider v3 — 4B

A decision-support LoRA adapter for [`unsloth/gemma-3-4b-it`](https://huggingface.co/unsloth/gemma-3-4b-it) trained with soft-label distillation on [`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3).

**First open Gemma-3-based decision model.**

- **Weights:** [hiteshluke/zyot-decider-v3-4b-lora](https://huggingface.co/hiteshluke/zyot-decider-v3-4b-lora)
- **Site:** [Zyot Lab](https://github.com/CodekinsTech/) · a Codekins Pvt Ltd model family
- **License:** Apache-2.0 (adapter); base is Gemma-licensed

## Headline numbers

| Benchmark | Zyot v3 4B | Notes |
|---|---|---|
| JevBench boolq (n=1000) | **0.857** | 95% Wilson CI [0.834, 0.877] |
| Boolq vs Kev-4B (n=200 subset) | **+26 pp** | Kev 0.595 → Ours 0.855 |
| jev-snake · Easy · avg food | **32.6** | 435 ms / move on GTX 1650 |
| jev-snake · Medium · avg food | **21.8** | 553 ms / move |
| jev-snake · Hard · avg food | **6.8** | 681 ms / move |

Statistically tied with Qwen 3.5 4B base (0.856) at the same size. Matches Laya (0.860) on boolq accuracy despite being a generative LLM rather than an encoder classifier.

## JevBench boolq — n = 1000

| Model | Params | Accuracy | Source |
|---|---|---|---|
| Jev 1.13 (TypeSafe API) | – | 0.917 | published |
| Gemma 4 12B (base) | 12B | 0.880 | published |
| Qwen 3.5 9B (base) | 9B | 0.861 | published |
| Laya (encoder classifier) | 0.42B | 0.860 | measured, n=200 |
| Qwen 3.5 4B (base) | 4B | 0.856 | published |
| **Zyot Decider v3 4B (this)** | 4B | **0.857** | CI [0.834, 0.877] |
| Kev-4B | 4B | 0.595 | measured, same 200-item subset |

Raw JSON: [`reports/boolq_1k_result.json`](reports/boolq_1k_result.json).

## Snake benchmark — iammusham/jev-snake harness

Ran on the official [iammusham/jev-snake](https://github.com/iammusham/jev-snake) engine (`snake-jev-v2`, 20×20 board, deterministic seeds), 3 tiers × 5 seeds = 15 games. First open scores published on this exact spec.

| Tier | Hazards | Avg food | Avg ticks | ms / move | Deaths |
|---|---|---|---|---|---|
| **Easy** | 0 / 0 | **32.6** | 475 | 435 | self:4, wall:1 |
| **Medium** | 8 obs / 3 decoys | **21.8** | 312 | 553 | self:4, obstacle:1 |
| **Hard** | 20 obs / 7 decoys | **6.8** | 488 | 681 | obstacle:2, self:1, max_ticks:2 |

Runs on a local `llama-server` (GTX 1650) with the Q4_K_M base + our LoRA GGUF.

### Head-to-head vs Kev-4B (same seeds, same harness)

| Tier | Zyot v3 4B — food | Kev-4B — food | Zyot ms / move | Kev ms / move |
|---|---|---|---|---|
| Easy | **32.6** | 0.2 | 435 | 1055 |
| Medium | **21.8** | 0.4 | 553 | 1371 |
| Hard | **6.8** | 1.0 | 681 | 1824 |

**Honest caveat.** Prompt shapes differ: Zyot is given a computed `preferred/safe` shortlist so the task becomes a copy-task matching its LoRA training. Kev-4B was called on its native `/v1/systemone` API with Jev's raw state JSON — the format it was built to eat. So this is "each model on its native prompt shape," not identical inputs. Kev may do better with a shortlist wrapper too.

Raw JSONs: [`reports/jev_snake_full_bench.json`](reports/jev_snake_full_bench.json), [`reports/kev_snake_bench.json`](reports/kev_snake_bench.json).

## Usage

### Python (transformers + PEFT)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = 'unsloth/gemma-3-4b-it'
adapter = 'hiteshluke/zyot-decider-v3-4b-lora'

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(model, adapter).eval()

T = tok.encode('T', add_special_tokens=False)[0]
F = tok.encode('F', add_special_tokens=False)[0]

def decide(passage, question):
    prompt = (f'State: {passage}\n\nQuestion: Based on the passage, is the answer to '
              f'"{question}" yes?\n\nOptions:\nT. Yes / True\nF. No / False\n\nAnswer:')
    ids = tok(prompt, return_tensors='pt', truncation=True, max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, -1]
    return {'T': logits[T].item(), 'F': logits[F].item()}
```

### Local llama-server (GGUF)

```bash
llama-server \
  -m gemma-3-4b-it-Q4_K_M.gguf \
  --lora zyot-decider-v3-4b-lora.gguf \
  -ngl 99 -c 2048 --host 127.0.0.1 --port 8081

# OpenAI-compatible endpoint
POST http://127.0.0.1:8081/v1/chat/completions
```

Build the LoRA GGUF from the HF adapter with [`scripts/make_gguf.py`](scripts/make_gguf.py).

## Reproducing the numbers

- **BoolQ** (n=1000): [`scripts/benchmark_boolq.py`](scripts/benchmark_boolq.py) — HF transformers, 4-bit on T4.
- **jev-snake** (3 tiers × 5 seeds): [`eval/run_jev_snake.py`](eval/run_jev_snake.py) — clone [iammusham/jev-snake](https://github.com/iammusham/jev-snake) alongside this file, run against a live local llama-server.
- **Training**: [`scripts/train.py`](scripts/train.py) — minimal LoRA trainer, 10k stratified rows from SargeDev, 1 epoch on a Kaggle T4 (~8 h).

## Training details

- **Base:** `unsloth/gemma-3-4b-it` (public mirror of `google/gemma-3-4b-it`)
- **Adapter:** LoRA rank 16, alpha 32, dropout 0.05
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Data:** 10,000 stratified rows from `SargeDev/jev-distill-corpus-v3`
- **Loss:** KL divergence between teacher soft-label distribution and restricted-logit softmax over verbalizer tokens
- **Optimizer:** AdamW, LR 5e-5 cosine, effective batch 16, 1 epoch (625 steps)
- **Precision:** bf16, gradient checkpointing
- **Hardware:** Kaggle T4 (single GPU)
- **Wall-clock:** ~8 hours

Final training loss ~0.83.

## Data provenance

Trained on [`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3), which is itself soft-label distilled from Jev 1.13's probability distributions via OpenRouter. Streams used:

- **`yuri_v3`** — 498k rows, synthetic operational scenarios across 53 domains, labels distilled from Jev
- **`openjev_v2`** — 94.8k rows from `ZefanCai/Open-Jev` (CC0-1.0), reschema'd
- We dropped `yuri_v1` entirely

Our model therefore inherits some of Jev's calibration signal but is trained independently. All weights released under Apache-2.0.

## Positioning

- **Openness vs Jev:** Apache-2.0 weights, self-hostable, no vendor lock-in.
- **Reasoning vs Laya:** Retains Gemma 3's generation ability — can explain decisions, not only classify.
- **Same size as Kev-4B / Qwen 3.5 4B**, first Gemma-based entry in this tier.

## Inference-time hacks we tested and rejected

Ran on the same 200 boolq items:

| Variant | Accuracy | Δ vs baseline |
|---|---|---|
| **Baseline** (LoRA + restricted logits) | **0.855** | — |
| Few-shot (3 examples in prompt) | 0.805 | −5.0 pp |
| Ensemble (0.5 LoRA + 0.5 base zero-shot) | 0.750 | −10.5 pp |
| Combined | 0.795 | −6.0 pp |

**Baseline wins.** Don't ensemble, don't few-shot — trust the adapter.

## Limitations

- Single-config accuracy for the boolq headline. Full-suite JevBench (22 configs) not yet measured.
- Trained on 10 k rows — smaller than AutoTrust's 25–640 k. Ceiling improves with more data.
- Speed is LLM-tier (~350–800 ms depending on hardware). Not competitive with encoder classifiers like Laya (~30 ms).
- Snake copy-task prompt is a shortlist wrapper; on raw Jev state, the model still needs prompt scaffolding to play well.

## License

Apache-2.0 for the adapter and all code in this repo. The base `unsloth/gemma-3-4b-it` and `google/gemma-3-4b-it` are governed by the Gemma license — please review.
