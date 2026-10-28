# Arbiter v4 — 12B

The next-generation Arbiter from **Zyot Lab (Codekins Pvt Ltd)**: **Gemma-4 12B IT + LoRA (r 32, α 64)** with a
fixed **28-slot decision head** read at the last token. One forward pass returns a calibrated probability for
yes/no, multiple-choice (up to 16 options) and ordinal scores. No sampling, no text parsing.

- **Weights (not stored in this repo):** [hiteshluke/arbiter-v4-12b](https://huggingface.co/hiteshluke/arbiter-v4-12b) — LoRA adapter + `head.pt`
- **Base model:** `google/gemma-4-12b-it` (Gemma Terms of Use)
- **Previous version:** [Arbiter v3.3 (4B)](../README.md), live on Ollaya: <https://ollaya.dev/library/arbiter>

## Benchmarks (our own runs)

| Benchmark | Result | Notes |
|---|---|---|
| BoolQ (validation, n = 3,270) | **89.2 %** | majority-class baseline 62.2 % |
| Ollaya typed-decisions (2,000 questions) | **0.640** | head-scored, calibration temperature 3.4 |
| ↳ Yes / No | 80.3 % | 482 / 600 |
| ↳ Multiple-choice | 66.3 % | 398 / 600 |
| ↳ Ordinal score | 50.1 % | 401 / 800 |

These numbers come from our own benchmark runs on Kaggle GPUs. **Arbiter v4 is not yet listed or evaluated on the
Ollaya leaderboard**; only v3.3 is. Raw results: [`results/benchmark_results.json`](results/benchmark_results.json).
Full write-up, including what did not work: [`BENCHMARK-AND-RETRAIN-PLAN.md`](BENCHMARK-AND-RETRAIN-PLAN.md).

## Run it yourself

The model is not packaged for Ollama or Ollaya yet. With a GPU you can load the base model, the LoRA and `head.pt`
with `transformers` + `peft` and read the head at the final prompt token, the same way as the
[v3.3 usage example](https://huggingface.co/hiteshluke/arbiter-4b). The benchmark scripts in [`scripts/`](scripts/)
show the exact prompt format and head read-out (set your own `HF_TOKEN` in the environment).

## Layout

- `scripts/` — benchmark scripts (Kaggle) with tokens removed
- `results/` — measured results
- `BENCHMARK-AND-RETRAIN-PLAN.md` — results, leaderboard analysis and the retrain plan
