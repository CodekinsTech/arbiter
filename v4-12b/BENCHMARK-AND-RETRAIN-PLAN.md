# Arbiter v4 12B — Benchmark Results, Why We're Low, and the Retrain Fix

**Date:** 2026-10-08
**Model:** Gemma 4 12B IT + LoRA (r=32, alpha=64) + 28-slot decision head
**Checkpoint benchmarked:** step-2500 (final; max_steps reached) — head `step-2500`, LoRA `step-2500`, calibration_temp 3.4
**Benchmark:** Ollaya typed-decisions (`LocalLLaMA/typed-decisions`, 400 states / ~2000 questions). Score = accuracy vs majority/gold label. Low annotator agreement → scores are comparative, ceiling is NOT 1.0.

---

## 1. Our results

| Method | Ollaya score | noul | choice | score |
|---|---|---|---|---|
| **Head-based** (how the model actually decides) | **0.6405** (1281/2000) | 0.803 | 0.663 | **0.501** |
| **Text-gen** (llama.cpp / deployed way) | **0.316** (632/2000) | 0.508 | 0.530 | **0.011** |

Text-gen had **781/2000 unparsed (39%)**.
(EMA-head head-based number: pending a quick re-run — expected ~0.64-0.65, text-gen is head-independent so unchanged.)

Results JSON: `benchmark_v4/results/v4-12b-bench/benchmark_results.json`

---

## 2. The leaderboard (typed-decisions, pulled from ollaya.dev/search 2026-10-08)

| Rank | Model | Score |
|---|---|---|
| 1 | winnow | 0.722 |
| 2 | clef | 0.703 |
| 3 | cygnet | 0.683 |
| 4 | **decider (4b)** | **0.680** |
| 4 | jeeves | 0.680 |
| 6 | kev | 0.669 |
| 7 | nimble | 0.665 |
| 8 | snap | 0.648 |
| — | **our v4 12B (head)** | **0.640** |
| 9 | jevk5 | 0.625 |
| 10 | decider (2b) | 0.591 |
| 11 | nli | 0.548 |
| 12 | decima | 0.495 |

We land ~#8, mid-pack. Top (winnow) = 0.722, gap **0.08**. No per-category breakdown is published on the leaderboard.

---

## 3. WHY we're low (root causes)

**A. `score` questions are the killer.** head-based `score` = 0.501 (coin-flip), and `score` is **800/2000 = 40% of the test**. If `score` matched `noul` (0.80), overall → **~0.76 = above winnow**. This one category is the whole gap.

**B. Two facts prove it's recipe, NOT model size:**
- We used **winnow's exact LoRA config** (r=32/alpha=64) — winnow is #1 at 0.722, we got 0.64. Same recipe ≠ same result → config is not the gap.
- **A 4B model (decider) beats our 12B** (0.680 vs 0.640). 3× the params, loses. So capacity is not the problem; training/method is. The 12B is being wasted.

**C. Concrete reasons `score` is weak:**
1. **Under-sampled in training.** `MIN_SCORE` = 5,000 vs `MIN_CHOICE` = 30,000 (see `training_v3_9_12b/kaggle_current.py` CONFIG). The model barely saw score examples.
2. **Non-ordinal head.** Score is ordinal (ranked levels 2-10), but the 28-slot head treats slots 18-27 as flat independent classes — no "4 is closer to 5 than to 9" structure. Modeling mismatch, not just a data gap.

**D. Separate deploy problem (text-gen).** The model was trained with the head, so it never learned to EMIT a clean bare answer token → 39% unparsed in text-gen, score type 0.011. This cripples llama.cpp/Ollama text-gen deployment (how Zyot Decider ships). The underlying reasoning (head) is fine.

---

## 4. WHAT makes it better (the retrain fix)

**Priority 1 — fix `score` (biggest lever, ~+0.12 overall):**
- **Rebalance data:** raise score examples massively — set `TARGET`/`MIN_SCORE` so score ≈ choice (e.g. 30k+ score rows). Pull more score-type rows from `tasksource-jev-typed-decisions` + `procedural-typed-decisions`.
- **Ordinal-aware score loss:** instead of flat cross-entropy over score slots, use an ordinal/soft-label loss (e.g. penalize by distance |pred_level - true_level|, or regression-to-level head, or CORN/CORAL ordinal). Keeps "near misses" cheaper than "far misses."

**Priority 2 — fix text-gen deploy (if shipping via llama.cpp):**
- Add **answer-token SFT**: a short finetune where the target is literally the bare answer token (letter/digit) after "Answer:", so the model emits parseable output. Kills the 39% unparsed.
- OR just **deploy WITH the head** (classifier inference) → get the full 0.64 immediately, skip text-gen entirely.

**Priority 3 — squeeze the rest toward winnow (0.72):**
- Longer/cleaner score training once ordinal loss is in.
- Revisit `choice` (0.66) — check option-shuffling / position bias.
- Consider self-consistency at inference (multi-sample vote) — earlier SC gave small gains.

---

## 5. Retrain time estimates

- **Delta fine-tune** (continue from step-2500, add rebalanced score data + ordinal loss): **~5-8h = 1 Kaggle session.** Do this first — if score → 0.75+, overall ≈ 0.76.
- **Full retrain from scratch** (rebalanced data + ordinal head): **~24h = 2 sessions (~2 days).** Only if delta underperforms.
- **Prep** (edit curation balance + wire ordinal loss in the training script): **~1-2h.**

---

## 6. Artifacts (everything saved)

- Training script: `training_v3_9_12b/kaggle_current.py` (also `models/training-scripts/arbiter-v4-12b-train.py` + NOTES)
- Benchmark script: `benchmark_v4/kaggle_benchmark.py` (mirrors training prompts/head/slots; head-based + text-gen)
- Benchmark results JSON: `benchmark_v4/results/v4-12b-bench/benchmark_results.json`
- HF checkpoints: `hiteshluke/arbiter-v4-12b-ckpts` — step-2500 (head+lora), step-ema-final (head only)
- Kaggle runs: kingk777/arbiter-v4-12b-benchmark (full), kingk777/arbiter-v4-12b-benchmark-ema (head-only)
- Leaderboard source: https://ollaya.dev/search

**When retraining:** start from this file. The target is `score` → ~0.80 via (1) data rebalance to ~30k score rows and (2) ordinal score loss. That alone should reach ~0.76 overall (#1). Deploy with the head, or add answer-token SFT before any text-gen deploy.
