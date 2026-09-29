# Zyot Decider v3.5 — training recipe

Overnight System One conversion of v3 (verbalizer) → v3.5 (pointer head).
Started 2026-09-29 night, targeting completion by morning of 2026-09-30.

## What this run does

Takes the v3 LoRA and bolts a fixed 24-slot pointer head (AutoTrust-style)
on top of Gemma 3 4B, then fine-tunes both head and LoRA together on 62k
curated rows across all three Jev decision primitives (`noul`, `choice`, `score`).

**Result target:** a real System One decision model — one forward pass produces
a typed probability distribution — not a verbalizer wrapped in different naming.

## Files

| File | Purpose |
|---|---|
| `train.py` | Kaggle-runnable training kernel (Unsloth + PEFT + custom head) |
| `curate_sargedev.py` | Streams SargeDev/jev-distill-corpus-v3 → 57k rows in canonical schema |
| `curate_v2_synth.py` | Regenerates 5.3k v2 hand-crafted rows (S1–S8: math, dates, routing, adversarial, abstention, severity, finance) |
| `kernel-metadata.json` | Kaggle kernel manifest — attaches adapter + curated dataset |
| `dataset-metadata.json` | Kaggle dataset manifest for the curated data |

## Architecture

- Base: `unsloth/gemma-3-4b-it` (Gemma 3 4B multimodal; text-only path used)
- LoRA: r=16, alpha=32, from v3 adapter (trainable)
- Head: `nn.Linear(2560, 24)` initialised from Gemma's LM head verbalizer rows
  (T, F, A-P, 0-5) — at step 0 head output == v3 verbalizer readout exactly
- Slot mapping: `0=T, 1=F, 2-17=A..P (choice), 18-23=0..5 (score)`

## Config (proven overnight config)

```python
NUM_SLOTS      = 24
BATCH          = 2
GA_STEPS       = 8            # effective batch 16
LR             = 3e-5         # conservative — head + LoRA together
MAX_STEPS      = 2500         # ~64% of one epoch on 62k rows @ effective batch 16
SAVE_EVERY     = 10           # rolling checkpoint
EVAL_EVERY     = 200          # named permanent snapshot
WARMUP_STEPS   = 100
MAX_LEN        = 768          # v2's proven config, ~30% faster than 1024
GRAD_CLIP      = 1.0
```

## Known failure modes and their fixes (from v2 and v3 iterations)

| Failure | Root cause | Fix in this recipe |
|---|---|---|
| Dataset attach race | Kernel starts before dataset ready | Wait for upload confirmation before `kaggle kernels push` |
| Wrong dataset path | Personal datasets mount at either `/kaggle/input/<slug>/` or `/kaggle/input/datasets/<owner>/<slug>/` | Multi-candidate probe in code |
| Gemma 3 gated on HF | `google/gemma-3-4b-it` requires acceptance | Use `unsloth/gemma-3-4b-it` public mirror |
| No chat template | `-pt` variant of Gemma has no chat template | Use `-it` |
| `warmup_ratio` TypeError | Newer transformers dropped that arg | Pin `transformers<5` |
| CUBLAS crash on T4×2 | QLoRA + DataParallel deadlock | `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` |
| torchao 0.10 incompat | Ships with newer wheels | `pip uninstall -y torchao` first |
| Loss oscillation | LR too high for combined LoRA+head training | Reduce to 3e-5, warmup 100 steps, grad clip 1.0 |
| Cancelled kernel wipes `/kaggle/working` | Kaggle deletes on cancel — completion or timeout preserves | Save every 10 steps + upload as Kaggle dataset backup |
| bf16 not supported on T4 | Compute capability 7.5 = no bf16 | Unsloth auto-falls to fp16 (warning is safe to ignore) |
| Gemma 3 `config.hidden_size` missing | Gemma 3 is multimodal — nested under `text_config` | `getattr(base.config, 'hidden_size') or base.config.text_config.hidden_size` |

## Data curation

**SargeDev (57k):**
- Only `kind in {noul, choice, score}`
- Only `source in {yuri_v3, openjev_v2}` (drop yuri_v1 — memory-relevance from 32B teacher, per SargeDev card)
- Teacher confidence threshold: `max(target) >= 0.6`
- Fuzzy dedup on `state[:200]`
- Length 50–4000 chars
- 70/30 yuri_v3 to openjev_v2 ratio

**v2 hand-crafted (5.3k):**
- S1 math (800), S2 date (600), S3 routing (800), S4 noise-resistance (700),
  S5 adversarial prompt injection (600) ← directly hits our v3 weakness,
  S6 abstention (500), S7 severity (800), S8 finance routing (500)
- All rendered as multi-choice → slots 2-17
- Adversarial + abstention are the highest-value contributions

## To run

1. Upload curated dataset (`train_final.jsonl`) as a Kaggle dataset — the manifest is `dataset-metadata.json`.
2. Ensure `nishanml/zyot-decider-v3-adapter` (or your equivalent) is available on Kaggle.
3. Push the kernel: `kaggle kernels push -p .` from the folder containing `train.py` and `kernel-metadata.json`.
4. Monitor first 15 min for setup errors, then leave to run ~5-9h.
5. Rolling checkpoints in `/kaggle/working/v3.5/head_ckpt/latest/` + `lora_ckpt/latest/`.

## To resume on a new machine (Colab A100 / new Kaggle account)

```bash
export ZYOT_RESUME=1
python train.py
```

Reads latest checkpoint automatically.

## Lessons from this build (add-only log)

- 2026-09-29: v3 verbalizer showed no measurable delta over base Qwen 3.5 4B — proved that a bigger data + real head is needed to differentiate.
- 2026-09-29: v2 12B research (TIDE, LayerSkip, R-Tuning) was never actually implemented — only paper-planned. v3.5 is the first Zyot model with real research-derived changes (pointer head + R-Tuning-style abstention data).
- 2026-09-29: Kev pointer head architecture chosen as reference (7.8k stars, Apache-2.0, most audited). Kev's *weights* rejected because trained on business decisions not Jev-distilled data.
- 2026-09-29: Adaptive depth / LayerSkip skipped for v3.5 — latency technique, wrong axis for a single-forward-pass decision model, breaks Unsloth speedup, no Gemma 3 port.
- 2026-09-29: Initial LR 5e-5 caused loss oscillation warning from user; reduced to 3e-5 + warmup 100 + grad clip 1.0.
- 2026-09-29: First MAX_STEPS=3900 gave 12.6h ETA (over Kaggle limit); reduced to 2500 + MAX_LEN 1024→768 for guaranteed completion.
