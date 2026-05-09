# computeruse

Monorepo for computer-use model robustness experiments. It currently contains two independent subprojects: a training-data generator for visual prompt-injection robustness and a defense/eval pipeline for fine-tuning a CUA model.

## Repository layout

- `kieran/` — Generates prompt-injection robustness training data by overlaying adversarial banners on desktop screenshots.
- `philip-cua-defense/` — Defense and evaluation pipeline for Northstar CUA (rendering, evals, training, and reports).

## Kieran: training data generator

**Purpose:** Create synthetic screenshot + instruction pairs with injected adversarial text.

**Dependencies:** Python 3, `Pillow`. The screenshot downloader also needs `datasets`.

**Common commands:**

```bash
cd kieran

# Download base screenshots (optional)
python download_base_screenshots.py --count 50

# Generate a dataset (includes 20% clean examples)
python generate_training_data.py --target 1000 --include-clean
```

Output lands in `kieran/training_data/` (images + `dataset.jsonl`). The generator auto-discovers any `.png` files in `kieran/base_screenshots/`.

## Philip CUA defense

**Purpose:** Render synthetic attack scenarios, evaluate baseline ASR, generate training data, and fine-tune a LoRA adapter for a CUA model.

**Quickstart (from `philip-cua-defense/NOTES.md`):**

```bash
cd philip-cua-defense

# Sanity render (no model needed)
python3 scripts/render.py

# Baseline ASR
python3 scripts/eval.py --tag baseline

# Generate training data
python3 scripts/gen_train.py --n 500 --out data/train.jsonl

# LoRA train
python3 scripts/train_lora.py --train data/train.jsonl --out outputs/lora-r16

# Re-eval
python3 scripts/eval.py --adapter outputs/lora-r16 --tag finetuned
```

See `philip-cua-defense/NOTES.md` for environment details, model assumptions, and baseline numbers.
