# Kieran's Training Data Generator

## What this is

Generates prompt-injection robustness training data for finetuning a computer-use model to ignore adversarial text overlaid on screenshots.

## Structure

```
kieran/
  generate_training_data.py   # Main generator — overlays injection banners on screenshots, outputs JSONL
  download_base_screenshots.py # Downloads real desktop screenshots from HuggingFace (ShowUI-desktop)
  base_screenshots/            # 50 real desktop PNGs (numbered base_000..base_049)
  training_data/
    images/                    # Generated augmented PNGs (not committed, regenerate with the script)
    dataset.jsonl              # One JSON record per example
  README.md                    # Full docs with CLI flags, injection categories, etc.
```

## Key design decisions

- Base images are decoupled from instructions — the generator auto-discovers any `.png` in `base_screenshots/` and randomly pairs them with the 20-task instruction pool.
- 72 injection texts across 15 attack categories (direct hijacks, fake errors, social engineering, multi-language, subtle/innocuous, chained multi-step, etc.)
- 12 visual banner styles (top/bottom bars, toasts, modals with drop shadows) so the model learns to recognize injections by content, not appearance.
- Generated images in `training_data/images/` are NOT committed (too large, fully reproducible). Run the script to regenerate.
- `--include-clean` adds 20% clean examples as negative samples.

## Common commands

```bash
# Regenerate the full dataset
python generate_training_data.py --target 1000 --include-clean

# Download more base screenshots
python download_base_screenshots.py --count 100

# Quick test run
python generate_training_data.py --target 20
```

## Dependencies

- Pillow (always needed)
- datasets (only for download_base_screenshots.py)
