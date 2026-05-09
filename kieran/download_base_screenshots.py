#!/usr/bin/env python3
"""Download diverse desktop screenshots from ShowUI-desktop for use as base images."""

import argparse
import random
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("base_screenshots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Number of base screenshots to download")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="base_screenshots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ShowUI-desktop dataset...")
    ds = load_dataset("showlab/ShowUI-desktop", split="train")
    total = len(ds)
    print(f"Total rows: {total}")

    rng = random.Random(args.seed)
    step = total // args.count
    indices = [i * step + rng.randint(0, step - 1) for i in range(args.count)]

    for i, idx in enumerate(indices):
        row = ds[idx]
        img = row["image"].convert("RGB")
        img = img.resize((1280, 800))
        dst = out_dir / f"base_{i:03d}.png"
        img.save(dst)
        print(f"Saved {dst}  (idx={idx})")

    print(f"\nDone — {args.count} screenshots saved to {out_dir}/")


if __name__ == "__main__":
    main()
