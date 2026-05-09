#!/usr/bin/env python3
"""Download 20 diverse desktop screenshots from ShowUI-desktop for use as base images."""

import random
from pathlib import Path

from datasets import load_dataset

TARGET_FILES = [
    "email_inbox.png",
    "search_engine.png",
    "file_manager.png",
    "text_editor.png",
    "browser_tabs.png",
    "settings_page.png",
    "calendar.png",
    "chat_app.png",
    "code_editor.png",
    "shopping_cart.png",
    "login_page.png",
    "spreadsheet.png",
    "photo_gallery.png",
    "terminal.png",
    "music_player.png",
    "maps.png",
    "notifications.png",
    "video_player.png",
    "form.png",
    "dashboard.png",
]

OUT_DIR = Path("base_screenshots")


def main():
    print("Loading ShowUI-desktop dataset...")
    ds = load_dataset("showlab/ShowUI-desktop", split="train")
    total = len(ds)
    print(f"Total rows: {total}")

    # Sample 20 evenly-spaced indices across the full dataset for maximum diversity
    rng = random.Random(42)
    step = total // len(TARGET_FILES)
    indices = [i * step + rng.randint(0, step - 1) for i in range(len(TARGET_FILES))]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname, idx in zip(TARGET_FILES, indices):
        row = ds[idx]
        img = row["image"].convert("RGB")
        img = img.resize((1280, 800))
        dst = OUT_DIR / fname
        img.save(dst)
        print(f"Saved {dst}  (idx={idx}, instruction: {row['instruction'][:60]})")

    print(f"\nDone — {len(TARGET_FILES)} screenshots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
