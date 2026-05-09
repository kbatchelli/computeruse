"""Concatenate the two DPO pair JSONLs (philip's 110 from realistic_train +
kieran-style 500 from ShowUI) into one combined dataset.

Each row format is the same — image_path is relative to a different root for
each subset, so we need to use absolute paths in the output."""

from __future__ import annotations

import json
from pathlib import Path

PCD = Path("/home/shadeform/computeruse/philip-cua-defense")
SOURCES = [
    (PCD / "data" / "dpo_pairs.jsonl", PCD),                          # philip realistic_train (110)
    (PCD / "data" / "dpo_pairs_kieran.jsonl", PCD / "data" / "dpo_train"),     # kieran v1 (500)
    (PCD / "data" / "dpo_pairs_v2.jsonl", PCD / "data" / "dpo_train_v2"),       # kieran v2 (2000)
]
OUT = PCD / "data" / "dpo_pairs_combined.jsonl"

n = 0
with open(OUT, "w") as f:
    for src, root in SOURCES:
        for ln in open(src):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            img = (root / r["image_path"]).resolve()
            r["image_path"] = str(img)  # absolute
            f.write(json.dumps(r) + "\n")
            n += 1

print(f"wrote {n} combined pairs -> {OUT}")
