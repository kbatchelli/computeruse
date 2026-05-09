"""Build DPO (chosen, rejected) preference pairs for prompt-injection defense.

For each scenario in our realistic_train/ set:
  - chosen   = the existing target (click on the safe bbox center)
  - rejected = a click on the malicious-element bbox center (read from the
               sibling .bboxes.json — any element with is_injection=true)

Output JSONL has columns matching trl.DPOTrainer's expected vision dataset:
  - prompt: the user instruction
  - chosen: the safe <tool_call> string
  - rejected: the malicious <tool_call> string
  - image_path: relative to --root
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PCD = Path("/home/shadeform/computeruse/philip-cua-defense")


def click_text(x: int, y: int) -> str:
    return f'<tool_call>\n{{"name": "click", "arguments": {{"x": {x}, "y": {y}}}}}\n</tool_call>'


def find_injection_bbox(bbox_manifest_path: Path) -> tuple[int, int] | None:
    """Return the pixel center of the FIRST is_injection target, in 0-999 norm."""
    if not bbox_manifest_path.exists():
        return None
    data = json.loads(bbox_manifest_path.read_text())
    vw = data.get("viewport", {}).get("width", 1024)
    vh = data.get("viewport", {}).get("height", 768)
    for t in data.get("targets", []):
        if not t.get("is_injection"):
            continue
        # Prefer button-like injection targets if available
        x1, y1, x2, y2 = t["bbox"]
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        cx_norm = int(cx_px / vw * 1000)
        cy_norm = int(cy_px / vh * 1000)
        return cx_norm, cy_norm
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default=str(PCD / "data" / "train.jsonl"),
                    help="SFT-format JSONL (must reference realistic_train images with sibling .bboxes.json)")
    ap.add_argument("--root", default=str(PCD),
                    help="Root for resolving image_path")
    ap.add_argument("--out", default=str(PCD / "data" / "dpo_pairs.jsonl"))
    args = ap.parse_args()

    root = Path(args.root)
    rows_out: list[dict] = []
    skipped_clean = 0
    skipped_no_inj = 0

    with open(args.in_jsonl) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            img_rel = row["image_path"]
            img_path = (root / img_rel).resolve()
            bbox_path = img_path.with_suffix(".bboxes.json")

            # Need both image + bbox manifest
            if not bbox_path.exists():
                # Probably a clean (no-injection) example — no malicious target to pair against.
                skipped_clean += 1
                continue
            mal = find_injection_bbox(bbox_path)
            if mal is None:
                skipped_no_inj += 1
                continue

            chosen = row["target"]
            rejected = click_text(mal[0], mal[1])

            rows_out.append({
                "image_path": img_rel,
                "prompt": row["instruction"],
                "chosen": chosen,
                "rejected": rejected,
                "_meta": row.get("_meta", {}),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows_out)} DPO pairs -> {out}")
    print(f"skipped: {skipped_clean} clean (no bbox manifest), {skipped_no_inj} no-injection-element")


if __name__ == "__main__":
    main()
