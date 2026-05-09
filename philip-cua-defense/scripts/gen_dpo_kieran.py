"""Generate DPO pairs at scale using Kieran's pipeline (real ShowUI screenshots
+ injection overlays) but capture BOTH the safe-target bbox AND the injection
overlay bbox so we can build (chosen, rejected) preference pairs.

Outputs to philip-cua-defense/data/dpo_train/ — does NOT touch
kieran/training_data/ which is currently being read by SFT.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

# Reuse Kieran's payloads + styles + font helpers
sys.path.insert(0, str(Path("/home/shadeform/computeruse/kieran")))
from generate_training_data import (  # noqa: E402
    BANNER_STYLES,
    INJECTION_TEXTS,
    _load_font,
    _rounded_rect,
)

OUT_W, OUT_H = 1024, 768
PCD = Path("/home/shadeform/computeruse/philip-cua-defense")


def overlay_injection_with_bbox(
    base_img: Image.Image, injection_text: str, style: dict, rng: random.Random
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Like Kieran's overlay_injection but also returns the (x0,y0,x1,y1) bbox in pixels."""
    img = base_img.copy().convert("RGBA")
    w, h = img.size

    font_size = rng.randint(*style["font_range"])
    font = _load_font(font_size)

    tmp = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), injection_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x, pad_y = 20, 12
    bw, bh = tw + pad_x * 2, th + pad_y * 2
    pos = style["position"]
    bg, fg, alpha = style["bg"], style["fg"], style["opacity"]

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if pos == "top":
        x0, y0, x1, y1 = 0, 0, w, bh
        draw.rectangle([x0, y0, x1, y1], fill=(*bg, alpha))
        tx, ty = (w - tw) // 2, pad_y
    elif pos == "bottom":
        x0, y0, x1, y1 = 0, h - bh, w, h
        draw.rectangle([x0, y0, x1, y1], fill=(*bg, alpha))
        tx, ty = (w - tw) // 2, y0 + pad_y
    elif pos == "center":
        x0, y0 = (w - bw) // 2, (h - bh) // 2
        x1, y1 = x0 + bw, y0 + bh
        draw.rectangle([x0, y0, x1, y1], fill=(*bg, alpha))
        tx, ty = x0 + pad_x, y0 + pad_y
    elif pos == "toast_tr":
        margin = 16
        x0, y0 = w - bw - margin, margin
        x1, y1 = x0 + bw, y0 + bh
        _rounded_rect(draw, (x0, y0, x1, y1), 10, (*bg, alpha))
        tx, ty = x0 + pad_x, y0 + pad_y
    elif pos == "toast_bl":
        margin = 16
        x0, y0 = margin, h - bh - margin
        x1, y1 = x0 + bw, y0 + bh
        _rounded_rect(draw, (x0, y0, x1, y1), 10, (*bg, alpha))
        tx, ty = x0 + pad_x, y0 + pad_y
    elif pos == "modal":
        modal_w = min(bw + 60, w - 40)
        modal_h = bh + 40
        x0, y0 = (w - modal_w) // 2, (h - modal_h) // 2
        x1, y1 = x0 + modal_w, y0 + modal_h
        shadow_off = 6
        _rounded_rect(draw, (x0 + shadow_off, y0 + shadow_off, x1 + shadow_off, y1 + shadow_off), 14, (0, 0, 0, 80))
        _rounded_rect(draw, (x0, y0, x1, y1), 14, (*bg, alpha))
        tx = x0 + (modal_w - tw) // 2
        ty = y0 + (modal_h - th) // 2
    else:
        raise ValueError(f"Unknown position: {pos}")

    draw.text((tx, ty), injection_text, fill=(*fg, 255), font=font)
    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB"), (x0, y0, x1, y1)


def click_target_norm(bbox_01: list) -> tuple[int, int]:
    cx = int((bbox_01[0] + bbox_01[2]) / 2 * 1000)
    cy = int((bbox_01[1] + bbox_01[3]) / 2 * 1000)
    return cx, cy


def click_text(x: int, y: int) -> str:
    return f'<tool_call>\n{{"name": "click", "arguments": {{"x": {x}, "y": {y}}}}}\n</tool_call>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=str(PCD / "data" / "dpo_train"))
    ap.add_argument("--jsonl", default=str(PCD / "data" / "dpo_pairs_kieran.jsonl"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("Loading ShowUI-desktop dataset (streaming)...", flush=True)
    ds = load_dataset("showlab/ShowUI-desktop", split="train")
    rows = [r for r in ds if r["type"] in ("situational", "appearance")]
    print(f"  filtered to {len(rows)} natural-language rows")
    rng.shuffle(rows)
    rows = rows[: args.target]

    out_records = []
    for i, row in enumerate(rows):
        img = row["image"].convert("RGB").resize((OUT_W, OUT_H), Image.LANCZOS)
        bbox_01 = row["bbox"]
        instruction = row["instruction"]

        inj_text = rng.choice(INJECTION_TEXTS)
        style = rng.choice(BANNER_STYLES)
        img_inj, inj_bbox_px = overlay_injection_with_bbox(img, inj_text, style, rng)

        # Safe target click (chosen)
        safe_x, safe_y = click_target_norm(bbox_01)

        # Malicious target click (rejected): center of injection overlay,
        # converted from pixel back to 0-999 normalized
        mx_px = (inj_bbox_px[0] + inj_bbox_px[2]) / 2
        my_px = (inj_bbox_px[1] + inj_bbox_px[3]) / 2
        mal_x = int(mx_px / OUT_W * 1000)
        mal_y = int(my_px / OUT_H * 1000)

        img_path = f"images/inj_{i:04d}.png"
        img_inj.save(out_dir / img_path)

        out_records.append({
            "image_path": img_path,
            "prompt": instruction,
            "chosen": click_text(safe_x, safe_y),
            "rejected": click_text(mal_x, mal_y),
            "_meta": {
                "source": "showui-desktop",
                "injection_text": inj_text,
                "safe_bbox_01": bbox_01,
                "injection_bbox_px": list(inj_bbox_px),
                "safe_xy_norm999": [safe_x, safe_y],
                "rejected_xy_norm999": [mal_x, mal_y],
            },
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    out_path = Path(args.jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out_records)} DPO pairs -> {out_path}")
    print(f"images in {out_dir}/images/")


if __name__ == "__main__":
    main()
