"""Verify Northstar is being invoked correctly per HF docs.

Runs three probes side-by-side on a single test image:

  A) HF-doc local style  : just messages with image + text, NO system, NO tools
  B) Our current setup    : system prompt + Qwen tools (what eval.py uses)
  C) CUA-action minimal   : no tools, lightweight system prompt asking for action

Reports raw output text + a sanity-check on the rendered "Sign in" button at
known pixel ~(510, 340).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cua_shared import MODEL_DIR, QWEN_TOOLS, SYS_PROMPT


def make_test_image(path: Path, w: int = 1024, h: int = 768) -> tuple[Image.Image, tuple[int, int]]:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        f = s = ImageFont.load_default()
    d.rectangle([0, 0, w, 60], fill="#2563eb")
    d.text((20, 15), "Acme Mail", fill="white", font=f)
    bbox = [350, 300, 670, 380]
    d.rectangle(bbox, outline="#2563eb", width=3, fill="#dbeafe")
    d.text((430, 322), "Sign in", fill="#1e3a8a", font=f)
    d.text((360, 240), "Welcome back. Click 'Sign in' to continue.", fill="black", font=s)
    img.save(path)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    return img, (cx, cy)


def run(processor, model, messages, tools, max_new_tokens, label):
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if tools is not None:
        kwargs["tools"] = tools
    text = processor.apply_chat_template(messages, **kwargs)
    images = []
    for m in messages:
        if isinstance(m["content"], list):
            for c in m["content"]:
                if c.get("type") == "image":
                    images.append(messages_image)  # set below
                    break
    inputs = processor(text=[text], images=[messages_image], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    out = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\n===== {label} =====")
    print("--- prompt head (first 280 chars) ---")
    print(text[:280])
    print("--- output (clean) ---")
    print(out)
    return out


if __name__ == "__main__":
    img_path = Path("/home/shadeform/computeruse/philip-cua-defense/outputs/probe/test_signin_v2.png")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    messages_image, (target_cx, target_cy) = make_test_image(img_path)
    print(f"[probe] test image saved -> {img_path}")
    print(f"[probe] 'Sign in' button center at pixel ({target_cx}, {target_cy})")
    print(f"[probe] expected normalized 0-999 click around ({int(target_cx/1024*1000)}, {int(target_cy/768*1000)})")

    print("\n[probe] loading model...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model.eval()

    user_text = "Click on the 'Sign in' button."

    # A) HF-doc local style — bare messages, no system, no tools, max_tokens=512
    msgs_a = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": user_text}
    ]}]
    run(processor, model, msgs_a, tools=None, max_new_tokens=512, label="A: HF-doc local style (no sys, no tools, max_tokens=512)")

    # B) Our current setup — system prompt + Qwen tools, max_tokens=128
    msgs_b = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    run(processor, model, msgs_b, tools=QWEN_TOOLS, max_new_tokens=128, label="B: Our current eval setup (sys + Qwen tools)")

    # C) CUA-minimal — no tools, lightweight CUA system prompt
    msgs_c = [
        {"role": "system", "content": "You are a computer-use agent. Output only a single click coordinate as (x, y) in 0-999 normalized space."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    run(processor, model, msgs_c, tools=None, max_new_tokens=64, label="C: CUA-minimal (sys for action format, no tools)")

    print("\n[probe] done.")
