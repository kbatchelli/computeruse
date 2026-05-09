"""Probe Northstar to see what action grammar it actually emits.

We don't have local docs for the exact tool schema the model was RL'd on.
The chat template is Qwen function-calling style. Try a few variants and dump
the raw decoded text so we can pin down the parser before writing the eval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_DIR = "/home/shadeform/northstar"
OUT_DIR = Path("/home/shadeform/computeruse/philip-cua-defense/outputs/probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_test_screenshot(path: Path, w: int = 1024, h: int = 768) -> None:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        small = font
    d.rectangle([0, 0, w, 60], fill="#2563eb")
    d.text((20, 15), "Acme Mail", fill="white", font=font)
    d.rectangle([350, 300, 670, 380], outline="#2563eb", width=3, fill="#dbeafe")
    d.text((430, 322), "Sign in", fill="#1e3a8a", font=font)
    d.text((360, 240), "Welcome back. Click 'Sign in' to continue.", fill="black", font=small)
    img.save(path)


def main() -> None:
    print("Loading model...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model.eval()
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    img_path = OUT_DIR / "test_signin.png"
    make_test_screenshot(img_path)
    img = Image.open(img_path).convert("RGB")

    instruction = "Click on the 'Sign in' button."

    qwen_tools = [
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click at normalized 0-999 coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "type",
                "description": "Type text into the focused field.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "navigate",
                "description": "Navigate the browser to a URL.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminate",
                "description": "End the task.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    variants = {
        "no_tools_no_sys": {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ]},
            ],
            "tools": None,
        },
        "with_tools_no_sys": {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ]},
            ],
            "tools": qwen_tools,
        },
        "with_sys_cua_directive": {
            "messages": [
                {"role": "system", "content": "You are a computer-use agent. Given a screenshot and a goal, emit a single action. Coordinates are normalized 0-999."},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ]},
            ],
            "tools": None,
        },
        "with_tools_and_sys": {
            "messages": [
                {"role": "system", "content": "You are a computer-use agent. Use the provided tools to interact. Coordinates are normalized 0-999."},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ]},
            ],
            "tools": qwen_tools,
        },
    }

    results = {}
    for name, cfg in variants.items():
        print(f"\n===== variant: {name} =====", flush=True)
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if cfg["tools"]:
            kwargs["tools"] = cfg["tools"]
        prompt_text = processor.apply_chat_template(cfg["messages"], **kwargs)
        print(f"--- prompt (first 600 chars) ---\n{prompt_text[:600]}\n--- end prompt head ---", flush=True)

        inputs = processor(
            text=[prompt_text],
            images=[img],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        decoded = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
        decoded_clean = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"--- raw output (with special tokens) ---\n{decoded}", flush=True)
        print(f"--- output (clean) ---\n{decoded_clean}", flush=True)
        results[name] = {"raw": decoded, "clean": decoded_clean}

    (OUT_DIR / "probe_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT_DIR / 'probe_results.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
