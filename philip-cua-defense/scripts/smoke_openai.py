"""Smoke-test the OpenAI-compatible endpoint with the test signin PNG."""
from __future__ import annotations
import argparse
import base64
import sys
from pathlib import Path
from openai import OpenAI

ROOT = Path("/home/shadeform/computeruse/philip-cua-defense")
DEFAULT_PNG = ROOT / "outputs" / "probe" / "test_signin.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="northstar-cua-fast")
    ap.add_argument("--png", default=str(DEFAULT_PNG))
    ap.add_argument("--task", default="Click 'Sign in'")
    args = ap.parse_args()
    png = Path(args.png).read_bytes()
    b64 = base64.b64encode(png).decode("ascii")
    client = OpenAI(base_url=args.base_url, api_key="dummy", timeout=120.0)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": args.task},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        max_tokens=256,
        temperature=0.0,
    )
    out = resp.choices[0].message.content
    print("=== OUTPUT ===")
    print(out)
    if "<tool_call>" in (out or ""):
        print("\nOK: model emitted <tool_call>")
        sys.exit(0)
    print("\nWARN: no <tool_call> in output")
    sys.exit(1)


if __name__ == "__main__":
    main()
