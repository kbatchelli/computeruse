"""Stitch the baseline + DPO-QA replay mp4s into a side-by-side video
with text labels.

Usage:
    python3 scripts/kernel_compose.py --scenario B2-fake-session-expired
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PCD = Path("/home/shadeform/computeruse/philip-cua-defense")
DEMO_DIR = PCD / "outputs" / "kernel_demo"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = DEMO_DIR / f"{args.scenario}_base.mp4"
    dpo = DEMO_DIR / f"{args.scenario}_dpo.mp4"
    for p in (base, dpo):
        if not p.exists():
            sys.exit(f"missing {p} — run kernel_demo.py first")

    out = Path(args.out) if args.out else DEMO_DIR / f"{args.scenario}_sxs.mp4"

    # Side-by-side stack with labels.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(base),
        "-i", str(dpo),
        "-filter_complex",
        (
            # Scale both to same height, draw text labels, then hstack
            "[0:v]scale=640:-2,drawtext=text='BASELINE Northstar':"
            "fontcolor=white:fontsize=22:box=1:boxcolor=black@0.6:boxborderw=8:x=20:y=20[a];"
            "[1:v]scale=640:-2,drawtext=text='DPO-QA finetuned':"
            "fontcolor=white:fontsize=22:box=1:boxcolor=black@0.6:boxborderw=8:x=20:y=20[b];"
            "[a][b]hstack=inputs=2"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    ]
    run(cmd)
    print(f"\nwrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
