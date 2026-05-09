"""Run all scenarios through the model, score ASR per family + overall.

Usage:
    python scripts/eval.py                 # baseline (no adapter)
    python scripts/eval.py --adapter outputs/lora-r16
    python scripts/eval.py --adapter outputs/lora-r16 --tag finetuned
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cua_shared import infer_action, load_northstar, load_scenarios, score_attack
from render import render_scenario

ROOT = Path("/home/shadeform/computeruse/philip-cua-defense")
SCEN_PATH = ROOT / "evals" / "scenarios.json"
RENDER_DIR = ROOT / "outputs" / "eval_renders"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    scenarios = load_scenarios(SCEN_PATH)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    lm = load_northstar(adapter_path=args.adapter)

    rows = []
    family_totals = defaultdict(lambda: [0, 0])  # family -> [success, total]
    overall = [0, 0]

    for scen in scenarios:
        img = render_scenario(scen)
        img_path = RENDER_DIR / f"{scen['id']}.png"
        img.save(img_path)

        result = infer_action(lm, img, scen["benign_task"])
        action = result["parsed"]
        attack_succeeded = score_attack(scen["success_criteria"], action)

        family_totals[scen["family"]][1] += 1
        family_totals[scen["family"]][0] += int(attack_succeeded)
        overall[1] += 1
        overall[0] += int(attack_succeeded)

        rows.append({
            "id": scen["id"],
            "family": scen["family"],
            "task": scen["benign_task"],
            "raw_output": result["raw"],
            "parsed_action": action,
            "attack_succeeded": bool(attack_succeeded),
        })
        flag = "ATTACK✓" if attack_succeeded else "safe   "
        print(f"  [{flag}] {scen['id']:35s}  {action.get('name') if action else '<unparsed>'}", flush=True)

    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "by_family_asr": {
            fam: {"success": s, "total": t, "asr": (s / t if t else 0.0)}
            for fam, (s, t) in family_totals.items()
        },
        "overall_asr": overall[0] / overall[1] if overall[1] else 0.0,
        "overall_success": overall[0],
        "overall_total": overall[1],
    }

    out = ROOT / "outputs" / f"eval_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
