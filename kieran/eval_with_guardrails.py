#!/usr/bin/env python3
"""
Eval Northstar + NeMo Guardrails output rail against Philip's realistic scenarios.

Reports two ASR numbers per run:
  - raw_asr:     model output scored directly (replicates Philip's baseline)
  - guarded_asr: model output passed through guardrail first; blocked actions
                 are treated as safe (ASR=0 for that scenario)

Usage (on the VM):
    python3 kieran/eval_with_guardrails.py --tag baseline
    python3 kieran/eval_with_guardrails.py --adapter outputs/lora-r16 --tag finetuned
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PHILIP_SCRIPTS = REPO_ROOT / "philip-cua-defense" / "scripts"
PHILIP_EVALS   = REPO_ROOT / "philip-cua-defense" / "evals"
PHILIP_OUTPUTS = REPO_ROOT / "philip-cua-defense" / "outputs"
KIERAN_ROOT    = Path(__file__).resolve().parent

sys.path.insert(0, str(PHILIP_SCRIPTS))
sys.path.insert(0, str(KIERAN_ROOT))

from cua_shared import infer_action, load_northstar, load_scenarios, score_attack  # Philip's, read-only
from guardrails.guardrail_wrapper import wrap as guardrail_wrap                    # ours


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="Path to LoRA adapter (optional)")
    ap.add_argument("--tag", default="baseline", help="Tag for output filename")
    ap.add_argument("--render-dir", default=None, help="Override realistic render dir")
    args = ap.parse_args()

    render_dir = Path(args.render_dir) if args.render_dir else PHILIP_OUTPUTS / "realistic_renders"
    scen_path  = PHILIP_EVALS / "scenarios.json"

    if not scen_path.exists():
        sys.exit(f"scenarios.json not found at {scen_path}")
    if not render_dir.exists():
        sys.exit(f"Render dir not found at {render_dir}. Run philip-cua-defense/realistic/render_scenarios.py first.")

    scenarios = load_scenarios(scen_path)
    lm = load_northstar(adapter_path=args.adapter)

    rows = []
    raw_by_family     = defaultdict(lambda: [0, 0])
    guarded_by_family = defaultdict(lambda: [0, 0])
    raw_overall       = [0, 0]
    guarded_overall   = [0, 0]

    for scen in scenarios:
        sid = scen["id"]
        png_path  = render_dir / f"{sid}.png"
        if not png_path.exists():
            print(f"  [skip ] {sid}: missing render")
            continue

        img    = Image.open(png_path).convert("RGB")
        result = infer_action(lm, img, scen["benign_task"])
        action = result["parsed"]
        crit   = scen["success_criteria"]

        # Score raw (no guardrail)
        raw_attack = score_attack(crit, action)

        # Score after guardrail
        raw_action_str = result["raw"]
        instruction    = scen["benign_task"]
        gr = guardrail_wrap(raw_action_str, instruction, verbose=False)
        if gr["blocked"]:
            # Guardrail intercepted — treat as safe
            guarded_attack = False
        else:
            guarded_attack = raw_attack  # guardrail passed it through unchanged

        fam = scen["family"]
        raw_by_family[fam][1]     += 1
        raw_by_family[fam][0]     += int(raw_attack)
        guarded_by_family[fam][1] += 1
        guarded_by_family[fam][0] += int(guarded_attack)
        raw_overall[1]     += 1
        raw_overall[0]     += int(raw_attack)
        guarded_overall[1] += 1
        guarded_overall[0] += int(guarded_attack)

        raw_flag     = "ATTK✓" if raw_attack     else "safe "
        guard_flag   = "BLOCK " if gr["blocked"]  else ("ATTK✓" if guarded_attack else "safe ")
        print(
            f"  raw=[{raw_flag}] guarded=[{guard_flag}]  {sid:35s}"
            + (f"  blocked: {gr['reason']}" if gr["blocked"] else ""),
            flush=True,
        )

        rows.append({
            "id": sid,
            "family": fam,
            "task": scen["benign_task"],
            "raw_output": result["raw"],
            "parsed_action": action,
            "raw_attack_succeeded": bool(raw_attack),
            "guardrail_blocked": bool(gr["blocked"]),
            "guardrail_reason": gr["reason"],
            "guarded_attack_succeeded": bool(guarded_attack),
        })

    def asr(counts):
        return counts[0] / counts[1] if counts[1] else 0.0

    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "raw_overall_asr":     asr(raw_overall),
        "guarded_overall_asr": asr(guarded_overall),
        "asr_reduction":       asr(raw_overall) - asr(guarded_overall),
        "guardrail_blocks":    sum(1 for r in rows if r["guardrail_blocked"]),
        "total_scenarios":     raw_overall[1],
        "by_family": {
            fam: {
                "raw_asr":     asr(raw_by_family[fam]),
                "guarded_asr": asr(guarded_by_family[fam]),
                "total":       raw_by_family[fam][1],
            }
            for fam in raw_by_family
        },
    }

    out = PHILIP_OUTPUTS / f"eval_guarded_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n=== SUMMARY ===")
    print(f"  Raw ASR:        {summary['raw_overall_asr']:.1%}  ({raw_overall[0]}/{raw_overall[1]})")
    print(f"  Guarded ASR:    {summary['guarded_overall_asr']:.1%}  ({guarded_overall[0]}/{guarded_overall[1]})")
    print(f"  ASR reduction:  {summary['asr_reduction']:.1%}")
    print(f"  Blocks fired:   {summary['guardrail_blocks']}/{raw_overall[1]}")
    print("\n  By family:")
    for fam, d in summary["by_family"].items():
        print(f"    {fam:15s}  raw={d['raw_asr']:.1%}  guarded={d['guarded_asr']:.1%}  (n={d['total']})")
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
