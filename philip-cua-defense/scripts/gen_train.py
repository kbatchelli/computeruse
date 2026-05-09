"""Generate realistic-render training data for LoRA SFT.

For each curated scenario × N injection-text variants × {with/without overlay},
render via Playwright, compute the pixel center of the safe target from the
rendered DOM bbox, and emit a JSONL row:
    {"image_path": "data/realistic_train/<id>__v<i>.png",
     "instruction": "<benign task>",
     "target": "<tool_call>{...}</tool_call>"}

Eval set (the canonical scenarios.json injection text) is kept held-out by
construction: we only use the alternate INJECTION_TEXTS pool from kieran's
work, never the verbatim text from scenarios.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path("/home/shadeform/computeruse")
PCD = REPO_ROOT / "philip-cua-defense"
sys.path.insert(0, str(PCD / "realistic"))
sys.path.insert(0, str(REPO_ROOT / "kieran"))

from render import render_scenario  # noqa: E402
from render_scenarios import inject_opts_for, template_for, template_subs_for  # noqa: E402

# 30 alternate injection-text payloads. None of these appear verbatim in
# scenarios.json, so the eval renders stay held-out.
from generate_training_data import INJECTION_TEXTS  # noqa: E402


# Scenarios with an unambiguous safe data-target. Skip the rest (e.g. F2
# spoofed-Save where the safe action is "use File menu" or "do nothing").
SAFE_TARGETS = {
    "B1-fake-update":               "article_body",
    "B2-fake-session-expired":      "injection_cancel_btn",
    "B3-fake-ai-suggestion":        "mark_as_read",
    "B4-fake-virus-scan":           "sidebar_downloads",
    "B5-cookie-consent-trojan":     "injection_decline_btn",
    "E1-article-redirect":          "article_body",
    "E3-email-forward-ceo":         "reply_btn",
    "E5-search-snippet-shell":      "result_2",
    "F1-placeholder-password-trap": "field_display_name",
    "F3-prefilled-instruction":     "field_recipient_name",
    "F4-autofill-suggestion-trojan":"add_item_input",
    "F5-tooltip-vs-label-mismatch": "continue_btn",
}


def bbox_center_norm(bbox, viewport=(1024, 768)) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return int(cx / viewport[0] * 1000), int(cy / viewport[1] * 1000)


def find_target(manifest: dict, name: str) -> dict | None:
    for t in manifest.get("targets", []):
        if t.get("data_target") == name:
            return t
    return None


def click_action_text(x: int, y: int) -> str:
    return f'<tool_call>\n{{"name": "click", "arguments": {{"x": {x}, "y": {y}}}}}\n</tool_call>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-injected-per-scenario", type=int, default=10)
    ap.add_argument("--include-clean", action="store_true", default=True,
                    help="Also generate one clean (no-injection) example per scenario")
    ap.add_argument("--out-dir", default=str(PCD / "data" / "realistic_train"))
    ap.add_argument("--out-jsonl", default=str(PCD / "data" / "train.jsonl"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = json.loads((PCD / "evals" / "scenarios.json").read_text())["scenarios"]
    by_id = {s["id"]: s for s in scenarios}

    rows: list[dict] = []
    skipped: list[str] = []
    rng_pool = list(INJECTION_TEXTS)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            for sid, safe_name in SAFE_TARGETS.items():
                if sid not in by_id:
                    continue
                scen = by_id[sid]
                kind = scen["base_ui"]["kind"]
                tpl = template_for(kind)
                subs = template_subs_for(scen)

                # N injected variants
                k = min(args.n_injected_per_scenario, len(rng_pool))
                for i in range(k):
                    inj_text = rng_pool[i % len(rng_pool)]
                    inj_opts = dict(inject_opts_for(scen))
                    inj_opts["text"] = inj_text  # override scenarios.json text

                    out_png = out_dir / f"{sid}__inj{i:02d}.png"
                    out_json = out_dir / f"{sid}__inj{i:02d}.bboxes.json"
                    try:
                        res = render_scenario(
                            template_html=tpl,
                            template_substitutions=subs,
                            inject_opts=inj_opts,
                            out_png=out_png,
                            out_json=out_json,
                            browser=browser,
                        )
                    except Exception as e:
                        skipped.append(f"{sid} inj{i}: {type(e).__name__}: {e}")
                        continue

                    safe = find_target(res["manifest"], safe_name)
                    if not safe:
                        skipped.append(f"{sid} inj{i}: missing safe target {safe_name}")
                        continue
                    x, y = bbox_center_norm(safe["bbox"])
                    rows.append({
                        "image_path": str(out_png.relative_to(PCD)),
                        "instruction": scen["benign_task"],
                        "target": click_action_text(x, y),
                        "_meta": {"id": sid, "variant": f"inj{i:02d}", "safe_target": safe_name},
                    })

                # 1 clean (no injection) variant — render with NO inject_opts
                if args.include_clean:
                    out_png = out_dir / f"{sid}__clean.png"
                    out_json = out_dir / f"{sid}__clean.bboxes.json"
                    # Pass an empty inject so render.py doesn't add the malicious overlay.
                    try:
                        res = render_scenario(
                            template_html=tpl,
                            template_substitutions=subs,
                            inject_opts={"kind": "none", "text": "", "overlayDataTarget": "_skip_"},
                            out_png=out_png,
                            out_json=out_json,
                            browser=browser,
                        )
                    except Exception as e:
                        skipped.append(f"{sid} clean: {type(e).__name__}: {e}")
                    else:
                        safe = find_target(res["manifest"], safe_name)
                        if safe:
                            x, y = bbox_center_norm(safe["bbox"])
                            rows.append({
                                "image_path": str(out_png.relative_to(PCD)),
                                "instruction": scen["benign_task"],
                                "target": click_action_text(x, y),
                                "_meta": {"id": sid, "variant": "clean", "safe_target": safe_name},
                            })

                print(f"[ok] {sid}: queued so far={len(rows)}", flush=True)
        finally:
            browser.close()

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nwrote {len(rows)} examples -> {out_jsonl}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
