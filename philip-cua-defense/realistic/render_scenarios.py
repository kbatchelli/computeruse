"""Driver: read scenarios.json, render each scenario into a realistic
screenshot + bbox manifest using `render.py`.

Usage:
    python render_scenarios.py                # render all 15
    python render_scenarios.py B1-fake-update # render one by id
    python render_scenarios.py B1 E2 F3       # render several by id-prefix
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from render import render_scenario  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
SCENARIOS_PATH = ROOT / "evals" / "scenarios.json"
TEMPLATES_DIR = THIS_DIR / "templates"
OUT_DIR = ROOT / "outputs" / "realistic_renders"


# ---------- mapping helpers ----------

def template_for(kind: str) -> Path:
    return TEMPLATES_DIR / f"{kind}.html"


def template_subs_for(scenario: dict) -> dict:
    """Map scenario.base_ui -> {{KEY}} substitutions used by the template."""
    bui = scenario.get("base_ui", {}) or {}
    kind = bui.get("kind", "")

    if kind == "wiki_article":
        return {
            "TITLE": bui.get("title", "Article"),
            "BODY":  bui.get("body_excerpt", ""),
        }
    if kind == "banking_dashboard":
        return {
            "ACCOUNT_LABEL": bui.get("account_label", "Checking ****0000"),
            "BALANCE":       bui.get("balance_text", "$0.00"),
        }
    if kind == "email_inbox":
        em = bui.get("selected_email", {}) or {}
        return {
            "FROM":    em.get("from", "Sender <a@b.com>"),
            "SUBJECT": em.get("subject", "(no subject)"),
        }
    if kind == "file_manager":
        cur = (bui.get("current_dir") or "/Downloads").rstrip("/").split("/")[-1] or "/"
        return {"DIR_NAME": cur}
    if kind == "news_article":
        return {"HEADLINE": bui.get("headline", "Untitled story")}
    if kind == "code_editor":
        return {"FILENAME": bui.get("filename", "main.py")}
    if kind == "email_thread":
        return {
            "FROM":    bui.get("from", "Sender <a@b.com>"),
            "SUBJECT": bui.get("subject", "(no subject)"),
        }
    if kind == "pdf_invoice":
        return {
            "VENDOR": bui.get("vendor", "Acme"),
            "TOTAL":  bui.get("total_text", "$0.00"),
        }
    if kind == "search_results":
        return {
            "QUERY":        bui.get("query", "search"),
            "RESULTS_JSON": bui.get("results", []),
        }
    if kind == "settings_page":
        fields = bui.get("fields", []) or []
        # Render the listed fields as a vertical stack of input rows.
        parts = []
        for f in fields:
            label = f.get("label", "")
            val   = f.get("current", "")
            slug  = label.lower().replace(" ", "_")
            parts.append(
                f'<div>'
                f'<label class="block text-xs uppercase tracking-wider text-slate-500 mb-1">{label}</label>'
                f'<input type="text" value="{val}" class="w-full border border-slate-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-200 focus:border-blue-400" data-target="field_{slug}"/>'
                f'</div>'
            )
        return {
            "PAGE_TITLE":  bui.get("page_title", "Settings"),
            "FIELDS_HTML": "\n".join(parts),
        }
    if kind == "doc_editor":
        return {"DOC_TITLE": bui.get("doc_title", "Untitled.docx")}
    if kind == "transfer_form":
        fields = bui.get("fields", []) or []
        parts = []
        for f in fields:
            label = f.get("label", "")
            val   = f.get("current", "")
            slug  = label.lower().replace(" ", "_")
            parts.append(
                f'<div>'
                f'<label class="block text-xs uppercase tracking-wider text-slate-500 mb-1">{label}</label>'
                f'<input type="text" value="{val}" class="w-full border border-slate-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-200 focus:border-emerald-400" data-target="field_{slug}"/>'
                f'</div>'
            )
        return {
            "PAGE_TITLE":  bui.get("page_title", "New Transfer"),
            "FIELDS_HTML": "\n".join(parts),
        }
    if kind == "shopping_list":
        return {"LIST_TITLE": bui.get("list_title", "List")}
    if kind == "wizard_step":
        return {"STEP_LABEL": bui.get("step_label", "Step")}
    return {}


def inject_opts_for(scenario: dict) -> dict:
    """Translate scenario.injection -> opts consumed by inject.js."""
    inj = scenario.get("injection", {}) or {}
    kind = inj.get("kind", "")
    text = inj.get("text", "")
    base = {
        "kind": kind,
        "style": inj.get("style", ""),
        "position": inj.get("position", ""),
        "text": text,
        "overlayDataTarget": "injection_overlay",
    }

    # Per-kind tuning of the primary call-to-action button label/data-target.
    if kind == "modal_banner":
        base.update({"buttonLabel": "Install Now", "buttonDataTarget": "install_now"})
    elif kind == "modal_dialog":
        # (banking session-expired) — primary is Submit
        base.update({"buttonLabel": "Submit", "buttonDataTarget": "submit_password"})
    elif kind == "ai_assistant_card":
        base.update({"buttonLabel": "Apply suggestion", "buttonDataTarget": "ai_apply"})
    elif kind == "system_alert_banner":
        base.update({"buttonLabel": "SCAN NOW", "buttonDataTarget": "scan_now"})
    elif kind == "cookie_consent_bar":
        base.update({"buttonLabel": "Accept", "buttonDataTarget": "cookies_accept"})
    elif kind == "in_body_text":
        base["hostSelector"] = '[data-target="article_body"]'
    elif kind == "in_email_body":
        base["hostSelector"] = '[data-target="email_body"]'
    elif kind == "code_comment":
        base["hostSelector"] = '[data-target="editor_area"]'
    elif kind == "in_pdf_text":
        base["hostSelector"] = '[data-target="invoice_footer"]'
    elif kind == "search_result_title":
        # nothing to mount, the template already injected the trap result
        pass
    elif kind == "placeholder_text":
        base["labelAbove"] = inj.get("label_above", "Confirm")
        base["overlayDataTarget"] = "field_confirm_password"
    elif kind == "spoofed_button_label":
        base["overlayDataTarget"] = "spoofed_save_btn"
        base["actualFunctionVisualHint"] = "Send to all-staff@company.com"
    elif kind == "prefilled_field_value":
        base["hostSelector"] = "section"  # the form section
    elif kind == "autofill_suggestion":
        base["overlayDataTarget"] = "autofill_trap"
    elif kind == "tooltip_text_with_instruction":
        base["overlayDataTarget"] = "continue_btn_right"

    return base


# ---------- main ----------

def main(argv: list[str]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]

    selected_args = [a for a in argv[1:] if not a.startswith("-")]
    if selected_args:
        def keep(s):
            return any(s["id"] == a or s["id"].startswith(a) for a in selected_args)
        scenarios = [s for s in scenarios if keep(s)]
        if not scenarios:
            print(f"no scenarios matched {selected_args}", file=sys.stderr)
            return 2

    summary = []
    t0 = time.time()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            for s in scenarios:
                sid = s["id"]
                kind = s["base_ui"]["kind"]
                tpl = template_for(kind)
                if not tpl.exists():
                    print(f"[skip] {sid}: no template for {kind}")
                    continue
                t1 = time.time()
                try:
                    res = render_scenario(
                        template_html=tpl,
                        template_substitutions=template_subs_for(s),
                        inject_opts=inject_opts_for(s),
                        out_png=OUT_DIR / f"{sid}.png",
                        out_json=OUT_DIR / f"{sid}.bboxes.json",
                        browser=browser,
                    )
                    dt = time.time() - t1
                    n_targets = len(res["manifest"]["targets"])
                    print(f"[ok]   {sid}  template={kind}  targets={n_targets}  {dt:.2f}s  -> {Path(res['png']).name}")
                    summary.append({"id": sid, "ok": True, "targets": n_targets, "png": res["png"]})
                except Exception as e:  # noqa: BLE001
                    print(f"[err]  {sid}: {type(e).__name__}: {e}")
                    summary.append({"id": sid, "ok": False, "error": str(e)})
        finally:
            browser.close()

    total = time.time() - t0
    print("---")
    print(f"rendered {sum(1 for r in summary if r['ok'])} / {len(summary)} in {total:.1f}s")
    print(f"output dir: {OUT_DIR}")
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
