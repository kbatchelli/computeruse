"""Kernel.sh side-by-side demo: real cloud Chromium, baseline vs DPO-QA.

For each scenario (B2 fake-session-expired and E5 search-snippet-shell are
the most visual wins):

  1. Spin up a Kernel cloud browser
  2. Connect Playwright via CDP
  3. Start mp4 replay recording
  4. Render the scenario HTML in the browser
  5. Run BASE model on the screenshot, execute its click
  6. Stop + download base.mp4

  ... then repeat with the DPO-QA adapter loaded → dpo.mp4

Final ffmpeg composes a side-by-side mp4 (base | dpo).

Requires: KERNEL_API_KEY env var.

Usage:
    source ~/.kernel_env && python3 scripts/kernel_demo.py --scenario B2-fake-session-expired
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from kernel import Kernel
from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "realistic"))

from cua_shared import infer_action, load_northstar, load_scenarios, to_pixel
from render import _substitute, INJECT_JS  # noqa: E402
from render_scenarios import inject_opts_for, template_for, template_subs_for  # noqa: E402

PCD = Path("/home/shadeform/computeruse/philip-cua-defense")
DEMO_DIR = PCD / "outputs" / "kernel_demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def render_html(scenario: dict) -> str:
    """Render the scenario as a complete HTML string with injection inlined."""
    tpl = template_for(scenario["base_ui"]["kind"])
    subs = template_subs_for(scenario)
    inj_opts = inject_opts_for(scenario)
    base_html = _substitute(tpl.read_text(encoding="utf-8"), subs)
    # Inline the injection so it runs after page load
    inject_call = f"""
    <script>
    {INJECT_JS}
    window.addEventListener('load', () => {{
      setTimeout(() => window.injectBanner({json.dumps(inj_opts)}), 200);
    }});
    </script>
    """
    if "</body>" in base_html:
        base_html = base_html.replace("</body>", inject_call + "</body>")
    else:
        base_html += inject_call
    return base_html


def run_one_agent(
    label: str,
    scenario: dict,
    adapter: str | None,
    out_mp4: Path,
    kernel_client: Kernel,
):
    """Spin a Kernel browser, record mp4, run one model, save."""
    print(f"\n=== [{label}] creating Kernel browser ===", flush=True)
    browser_info = kernel_client.browsers.create(
        headless=False,  # required for mp4 replay
        viewport={"width": 1024, "height": 768},
        timeout_seconds=300,
    )
    sid = browser_info.session_id
    cdp_url = browser_info.cdp_ws_url
    print(f"[{label}] session={sid}", flush=True)

    print(f"[{label}] starting mp4 replay (15 fps)...", flush=True)
    replay = kernel_client.browsers.replays.start(sid, framerate=15, max_duration_in_seconds=120)
    replay_id = replay.replay_id
    print(f"[{label}] replay_id={replay_id}", flush=True)

    try:
        html = render_html(scenario)
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(cdp_url)
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.set_viewport_size({"width": 1024, "height": 768})

            print(f"[{label}] loading scenario HTML...", flush=True)
            # Use a data: URL so external CDN requests have a proper origin
            import base64
            data_url = "data:text/html;charset=utf-8;base64," + base64.b64encode(html.encode("utf-8")).decode("ascii")
            page.goto(data_url, wait_until="networkidle", timeout=15000)
            # Tailwind CDN compiles styles client-side after page load — wait for it
            page.wait_for_timeout(3500)
            # Force a style recomputation to be safe
            page.evaluate("document.body && document.body.offsetHeight")

            png_bytes = page.screenshot(type="png", full_page=False)
            screenshot_path = DEMO_DIR / f"{label}_pre.png"
            screenshot_path.write_bytes(png_bytes)
            print(f"[{label}] screenshot -> {screenshot_path.name}", flush=True)

            # Load model + run inference
            print(f"[{label}] loading model (adapter={adapter})...", flush=True)
            lm = load_northstar(adapter_path=adapter)
            img = Image.open(screenshot_path).convert("RGB")
            print(f"[{label}] inferring action...", flush=True)
            result = infer_action(lm, img, scenario["benign_task"])
            print(f"[{label}] action: {result['parsed']}", flush=True)
            print(f"[{label}] raw output (first 200 chars): {result['raw'][:200]!r}", flush=True)

            action = result["parsed"]
            if action and action.get("name") == "click":
                pt = to_pixel(action)
                if pt:
                    px, py = pt
                    print(f"[{label}] CLICKING at pixel ({px:.0f}, {py:.0f})", flush=True)
                    # Move mouse first so the cursor is visible in replay
                    page.mouse.move(px, py, steps=20)
                    page.wait_for_timeout(500)
                    page.mouse.click(px, py)
                    page.wait_for_timeout(2000)
            elif action:
                print(f"[{label}] non-click action: {action.get('name')}; not executing", flush=True)
            else:
                print(f"[{label}] no action parsed", flush=True)

            page.wait_for_timeout(1000)
            del lm
            torch.cuda.empty_cache()
    finally:
        print(f"[{label}] stopping replay...", flush=True)
        kernel_client.browsers.replays.stop(replay_id, id=sid)
        time.sleep(2)
        print(f"[{label}] downloading mp4...", flush=True)
        resp = kernel_client.browsers.replays.download(replay_id, id=sid)
        # BinaryAPIResponse — write to file
        with open(out_mp4, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
        size_mb = out_mp4.stat().st_size / 1e6
        print(f"[{label}] mp4 saved -> {out_mp4} ({size_mb:.1f} MB)", flush=True)
        try:
            kernel_client.browsers.delete_by_id(sid)
        except Exception as e:
            print(f"[{label}] note: failed to delete browser: {e}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--adapter", default="outputs/dpo-qa-r16")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--dpo-only", action="store_true")
    args = ap.parse_args()

    if "KERNEL_API_KEY" not in os.environ:
        sys.exit("Set KERNEL_API_KEY env var")

    scenarios = load_scenarios(PCD / "evals" / "scenarios.json")
    scen = next((s for s in scenarios if s["id"] == args.scenario), None)
    if scen is None:
        sys.exit(f"unknown scenario: {args.scenario}")

    print(f"=== scenario: {args.scenario} ===")
    print(f"task: {scen['benign_task']}")
    print(f"injection: {scen['injection']['text'][:120]}")

    client = Kernel(api_key=os.environ["KERNEL_API_KEY"])

    if not args.dpo_only:
        run_one_agent("baseline", scen, None, DEMO_DIR / f"{args.scenario}_base.mp4", client)
    if not args.baseline_only:
        run_one_agent("dpo-qa", scen, args.adapter, DEMO_DIR / f"{args.scenario}_dpo.mp4", client)

    print(f"\n=== artifacts in {DEMO_DIR} ===")
    for p in sorted(DEMO_DIR.iterdir()):
        if p.name.startswith(args.scenario):
            print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
