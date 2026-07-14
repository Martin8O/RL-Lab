"""Reliable screenshot of the running app — the escape hatch for the MCP ``preview_screenshot`` timeout.

WHY THIS EXISTS: the MCP ``preview_screenshot`` times out (30 s) whenever another
dev session also runs a preview server — this session's preview tab is then a *background* surface
(``document.visibilityState==="hidden"``), the browser pauses ``requestAnimationFrame`` / compositor
frames, and the tool's wait-for-a-fresh-frame never resolves. A **headless Playwright page is never a
hidden background tab**, so it always paints and always captures — regardless of how many other
sessions are up. Use THIS for a real image; keep DOM inspection (``preview_inspect``/``snapshot``/
``eval``) for verifying values. Playwright ships in the dev venv already (it renders the README GIFs).

USAGE (from repo root; find the app port via the preview tool or the backend log):
  .venv/Scripts/python.exe tools/shot.py http://localhost:8000/ out.png
  .venv/Scripts/python.exe tools/shot.py http://localhost:8000/ picker.png --selector "[role=menu]"
  .venv/Scripts/python.exe tools/shot.py http://localhost:8000/ full.png --full
  .venv/Scripts/python.exe tools/shot.py http://localhost:8000/ menu.png \
        --eval "document.querySelector('button.lab-trigger').click()" --wait 900
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless screenshot of a running page (ADR-099 escape hatch).")
    ap.add_argument("url", help="page URL, e.g. http://localhost:8000/")
    ap.add_argument("out", help="output PNG path")
    ap.add_argument("--selector", help="CSS selector — capture just this element")
    ap.add_argument("--clip", help="x,y,w,h — capture just this rectangle (CSS px)")
    ap.add_argument("--full", action="store_true", help="full-page screenshot")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--scale", type=float, default=2.0, help="device scale factor (crispness)")
    ap.add_argument("--wait", type=int, default=1200, help="ms to settle after load")
    ap.add_argument("--eval", dest="js", help="JS to run before the shot (e.g. open a menu)")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright missing — `pip install playwright && playwright install chromium`", file=sys.stderr)
        return 2

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=args.scale,
        )
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_timeout(args.wait)
        if args.js:
            page.evaluate(args.js)
            page.wait_for_timeout(args.wait)

        shot_kw: dict = {"path": args.out}
        if args.clip:
            x, y, w, h = (float(v) for v in args.clip.split(","))
            shot_kw["clip"] = {"x": x, "y": y, "width": w, "height": h}
        elif args.full:
            shot_kw["full_page"] = True

        if args.selector:
            page.locator(args.selector).first.screenshot(path=args.out)
        else:
            page.screenshot(**shot_kw)
        browser.close()

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
