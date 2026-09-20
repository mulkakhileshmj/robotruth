"""Build the scenario browser pages from a pulled results folder.

Reads only the manifests and the battery that the box already produced, and links the
videos that came back with them. No simulation, no policy, no compute: this is report
rendering over finished evidence.

Usage: python policyci/ops/build_browser.py D:/robotruth/results/pci_2
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from policyci.browser import render_browser
from policyci.passport import build_passport, verify_passport, write_passport
from policyci.regression import ComparabilityError, diff_runs, load_manifest
from policyci.report import render_diff

BASELINE = "act_v18"
NOISE_REF = "act_v18_s1"


def main(root: str) -> int:
    root = Path(root)
    merged = root / "merged"
    battery = root / "battery_b1.jsonl"
    if not merged.is_dir():
        print(f"no merged/ under {root}", file=sys.stderr)
        return 2

    runs = {p.name: p / "run_manifest.json" for p in sorted(merged.iterdir())
            if (p / "run_manifest.json").exists()}
    if BASELINE not in runs:
        print(f"no baseline run {BASELINE} in {merged}", file=sys.stderr)
        return 2
    a = load_manifest(runs[BASELINE])
    a2 = load_manifest(runs[NOISE_REF]) if NOISE_REF in runs else None

    # collect every run's videos under one tree the pages can link into
    vroot = root / "videos"
    for name, mpath in runs.items():
        src = mpath.parent / "videos"
        if src.is_dir():
            dst = vroot / name
            dst.mkdir(parents=True, exist_ok=True)
            for mp4 in src.glob("*.mp4"):
                shutil.copy2(mp4, dst / mp4.name)

    pages = []
    targets = [(NOISE_REF, "noise_control")] if a2 else []
    targets += [(n, n) for n in runs if n not in (BASELINE, NOISE_REF)]

    for name, slug in targets:
        b = load_manifest(runs[name])
        # the negative control is diffed WITHOUT a floor: it is the thing that measures it
        ref = None if name == NOISE_REF else ((a, a2) if a2 else None)
        try:
            d = diff_runs(a, b, noise_ref=ref)
        except ComparabilityError as e:
            print(f"REFUSED {name}: {e}", file=sys.stderr)
            continue
        render_diff(d, a, b, root / f"diff_{slug}.md")
        page = render_browser(d, a, b, battery, root / f"browser_{slug}.html", video_rel="videos")
        pages.append(page)
        pp = build_passport(a, b, d, battery_path=battery)
        ppath = write_passport(pp, root / f"passport_{slug}.json")
        assert verify_passport(ppath), "passport failed its own digest check"
        print(f"  passport: {pp['decision']['recommendation']:24s} {ppath.name}")
        floor = "no floor (this run IS the floor)" if ref is None else f"floor {d.noise_flips}"
        print(f"{name:22s} {d.sequential.decision:28s} A={d.a_rate.estimate:.3f} B={d.b_rate.estimate:.3f} "
              f"broken={len(d.newly_broken)} fixed={len(d.fixed)} {floor} -> {page.name}")

    index = root / "index.html"
    links = "\n".join(
        f"<li><a href='{p.name}'>{p.stem.replace('browser_', '')}</a></li>" for p in pages)
    index.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Policy CI results</title>"
        "<style>body{font:15px/1.6 ui-sans-serif,system-ui,sans-serif;max-width:680px;"
        "margin:40px auto;padding:0 16px;background:#fff;color:#14161a}"
        "@media(prefers-color-scheme:dark){body{background:#0f1115;color:#e8eaed}"
        "a{color:#60a5fa}}</style></head><body>"
        f"<h1>Policy CI results</h1><p>Battery from <code>{battery.name}</code>.</p><ul>{links}</ul>"
        "</body></html>", encoding="utf-8")
    print(f"index -> {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
