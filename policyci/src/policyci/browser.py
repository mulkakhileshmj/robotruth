"""Scenario browser: the developer surface.

A regression count is a complaint; a replayable scenario is a bug report. This renders a
self-contained HTML page for one diff: the headline numbers with their evidence, then every
newly-broken scenario with its seed, its hashes, its per-gate breakdown and its replay video
side by side with the baseline's.

It is a static file on purpose. It opens from the results folder with no server, and it
carries the pins and hashes it was built from, so a page can always be traced back to the
runs that produced it.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from policyci.regression import Diff

_CSS = """
:root{--bg:#ffffff;--fg:#14161a;--muted:#5b6470;--line:#e3e6ea;--card:#f7f8fa;
--ok:#15803d;--bad:#b91c1c;--warn:#a16207;--accent:#1d4ed8;}
:root:not([data-theme="light"]){@media (prefers-color-scheme:dark){
:root{--bg:#0f1115;--fg:#e8eaed;--muted:#9aa3af;--line:#252a32;--card:#161a21;
--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--accent:#60a5fa;}}}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0f1115;--fg:#e8eaed;--muted:#9aa3af;--line:#252a32;--card:#161a21;
--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--accent:#60a5fa;}}
:root[data-theme="dark"]{--bg:#0f1115;--fg:#e8eaed;--muted:#9aa3af;--line:#252a32;
--card:#161a21;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--accent:#60a5fa;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:0 16px 64px;
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1040px;margin:0 auto}
h1{font-size:26px;margin:28px 0 4px;letter-spacing:-.01em}
h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);font-size:13px;margin:0 0 20px}
.verdict{display:inline-block;padding:7px 14px;border-radius:7px;font-weight:600;font-size:15px}
.v-worse{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.v-better{background:color-mix(in srgb,var(--ok) 16%,transparent);color:var(--ok)}
.v-none{background:color-mix(in srgb,var(--muted) 16%,transparent);color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h3{margin:0 0 6px;font-size:14px;font-family:ui-monospace,monospace}
.vids{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.vids figure{margin:0}
.vids figcaption{font-size:11px;color:var(--muted);margin-bottom:3px}
video{width:100%;border-radius:6px;background:#000;display:block}
.none{color:var(--muted);font-size:13px;font-style:italic}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;font-weight:600}
.p-pass{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
.p-fail{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.p-na{background:color-mix(in srgb,var(--muted) 15%,transparent);color:var(--muted)}
.note{background:var(--card);border-left:3px solid var(--accent);padding:10px 14px;
border-radius:0 7px 7px 0;font-size:13px;color:var(--muted);margin:14px 0}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);
font-size:12px;color:var(--muted)}
@media(max-width:560px){.vids{grid-template-columns:1fr}}
"""

_VERDICT = {
    "B_better": ("BETTER", "v-better"),
    "A_better": ("WORSE", "v-worse"),
    "no_difference_within_margin": ("NO MEANINGFUL DIFFERENCE", "v-none"),
    "undecided": ("INCONCLUSIVE, more trials needed", "v-none"),
}


def _e(s) -> str:
    return html.escape(str(s))


def _iv(iv) -> str:
    return f"{100 * iv.estimate:.1f}% <span style='color:var(--muted)'>[{100 * iv.lower:.1f}, {100 * iv.upper:.1f}]</span>"


def _gate_pills(gates: dict) -> str:
    out = []
    for k, v in (gates or {}).items():
        cls = {"pass": "p-pass", "fail": "p-fail"}.get(v, "p-na")
        out.append(f"<span class='pill {cls}'>{_e(k)}: {_e(v)}</span>")
    return " ".join(out) or "<span class='none'>no gate detail</span>"


def render_browser(d: Diff, a: dict, b: dict, battery_path: str | Path | None,
                   out_path: str | Path, video_rel: str = "videos") -> Path:
    """Write the scenario browser page. `video_rel` is where the mp4s sit relative to it."""
    seeds: dict[str, dict] = {}
    if battery_path and Path(battery_path).exists():
        for line in Path(battery_path).read_text(encoding="utf-8").splitlines()[1:]:
            if line.strip():
                s = json.loads(line)
                seeds[s["hash"]] = s["params"]

    label, cls = _VERDICT.get(d.sequential.decision, (d.sequential.decision, "v-none"))
    seq = d.sequential
    p = []
    p.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    p.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    p.append(f"<title>{_e(d.b_name)} vs {_e(d.a_name)}</title><style>{_CSS}</style></head><body><div class='wrap'>")
    p.append(f"<h1>{_e(d.b_name)} vs {_e(d.a_name)}</h1>")
    p.append(f"<p class='sub'>Battery <code>{_e(a['battery_hash'][:16])}</code> &middot; {d.n} scenarios &middot; "
             f"task <code>{_e(a['task'])}</code> &middot; evaluator <code>{_e(a['evaluator_version'])}</code></p>")
    p.append(f"<p><span class='verdict {cls}'>{_e(label)}</span></p>")

    p.append("<h2>Headline</h2><table><tr><th>metric</th>"
             f"<th>{_e(d.a_name)}</th><th>{_e(d.b_name)}</th></tr>")
    p.append(f"<tr><td>Success rate</td><td>{_iv(d.a_rate)}</td><td>{_iv(d.b_rate)}</td></tr></table>")
    p.append(f"<p>Paired difference (B minus A): <strong>{_iv(d.paired)}</strong><br>"
             f"Anytime-valid sequential estimate: {seq.estimate:+.3f} "
             f"[{seq.lower:+.3f}, {seq.upper:+.3f}] at n={seq.n}.</p>")

    p.append("<h2>Scenario diff</h2><table><tr><th>quantity</th><th>value</th><th>what it means</th></tr>")
    p.append(f"<tr><td>Newly broken</td><td><strong>{len(d.newly_broken)}</strong></td>"
             "<td>passed under A, failed under B</td></tr>")
    if d.noise_flips is not None:
        p.append(f"<tr><td>Noise floor</td><td><strong>{d.noise_flips}</strong></td>"
                 "<td>expected one-directional flips from running the same policy twice</td></tr>")
        p.append(f"<tr><td>Beyond noise</td><td><strong>{d.significant_regressions}</strong></td>"
                 "<td>what is left once the floor is subtracted</td></tr>")
    else:
        p.append("<tr><td>Noise floor</td><td><strong>not measured</strong></td>"
                 "<td>run the same policy twice before trusting any count here</td></tr>")
    p.append(f"<tr><td>Fixed</td><td><strong>{len(d.fixed)}</strong></td>"
             "<td>failed under A, passed under B</td></tr></table>")

    p.append("<div class='note'>A flip count on its own is not a regression. The verdict above "
             "comes from a paired, anytime-valid test over matched scenarios, and the floor is "
             "measured by running one policy against itself on these same scenes.</div>")

    p.append(f"<h2>Newly broken scenarios ({len(d.newly_broken)})</h2>")
    if not d.newly_broken:
        p.append("<p class='none'>None. Nothing that passed under the baseline fails here.</p>")
    else:
        p.append("<div class='cards'>")
        for h in d.newly_broken:
            rb = b["results"][h]
            ra = a["results"][h]
            short = h[:12]
            params = seeds.get(h, {})
            p.append("<div class='card'>")
            p.append(f"<h3>{_e(short)} <span style='color:var(--muted)'>#{rb.get('index')}</span></h3>")
            if params:
                p.append(f"<div><code>{_e(json.dumps(params))}</code></div>")
            p.append(f"<div style='margin-top:6px'>B failure: <strong>{_e(rb.get('failure') or 'fail')}</strong> "
                     f"&middot; reward {_e(ra.get('max_reward'))} &rarr; {_e(rb.get('max_reward'))}</div>")
            p.append(f"<div style='margin-top:6px'>{_gate_pills(rb.get('gates'))}</div>")
            va = f"{video_rel}/{d.a_name}/{short}_{d.a_name}.mp4"
            vb = f"{video_rel}/{d.b_name}/{short}_{d.b_name}.mp4"
            p.append("<div class='vids'>")
            p.append(f"<figure><figcaption>{_e(d.a_name)} (passed)</figcaption>"
                     f"<video controls preload='none' src='{_e(va)}'></video></figure>")
            p.append(f"<figure><figcaption>{_e(d.b_name)} (failed)</figcaption>"
                     f"<video controls preload='none' src='{_e(vb)}'></video></figure>")
            p.append("</div></div>")
        p.append("</div>")

    if d.warnings:
        p.append("<h2>Warnings</h2><ul>")
        for w in d.warnings:
            p.append(f"<li>{_e(w)}</li>")
        p.append("</ul>")

    p.append("<footer>")
    p.append(f"Run manifests <code>{_e(a['manifest_hash'][:16])}</code> and <code>{_e(b['manifest_hash'][:16])}</code>. "
             f"Pins <code>{_e(a['pins_hash'][:16])}</code>. "
             f"Simulator: {_e(a['pins'].get('mujoco', '?'))} MuJoCo, {_e(a['pins'].get('gpu', '?'))}.<br>"
             "A number without an interval is not a result.")
    p.append("</footer></div></body></html>")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p), encoding="utf-8")
    return out
