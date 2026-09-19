"""Report generation. House rule: no success rate is ever printed without an interval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import html as _html
import re as _re

from jinja2 import Environment, BaseLoader
from markupsafe import Markup

from robotruth import __version__


@dataclass
class Section:
    title: str
    body_md: str
    table: Optional[list[dict[str, Any]]] = None
    verdict: Optional[str] = None  # PASS / FAIL / WARN / INFO


@dataclass
class Report:
    title: str
    sections: list[Section] = field(default_factory=list)
    verdict: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    def add(self, section: Section) -> None:
        self.sections.append(section)

    def to_markdown(self) -> str:
        out = [f"# {self.title}", "", f"robotruth {__version__}, generated {self.created_at}", ""]
        if self.verdict:
            out += [f"**Verdict: {self.verdict}**", ""]
        for s in self.sections:
            head = f"## {s.title}" + (f" [{s.verdict}]" if s.verdict else "")
            out += [head, "", s.body_md.strip(), ""]
            if s.table:
                cols = list(s.table[0].keys())
                out.append("| " + " | ".join(cols) + " |")
                out.append("|" + "|".join("---" for _ in cols) + "|")
                for row in s.table:
                    out.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
                out.append("")
        return "\n".join(out)

    def to_html(self) -> str:
        return _HTML.render(r=self, fmt=_fmt, md=_md_html, cap=_cap, anchor=_anchor, version=__version__)

    def write(self, out_dir: Path | str, stem: str = "report") -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md = out_dir / f"{stem}.md"
        html = out_dir / f"{stem}.html"
        md.write_text(self.to_markdown(), encoding="utf-8")
        html.write_text(self.to_html(), encoding="utf-8")
        return md, html


def _md_html(text: str) -> Markup:
    """Render the small markdown subset the reports use (bold, code, links, bullets,
    paragraphs) to HTML. Input is escaped first, so report content cannot inject markup."""

    def inline(t: str) -> str:
        t = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = _re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = _re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', t)
        return t

    out: list[str] = []
    para: list[str] = []
    items: list[str] = []

    def flush_para() -> None:
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    def flush_items() -> None:
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in _html.escape(text or "", quote=False).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            flush_para()
            items.append(stripped[2:])
        elif not stripped:
            flush_para()
            flush_items()
        else:
            flush_items()
            para.append(stripped)
    flush_para()
    flush_items()
    return Markup("\n".join(out))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


_ACRONYMS = {"fps": "FPS", "id": "ID", "url": "URL", "vlm": "VLM", "api": "API", "ms": "ms"}


def _cap(text: Any) -> str:
    """Sentence-case a column header or title, keeping known acronyms upper-case."""
    words = str(text).split()
    out = []
    for i, w in enumerate(words):
        low = w.lower()
        if low in _ACRONYMS:
            out.append(_ACRONYMS[low])
        elif i == 0:
            out.append(w[:1].upper() + w[1:])
        else:
            out.append(w)
    return " ".join(out)


def _anchor(text: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


_HTML = Environment(loader=BaseLoader(), autoescape=True).from_string("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ cap(r.title) }}</title>
<style>
:root{
  --ink:#1a1c22; --muted:#5b6472; --line:#e3e6ea; --bg:#f6f7f9; --card:#ffffff;
  --accent:#1f4e79; --accent-soft:#eaf1f7;
}
*{box-sizing:border-box}
body{font-family:"Segoe UI",system-ui,Roboto,Helvetica,Arial,sans-serif;line-height:1.55;
  margin:0;color:var(--ink);background:var(--bg)}
.page{max-width:1080px;margin:0 auto;padding:2.5rem 1.5rem 4rem}
header.doc{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:1.75rem 2rem;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(16,24,40,.06)}
header.doc h1{margin:0 0 .35rem;font-size:1.75rem;letter-spacing:-.01em;color:var(--accent)}
.meta{color:var(--muted);font-size:.9rem;margin:0}
.headline{margin:.9rem 0 0;font-size:1rem}
nav.toc{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:1rem 1.5rem;margin-bottom:1.5rem;font-size:.92rem}
nav.toc strong{display:block;margin-bottom:.4rem;color:var(--muted);
  text-transform:uppercase;font-size:.75rem;letter-spacing:.08em}
nav.toc ol{margin:0;padding-left:1.2rem}
nav.toc a{color:var(--accent);text-decoration:none}
nav.toc a:hover{text-decoration:underline}
section.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:1.5rem 2rem;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(16,24,40,.05)}
h2{font-size:1.2rem;margin:.1rem 0 .9rem;letter-spacing:-.005em;color:var(--accent);
  display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
h2 .no{color:var(--muted);font-weight:600}
p{margin:.6rem 0}
ul{margin:.5rem 0 .8rem;padding-left:1.35rem}
li{margin:.3rem 0}
li::marker{color:var(--accent)}
code{background:var(--accent-soft);border-radius:4px;padding:.08rem .35rem;
  font-family:Consolas,"SF Mono",Menlo,monospace;font-size:.85em}
a{color:var(--accent)}
.verdict{display:inline-block;padding:.15rem .6rem;border-radius:99px;font-weight:600;
  font-size:.72rem;letter-spacing:.06em;text-transform:uppercase}
.PASS{background:#d9f2e0;color:#155e37}.FAIL{background:#fbdcdc;color:#8a1f1f}
.WARN{background:#fdf0cd;color:#7a5b00}.INFO{background:#e7eaee;color:#454e5c}
.tablewrap{overflow-x:auto;margin:.6rem 0 .3rem}
table{border-collapse:collapse;width:100%;font-size:.87rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:.5rem .7rem;text-align:left;vertical-align:top}
th{background:var(--accent-soft);color:var(--accent);font-weight:600;white-space:nowrap;
  position:sticky;top:0}
tbody tr:nth-child(even){background:#fafbfc}
tbody tr:hover{background:var(--accent-soft)}
td.fatal{color:#8a1f1f;font-weight:600} td.warn{color:#7a5b00}
footer{color:var(--muted);font-size:.82rem;text-align:center;margin-top:2rem}
@media print{body{background:#fff}.page{padding:0}
  section.card,header.doc,nav.toc{box-shadow:none;border-color:#ccc;break-inside:avoid}}
</style></head><body><div class="page">
<header class="doc">
<h1>{{ cap(r.title) }}</h1>
<p class="meta">robotruth {{ version }} &middot; generated {{ r.created_at }}</p>
{% if r.verdict %}<p class="headline"><span class="verdict INFO">Summary</span>&ensp;{{ cap(r.verdict) }}</p>{% endif %}
</header>
<nav class="toc"><strong>Contents</strong><ol>
{% for s in r.sections %}<li><a href="#{{ anchor(s.title) }}">{{ cap(s.title) }}</a></li>
{% endfor %}</ol></nav>
{% for s in r.sections %}
<section class="card" id="{{ anchor(s.title) }}">
<h2><span class="no">{{ loop.index }}.</span> {{ cap(s.title) }}{% if s.verdict %} <span class="verdict {{ s.verdict }}">{{ s.verdict }}</span>{% endif %}</h2>
{{ md(s.body_md) }}
{% if s.table %}<div class="tablewrap"><table><thead><tr>{% for c in s.table[0].keys() %}<th>{{ cap(c) }}</th>{% endfor %}</tr></thead><tbody>
{% for row in s.table %}<tr>{% for c, v in row.items() %}<td class="{{ (row.get('severity') or '')|lower }}">{{ fmt(v) }}</td>{% endfor %}</tr>{% endfor %}
</tbody></table></div>{% endif %}
</section>
{% endfor %}
<footer>Generated by robotruth {{ version }}. Every rate carries its interval; anything unmeasured is stated as such.</footer>
</div></body></html>""")
