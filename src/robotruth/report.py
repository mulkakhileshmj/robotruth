"""Report generation. House rule: no success rate is ever printed without an interval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, BaseLoader

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
        return _HTML.render(r=self, fmt=_fmt, version=__version__)

    def write(self, out_dir: Path | str, stem: str = "report") -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md = out_dir / f"{stem}.md"
        html = out_dir / f"{stem}.html"
        md.write_text(self.to_markdown(), encoding="utf-8")
        html.write_text(self.to_html(), encoding="utf-8")
        return md, html


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


_HTML = Environment(loader=BaseLoader(), autoescape=True).from_string("""<!doctype html>
<html><head><meta charset="utf-8"><title>{{ r.title }}</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1b1b1f;background:#fafafa}
h1{font-size:1.6rem} h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.25rem}
.verdict{display:inline-block;padding:.2rem .6rem;border-radius:.4rem;font-weight:600}
.PASS{background:#d8f3dc;color:#1b4332}.FAIL{background:#ffd6d6;color:#7a1f1f}.WARN{background:#fff3cd;color:#664d03}.INFO{background:#e2e3e5;color:#41464b}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0 1rem}
th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:left;vertical-align:top}
th{background:#f0f0f0} td.fatal{color:#7a1f1f;font-weight:600} td.warn{color:#664d03}
pre{background:#f0f0f0;padding:.6rem;overflow-x:auto} .meta{color:#666;font-size:.85rem}
</style></head><body>
<h1>{{ r.title }}</h1>
<p class="meta">robotruth {{ version }}, generated {{ r.created_at }}</p>
{% if r.verdict %}<p><span class="verdict {{ r.verdict }}">{{ r.verdict }}</span></p>{% endif %}
{% for s in r.sections %}
<h2>{{ s.title }} {% if s.verdict %}<span class="verdict {{ s.verdict }}">{{ s.verdict }}</span>{% endif %}</h2>
<pre style="white-space:pre-wrap;background:transparent;padding:0">{{ s.body_md }}</pre>
{% if s.table %}<table><thead><tr>{% for c in s.table[0].keys() %}<th>{{ c }}</th>{% endfor %}</tr></thead><tbody>
{% for row in s.table %}<tr>{% for c, v in row.items() %}<td class="{{ (row.get('severity') or '')|lower }}">{{ fmt(v) }}</td>{% endfor %}</tr>{% endfor %}
</tbody></table>{% endif %}
{% endfor %}
</body></html>""")
