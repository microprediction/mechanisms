#!/usr/bin/env python3
"""Render papers/*.md into readable KaTeX pages under docs/papers/.

The site's "full draft" links should point at rendered, readable pages — not
raw markdown on GitHub. Like build_bibliography.py this is a generator with a
--check staleness mode, so the markdown stays the single source of truth and
the HTML can never silently drift.

  python scripts/build_papers.py           # write docs/papers/<slug>.html
  python scripts/build_papers.py --check   # exit 1 if any rendered page is stale

Scope: the constrained markdown these papers actually use — headings, bold,
italics, links, inline code, blockquotes, lists, horizontal rules, display and
inline TeX math. Math spans are extracted before any other processing and
restored verbatim afterwards, so `$p^*$`-style asterisks can never be mangled
into emphasis; KaTeX renders them in the browser. Dependency-free.
"""
from __future__ import annotations

import argparse
import html as html_mod
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "papers"
OUT = ROOT / "docs" / "papers"

PAPERS = [
    "nearest-the-pin-parimutuel.md",
    "scoring-point-cloud-distributional-submissions.md",
]

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Mechanisms — {title}</title>
  <link rel="stylesheet" href="../style.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js" crossorigin="anonymous"
    onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
</head>
<body>
  <header class="site-header">
    <div class="nav-inner">
      <a class="brand" href="../index.html">mechanisms</a>
      <nav>
        <a href="../catalog.html">Catalog</a>
        <a href="../demos.html">Demos</a>
        <a href="../map.html">Map</a>
        <a href="../connections.html">Connections</a>
        <a href="../implementations.html">Implementations</a>
        <a href="../papers.html">Papers</a>
        <a href="../bibliography.html">Bibliography</a>
        <a href="../world-cup.html">World Cup</a>
        <a href="https://github.com/microprediction/mechanisms">GitHub</a>
      </nav>
    </div>
  </header>

  <main>
    <p class="pagenav"><a href="../papers.html">&larr; Papers</a> &middot;
      <a href="./{slug}.pdf">pdf</a> &middot;
      <a href="https://github.com/microprediction/mechanisms/blob/main/papers/{src}">markdown source</a></p>
{body}
  </main>

  <footer>
    <a href="https://github.com/microprediction/mechanisms">Source</a> &middot;
    Maintained by <a href="https://github.com/microprediction">Peter Cotton</a>.
  </footer>
</body>
</html>
"""

MATH_TOKEN = "\x00MATH{}\x00"


def _extract_math(text: str):
    """Replace $$...$$ and $...$ spans with tokens; return (text, spans)."""
    spans: list[str] = []

    def stash(m: "re.Match") -> str:
        spans.append(m.group(0))
        return MATH_TOKEN.format(len(spans) - 1)

    text = re.sub(r"\$\$.*?\$\$", stash, text, flags=re.S)
    text = re.sub(r"\$[^$\n][^$]*?\$", stash, text)
    return text, spans


def _restore_math(text: str, spans: list[str]) -> str:
    for i, s in enumerate(spans):
        text = text.replace(MATH_TOKEN.format(i), html_mod.escape(s, quote=False))
    return text


def _inline(s: str) -> str:
    """Inline markdown -> HTML on math-free text (escape first, then mark up)."""
    s = html_mod.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
               lambda m: f'<a href="{_href(m.group(2))}">{m.group(1)}</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def _href(url: str) -> str:
    """Rewrite repo-relative markdown links to work from docs/papers/."""
    if url.startswith(("http://", "https://", "#")):
        return url
    if url.endswith(".md") or ".md#" in url:  # sibling papers / research notes
        base, _, frag = url.partition("#")
        name = Path(base).name[:-3]
        if (SRC / (name + ".md")).exists():   # sibling rendered paper
            return f"./{name}.html" + (f"#{frag}" if frag else "")
        return "https://github.com/microprediction/mechanisms/blob/main/" + \
               base.lstrip("./").replace("../", "") + (f"#{frag}" if frag else "")
    # code / tests / other repo files
    return "https://github.com/microprediction/mechanisms/blob/main/" + \
        url.lstrip("./").replace("../", "")


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def md_to_html(md: str) -> tuple[str, str]:
    """Return (title, body html)."""
    text, spans = _extract_math(md)
    lines = text.split("\n")
    out: list[str] = []
    title = ""
    para: list[str] = []
    in_list = in_quote = False

    def flush_para():
        nonlocal para
        if para:
            out.append("    <p>" + _inline(" ".join(para)) + "</p>")
            para = []

    def close_blocks():
        nonlocal in_list, in_quote
        flush_para()
        if in_list:
            out.append("    </ul>")
            in_list = False
        if in_quote:
            out.append("    </blockquote>")
            in_quote = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if in_quote and not para:
                pass
            close_blocks() if not in_quote else flush_para()
            continue
        if stripped.startswith("---") and set(stripped) <= {"-"}:
            close_blocks(); out.append("    <hr />"); continue
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            close_blocks()
            level, txt = len(m.group(1)), m.group(2)
            if level == 1 and not title:
                title = re.sub(r"[*`]", "", txt)
            anchor = _slugify(re.sub(MATH_TOKEN.replace("{}", r"\d+"), "", txt))
            out.append(f'    <h{level} id="{anchor}">{_inline(txt)}</h{level}>')
            continue
        if stripped.startswith(">"):
            flush_para()
            if not in_quote:
                if in_list:
                    out.append("    </ul>"); in_list = False
                out.append("    <blockquote>"); in_quote = True
            content = stripped.lstrip(">").strip()
            if content:
                para.append(content)
            else:
                flush_para()
            continue
        if in_quote and not stripped.startswith(">"):
            para.append(stripped)  # lazy blockquote continuation
            continue
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m and not stripped.startswith("**"):
            flush_para()
            if not in_list:
                out.append("    <ul>"); in_list = True
            out.append("      <li>" + _inline(m.group(1)) + "</li>")
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            flush_para()
            if not in_list:
                out.append("    <ul>"); in_list = True
            out.append(f"      <li><strong>{m.group(1)}.</strong> "
                       + _inline(m.group(2)) + "</li>")
            continue
        if in_list and raw.startswith(("   ", "\t")):   # list-item continuation
            out[-1] = out[-1][:-5] + " " + _inline(stripped) + "</li>"
            continue
        if in_list:
            out.append("    </ul>"); in_list = False
        para.append(stripped)
    close_blocks()
    body = _restore_math("\n".join(out), spans)
    return title, body


def fold_status(body: str) -> str:
    """Collapse the Status blockquote into a <details> so readers can skip it."""
    return re.sub(
        r"(<blockquote>\s*<p><strong>Status\.</strong>[\s\S]*?</blockquote>)",
        '    <details class="status-note">\n'
        "      <summary>Status: working draft — scope, assumptions, changelog"
        "</summary>\n\\1\n    </details>",
        body, count=1)


def build_one(name: str) -> tuple[Path, str]:
    md = (SRC / name).read_text(encoding="utf-8")
    title, body = md_to_html(md)
    body = fold_status(body)
    page = PAGE.format(title=html_mod.escape(title), src=name,
                       slug=name[:-3], body=body)
    return OUT / (name[:-3] + ".html"), page


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any rendered paper is stale")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stale = []
    for name in PAPERS:
        path, page = build_one(name)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != page:
                stale.append(path.name)
        else:
            path.write_text(page, encoding="utf-8")
            print(f"wrote {path}")
    if args.check:
        if stale:
            print("STALE rendered papers:", ", ".join(stale),
                  "— run: python scripts/build_papers.py")
            return 1
        print("rendered papers are up to date ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
