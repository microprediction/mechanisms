#!/usr/bin/env python3
"""Generate docs/bibliography.html from research/bibliography.bib.

Makes the .bib the single source of truth and retires the hand-sync. The page
chrome (head, nav, footer) and the closing note are preserved from the existing
file; everything between the <h1> and that closing note — the subtitle and every
<h3> section with its <ul> of entries — is regenerated from the .bib.

  python scripts/build_bibliography.py            # write docs/bibliography.html
  python scripts/build_bibliography.py --check     # exit 1 if the file is stale

Sections come from the ``% ===== ... =====`` / ``% --- ... ---`` comments in the
.bib (file order preserved). Dependency-free: standard library only.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bibtools import parse_bib, latex_to_unicode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "research" / "bibliography.bib"
HTML = ROOT / "docs" / "bibliography.html"

H1 = "<h1>Bibliography</h1>"
TAIL_MARKER = '    <p class="muted" style="margin-top:18px;">'

# Prose intros that are not derivable from the .bib, keyed by section title.
SECTION_INTROS = {
    "Generalizations: prior art for the candidate-original notes":
        '    <p class="subtitle" style="margin-top:28px;">The sections below '
        'collect the prior art for the three candidate-original generalizations '
        'in <a href="https://github.com/microprediction/mechanisms/tree/main/'
        'research">research/</a>. Exhaustive references are the point.</p>\n',
}


def format_authors(field: str) -> str:
    out = []
    for person in (p.strip() for p in field.split(" and ") if p.strip()):
        if "," in person:
            last, given = person.split(",", 1)
            given = re.sub(r",\s*(Jr|Sr|II|III|IV)\.?\s*$", "", given.strip())  # drop generational suffix
            inits = " ".join(w[0] + "." for w in re.findall(r"[^\s.,\-]+",
                                                            latex_to_unicode(given)))
            last = latex_to_unicode(last).strip()
            out.append(f"{last}, {inits}" if inits else last)
        else:
            toks = latex_to_unicode(person).split()
            out.append(f"{toks[-1]}, " + " ".join(w[0] + "." for w in toks[:-1])
                       if len(toks) > 1 else (toks[0] if toks else person))
    if len(out) <= 1:
        return out[0] if out else ""
    return " & ".join(out) if len(out) == 2 else ", ".join(out[:-1]) + " & " + out[-1]


def _em(s: str) -> str:
    return f"<em>{html.escape(latex_to_unicode(s))}</em>"


def venue(e) -> str:
    f, t = e.fields, e.etype
    if t == "article":
        s = _em(f.get("journal", ""))
        if f.get("volume"):
            s += " " + f["volume"] + (f"({f['number']})" if f.get("number") else "")
        return s + "."
    if t in ("inproceedings", "conference"):
        return _em(f.get("booktitle", "")) + "."
    if t == "incollection":
        s = _em(f.get("booktitle", ""))
        return s + (", " + html.escape(latex_to_unicode(f["publisher"])) if f.get("publisher") else "") + "."
    if t == "book":
        return html.escape(latex_to_unicode(f.get("publisher", ""))) + "."
    if t == "techreport":
        return " ".join(x for x in (f.get("institution", ""), f.get("type", ""),
                                    f.get("number", "")) if x) + "."
    if t == "misc":
        hp = f.get("howpublished", "")
        return html.escape(latex_to_unicode(hp)) + "." if hp else ""
    return ""


def render_li(e) -> str:
    authors = html.escape(format_authors(e.fields.get("author") or e.fields.get("editor", "")))
    year = e.year or "n.d."
    title = "&ldquo;" + html.escape(latex_to_unicode(e.title).rstrip(".")) + ".&rdquo;"
    href = (("https://doi.org/" + e.doi) if e.doi
            else e.url or (("https://arxiv.org/abs/" + e.arxiv) if e.arxiv else None))
    if href:
        title = f'<a href="{html.escape(href)}">{title}</a>'
    bits = [f"{authors} ({year}).", title]
    v = venue(e)
    if v:
        bits.append(v)
    li = " ".join(bits)
    if e.fields.get("note"):
        li += f' <span class="muted">{html.escape(latex_to_unicode(e.fields["note"]))}</span>'
    return f"      <li>{li}</li>"


def section_title(name: str) -> str:
    return html.escape(name.replace("<->", "⇄"))


def build() -> str:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    n = len(entries)
    parts = [
        H1, "\n",
        f'    <p class="subtitle">The literature behind each mechanism, {n} '
        "references, organized by topic. Titles link to their DOI or to the "
        "canonical source. A machine-readable version lives in\n"
        '    <a href="https://github.com/microprediction/mechanisms/blob/main/'
        'research/bibliography.bib">research/bibliography.bib</a>.</p>\n',
    ]
    seen_intro: set[str] = set()
    cur = None
    for e in entries:
        title = e.subsection or e.section
        if e.section in SECTION_INTROS and e.section not in seen_intro:
            if cur is not None:
                parts.append("    </ul>\n")
                cur = None
            parts.append("\n" + SECTION_INTROS[e.section])
            seen_intro.add(e.section)
        if title != cur:
            if cur is not None:
                parts.append("    </ul>\n")
            parts.append(f"\n    <h3>{section_title(title)}</h3>\n    <ul>\n")
            cur = title
        parts.append(render_li(e) + "\n")
    if cur is not None:
        parts.append("    </ul>\n\n")

    old = HTML.read_text(encoding="utf-8")
    head = old.split(H1)[0]
    tail = TAIL_MARKER + old.split(TAIL_MARKER, 1)[1]
    return head + "".join(parts) + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if docs/bibliography.html is out of date")
    args = ap.parse_args()
    generated = build()
    if args.check:
        if HTML.read_text(encoding="utf-8") != generated:
            print("docs/bibliography.html is STALE — run: python scripts/build_bibliography.py")
            return 1
        print("docs/bibliography.html is up to date ✓")
        return 0
    HTML.write_text(generated, encoding="utf-8")
    print(f"wrote {HTML} ({generated.count('<li>')} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
