#!/usr/bin/env python3
"""Detect drift between research/bibliography.bib and docs/bibliography.html.

The two are maintained by hand (there is no generator), so they drift silently.
This flags entries present in one but not the other, matching heuristically on
(first-author surname, year). Advisory by default (exit 0); pass --strict to fail.

The right long-term fix is to generate the HTML from the .bib; until then this is
the guard rail. Dependency-free: standard library only.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bibtools import parse_bib, _fold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "research" / "bibliography.bib"
HTML = ROOT / "docs" / "bibliography.html"

LI_RE = re.compile(r"<li>(.*?)</li>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
HTML_HEAD_RE = re.compile(r"^\s*([^,(<]+),")              # surname before 1st comma
YEAR_RE = re.compile(r"\((\d{4})[a-z]?\)")


def sig(surname: str, year: str | None) -> tuple[str, str]:
    return (_fold(surname).lower().strip(), year or "")


def bib_sigs() -> dict[tuple[str, str], str]:
    out = {}
    for e in parse_bib(BIB.read_text(encoding="utf-8")):
        if not e.year or not e.first_author_surname:
            continue  # undated / org-authored: not matchable by (surname, year)
        out[sig(e.first_author_surname, e.year)] = e.key
    return out


def html_sigs() -> dict[tuple[str, str], str]:
    out = {}
    for raw in LI_RE.findall(HTML.read_text(encoding="utf-8")):
        text = html.unescape(TAG_RE.sub("", raw)).strip()
        m_auth = HTML_HEAD_RE.match(text)
        m_year = YEAR_RE.search(text)
        if not m_auth or not m_year:
            continue  # org-authored / undated entry — not matchable
        out[sig(m_auth.group(1), m_year.group(1))] = text[:60]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit non-zero on drift")
    args = ap.parse_args()

    b, h = bib_sigs(), html_sigs()
    only_bib = sorted(set(b) - set(h))
    only_html = sorted(set(h) - set(b))

    print(f".bib: {len(b)} matchable entries   .html: {len(h)} matchable entries")
    if only_bib:
        print(f"\nIn .bib but not on the site ({len(only_bib)}):")
        for k in only_bib:
            print(f"  - {b[k]}  ({k[0]} {k[1]})")
    if only_html:
        print(f"\nOn the site but not in .bib ({len(only_html)}):")
        for k in only_html:
            print(f"  - {k[0]} {k[1]}: {h[k]}")
    if not only_bib and not only_html:
        print("in sync (by author+year) ✓")

    drift = len(only_bib) + len(only_html)
    return 1 if (args.strict and drift) else 0


if __name__ == "__main__":
    raise SystemExit(main())
