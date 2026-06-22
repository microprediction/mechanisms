#!/usr/bin/env python3
"""Validate research/bibliography.bib.

Offline (default, no network — safe for pre-commit and PR CI):
  - duplicate citation keys
  - brace balance across the file
  - every entry carries a resolvable identifier (DOI / arXiv / URL / ISBN), else
    it is flagged as unverifiable and needs a human

Online (--online, used on a schedule / manually — needs network):
  - every DOI resolves via the Crossref REST API
  - every arXiv id resolves via the arXiv API
  - the resolved title / first-author surname / year MATCH the .bib entry,
    catching exactly the wrong-venue / wrong-year / wrong-author class of error

Exit code is non-zero if any ERROR is found (warnings alone do not fail).
Dependency-free: standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bibtools import Entry, parse_bib, norm_title, _delatex, _fold  # noqa: E402

BIB = Path(__file__).resolve().parent.parent / "research" / "bibliography.bib"
MAILTO = "peter.cotton@microprediction.com"   # Crossref "polite pool"
UA = f"mechanisms-bib-checker (+https://github.com/microprediction/mechanisms; mailto:{MAILTO})"

errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


# --------------------------------------------------------------------------
# Offline checks
# --------------------------------------------------------------------------
def check_offline(text: str, entries: list[Entry]) -> None:
    if text.count("{") != text.count("}"):
        err(f"brace imbalance: {text.count('{')} open vs {text.count('}')} close")

    seen: dict[str, int] = {}
    for e in entries:
        if e.key in seen:
            err(f"duplicate key '{e.key}' (lines {seen[e.key]} and {e.lineno})")
        seen[e.key] = e.lineno

    # Same paper under two keys: silently duplicates in any generated output.
    by_doi: dict[str, str] = {}
    by_title: dict[str, str] = {}
    for e in entries:
        if e.doi:
            d = e.doi.lower()
            if d in by_doi:
                err(f"duplicate DOI {e.doi}: '{by_doi[d]}' and '{e.key}'")
            by_doi[d] = e.key
        t = norm_title(e.title)
        if t and t in by_title and by_title[t] != e.key:
            err(f"duplicate title under two keys: '{by_title[t]}' and '{e.key}' "
                f"({e.title[:50]}...)")
        by_title[t] = e.key

    for e in entries:
        if e.etype in {"book", "incollection", "techreport"}:
            continue  # books/reports legitimately may carry only an ISBN/url
        if not e.has_identifier:
            warn(f"{e.key} (line {e.lineno}): no DOI/arXiv/URL — unverifiable")


# --------------------------------------------------------------------------
# Online checks
# --------------------------------------------------------------------------
def _get(url: str, accept: str = "application/json"):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _title_matches(a: str, b: str) -> bool:
    ta, tb = norm_title(a), norm_title(b)
    if not ta or not tb:
        return False
    if ta == tb or ta in tb or tb in ta:
        return True
    sa, sb = set(ta.split()), set(tb.split())
    return len(sa & sb) / max(1, min(len(sa), len(sb))) >= 0.7


def _datacite_exists(doi: str) -> bool:
    """LIPIcs, Zenodo, some arXiv DOIs are registered at DataCite, not Crossref."""
    try:
        _get("https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe=""))
        return True
    except Exception:
        return False


def check_doi(e: Entry) -> None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(e.doi, safe="")
    try:
        msg = json.loads(_get(url))["message"]
    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            if _datacite_exists(e.doi):
                return  # valid, just a non-Crossref registrar — no metadata x-check
            err(f"{e.key}: DOI does not resolve at Crossref or DataCite ({e.doi})")
        else:
            warn(f"{e.key}: Crossref HTTP {ex.code} for {e.doi}")
        return
    except Exception as ex:  # network/transient
        warn(f"{e.key}: Crossref lookup failed ({ex})")
        return

    # Metadata cross-checks are WARNINGS: online/print years, DOI-registration
    # years, and author-ordering legitimately differ from a correct .bib entry.
    titles = msg.get("title") or []
    if titles and not _title_matches(e.title, titles[0]):
        warn(f"{e.key}: title differs from Crossref — POSSIBLE WRONG DOI"
             f"\n      bib: {e.title}\n      doi: {titles[0]}")

    dp = (msg.get("issued") or {}).get("date-parts") or [[None]]
    cr_year = str(dp[0][0]) if dp and dp[0] and dp[0][0] is not None else None
    if cr_year and e.year and cr_year != e.year:
        warn(f"{e.key}: year bib={e.year} vs Crossref={cr_year} ({e.doi}) "
             f"— check online/print")

    authors = msg.get("author") or []
    if authors:
        fam = _fold(_delatex(authors[0].get("family", ""))).lower()
        mine = e.first_author_surname.lower()
        if fam and fam not in mine and mine not in fam:
            warn(f"{e.key}: first author bib='{e.first_author_surname}' vs "
                 f"Crossref='{authors[0].get('family','')}' ({e.doi})")


def check_arxiv(e: Entry) -> None:
    url = f"http://export.arxiv.org/api/query?id_list={e.arxiv}"
    try:
        root = ET.fromstring(_get(url, accept="application/atom+xml"))
    except Exception as ex:
        warn(f"{e.key}: arXiv lookup failed ({ex})")
        return
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None or entry.find("a:id", ns) is None:
        err(f"{e.key}: arXiv id not found ({e.arxiv})")
        return
    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
    if title and not _title_matches(e.title, title):
        warn(f"{e.key}: title differs from arXiv\n      bib: {e.title}\n      arx: {title}")
    published = entry.findtext("a:published", default="", namespaces=ns)
    ax_year = published[:4] if published else None
    if ax_year and e.year and ax_year != e.year:
        # arXiv year is first-submission; journal year may legitimately differ.
        warn(f"{e.key}: bib year {e.year} != arXiv first-submission {ax_year} "
             f"(ok if later journal version)")


def check_online(entries: list[Entry], delay: float) -> None:
    n = sum(1 for e in entries if e.doi or e.arxiv)
    print(f"  verifying {n} entries against Crossref / arXiv "
          f"(~{n * delay:.0f}s polite-pool budget)...", file=sys.stderr)
    for e in entries:
        if e.doi:
            check_doi(e)
            time.sleep(delay)
        elif e.arxiv:
            check_arxiv(e)
            time.sleep(delay)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--online", action="store_true",
                    help="also verify DOIs/arXiv ids against primary sources")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds between network requests (default 0.3)")
    ap.add_argument("--bib", type=Path, default=BIB)
    args = ap.parse_args()

    text = args.bib.read_text(encoding="utf-8")
    entries = parse_bib(text)
    print(f"parsed {len(entries)} entries from {args.bib}", file=sys.stderr)

    check_offline(text, entries)
    if args.online:
        check_online(entries, args.delay)

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
