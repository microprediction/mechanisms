#!/usr/bin/env python3
"""Build PDF versions of papers/*.md into docs/papers/ via pandoc + xelatex.

Run locally (needs pandoc and a TeX distribution; not wired into CI):

  python scripts/build_pdfs.py

Repo-relative markdown links are rewritten to absolute URLs (sibling papers to
their rendered site pages, code and notes to GitHub) so the links work from
inside a PDF. Math passes through to LaTeX untouched.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "papers"
OUT = ROOT / "docs" / "papers"
SITE = "https://mechanisms.microprediction.org"
GITHUB = "https://github.com/microprediction/mechanisms/blob/main"

PAPERS = [
    "scoring-point-cloud-distributional-submissions.md",
    "composition-and-the-algebra-of-mechanisms.md",
    "multi-stage-solicitation.md",
    "likelihood-versus-crps.md",
    "non-convex-market-makers.md",
]

# The markdown master opens with title / subtitle / byline / hr / Abstract /
# hr; sections are manually numbered "## N. Title" so the site HTML can render
# them verbatim. For the PDF this block becomes real pandoc metadata (yielding
# \maketitle and an abstract environment) and the manual numbers give way to
# --number-sections, which reproduces them one for one.
HEADER = re.compile(
    r"\A# (?P<title>.+)\n+"
    r"### (?P<subtitle>.+)\n+"
    r"(?P<byline>.+)\n+"
    r"---\n+"
    r"## Abstract\n+"
    r"(?P<abstract>[\s\S]+?)\n+---\n+")


def parse_byline(byline: str) -> tuple[str, str, str, str]:
    """Split a `·`-separated byline into (author, affiliation, email, date).

    The author is the first segment. A segment with an `@` and no space is an
    email; an italic `*...*` segment or a bare year is date; the remaining free
    segment (if any) is the affiliation. Both papers parse: composition's
    ``Peter Cotton · *Working draft v0.4* · 2026`` yields no affiliation/email
    and date ``Working draft v0.4 · 2026``; the SSRN byline adds the two.
    """
    segs = [s.strip() for s in byline.split("·")]
    author, affiliation, email, date_parts = segs[0], "", "", []
    for s in segs[1:]:
        if "@" in s and " " not in s:
            email = s
        elif s.startswith("*") and s.endswith("*"):
            date_parts.append(s.strip("*").strip())
        elif re.fullmatch(r"\d{4}", s):
            date_parts.append(s)
        elif not affiliation and not date_parts:
            affiliation = s
        else:
            date_parts.append(s)
    return author, affiliation, email, " · ".join(date_parts)


def paperize(md: str) -> str:
    m = HEADER.match(md)
    if not m:
        raise ValueError("paper header block not in the expected form")
    body = md[m.end():]
    body = re.sub(r"(?m)^## \d+\.\s+", "## ", body)
    body = re.sub(r"(?m)^## References$", "## References {-}", body)
    author, affiliation, email, date = parse_byline(m["byline"])
    if affiliation or email:            # SSRN title page: name carries a footnote
        note = " ".join(x for x in (affiliation + "." if affiliation else "",
                                    f"Email: <{email}>." if email else "") if x)
        author = f"{author}^[{note}]"
    abstract = "\n".join("  " + ln for ln in m["abstract"].splitlines())
    meta = (
        "---\n"
        f'title: "{m["title"]}"\n'
        f'subtitle: "{m["subtitle"]}"\n'
        f'author: "{author}"\n'
        f'date: "{date}"\n'
        f"abstract: |\n{abstract}\n"
        "numbersections: true\n"
        "indent: true\n"
        "---\n\n")
    return meta + body


def absolutize(md: str) -> str:
    def repl(m: "re.Match") -> str:
        text, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://", "#")):
            return m.group(0)
        base, _, frag = url.partition("#")
        frag = f"#{frag}" if frag else ""
        name = Path(base).name
        if base.endswith(".md") and (SRC / name).exists():
            return f"[{text}]({SITE}/papers/{name[:-3]}.html{frag})"
        return f"[{text}]({GITHUB}/{base.lstrip('./').replace('../', '')}{frag})"
    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", repl, md)  # skip ![img](...)


def build_one(name: str) -> Path:
    md = paperize(absolutize((SRC / name).read_text(encoding="utf-8")))
    pdf = OUT / (name[:-3] + ".pdf")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(md)
        tmp = fh.name
    cmd = [
        "pandoc", tmp, "-o", str(pdf),
        "-f", "markdown+tex_math_dollars",
        "--shift-heading-level-by=-1",
        "--citeproc",
        "--bibliography", str(ROOT / "research" / "bibliography.bib"),
        "--metadata", "link-citations=true",
        "--resource-path", str(SRC),          # resolve figures/<name>.pdf
        "--default-image-extension=pdf",
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1.1in",
        "-V", "fontsize=11pt",
        "-V", "mainfont=Palatino",
        "-V", "colorlinks=true",
        "-V", "linkcolor=NavyBlue",
        "-V", "urlcolor=NavyBlue",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    Path(tmp).unlink()
    return pdf


def main() -> int:
    if shutil.which("pandoc") is None:
        print("pandoc not found — install pandoc + a TeX distribution")
        return 1
    OUT.mkdir(exist_ok=True)
    for name in PAPERS:
        try:
            pdf = build_one(name)
            kb = pdf.stat().st_size // 1024
            print(f"wrote {pdf} ({kb} KB)")
        except subprocess.CalledProcessError as e:
            print(f"FAILED {name}:\n{e.stderr[-2000:]}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
