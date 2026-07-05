#!/usr/bin/env python3
"""Compile papers/figures/*.tex (standalone TikZ) to a PDF (for the paper PDF
build) and an SVG (for the site), via pdflatex + dvisvgm.

  python scripts/build_figures.py

The .pdf stays next to the source in papers/figures/ so the pandoc/xelatex
build can \\includegraphics it; the .svg is written to docs/papers/figures/ for
the HTML pages. Run locally; not wired into CI (needs a TeX distribution).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "papers" / "figures"
OUT_SVG = ROOT / "docs" / "papers" / "figures"


def build_one(tex: Path) -> None:
    name = tex.stem
    # PDF (standalone-cropped) for the xelatex paper build (\includegraphics)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=FIG, check=True, capture_output=True, text=True)
    # PDF -> SVG for the site; pdftocairo honours the PDF's crop box, so the
    # SVG comes out the same size as the figure (the dvisvgm DVI route did not)
    subprocess.run(
        ["pdftocairo", "-svg",
         str(FIG / f"{name}.pdf"), str(OUT_SVG / f"{name}.svg")],
        check=True, capture_output=True, text=True)
    print(f"built {name}: {name}.pdf + figures/{name}.svg")


def main() -> int:
    for tool in ("pdflatex", "pdftocairo"):
        if shutil.which(tool) is None:
            print(f"{tool} not found — need a TeX distribution and poppler")
            return 1
    OUT_SVG.mkdir(parents=True, exist_ok=True)
    texs = sorted(FIG.glob("*.tex"))
    if not texs:
        print("no figures in papers/figures/")
        return 0
    for tex in texs:
        build_one(tex)
    for ext in ("aux", "log", "out", "dvi"):
        for f in FIG.glob(f"*.{ext}"):
            f.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
