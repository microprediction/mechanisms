"""Tiny terminal-visualisation helpers shared by the simulation demos.

numpy-only, no matplotlib — these print to the console so every demo runs
anywhere. Imported by the ``sim_*.py`` scripts.
"""

from __future__ import annotations

import numpy as np

_BLOCKS = "▁▂▃▄▅▆▇█"


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def bar(value: float, vmax: float = 1.0, width: int = 40, char: str = "█") -> str:
    """A horizontal bar of length proportional to ``value / vmax``."""
    if vmax <= 0:
        return ""
    n = int(round(width * max(0.0, min(1.0, value / vmax))))
    return char * n


def sparkline(values, width: int = 70) -> str:
    """A one-line unicode sparkline of a sequence (subsampled to ``width``)."""
    v = np.asarray(values, float)
    if v.size == 0:
        return ""
    if v.size > width:                       # subsample evenly to keep it short
        idx = np.linspace(0, v.size - 1, width).round().astype(int)
        v = v[idx]
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return _BLOCKS[len(_BLOCKS) // 2] * len(v)
    idx = ((v - lo) / (hi - lo) * (len(_BLOCKS) - 1)).round().astype(int)
    return "".join(_BLOCKS[i] for i in idx)


def labeled_bars(pairs, vmax=None, width: int = 36, fmt="{:.3f}") -> None:
    """Print ``[(label, value), ...]`` as aligned labelled bars."""
    pairs = list(pairs)
    if vmax is None:
        vmax = max((v for _, v in pairs), default=1.0) or 1.0
    wlab = max((len(str(l)) for l, _ in pairs), default=0)
    for label, value in pairs:
        print(f"  {str(label):<{wlab}}  {bar(value, vmax, width):<{width}} {fmt.format(value)}")
