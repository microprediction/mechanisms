"""Calibration and sharpness diagnostics for probabilistic forecasts.

Tooling that sits *around* the mechanisms: once a market or model emits
probabilities, are they any good? Two complementary properties:

- **Calibration / reliability** — of the events assigned probability ~p, do a
  fraction ~p actually occur?
- **Sharpness / resolution** — how far from the base rate are the forecasts?
  Sharp, calibrated forecasts are the goal.

References
----------
- Murphy, A. (1973). "A New Vector Partition of the Probability Score." J. Appl. Met.
- Gneiting, T., Balabdaoui, F. & Raftery, A. (2007). "Probabilistic forecasts,
  calibration and sharpness." JRSS-B 69(2).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["reliability_diagram", "expected_calibration_error", "brier_decomposition"]


def reliability_diagram(probs, outcomes, n_bins: int = 10):
    """Bin binary forecasts and return ``(bin_centers, mean_pred, emp_freq, counts)``.

    ``probs`` are forecast probabilities of the positive class in ``[0, 1]``;
    ``outcomes`` are 0/1 realisations. A perfectly calibrated forecaster has
    ``mean_pred == emp_freq`` in every populated bin (the diagonal).
    """
    p = np.asarray(probs, float)
    y = np.asarray(outcomes, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_pred = np.full(n_bins, np.nan)
    emp_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, int)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            mean_pred[b] = p[mask].mean()
            emp_freq[b] = y[mask].mean()
    return centers, mean_pred, emp_freq, counts


def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    r"""Expected Calibration Error: count-weighted mean ``|mean_pred - emp_freq|``."""
    _, mean_pred, emp_freq, counts = reliability_diagram(probs, outcomes, n_bins)
    total = counts.sum()
    if total == 0:
        return float("nan")
    gap = np.abs(mean_pred - emp_freq)
    mask = counts > 0
    return float(np.sum(counts[mask] * gap[mask]) / total)


def brier_decomposition(probs, outcomes, n_bins: int = 10):
    r"""Murphy's reliability-resolution-uncertainty decomposition of the Brier score.

    ``BS = reliability - resolution + uncertainty`` (for binary outcomes). Lower
    reliability is better (perfect calibration => 0); higher resolution is better.
    Returns ``(reliability, resolution, uncertainty)``.
    """
    p = np.asarray(probs, float)
    y = np.asarray(outcomes, float)
    n = len(y)
    base = y.mean()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        nk = mask.sum()
        if nk == 0:
            continue
        pk = p[mask].mean()
        ok = y[mask].mean()
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - base) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base * (1.0 - base)
    return float(reliability), float(resolution), float(uncertainty)
