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

from typing import NamedTuple

import numpy as np

__all__ = ["BrierDecomposition", "reliability_diagram",
           "expected_calibration_error", "brier_decomposition"]


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


class BrierDecomposition(NamedTuple):
    """Murphy decomposition with the term that makes it close exactly."""

    reliability: float
    resolution: float
    uncertainty: float
    within_bin: float

    @property
    def brier(self) -> float:
        return self.reliability - self.resolution + self.uncertainty + self.within_bin


def brier_decomposition(probs, outcomes, n_bins=10) -> BrierDecomposition:
    r"""Murphy's reliability-resolution-uncertainty decomposition, plus residual.

    ``BS = reliability - resolution + uncertainty + within_bin`` exactly, for
    binary outcomes. Lower reliability is better (perfect calibration gives 0);
    higher resolution is better.

    The three-term identity holds when every forecast in a bin is equal. With
    distinct forecasts grouped into a bin, a within-bin term remains, and it is
    not small: ``probs = [0.1, 0.9]`` with ``outcomes = [0, 1]`` in one bin has
    reliability 0, resolution 0, uncertainty 0.25, implying a Brier score of
    0.25, while the actual score is 0.01. Writing
    :math:`\delta_i = p_i - \bar p_{k(i)}` for the deviation from a forecast's
    own bin mean, the missing term is
    :math:`n^{-1}\sum_i (\delta_i^2 - 2\delta_i y_i)`, returned here as
    ``within_bin``.

    Pass ``n_bins=None`` to group by distinct forecast values instead, which
    makes ``within_bin`` exactly zero and the classical three-term identity
    exact.
    """
    p = np.asarray(probs, float)
    y = np.asarray(outcomes, float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("probs and outcomes must be 1-D of the same length")
    if p.size == 0:
        raise ValueError("need at least one observation")
    if not np.all(np.isfinite(p)) or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("probs must be finite and in [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("outcomes must be binary")

    n = len(y)
    base = y.mean()
    if n_bins is None:
        _, idx = np.unique(p, return_inverse=True)
        n_groups = idx.max() + 1
    else:
        n_bins = int(n_bins)
        if n_bins < 1:
            raise ValueError("n_bins must be a positive integer or None")
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        n_groups = n_bins

    reliability = resolution = 0.0
    deviations = np.zeros_like(p)
    for b in range(n_groups):
        mask = idx == b
        nk = int(mask.sum())
        if nk == 0:
            continue
        pk = p[mask].mean()
        ok = y[mask].mean()
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - base) ** 2
        deviations[mask] = p[mask] - pk
    reliability /= n
    resolution /= n
    uncertainty = base * (1.0 - base)
    within = float(np.mean(deviations ** 2 - 2.0 * deviations * y))
    return BrierDecomposition(float(reliability), float(resolution),
                              float(uncertainty), within)
