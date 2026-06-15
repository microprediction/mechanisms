"""Aggregating probabilistic forecasts — opinion pools.

Once many forecasts have been elicited (by markets, scoring rules, or peer
mechanisms), they must be combined into a single consensus distribution. The two
canonical pooling operators:

- :func:`linear_opinion_pool` — a weighted arithmetic mean of the forecasts. It
  is *externally Bayesian*-friendly, preserves the support, and is the natural
  pool when forecasts are exchangeable. Tends to be over-dispersed (under-confident).
- :func:`logarithmic_opinion_pool` — a weighted geometric mean (normalised). It
  is sharper / more confident, multiplicatively combines evidence, and zeroes out
  any outcome that *any* forecaster rules out.

Also provided: :func:`depth_trimmed_mean`, a robust pool that drops the most
extreme forecasts (by distance to the coordinate-wise median) before averaging —
a simple stand-in for the statistical-depth trimming used in robust forecast
aggregation.

References
----------
- Genest, C. & Zidek, J. (1986). "Combining Probability Distributions: A Critique
  and an Annotated Bibliography." Statistical Science 1(1).
"""

from __future__ import annotations

import numpy as np

__all__ = ["linear_opinion_pool", "logarithmic_opinion_pool", "depth_trimmed_mean"]


def _stack_and_weights(forecasts, weights):
    P = np.atleast_2d(np.asarray(forecasts, dtype=float))
    m = P.shape[0]
    if weights is None:
        w = np.full(m, 1.0 / m)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (m,):
            raise ValueError("weights must align with forecasts")
        w = w / w.sum()
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each forecast must be a probability vector summing to 1")
    return P, w


def linear_opinion_pool(forecasts, weights=None) -> np.ndarray:
    r"""Weighted arithmetic mean ``sum_i w_i p_i`` of categorical forecasts."""
    P, w = _stack_and_weights(forecasts, weights)
    return w @ P


def logarithmic_opinion_pool(forecasts, weights=None) -> np.ndarray:
    r"""Normalised weighted geometric mean ``prop_to prod_i p_i^{w_i}``."""
    P, w = _stack_and_weights(forecasts, weights)
    log_pool = w @ np.log(np.clip(P, 1e-15, 1.0))
    pool = np.exp(log_pool - log_pool.max())
    return pool / pool.sum()


def depth_trimmed_mean(forecasts, trim: float = 0.2) -> np.ndarray:
    r"""Robust linear pool: drop the ``trim`` fraction of most-outlying forecasts.

    "Outlyingness" is the L1 distance from the coordinate-wise median forecast (a
    simple proxy for low statistical depth). The deepest ``1 - trim`` fraction is
    averaged. With ``trim = 0`` this is the plain linear pool.
    """
    if not (0.0 <= trim < 1.0):
        raise ValueError("trim must be in [0, 1)")
    P = np.atleast_2d(np.asarray(forecasts, dtype=float))
    m = P.shape[0]
    med = np.median(P, axis=0)
    dist = np.abs(P - med).sum(axis=1)
    keep = max(1, int(round(m * (1.0 - trim))))
    idx = np.argsort(dist)[:keep]
    pooled = P[idx].mean(axis=0)
    return pooled / pooled.sum()
