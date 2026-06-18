"""kelly — Kelly (log-optimal) bet sizing, the agent-side dual of the log score.

Kelly sizing is *not* a market mechanism (a rule that aggregates, prices, or
allocates across agents); it is the agent-side primitive that mechanisms compose
with, so it lives here as a small library rather than as a catalog mechanism.

Three facts tie it to the rest of this package:

* **Sizing.** For a single binary bet at net odds ``b`` with win probability
  ``p``, the bankroll fraction maximising long-run log-growth is
  ``f* = p - (1-p)/b`` (Kelly 1956; Breiman 1961; Thorp 2006).

* **Duality with the log score.** Betting beliefs ``p`` against complete-market
  prices ``pi`` (both probability vectors), a log-wealth maximiser invests the
  whole bankroll as ``b_i = p_i`` and grows at rate
  ``sum_i p_i log(p_i / pi_i) = KL(p || pi)`` per round — exactly the regret of
  the logarithmic score, the Bregman divergence of negative entropy. Sizing and
  proper scoring are the same convex object read from the two sides.

* **Ensembling.** Iterating that bet over a panel of forecasters is an *ensemble*
  mechanism: weight each member by its wealth, and wealth compounds by the
  likelihood it assigned to what happened, ``w_i <- w_i * l_i / sum_j w_j l_j``.
  Accurate members compound, the wealth-weighted consensus self-corrects. This is
  Bayesian model averaging / Cover's universal portfolio, and it is the update
  driving the wealth-weighted pool in ``examples/sim_pipeline.py``.

Functions
---------
kelly_fraction          binary Kelly fraction ``f* = p - (1-p)/b``
kelly_growth_binary     expected log-growth of a binary bet at fraction ``f``
kelly_allocation        log-optimal stake vector over a complete market (= ``p``)
kelly_growth_rate       expected log-growth of betting ``p`` vs ``pi`` = ``KL(p||pi)``
fractional_kelly        scaled (lower-variance) Kelly bet
wealth_weighted_update  one round of the wealth-weighted ensemble update
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kelly_fraction",
    "kelly_growth_binary",
    "kelly_allocation",
    "kelly_growth_rate",
    "fractional_kelly",
    "wealth_weighted_update",
]


def kelly_fraction(p: float, b: float) -> float:
    r"""Growth-optimal bankroll fraction for a binary bet.

    Win probability ``p`` at net decimal odds ``b``: a unit stake returns ``b`` in
    profit on a win and loses the stake on a loss. The log-growth-optimal fraction
    is ``f* = p - (1-p)/b = (p b - (1-p)) / b``. A non-positive ``f*`` means the
    bet has no edge; it is clamped to ``0`` (do not bet).
    """
    p = float(p)
    b = float(b)
    if b <= 0:
        raise ValueError("net odds b must be positive")
    return max(0.0, p - (1.0 - p) / b)


def kelly_growth_binary(p: float, b: float, f: float | None = None) -> float:
    r"""Expected per-bet log-growth ``g(f) = p\log(1+fb) + (1-p)\log(1-f)``.

    With ``f=None`` the optimal ``f*`` from :func:`kelly_fraction` is used. The
    function is concave in ``f`` and maximised at ``f*``.
    """
    p = float(p)
    b = float(b)
    if f is None:
        f = kelly_fraction(p, b)
    f = float(f)
    if f < 0.0 or f >= 1.0:
        raise ValueError("bet fraction f must lie in [0, 1)")
    return float(p * np.log1p(f * b) + (1.0 - p) * np.log1p(-f))


def kelly_allocation(p) -> np.ndarray:
    r"""Log-optimal stake vector over a complete market of exclusive outcomes.

    In a complete market (one Arrow--Debreu claim per outcome, prices summing to
    one) a log-wealth maximiser invests the whole bankroll and stakes fraction
    ``p_i`` on outcome ``i`` — independent of the prices it pays. So the optimal
    allocation is simply the (normalised) belief vector: *bet your beliefs*.
    """
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        raise ValueError("belief vector must have positive mass")
    return p / s


def kelly_growth_rate(p, pi) -> float:
    r"""Expected per-round log-growth of betting beliefs ``p`` against prices ``pi``.

    Both are probability vectors over the same outcomes. The growth rate is
    ``sum_i p_i log(p_i / pi_i) = KL(p || pi)`` — the regret of the log score and
    the Bregman divergence of negative entropy. Non-negative, and zero iff
    ``p == pi`` (no edge over the market).
    """
    p = np.asarray(p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    p = p / p.sum()
    pi = pi / pi.sum()
    mask = p > 0
    if np.any(pi[mask] <= 0):
        raise ValueError("prices must be positive wherever beliefs are")
    return float(np.sum(p[mask] * np.log(p[mask] / pi[mask])))


def fractional_kelly(f, lam: float = 0.5):
    r"""Scaled Kelly stake ``lam * f`` (e.g. ``lam=0.5`` is half-Kelly).

    Fractional Kelly gives up a little growth for a large reduction in variance and
    drawdown. Accepts a scalar or an array of fractions.
    """
    return float(lam) * np.asarray(f, dtype=float)


def wealth_weighted_update(weights, likelihoods) -> np.ndarray:
    r"""One round of the wealth-weighted ensemble update.

    Member ``i`` holds wealth share ``weights[i]`` and assigned probability
    ``likelihoods[i]`` to the outcome that occurred. Log-optimal (Kelly) betting
    against the consensus reallocates wealth multiplicatively by the likelihood, so
    the post-round shares are ``w_i l_i / sum_j w_j l_j``. Iterating is Bayesian
    model averaging / Cover's universal-portfolio ensemble: accurate members
    compound and the wealth-weighted consensus self-corrects.
    """
    w = np.asarray(weights, dtype=float)
    l = np.asarray(likelihoods, dtype=float)
    if w.shape != l.shape:
        raise ValueError("weights and likelihoods must align")
    nw = w * l
    s = nw.sum()
    if s <= 0:
        raise ValueError("at least one member must give the outcome positive mass")
    return nw / s
