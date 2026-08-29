"""Non-convex cost-function makers: coherence, envelopes, and frictions.

A path-independent maker charges ``C(q+s) - C(q)``, which telescopes over any
closed path whatever ``C`` is: cycles are refunds, convex or not. The
no-sure-profit condition is therefore not convexity but a chord condition —
every chord slope of ``C`` must lie in the convex hull of payoffs (for the
scalar maker settling in ``[-1, 1]``, ``C`` must be 1-Lipschitz). A
non-convex maker with coherent slopes admits no arbitrage.

What non-convexity does change is which states rational flow visits. A
myopic risk-neutral trader's optimal fill lands exactly on the contact set
where ``C`` meets its lower convex envelope, with maximal profit equal to
the envelope profit plus the gap ``C - conv(C)`` at the starting state: the
market trades the biconjugate ``C**``, and the concave stretches become
unquoted gaps — holes in the book. Whoever lands off-contact (noise)
overpays by the gap; the next rational trader recoups it; the maker is a
conduit that keeps only envelope differences over rational-to-rational
spans.

Two frictions buy back what non-convexity spends. A proportional fee ``f``
tolerates chord slopes exiting the payoff hull by up to ``f`` (bounded
incoherence is priced, the scalar form of the fee-spread lemma). And merging
with a quadratic co-quoter of depth ``lam`` (liquidity ``lam``) is infimal
convolution with ``(.)^2 / (2 lam)`` — the Moreau envelope — which becomes
convex once the co-quoter is deep enough: a deep convex co-quoter fills the
gaps in the book. The required depth is set by the global geometry of the
gaps (how wide a concave stretch the quadratic must bridge), not by the
local weak-convexity constant, which governs prox uniqueness instead.

See ``research/predictors-as-markets.md``.

References
----------
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). "Efficient Market
  Making via Convex Optimization, and a Connection to Online Learning."
  ACM TEAC 1(2).
- Rockafellar, R. T. (1970). *Convex Analysis.* Princeton. (Biconjugation,
  Moreau envelopes, infimal convolution.)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "NonconvexMaker",
    "lower_convex_envelope",
    "max_sure_profit",
    "moreau_envelope",
]


class NonconvexMaker:
    """Scalar path-independent maker with arbitrary cost and proportional fee.

    Settlement is assumed to lie in ``[-1, 1]``; ``cost`` is any callable
    (convexity not required).
    """

    def __init__(self, cost: Callable[[float], float], fee: float = 0.0, q0: float = 0.0):
        if fee < 0:
            raise ValueError("fee must be non-negative")
        self._cost = cost
        self.fee = float(fee)
        self.q = float(q0)

    def cost(self, q: float) -> float:
        return float(self._cost(q))

    def trade_cost(self, s: float) -> float:
        return self.cost(self.q + s) - self.cost(self.q) + self.fee * abs(s)

    def apply_fill(self, s: float) -> float:
        charge = self.trade_cost(s)
        self.q += s
        return charge


def lower_convex_envelope(xs, ys) -> np.ndarray:
    """Values of the lower convex envelope of the graph ``(xs, ys)`` at ``xs``.

    ``xs`` must be increasing. Andrew's monotone-chain lower hull.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    hull: list[tuple[float, float]] = []
    for p in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) >= (p[1] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(p)
    hx, hy = (np.array(v) for v in zip(*hull))
    return np.interp(xs, hx, hy)


def max_sure_profit(xs, ys, fee: float = 0.0, stride: int = 1) -> float:
    """Largest outcome-independent profit of a single fill against the maker.

    For settlement in ``[-1, 1]`` the worst-case payoff of a fill ``s`` is
    ``-|s|``, so the sure profit of moving from ``x_i`` to ``x_j`` is
    ``-|x_j - x_i| - (y_j - y_i) - fee |x_j - x_i|``. Non-positive for every
    pair iff the fee-widened chord condition holds.
    """
    xs = np.asarray(xs, float)[::stride]
    ys = np.asarray(ys, float)[::stride]
    S = xs[None, :] - xs[:, None]
    dC = ys[None, :] - ys[:, None]
    prof = -np.abs(S) - dC - fee * np.abs(S)
    return float(prof.max())


def moreau_envelope(xs, ys, lam: float) -> np.ndarray:
    """Numeric Moreau envelope ``min_y ys(y) + (x - y)^2 / (2 lam)`` on ``xs``.

    The merged venue of the ``ys`` maker and a quadratic co-quoter of depth
    ``lam`` (their infimal convolution).
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    out = np.empty_like(ys)
    chunk = 512
    for a in range(0, len(xs), chunk):
        block = xs[a:a + chunk, None] - xs[None, :]
        out[a:a + chunk] = (ys[None, :] + block * block / (2.0 * lam)).min(axis=1)
    return out
