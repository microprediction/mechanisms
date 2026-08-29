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
an excluded inventory range, which the book shows as a block at the
single supporting price, not as a missing price range. Whoever lands
off-contact (noise)
overpays by the gap; the next rational trader recoups it; the maker is a
conduit that keeps only envelope differences over rational-to-rational
spans.

Two frictions buy back what non-convexity spends. A proportional fee ``f``
tolerates chord slopes exiting the payoff hull by up to ``f`` (bounded
incoherence is priced, the scalar form of the fee-spread lemma). And merging
with a quadratic co-quoter of depth ``lam`` (liquidity ``lam``) is infimal
convolution with ``(.)^2 / (2 lam)`` — the Moreau envelope — which
preserves chord coherence at every depth and attenuates the excluded
range like ``1/lam``. It does not fill that range at finite depth: for the
symmetric double well the midpoint stays strictly off contact for every
finite ``lam``, and it does not convexify the venue either. Apparent
convexification on a bounded window is an artifact: for large ``lam`` the
infimum is attained at the window edge and the envelope is dominated by
the boundary parabola, so widening the window at fixed ``lam`` brings the
non-convexity back (tested). What the depth buys is attenuation of the
envelope gap, not a change of kind.

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

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "ChordCertificate",
    "NonconvexMaker",
    "chord_bounds",
    "lower_convex_envelope",
    "max_sure_profit",
    "min_tenable_fee",
    "moreau_envelope",
    "sure_profit_certificate",
    "worst_case_loss",
]


def _validated_grid(xs, ys):
    """Shared input checks: 1-D, equal length, sorted, finite."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1:
        raise ValueError("xs and ys must be one-dimensional")
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same length")
    if xs.size < 2:
        raise ValueError("need at least two grid points")
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("xs and ys must be finite")
    if not np.all(np.diff(xs) > 0):
        raise ValueError("xs must be strictly increasing")
    return xs, ys


def _validated_fee(fee) -> float:
    fee = float(fee)
    if not np.isfinite(fee) or fee < 0:
        raise ValueError("fee must be finite and non-negative")
    return fee


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

    ``xs`` must be strictly increasing. Andrew's monotone-chain lower hull
    of the sampled points, which is the envelope of the sample rather than
    of the underlying continuous function.
    """
    xs, ys = _validated_grid(xs, ys)
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


@dataclass(frozen=True)
class ChordCertificate:
    """Witness for the worst fee-adjusted sure profit on a grid."""

    max_profit: float
    start_index: int
    end_index: int
    chord_slope: float
    required_fee: float


def sure_profit_certificate(xs, ys, fee: float = 0.0,
                            hull: tuple = (-1.0, 1.0)) -> ChordCertificate:
    """Exact worst sure profit over all grid chords, with its witness.

    For payoff hull ``[a, b]`` the worst-case payoff of a fill ``s`` is
    ``a s`` when ``s > 0`` and ``b s`` when ``s < 0``, so the sure profit of
    moving from ``x_i`` to ``x_j`` is, writing ``A_i = y_i - (a - f) x_i``
    and ``B_i = y_i - (b + f) x_i``,

        i < j (buy):   A_i - A_j          j > i (sell back):  B_j - B_i

    so one monotone scan keeping the running maximum of ``A`` and minimum of
    ``B`` finds the worst chord exactly, in ``O(n)``. This replaces an
    ``O(n^2)`` scan with a ``stride`` argument, which could skip the single
    violating chord and report coherence that is not there.
    """
    xs, ys = _validated_grid(xs, ys)
    a, b = float(hull[0]), float(hull[1])
    if not a < b:
        raise ValueError("hull must satisfy a < b")
    f = _validated_fee(fee)

    A = ys - (a - f) * xs
    B = ys - (b + f) * xs
    best = -np.inf
    wi = wj = 0
    max_A, i_A = A[0], 0
    min_B, i_B = B[0], 0
    for j in range(1, len(xs)):
        buy = max_A - A[j]
        if buy > best:
            best, wi, wj = buy, i_A, j
        sell = B[j] - min_B
        if sell > best:
            best, wi, wj = sell, j, i_B      # fill runs from x_j back to x_i
        if A[j] > max_A:
            max_A, i_A = A[j], j
        if B[j] < min_B:
            min_B, i_B = B[j], j

    s = xs[wj] - xs[wi] if wj != wi else 0.0
    slope = (ys[wj] - ys[wi]) / s if s != 0 else float("nan")
    excess = (a - slope) if s > 0 else (slope - b)
    return ChordCertificate(float(best), int(wi), int(wj), float(slope),
                            float(max(0.0, excess)))


def max_sure_profit(xs, ys, fee: float = 0.0, hull: tuple = (-1.0, 1.0)) -> float:
    """Largest outcome-independent profit of a single fill against the maker.

    Non-positive for every chord iff the fee-widened chord condition holds.
    Exact on the grid; see :func:`sure_profit_certificate` for the witness.
    """
    return sure_profit_certificate(xs, ys, fee=fee, hull=hull).max_profit


def chord_bounds(xs, ys, index: int) -> tuple:
    """One-sided chord bounds ``(L, U)`` at ``xs[index]`` (Lemma 5).

    ``L = sup_{s<0} d_q(s)`` and ``U = inf_{s>0} d_q(s)``; the state is a
    contact point of the convex envelope iff ``L <= U``, with no attainment
    hypothesis needed.
    """
    xs, ys = _validated_grid(xs, ys)
    i = int(index)
    if not 0 <= i < len(xs):
        raise ValueError("index out of range")
    L = -np.inf if i == 0 else float(np.max((ys[:i] - ys[i]) / (xs[:i] - xs[i])))
    U = np.inf if i == len(xs) - 1 else \
        float(np.min((ys[i + 1:] - ys[i]) / (xs[i + 1:] - xs[i])))
    return L, U


def min_tenable_fee(xs, ys, index: int, hull: tuple = (-1.0, 1.0)) -> float:
    """Smallest fee making some admissible belief decline to trade at a state.

    The fee-widened interval ``[L - f, U + f]`` meets the hull ``[a, b]``
    exactly when ``L - f <= U + f``, ``L - f <= b`` and ``a <= U + f``, so

        f_min = max(0, L - b, a - U, (L - U) / 2).

    Under chord coherence ``L, U`` lie in the hull and this reduces to
    ``max(0, (L - U) / 2)``.
    """
    a, b = float(hull[0]), float(hull[1])
    if not a < b:
        raise ValueError("hull must satisfy a < b")
    L, U = chord_bounds(xs, ys, index)
    return float(max(0.0, L - b, a - U, (L - U) / 2.0))


def worst_case_loss(xs, ys, index: int, hull: tuple = (-1.0, 1.0)) -> float:
    """Maker's worst-case loss from state ``xs[index]``, over grid fills.

    ``sup_{z in K, s} [ z s - C(q+s) + C(q) ]``. Coherence does not bound
    this: ``C = 0`` is coherent yet loses without bound, and a cost whose
    slopes stay strictly inside the hull has infinite worst-case loss
    because its conjugate is infinite at the hull's endpoints.
    """
    xs, ys = _validated_grid(xs, ys)
    a, b = float(hull[0]), float(hull[1])
    i = int(index)
    s = xs - xs[i]
    dC = ys - ys[i]
    return float(np.max(np.maximum(a * s, b * s) - dC))


def moreau_envelope(xs, ys, lam: float) -> np.ndarray:
    """Numeric Moreau envelope ``min_y ys(y) + (x - y)^2 / (2 lam)`` on ``xs``.

    The merged venue of the ``ys`` maker and a quadratic co-quoter of depth
    ``lam`` (their infimal convolution). Minimizes over the *sampled* grid,
    so it is the envelope of the sample, not of the underlying continuous
    function; ``tests/oracles/piecewise_linear.py`` supplies the exact
    piecewise-linear oracle used to check it.
    """
    xs, ys = _validated_grid(xs, ys)
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive")
    out = np.empty_like(ys)
    chunk = 512
    for a in range(0, len(xs), chunk):
        block = xs[a:a + chunk, None] - xs[None, :]
        out[a:a + chunk] = (ys[None, :] + block * block / (2.0 * lam)).min(axis=1)
    return out
