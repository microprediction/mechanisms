"""Exact oracles for the piecewise-linear interpolant of a sampled cost.

Given knots ``(xs, ys)`` with ``xs`` strictly increasing, write ``C`` for the
piecewise-linear function interpolating them. Each function below returns the
exact value of a quantity for that ``C`` over the *continuum*, by a route that
does not resemble the library's.

The independence is deliberate and is the point:

- :func:`exact_lower_convex_envelope` solves the defining linear program by
  enumerating pairs of knots, where the library walks a monotone chain.
- :func:`exact_moreau_envelope` minimises analytically on each linear piece,
  where the library minimises over sampled points.
- :func:`exact_chord_bounds` and :func:`exact_max_sure_profit` enumerate all
  pairs, where the library uses a single ``O(n)`` running scan.

All are ``O(n^2)`` or worse and are for testing only.
"""

from __future__ import annotations

import numpy as np


def _knots(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if xs.ndim != 1 or xs.shape != ys.shape or xs.size < 2:
        raise ValueError("need at least two aligned knots")
    if not np.all(np.diff(xs) > 0):
        raise ValueError("xs must be strictly increasing")
    return xs, ys


def exact_lower_convex_envelope(xs, ys) -> np.ndarray:
    r"""The convex envelope of the piecewise-linear interpolant, at ``xs``.

    Straight from the definition as a linear program,

    .. math::
        \hat C(q) = \min\Big\{\textstyle\sum_j \lambda_j y_j \;:\;
        \sum_j \lambda_j x_j = q,\ \sum_j \lambda_j = 1,\ \lambda \ge 0\Big\},

    whose optimal basic solutions in one dimension put weight on at most two
    knots. So the exact answer is the smallest value any chord between two
    knots straddling ``q`` takes at ``q``, and enumerating those pairs is
    exact rather than approximate. The library instead walks Andrew's monotone
    chain, which is a different algorithm reaching the same set.
    """
    xs, ys = _knots(xs, ys)
    n = xs.size
    out = np.full(n, np.inf)
    for j in range(n):
        for k in range(j, n):
            if j == k:
                candidate_x, candidate_y = xs[j:j + 1], ys[j:j + 1]
                mask = xs == xs[j]
            else:
                span = xs[k] - xs[j]
                mask = (xs >= xs[j]) & (xs <= xs[k])
                t = (xs[mask] - xs[j]) / span
                candidate_y = ys[j] * (1.0 - t) + ys[k] * t
            out[mask] = np.minimum(out[mask], candidate_y)
    return out


def exact_moreau_envelope(xs, ys, lam: float) -> np.ndarray:
    r"""The Moreau envelope of the piecewise-linear interpolant, at ``xs``.

    ``e_lam C(q) = min_x [C(x) + (x-q)^2 / (2 lam)]``, minimised over the whole
    line rather than over the sampled points. On the piece ``[x_j, x_{j+1}]``,
    ``C(x) = y_j + b_j (x - x_j)`` and the minimand is a parabola with vertex at
    ``x = q - lam b_j``; clipping that vertex to the piece and taking the least
    value over pieces is exact in closed form.

    The library minimises over the grid, so it can only ever return a value at
    a sampled point. The gap between the two is the discretisation error the
    library's docstring warns about, and it is what makes a self-comparison
    unable to detect a wrong Moreau envelope.
    """
    xs, ys = _knots(xs, ys)
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive")

    q = xs[:, None]
    x_lo, x_hi = xs[:-1][None, :], xs[1:][None, :]
    y_lo = ys[:-1][None, :]
    slope = ((ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1]))[None, :]

    vertex = np.clip(q - lam * slope, x_lo, x_hi)
    value = y_lo + slope * (vertex - x_lo) + (vertex - q) ** 2 / (2.0 * lam)
    return value.min(axis=1)


def exact_chord_bounds(xs, ys, index: int):
    r"""One-sided chord bounds ``(L, U)`` at ``xs[index]``, over the continuum.

    ``L = sup_{s<0} d(s)`` and ``U = inf_{s>0} d(s)`` for
    ``d(s) = [C(q+s) - C(q)] / s``. On any linear piece ``d(s) = b + a/s`` is
    monotone in ``s``, so every extremum over the continuum is attained at a
    knot and enumerating knots is exact, not a sample. Returns
    ``(-inf, +inf)`` on the sides where no knot lies.
    """
    xs, ys = _knots(xs, ys)
    i = int(index)
    if not (0 <= i < xs.size):
        raise ValueError("index out of range")
    q, c = xs[i], ys[i]
    left = [(ys[j] - c) / (xs[j] - q) for j in range(xs.size) if j < i]
    right = [(ys[j] - c) / (xs[j] - q) for j in range(xs.size) if j > i]
    lower = max(left) if left else float("-inf")
    upper = min(right) if right else float("inf")
    return float(lower), float(upper)


def exact_max_sure_profit(xs, ys, fee: float = 0.0, hull=(-1.0, 1.0)) -> float:
    r"""Largest sure profit available, by brute-force enumeration of all fills.

    A fill from knot ``i`` to knot ``j`` earns
    ``min_mu in hull [mu (x_j - x_i)] - [C(x_j) - C(x_i)] - fee |x_j - x_i|``,
    and the worst-case belief is the hull endpoint matching the sign of the
    fill. Enumerating every ordered pair is ``O(n^2)`` and obviously correct;
    the library gets the same answer from a single ``O(n)`` running scan, which
    is where an error could hide.

    The value is *not* floored at zero. A coherent maker has every chord losing
    money, so the maximum is negative and its magnitude is the margin by which
    coherence holds. Flooring it would discard exactly that.
    """
    xs, ys = _knots(xs, ys)
    fee = float(fee)
    lo, hi = float(hull[0]), float(hull[1])
    best = -np.inf
    for i in range(xs.size):
        for j in range(xs.size):
            s = xs[j] - xs[i]
            if s == 0.0:
                continue
            worst_payoff = (lo if s > 0 else hi) * s
            profit = worst_payoff - (ys[j] - ys[i]) - fee * abs(s)
            best = max(best, profit)
    return float(best)
