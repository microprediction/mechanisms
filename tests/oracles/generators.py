"""Structured generators for cases that random sampling rarely produces.

Uniform random inputs concentrate on the easy interior of a problem. Every
error the audits found lived somewhere a uniform sampler almost never goes: a
threshold's far side, a cost whose violation is exactly zero, a tie between
makers, a matrix on the boundary of a cone. These build such cases on purpose,
parameterised by the quantity that decides the regime, so a test can sweep
across a boundary rather than hope to land near one.
"""

from __future__ import annotations

import itertools

import numpy as np


def controlled_violation_cost(xs, excess: float, hull=(-1.0, 1.0)):
    r"""A cost whose worst chord slope exceeds the hull by exactly ``excess``.

    Built by integrating a slope that is ``hull[1] + excess`` on one interval
    and inside the hull elsewhere, so the arbitrage depth is ``excess`` by
    construction and ``excess = 0`` sits exactly on the coherence boundary.
    A uniform random cost lands on that boundary with probability zero.
    """
    xs = np.asarray(xs, float)
    lo, hi = float(hull[0]), float(hull[1])
    slope = np.full(xs.size, 0.5 * (lo + hi))
    middle = (xs > -1.0) & (xs < 1.0)
    slope[middle] = hi + float(excess)
    dx = np.diff(xs)
    values = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * dx)])
    return values - values[xs.size // 2]


def multi_well_cost(xs, n_wells: int = 3, depth: float = 1.0):
    """A cost with ``n_wells`` separated local minima of comparable depth.

    Multi-well costs are where the convex envelope has several distinct
    components of ``{C > C-hat}`` and where the "between contact points" claim
    of M4 needs its per-component restriction. Near-equal well depths also make
    the argmin set non-unique, which is the case a single-argmin test misses.
    """
    xs = np.asarray(xs, float)
    centres = np.linspace(-3.0, 3.0, int(n_wells))
    wells = -float(depth) * sum(np.exp(-2.0 * (xs - c) ** 2) for c in centres)
    return 0.05 * xs ** 2 + wells


def non_attained_threshold_cost(xs, asymptote: float = 0.6):
    """A cost whose slope approaches ``asymptote`` without reaching it.

    ``C' = asymptote * tanh(x)`` never attains its supremum, so the conjugate
    is finite on the closed interval but the maximiser escapes to infinity at
    the endpoints. Tests that assume an attained optimum fail here, which is
    the point.
    """
    xs = np.asarray(xs, float)
    return float(asymptote) * np.log(np.cosh(xs))


def tied_makers(n: int = 3, b: float = 1.0, fee: float = 0.1):
    """``n`` makers with identical parameters, so every split is a tie.

    Ties are where a routing rule's tie-breaking shows, where the "sparse
    allocation" claim of M5 fails, and where a residual-assignment bug would
    concentrate the whole rounding error on one maker.
    """
    from mechanisms.fee_routing import LogCoshMaker

    return [LogCoshMaker(b=b, fee=fee) for _ in range(int(n))]


def near_singular_correlation(n: int = 3, slack: float = 1e-6):
    """An equicorrelation matrix whose smallest eigenvalue is about ``slack``.

    For a common ``rho`` the eigenvalues are ``1 - rho`` and ``1 + (n-1) rho``,
    so ``rho = (slack - 1) / (n - 1)`` puts the matrix ``slack`` inside the PSD
    cone. Sweeping ``slack`` through zero crosses the boundary exactly.
    """
    rho = (float(slack) - 1.0) / (int(n) - 1)
    matrix = np.full((int(n), int(n)), rho)
    np.fill_diagonal(matrix, 1.0)
    return matrix, rho


def binary_support(n: int = 3):
    """Every ``+-1`` assignment and the payoffs of all pairwise products.

    Exhaustive rather than sampled: for small ``n`` the outcome space is small
    enough to enumerate, so a claim about "every outcome" can be checked
    against every outcome instead of a few hundred draws.
    """
    outcomes = np.array(list(itertools.product([-1, 1], repeat=int(n))))
    pairs = list(itertools.combinations(range(int(n)), 2))
    payoffs = np.array([[o[i] * o[j] for i, j in pairs] for o in outcomes], float)
    return outcomes, payoffs, pairs
