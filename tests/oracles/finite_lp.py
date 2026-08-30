"""An exact linear-programming oracle for arbitrage on a finite outcome space.

Propositions 1, 3 and 11 of *Non-Convex Market Makers* are three readings of one
question: given quoted prices and a payoff matrix, is there a portfolio whose
price is below its worst-case payoff? On a finite outcome space that is a linear
program, so it can be answered exactly rather than searched for.

Solving it once settles all three at the same time:

- **Proposition 1** (no-arbitrage is the chord condition) is the ``fee = 0``,
  scalar case, where the portfolio is a single signed fill.
- **Proposition 3** (a fee of at least the arbitrage depth restores coherence)
  is the same LP with the fee subtracted from the margin.
- **Proposition 11** (arbitrage is separation) is the LP's *dual*: the optimal
  ``y`` is exactly the separating functional, and the optimal value is the
  margin by which the quote sits outside the payoff hull.

Uses ``scipy.optimize.linprog``, a test-only dependency. The library itself
stays numpy-only.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ArbitrageCertificate(NamedTuple):
    """The LP's answer, with the witness that makes it checkable."""

    margin: float           # worst-case payoff minus price, at the optimum
    portfolio: np.ndarray   # the separating functional y
    exists: bool            # margin > tolerance

    def verify(self, prices, payoffs, fee: float = 0.0, tol: float = 1e-9) -> bool:
        """Re-price the witness from scratch. A certificate must be checkable."""
        y = np.asarray(self.portfolio, float)
        x = np.asarray(prices, float)
        phi = np.atleast_2d(np.asarray(payoffs, float))
        worst = float(np.min(phi @ y))
        cost = float(x @ y) + fee * float(np.sum(np.abs(y)))
        return abs((worst - cost) - self.margin) <= tol


def arbitrage_lp(prices, payoffs, fee: float = 0.0) -> ArbitrageCertificate:
    r"""Largest sure profit from a unit-sized portfolio, exactly.

    Solves

    .. math::
        \max_{y,\,t}\; t - x^\top y - f\lVert y\rVert_1
        \quad\text{s.t.}\quad t \le \Phi y,\ \lVert y\rVert_1 \le 1,

    where ``x`` are the quoted prices and ``Phi[k]`` is the payoff vector in
    outcome ``k``. The ``l1`` bound is a normalisation: a separating functional
    can be scaled freely, so without it the value is 0 or unbounded and carries
    no information about depth. With it the optimal value is the *margin per
    unit of position*, which is the arbitrage depth of §4.

    Splitting ``y = y_+ - y_-`` with both non-negative makes the ``l1`` norm and
    the fee linear, so this is an honest LP with no smoothing or search.
    """
    from scipy.optimize import linprog

    x = np.asarray(prices, float)
    phi = np.atleast_2d(np.asarray(payoffs, float))
    fee = float(fee)
    if phi.ndim != 2 or phi.shape[1] != x.size:
        raise ValueError("payoffs must be (n_outcomes, n_securities) matching prices")
    if fee < 0:
        raise ValueError("fee must be non-negative")

    n = x.size
    # variables: [y_plus (n), y_minus (n), t]
    c = np.concatenate([x + fee, -x + fee, [-1.0]])          # minimise -objective
    # t <= phi @ y  ->  -phi y_plus + phi y_minus + t <= 0
    a_ub = np.hstack([-phi, phi, np.ones((phi.shape[0], 1))])
    b_ub = np.zeros(phi.shape[0])
    # ||y||_1 <= 1  ->  sum(y_plus + y_minus) <= 1
    a_ub = np.vstack([a_ub, np.concatenate([np.ones(2 * n), [0.0]])])
    b_ub = np.append(b_ub, 1.0)

    bounds = [(0.0, None)] * (2 * n) + [(None, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    y = res.x[:n] - res.x[n:2 * n]
    margin = float(-res.fun)
    return ArbitrageCertificate(margin, y, margin > 1e-9)


def arbitrage_depth_lp(prices, payoffs) -> float:
    """The smallest fee that removes every arbitrage: the arbitrage depth.

    Equal to the zero-fee margin, because the fee enters the objective as
    ``-f ||y||_1`` and the normalisation pins ``||y||_1`` at 1 whenever a
    profitable direction exists. So one LP gives both the violation and the
    fee that prices it, which is the content of Proposition 3.
    """
    return max(0.0, arbitrage_lp(prices, payoffs, fee=0.0).margin)
