"""Generic convex cost-function market maker (CMM).

Abernethy, Chen & Wortman Vaughan (2013) showed that a large class of automated
market makers is described by a single convex *potential* ``C(q)`` over the
outstanding share vector ``q``: prices are the gradient ``grad C(q)``, convexity
gives no-arbitrage, and a bounded gradient range gives bounded loss. LMSR is the
special case ``C(q) = b log sum exp(q_i / b)``. This module provides the generic
maker and two instances (LMSR and a quadratic / "sum of squares" maker) so the
shared structure is explicit.

Prices here are computed by automatic-difference-free numerical gradient unless
the instance supplies an analytic gradient.

References
----------
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). "Efficient Market
  Making via Convex Optimization, and a Connection to Online Learning."
  ACM TEAC 1(2), Article 12.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["CostFunctionMarketMaker", "lmsr_potential", "quadratic_potential"]


class CostFunctionMarketMaker:
    """Market maker defined by a convex potential ``C`` (and optional gradient).

    Parameters
    ----------
    n_outcomes : number of outcomes / share types.
    cost       : callable ``q -> float``, a convex potential.
    grad       : optional callable ``q -> ndarray`` giving prices ``grad C(q)``.
                 If omitted, a central finite-difference is used.
    """

    def __init__(
        self,
        n_outcomes: int,
        cost: Callable[[np.ndarray], float],
        grad: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        q0=None,
    ):
        self.n = int(n_outcomes)
        self._cost = cost
        self._grad = grad
        self.q = np.zeros(self.n) if q0 is None else np.asarray(q0, float).copy()

    def cost(self, q=None) -> float:
        q = self.q if q is None else np.asarray(q, float)
        return float(self._cost(q))

    def prices(self, q=None) -> np.ndarray:
        q = self.q if q is None else np.asarray(q, float)
        if self._grad is not None:
            return np.asarray(self._grad(q), float)
        # central finite difference fallback
        eps = 1e-6
        g = np.empty(self.n)
        for i in range(self.n):
            e = np.zeros(self.n)
            e[i] = eps
            g[i] = (self._cost(q + e) - self._cost(q - e)) / (2 * eps)
        return g

    def cost_to_trade(self, delta) -> float:
        delta = np.asarray(delta, float)
        return self.cost(self.q + delta) - self.cost(self.q)

    def buy(self, outcome: int, shares: float) -> float:
        delta = np.zeros(self.n)
        delta[int(outcome)] = float(shares)
        c = self.cost_to_trade(delta)
        self.q = self.q + delta
        return c


def lmsr_potential(b: float):
    """Return ``(cost, grad)`` for the LMSR potential with liquidity ``b``."""

    def cost(q):
        z = np.asarray(q, float) / b
        m = np.max(z)
        return b * (m + np.log(np.sum(np.exp(z - m))))

    def grad(q):
        z = np.asarray(q, float) / b
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    return cost, grad


def quadratic_potential(alpha: float = 1.0):
    r"""Return ``(cost, grad)`` for a simple quadratic potential.

    ``C(q) = (alpha/2) * ||q||^2``. Prices are ``alpha * q`` — not normalised to a
    probability simplex, so this is a *bounded-budget* maker rather than a
    probability market, included to show that LMSR's softmax is one choice of
    potential among many.
    """

    def cost(q):
        q = np.asarray(q, float)
        return 0.5 * alpha * float(q @ q)

    def grad(q):
        return alpha * np.asarray(q, float)

    return cost, grad
