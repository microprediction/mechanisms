"""Merging quadratic market makers is Gaussian precision fusion.

Merging cost-function makers is the infimal convolution of their costs, under
which inverse Hessians — liquidity matrices — add. For quadratic costs the
correspondence with Bayesian fusion is exact rather than local: the merged
market's clearing price is the precision-weighted average of the makers'
marginal prices, and its price impact is the inverse of the summed
precisions. Read each maker as an independent Gaussian source whose mean is
its quote and whose precision is its capital-scaled liquidity matrix; the
market quotes the posterior mean and its price impact is the posterior
covariance. Where nobody quotes confidently, impact is high: the market
reports a wide posterior.

Two consequences, developed in
``research/liquidity-is-precision.md``:

- A participant quoting coordinates independently (a diagonal prior, no
  correlation model) blends the aggregate toward a diagonal target: the
  Ledoit-Wolf shrinkage form with intensity equal to the capital share of the
  naive quoter, tuned thereafter by profit and loss rather than by formula.
- Inventory shifts a maker's quote, never its weight: for any fixed cost
  function the supply slope at a given price is the conjugate Hessian at that
  price, independent of the book. Capital is precision; inventory is mean
  shift.

References
----------
- Bhaskara, A., Frongillo, R., Lindgren, E. & Papireddygari, M. (2023). "A
  General Theory of Liquidity Provisioning for Prediction Markets."
  arXiv:2311.08725.
- Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for
  Large-Dimensional Covariance Matrices." J. Multivariate Analysis 88(2).
- Storkey, A. (2011). "Machine Learning Markets." AISTATS.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["QuadraticMaker", "fuse"]


class QuadraticMaker:
    """Vector market maker with quadratic cost and prior quote.

    Cost ``C(q) = m0 . q + q . H q / 2`` where ``H = Lambda^{-1}`` and
    ``Lambda`` is the maker's liquidity (precision) matrix, scaled by its
    capital. The marginal price at inventory ``q`` is ``m0 + H q``; the
    supply offered at external price ``p`` is ``Lambda (p - price)``.

    Parameters
    ----------
    prior_price : quote vector at an empty book (the maker's estimate).
    liquidity   : symmetric positive-definite precision matrix ``Lambda``.
    """

    def __init__(self, prior_price, liquidity):
        self.m0 = np.asarray(prior_price, float).copy()
        self.L = np.asarray(liquidity, float).copy()
        if self.L.shape != (self.m0.size, self.m0.size):
            raise ValueError("liquidity must be square and match prior_price")
        if not np.allclose(self.L, self.L.T):
            raise ValueError("liquidity must be symmetric")
        if np.any(np.linalg.eigvalsh(self.L) <= 0):
            raise ValueError("liquidity must be positive definite")
        self.H = np.linalg.inv(self.L)
        self.q = np.zeros(self.m0.size)

    @property
    def price(self) -> np.ndarray:
        """Marginal price ``m0 + H q`` at the current inventory."""
        return self.m0 + self.H @ self.q

    def cost(self, q) -> float:
        q = np.asarray(q, float)
        return float(self.m0 @ q + 0.5 * q @ self.H @ q)

    def trade_cost(self, s) -> float:
        """Charge ``C(q + s) - C(q)`` for a fill ``s``."""
        return self.cost(self.q + s) - self.cost(self.q)

    def supply(self, p) -> np.ndarray:
        """Fill the maker optimally provides at price ``p``: ``Lambda (p - price)``."""
        return self.L @ (np.asarray(p, float) - self.price)

    def apply_fill(self, s) -> float:
        """Execute a fill, mutating inventory; return the charge collected."""
        charge = self.trade_cost(s)
        self.q = self.q + np.asarray(s, float)
        return charge


def fuse(makers: Sequence[QuadraticMaker], demand=None):
    """Clear a net ``demand`` against the merged market; ``None`` means zero.

    Returns ``(price, precision, fills)``: the clearing price
    ``(sum_i Lambda_i)^{-1} (sum_i Lambda_i price_i + demand)``, the aggregate
    precision ``sum_i Lambda_i`` (whose inverse is the merged market's price
    impact, the posterior covariance), and each maker's fill. With zero
    demand this is exactly the Bayesian fusion of independent Gaussian
    sources: precision-weighted mean, precisions added.
    """
    if not makers:
        raise ValueError("need at least one maker")
    n = makers[0].m0.size
    demand = np.zeros(n) if demand is None else np.asarray(demand, float)
    precision = np.sum([m.L for m in makers], axis=0)
    p = np.linalg.solve(precision, np.sum([m.L @ m.price for m in makers], axis=0) + demand)
    fills = [m.supply(p) for m in makers]
    return p, precision, fills
