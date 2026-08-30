"""Hanson's Logarithmic Market Scoring Rule (LMSR).

A market scoring rule turns a strictly proper scoring rule into an automated
market maker with infinite liquidity. The LMSR uses the logarithmic score and
admits the closed-form cost function

.. math::
    C(q) = b \\, \\log \\sum_i \\exp(q_i / b),

where ``q_i`` is the number of outstanding shares of outcome ``i`` (each share
pays \\$1 if that outcome occurs) and ``b > 0`` is the liquidity parameter. A
trader moving the share vector from ``q`` to ``q'`` pays ``C(q') - C(q)``. The
instantaneous prices are the softmax gradient

.. math::
    p_i(q) = \\frac{\\exp(q_i / b)}{\\sum_j \\exp(q_j / b)},

which are non-negative and sum to one — the market's implied probabilities. The
market maker's worst-case loss is bounded by ``b * log(n)``.

References
----------
- Hanson, R. (2003). "Combinatorial Information Market Design." ISF 5(1).
- Hanson, R. (2007). "Logarithmic Market Scoring Rules..." J. Prediction Markets 1(1).
- Abernethy, Chen & Wortman Vaughan (2013). "Efficient Market Making via
  Convex Optimization." ACM TEAC 1(2).
"""

from __future__ import annotations

import numpy as np

__all__ = ["LMSR"]


class LMSR:
    """Logarithmic Market Scoring Rule market maker over ``n`` outcomes."""

    def __init__(self, n_outcomes: int, b: float = 100.0, q0=None):
        if n_outcomes < 2:
            raise ValueError("need at least 2 outcomes")
        if b <= 0:
            raise ValueError("liquidity parameter b must be positive")
        self.n = int(n_outcomes)
        self.b = float(b)
        self.q = np.zeros(self.n) if q0 is None else np.asarray(q0, float).copy()
        if self.q.shape != (self.n,):
            raise ValueError("q0 must have shape (n_outcomes,)")
        if not np.all(np.isfinite(self.q)):
            raise ValueError("q0 must be finite")
        self.q_start = self.q.copy()

    # -- core cost function ------------------------------------------------
    def cost(self, q=None) -> float:
        r"""Cost function ``C(q) = b log sum_i exp(q_i / b)`` (log-sum-exp, stable)."""
        q = self.q if q is None else np.asarray(q, float)
        z = q / self.b
        m = np.max(z)
        return float(self.b * (m + np.log(np.sum(np.exp(z - m)))))

    def prices(self, q=None) -> np.ndarray:
        """Instantaneous prices = implied probabilities (softmax of ``q / b``)."""
        q = self.q if q is None else np.asarray(q, float)
        z = q / self.b
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    # -- trading -----------------------------------------------------------
    def cost_to_trade(self, delta) -> float:
        """Cost to change holdings by ``delta`` (a length-n vector of shares).

        Positive entries buy shares of that outcome; negative entries sell.
        """
        delta = np.asarray(delta, float)
        if delta.shape != (self.n,):
            raise ValueError("delta must have shape (n_outcomes,)")
        return self.cost(self.q + delta) - self.cost(self.q)

    def buy(self, outcome: int, shares: float) -> float:
        """Buy ``shares`` of a single ``outcome``; apply the trade; return cost."""
        delta = np.zeros(self.n)
        delta[int(outcome)] = float(shares)
        cost = self.cost_to_trade(delta)
        self.q = self.q + delta
        return cost

    def shares_for_budget(self, outcome: int, budget: float) -> float:
        r"""Shares of ``outcome`` obtainable for exactly ``budget`` of cash.

        Inverting the cost function for a single-outcome buy gives a closed form.
        Let ``S = sum_j exp(q_j / b)`` and ``e = exp(q_o / b)``. Spending ``m`` on
        outcome ``o`` buys ``x`` shares where
        ``m = b log( (S - e + e*exp(x/b)) / S )``, so

        .. math::
            x = b \\log\\!\\Big( \\tfrac{S}{e}\\big(e^{m/b} - 1\\big) + 1 \\Big).
        """
        o = int(outcome)
        z = self.q / self.b
        mx = np.max(z)
        S = np.sum(np.exp(z - mx))           # = (sum exp(q/b)) / exp(mx)
        e = np.exp(z[o] - mx)
        return float(self.b * np.log((S / e) * (np.expm1(budget / self.b)) + 1.0))

    # -- bookkeeping -------------------------------------------------------
    @property
    def max_loss(self) -> float:
        """Market maker's worst-case subsidy, ``b * log(n)``."""
        return float(self.b * np.log(self.n))

    def payout(self, winning_outcome: int) -> float:
        """Total paid to share-holders if ``winning_outcome`` occurs: ``q[winner]``."""
        return float(self.q[int(winning_outcome)])

    def realized_pnl(self, winning_outcome: int) -> float:
        """Market-maker profit/loss since inception: ``C(q) - C(0) - q[winner]``.

        Revenue is measured from the *empty* book, not from ``q0``, so a market
        opened with ``q0 != 0`` is being read as one where those shares were sold
        earlier and their revenue ``C(q0) - C(0)`` was banked at the time. Under
        that reading the loss is bounded below by ``-max_loss``, and
        :meth:`bounded_loss_holds` checks it.

        If instead ``q0`` was granted without collecting anything for it — a
        seeded position rather than prior issuance — the maker also owes
        ``q0[winner]`` it was never paid for, the bound does not apply, and
        :meth:`session_pnl` is the figure you want.
        """
        revenue = self.cost(self.q) - self.cost(np.zeros(self.n))
        return float(revenue - self.payout(winning_outcome))

    def session_pnl(self, winning_outcome: int) -> float:
        """P&L over this object's own trades only: ``C(q) - C(q0) - (q - q0)[winner]``.

        The right figure when ``q0`` is a seeded starting inventory rather than
        prior issuance whose revenue was collected. Equal to
        :meth:`realized_pnl` when ``q0`` is zero.
        """
        revenue = self.cost(self.q) - self.cost(self.q_start)
        issued = float(self.q[int(winning_outcome)]
                       - self.q_start[int(winning_outcome)])
        return float(revenue - issued)

    def bounded_loss_holds(self, winning_outcome: int) -> bool:
        """Is the inception-to-date loss within the ``b log n`` bound?"""
        return self.realized_pnl(winning_outcome) >= -self.max_loss - 1e-9
