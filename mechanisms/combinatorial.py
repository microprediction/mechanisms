r"""Combinatorial and conditional prediction markets via LMSR.

Hanson's market scoring rule was designed for *modular combinatorial* information
aggregation: a single market maker over the joint outcome space of several base
variables prices every logical combination of them at once, and keeps the implied
joint distribution coherent (no Dutch book across related securities).

This module runs an LMSR over the $2^n$ joint states of $n$ binary variables. From
the joint prices it reads off:

- the **probability of any event** (any subset of states, e.g. a logical formula),
- **marginals** $P(X_i = 1)$,
- **conditionals** $P(A \mid B) = P(A \cap B)/P(B)$ — i.e. "if $B$ then $A$"
  markets — which is what makes the market *combinatorial* rather than just a bag
  of independent binaries: trading a joint event moves the relevant conditionals.

Events are length-$2^n$ boolean masks over states; build them with :func:`var` and
combine with numpy ``&``, ``|``, ``~``. Worst-case maker loss is $b\log(2^n)$.

References
----------
- Hanson, R. (2003). "Combinatorial Information Market Design." Info. Sys. Frontiers.
- Hanson, R. (2007). "Logarithmic Market Scoring Rules ..." J. Prediction Markets.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CombinatorialMarket", "var"]


def _states(n_vars: int) -> np.ndarray:
    """Boolean array of shape (2**n_vars, n_vars): row s is the bit pattern of s."""
    s = np.arange(2 ** n_vars)
    return ((s[:, None] >> np.arange(n_vars)[None, :]) & 1).astype(bool)


class CombinatorialMarket:
    """LMSR over the joint states of ``n_vars`` binary variables."""

    def __init__(self, n_vars: int, b: float = 100.0):
        if n_vars < 1:
            raise ValueError("need at least one variable")
        if b <= 0:
            raise ValueError("liquidity parameter b must be positive")
        self.n_vars = int(n_vars)
        self.n_states = 2 ** self.n_vars
        self.b = float(b)
        self.q = np.zeros(self.n_states)
        self.bits = _states(self.n_vars)  # (n_states, n_vars)

    # -- cost / prices over joint states ----------------------------------
    def cost(self, q=None) -> float:
        q = self.q if q is None else np.asarray(q, float)
        z = q / self.b
        m = np.max(z)
        return float(self.b * (m + np.log(np.sum(np.exp(z - m)))))

    def prices(self, q=None) -> np.ndarray:
        """Implied joint distribution over the ``2**n_vars`` states (softmax)."""
        q = self.q if q is None else np.asarray(q, float)
        z = q / self.b
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    # -- reading the distribution -----------------------------------------
    def prob(self, event) -> float:
        """Implied probability of an event (a boolean mask over states)."""
        event = np.asarray(event, bool)
        return float(self.prices()[event].sum())

    def marginal(self, i: int) -> float:
        """Implied marginal probability that variable ``i`` is 1."""
        return self.prob(self.bits[:, int(i)])

    def conditional(self, a, b) -> float:
        r"""Implied conditional ``P(A | B) = P(A and B) / P(B)``."""
        a = np.asarray(a, bool)
        b = np.asarray(b, bool)
        pb = self.prob(b)
        if pb <= 0:
            return float("nan")
        return self.prob(a & b) / pb

    # -- trading ----------------------------------------------------------
    def buy_event(self, event, shares: float) -> float:
        """Buy ``shares`` of a security paying 1 in every state of ``event``.

        Adds ``shares`` to ``q`` on those states and returns the LMSR cost. This is
        the combinatorial security: betting on a logical formula updates all of its
        marginals and conditionals coherently.
        """
        event = np.asarray(event, bool)
        delta = np.zeros(self.n_states)
        delta[event] = float(shares)
        c = self.cost(self.q + delta) - self.cost(self.q)
        self.q = self.q + delta
        return c

    @property
    def max_loss(self) -> float:
        """Worst-case maker loss: ``b log(2**n_vars)``."""
        return self.b * np.log(self.n_states)

    # -- event helpers ----------------------------------------------------
    def var(self, i: int, value: int = 1) -> np.ndarray:
        """Boolean mask: states where variable ``i`` equals ``value`` (default 1)."""
        col = self.bits[:, int(i)]
        return col if value else ~col


def var(market: CombinatorialMarket, i: int, value: int = 1) -> np.ndarray:
    """Free-function form of :meth:`CombinatorialMarket.var`."""
    return market.var(i, value)
