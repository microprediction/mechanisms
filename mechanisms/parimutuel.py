"""Parimutuel (pool betting) markets.

In a parimutuel market all wagers on a set of mutually exclusive, exhaustive
outcomes are pooled. After the outcome is known, the operator removes a fixed
fraction (the *takeout* ``tau``) and distributes the remaining pool pro rata to
the holders of winning tickets. Crucially the odds are *endogenous*: they are
fixed only when betting closes, and the operator carries no outcome risk — the
bettors collectively fund every payout.

Invented by Pierre Oller in the 1860s; mechanised by the totalisator. The pool
fractions ``W_i / W`` are the market's implied probabilities, which makes the
parimutuel a simple information-aggregation mechanism (subject to the well-known
favourite-longshot bias; see Thaler & Ziemba 1988, Snowberg & Wolfers 2010).

References
----------
- Thaler, R. & Ziemba, W. (1988). "Anomalies: Parimutuel Betting Markets."
  J. Economic Perspectives 2(2).
- Pennock, D. (2004). "A Dynamic Pari-Mutuel Market..." ACM EC'04.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

__all__ = ["ParimutuelPool", "DynamicParimutuelMarket"]


@dataclass
class ParimutuelPool:
    """A single static parimutuel pool over ``n`` outcomes.

    Parameters
    ----------
    n_outcomes : number of mutually exclusive outcomes.
    takeout    : fraction ``tau in [0, 1)`` removed from the pool before payout.
    """

    n_outcomes: int
    takeout: float = 0.0
    # stake[outcome] -> list of (bettor, amount)
    _stakes: List[List] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.n_outcomes < 2:
            raise ValueError("need at least 2 outcomes")
        if not (0.0 <= self.takeout < 1.0):
            raise ValueError("takeout must be in [0, 1)")
        self._stakes = [[] for _ in range(self.n_outcomes)]

    def bet(self, bettor: str, outcome: int, amount: float) -> None:
        """Record a wager of ``amount`` by ``bettor`` on ``outcome``."""
        if not (0 <= outcome < self.n_outcomes):
            raise ValueError("outcome out of range")
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._stakes[outcome].append((bettor, float(amount)))

    def pool_on(self, outcome: int) -> float:
        """Total amount wagered on ``outcome``."""
        return float(sum(a for _, a in self._stakes[outcome]))

    @property
    def total_pool(self) -> float:
        return float(sum(self.pool_on(i) for i in range(self.n_outcomes)))

    def implied_probabilities(self) -> np.ndarray:
        """Market-implied probabilities ``W_i / W`` (zeros if pool empty)."""
        w = np.array([self.pool_on(i) for i in range(self.n_outcomes)], float)
        total = w.sum()
        if total == 0:
            return np.zeros(self.n_outcomes)
        return w / total

    def decimal_odds(self, outcome: int) -> float:
        """Current implied decimal odds (payout per unit) on ``outcome``.

        ``(1 - tau) * W / W_i``. Returns ``inf`` if nothing is staked on it.
        """
        w_i = self.pool_on(outcome)
        if w_i == 0:
            return float("inf")
        return (1.0 - self.takeout) * self.total_pool / w_i

    def settle(self, winning_outcome: int) -> Dict[str, float]:
        """Distribute the net pool to winners; return per-bettor payouts.

        A bettor staking ``w`` on the winning outcome receives
        ``w * (1 - tau) * W / W_j``. If nobody backed the winner the net pool is
        returned to all bettors pro rata to their total stake (a common refund
        rule); here we credit it back to backers of the winning outcome only,
        which is empty, so we refund everyone proportionally.
        """
        net = (1.0 - self.takeout) * self.total_pool
        winners = self._stakes[winning_outcome]
        payouts: Dict[str, float] = {}
        w_j = self.pool_on(winning_outcome)
        if w_j == 0:
            # No winning tickets: refund the net pool pro rata across all stakes.
            total = self.total_pool
            if total == 0:
                return payouts
            for outcome in self._stakes:
                for bettor, amt in outcome:
                    payouts[bettor] = payouts.get(bettor, 0.0) + net * amt / total
            return payouts
        for bettor, amt in winners:
            payouts[bettor] = payouts.get(bettor, 0.0) + amt * net / w_j
        return payouts


@dataclass
class DynamicParimutuelMarket:
    """Pennock's (2004) dynamic parimutuel market, share-ratio variant.

    Traders buy *shares* of outcomes; the instantaneous price of an outcome is
    its share of the total money in the pool. Unlike a static pool, prices move
    continuously as shares are bought, so early buyers at low prices profit when
    later buying raises the price — combining parimutuel's no-operator-risk
    property with continuous price discovery. On resolution the entire pool is
    divided among holders of winning shares pro rata.

    This is a teaching implementation of the *total-money-redistributed* payout
    rule. ``money`` is the cash in the pool; ``shares[i]`` the outstanding shares
    of outcome ``i``. The price of outcome ``i`` is ``money * shares_i / sum_k shares_k^2``
    in the "share-ratio" price function; here we use the simpler instantaneous
    price ``p_i = shares_i / sum_k shares_k`` and cost = price for a small buy,
    which is adequate for illustrating the dynamics.
    """

    n_outcomes: int
    shares: np.ndarray = field(default=None, repr=False)
    money: float = 0.0
    _holdings: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.n_outcomes < 2:
            raise ValueError("need at least 2 outcomes")
        if self.shares is None:
            self.shares = np.ones(self.n_outcomes)  # start at uniform prices

    def prices(self) -> np.ndarray:
        s = self.shares.sum()
        return self.shares / s if s > 0 else np.full(self.n_outcomes, 1.0 / self.n_outcomes)

    def buy(self, trader: str, outcome: int, spend: float) -> float:
        """Spend ``spend`` cash to buy shares of ``outcome``; return shares bought.

        Shares bought = ``spend / price`` at the pre-trade price (a linear
        approximation; a production DPM integrates the cost function).
        """
        if not (0 <= outcome < self.n_outcomes):
            raise ValueError("outcome out of range")
        if spend <= 0:
            raise ValueError("spend must be positive")
        price = self.prices()[outcome]
        bought = spend / max(price, 1e-12)
        self.shares[outcome] += bought
        self.money += spend
        h = self._holdings.setdefault(trader, np.zeros(self.n_outcomes))
        h[outcome] += bought
        return float(bought)

    def settle(self, winning_outcome: int) -> Dict[str, float]:
        """Distribute the whole money pool to winning-share holders pro rata."""
        total_winning = self.shares[winning_outcome]
        payouts: Dict[str, float] = {}
        if total_winning == 0:
            return payouts
        for trader, h in self._holdings.items():
            frac = h[winning_outcome] / total_winning
            if frac > 0:
                payouts[trader] = self.money * frac
        return payouts
