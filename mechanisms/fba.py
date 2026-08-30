"""Frequent Batch Auction (FBA) — uniform-price discrete-time clearing.

Budish, Cramton & Shim (2015) argue the continuous-time limit order book
(:mod:`mechanisms.cda`) creates a socially wasteful high-frequency "arms race":
fast traders snipe stale quotes before slower participants can cancel. The
**frequent batch auction** discretises time — orders submitted within a short
interval (a fraction of a second, or one blockchain block) are collected and
cleared together at a single **uniform price** that maximises executed volume.
Because all orders in a batch clear at the same price regardless of arrival
order, microsecond speed advantages (and, on-chain, transaction-ordering / MEV
attacks) lose their value.

This module computes the uniform clearing price and matched quantity from a set
of bids and asks by intersecting aggregate demand and supply.

References
----------
- Budish, E., Cramton, P. & Shim, J. (2015). "The High-Frequency Trading Arms
  Race: Frequent Batch Auctions as a Market Design Response." QJE 130(4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

__all__ = ["BatchAuction", "clear_uniform_price"]


def clear_uniform_price(bids: List[Tuple[float, float]],
                        asks: List[Tuple[float, float]]):
    """Clear a batch and return ``(price, quantity)`` (or ``(None, 0)``).

    ``bids`` and ``asks`` are lists of ``(price, qty)``. Demand at price ``p`` is
    the total quantity of bids with price ``>= p``; supply is the total asks with
    price ``<= p``. The clearing price maximises matched volume
    ``min(demand, supply)``, and is the midpoint of the executable band, the
    interval of prices at which that volume can actually trade.

    The band is set by the *marginal* accepted orders, not the extreme ones.
    Taking the highest crossing bid and lowest crossing ask returns a price at
    which the claimed volume cannot trade: with bids at 100 and 1000 and asks at
    1 and 99, two units cross, but the midpoint of (1, 1000) is 500.5, where
    demand is only one unit. The marginal accepted bid is 100 and the marginal
    accepted ask is 99, so the executable band is [99, 100].
    """
    if not bids or not asks:
        return None, 0.0
    candidates = sorted({p for p, _ in bids} | {p for p, _ in asks})
    best_price, vol = None, 0.0
    for p in candidates:
        demand = sum(q for bp, q in bids if bp >= p)
        supply = sum(q for ap, q in asks if ap <= p)
        v = min(demand, supply)
        if v > vol + 1e-12:
            best_price, vol = p, v
    if vol <= 0:
        return None, 0.0

    lo_band, hi_band = _executable_band(bids, asks, vol)
    return float(0.5 * (lo_band + hi_band)), float(vol)


def _executable_band(bids, asks, vol: float):
    """Prices at which ``vol`` units can trade: [marginal ask, marginal bid].

    Walk demand down from the best bid and supply up from the best ask until
    each reaches ``vol``; the prices at which they get there are the marginal
    accepted orders, and any price between them supports the full volume.
    """
    def marginal(orders, reverse):
        cum = 0.0
        for price, qty in sorted(orders, key=lambda t: t[0], reverse=reverse):
            cum += qty
            if cum >= vol - 1e-12:
                return price
        return price

    marginal_bid = marginal(bids, reverse=True)    # lowest accepted bid
    marginal_ask = marginal(asks, reverse=False)   # highest accepted ask
    return float(marginal_ask), float(marginal_bid)


def clear_with_allocations(bids, asks):
    """Clear and return ``(price, volume, bid_fills, ask_fills)``.

    Orders strictly better than the clearing price fill completely; orders at
    the price are rationed pro rata over the residual, which is the usual
    convention when the marginal price is tied. ``bid_fills`` and ``ask_fills``
    align with the input order lists.
    """
    price, vol = clear_uniform_price(bids, asks)
    if price is None:
        return None, 0.0, [0.0] * len(bids), [0.0] * len(asks)

    def fills(orders, crosses_strictly, at_price):
        out = [0.0] * len(orders)
        inside = [i for i, (p, _) in enumerate(orders) if crosses_strictly(p)]
        marginal = [i for i, (p, _) in enumerate(orders) if at_price(p)]
        taken = 0.0
        for i in inside:
            out[i] = orders[i][1]
            taken += orders[i][1]
        residual = max(0.0, vol - taken)
        pool = sum(orders[i][1] for i in marginal)
        if pool > 0 and residual > 0:
            for i in marginal:                      # pro-rata rationing
                out[i] = residual * orders[i][1] / pool
        return out

    tol = 1e-12
    bid_fills = fills(bids, lambda p: p > price + tol, lambda p: abs(p - price) <= tol)
    ask_fills = fills(asks, lambda p: p < price - tol, lambda p: abs(p - price) <= tol)
    return price, vol, bid_fills, ask_fills


@dataclass
class BatchAuction:
    """Accumulate orders within an interval, then clear at a uniform price."""

    _bids: List[Tuple[float, float]] = None
    _asks: List[Tuple[float, float]] = None

    def __post_init__(self):
        self._bids = []
        self._asks = []

    def submit(self, side: str, price: float, qty: float) -> None:
        if side == "buy":
            self._bids.append((float(price), float(qty)))
        elif side == "sell":
            self._asks.append((float(price), float(qty)))
        else:
            raise ValueError("side must be 'buy' or 'sell'")

    def clear(self):
        """Return ``(clearing_price, matched_qty)`` for the accumulated batch."""
        return clear_uniform_price(self._bids, self._asks)
