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
    ``min(demand, supply)``; we report the midpoint of the overlap between the
    highest matched bid and lowest matched ask.
    """
    if not bids or not asks:
        return None, 0.0
    # Candidate prices: all submitted limit prices.
    candidates = sorted({p for p, _ in bids} | {p for p, _ in asks})
    best = (None, 0.0, 0.0)  # (price, volume, -spreadiness)
    for p in candidates:
        demand = sum(q for bp, q in bids if bp >= p)
        supply = sum(q for ap, q in asks if ap <= p)
        vol = min(demand, supply)
        if vol > best[1] + 1e-12:
            best = (p, vol, 0.0)
    price, vol, _ = best
    if vol <= 0:
        return None, 0.0
    # Refine to the midpoint of the executable band: highest bid and lowest ask
    # that can transact `vol`.
    crossing_bids = sorted((bp for bp, _ in bids if bp >= price), reverse=True)
    crossing_asks = sorted(ap for ap, _ in asks if ap <= price)
    lo = crossing_asks[0]
    hi = crossing_bids[0]
    return float(0.5 * (lo + hi)), float(vol)


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
