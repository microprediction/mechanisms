"""Continuous double auction (CDA) with a price-time-priority limit order book.

The CDA is the dominant exchange mechanism: buyers and sellers post limit orders
into a book, and an incoming order matches against the best available opposite
orders by **price priority** (best price first) then **time priority** (earliest
first at a given price). Marketable orders execute immediately; the remainder
rests in the book. This is a minimal, readable matching engine — not a
production exchange — intended to illustrate the mechanism and support
experiments with simple traders (e.g. Gode & Sunder's zero-intelligence agents).

References
----------
- Smith, V. (1962). "An Experimental Study of Competitive Market Behavior." JPE.
- Gode, D. & Sunder, S. (1993). "Allocative Efficiency of Markets with
  Zero-Intelligence Traders." JPE 101(1).
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = ["Order", "Trade", "LimitOrderBook"]


@dataclass
class Order:
    side: str            # "buy" or "sell"
    price: float
    qty: float
    trader: str = "anon"
    id: int = -1

    def __post_init__(self):
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if self.price <= 0 or self.qty <= 0:
            raise ValueError("price and qty must be positive")


@dataclass(frozen=True)
class Trade:
    price: float
    qty: float
    buyer: str
    seller: str
    taker_side: str      # side of the incoming (aggressing) order


@dataclass
class LimitOrderBook:
    """A price-time-priority limit order book for a single instrument.

    Bids are kept in a max-heap (by price, then time); asks in a min-heap. An
    incoming order is matched greedily against the opposite side while prices
    cross, and any residual quantity rests in the book.
    """

    # heaps store (sort_key, seq, Order); seq breaks ties by arrival time.
    _bids: list = field(default_factory=list, repr=False)
    _asks: list = field(default_factory=list, repr=False)
    _seq: itertools.count = field(default_factory=itertools.count, repr=False)
    _next_id: itertools.count = field(default_factory=lambda: itertools.count(1), repr=False)

    def best_bid(self) -> Optional[float]:
        self._clean(self._bids)
        return -self._bids[0][0] if self._bids else None

    def best_ask(self) -> Optional[float]:
        self._clean(self._asks)
        return self._asks[0][0] if self._asks else None

    def spread(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else a - b

    def mid(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else 0.5 * (a + b)

    @staticmethod
    def _clean(heap) -> None:
        # drop fully-filled (qty == 0) orders that bubbled to the top
        while heap and heap[0][2].qty <= 0:
            heapq.heappop(heap)

    def submit(self, order: Order) -> List[Trade]:
        """Submit ``order``; return the list of trades it generates."""
        if order.id < 0:
            order.id = next(self._next_id)
        trades: List[Trade] = []
        if order.side == "buy":
            trades = self._match(order, self._asks, taker_is_buy=True)
            if order.qty > 0:
                heapq.heappush(self._bids, (-order.price, next(self._seq), order))
        else:
            trades = self._match(order, self._bids, taker_is_buy=False)
            if order.qty > 0:
                heapq.heappush(self._asks, (order.price, next(self._seq), order))
        return trades

    def _match(self, incoming: Order, book: list, taker_is_buy: bool) -> List[Trade]:
        trades: List[Trade] = []
        while incoming.qty > 0 and book:
            self._clean(book)
            if not book:
                break
            _, _, resting = book[0]
            resting_price = resting.price
            crosses = (incoming.price >= resting_price) if taker_is_buy else (incoming.price <= resting_price)
            if not crosses:
                break
            fill = min(incoming.qty, resting.qty)
            # Trades execute at the resting (maker) price — standard CDA rule.
            if taker_is_buy:
                buyer, seller = incoming.trader, resting.trader
            else:
                buyer, seller = resting.trader, incoming.trader
            trades.append(Trade(price=resting_price, qty=fill, buyer=buyer,
                                seller=seller, taker_side=incoming.side))
            incoming.qty -= fill
            resting.qty -= fill
            if resting.qty <= 0:
                heapq.heappop(book)
        return trades

    def depth(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return ``(bids, asks)`` as sorted ``[(price, qty), ...]`` ladders."""
        bids: dict = {}
        for _, _, o in self._bids:
            if o.qty > 0:
                bids[o.price] = bids.get(o.price, 0.0) + o.qty
        asks: dict = {}
        for _, _, o in self._asks:
            if o.qty > 0:
                asks[o.price] = asks.get(o.price, 0.0) + o.qty
        return (sorted(bids.items(), reverse=True), sorted(asks.items()))
