"""Hybrid CLOB + AMM market for a binary outcome.

An automated market maker guarantees liquidity but charges slippage on every
trade (the price moves as you sweep the curve). A central limit order book offers
zero-slippage fills but only when a counterparty happens to be resting. A *hybrid*
takes the best of both: match a taker against resting limit orders first, and use
the AMM only for the unmatched remainder.

For a binary market the matchable counterparties are *complementary*: a buyer of
YES at price $a$ and a buyer of NO at price $b$ can be paired whenever
$a + b \\ge 1$ — together they fund the mint of one YES+NO pair (which costs \\$1),
each receiving the side they wanted, with **no AMM involvement and no slippage**.
Concretely, a YES taker matched against a resting NO bid at price $b$ pays $1 - b$
per share. Whatever the book cannot fill is routed to the AMM backstop (an
:class:`~mechanisms.lmsr.LMSR` over the two outcomes), which always quotes a price.

This is the architecture behind on-chain prediction venues that layer a P2P order
book over an AMM (e.g. PulsePlay over Yellow Network); the order book removes the
spread/slippage that deters size, the AMM removes the empty-book problem.

References: Hanson (2007) for the LMSR backstop; see also research/gaps-and-roadmap.md.
"""

from __future__ import annotations

from typing import List

from .lmsr import LMSR

__all__ = ["HybridBinaryMarket"]

YES, NO = 0, 1


class HybridBinaryMarket:
    """Binary market: a resting limit-order book over an LMSR AMM backstop."""

    def __init__(self, b: float = 100.0):
        self.amm = LMSR(n_outcomes=2, b=b)
        # resting BUY orders per side, each [price, size]; price is max willing to pay
        self.resting: dict = {YES: [], NO: []}

    # -- book ------------------------------------------------------------
    def rest(self, side: int, price: float, size: float) -> None:
        """Post a resting buy order: willing to pay up to ``price`` for ``size`` shares."""
        if not (0.0 < price < 1.0):
            raise ValueError("limit price must be in (0, 1)")
        if size <= 0:
            raise ValueError("size must be positive")
        self.resting[int(side)].append([float(price), float(size)])

    def amm_price(self, side: int = YES) -> float:
        return float(self.amm.prices()[int(side)])

    # -- taker -----------------------------------------------------------
    def market_buy(self, side: int, qty: float) -> dict:
        """Buy ``qty`` shares of ``side``: complementary book first, AMM remainder.

        Returns a report dict with the split (P2P vs AMM), the average price paid,
        and the AMM price before/after — so callers can see the slippage avoided.
        """
        side = int(side)
        opp = NO if side == YES else YES
        remaining = float(qty)

        # 1) complementary matching: pair with the highest-priced opposite bids
        #    (highest b => cheapest 1-b for us), zero slippage.
        book = sorted(self.resting[opp], key=lambda o: -o[0])
        p2p_cost = 0.0
        p2p_filled = 0.0
        for order in book:
            if remaining <= 1e-12:
                break
            b_price, b_size = order
            our_price = 1.0 - b_price            # complementary pair funds the $1 mint
            take = min(remaining, b_size)
            p2p_cost += take * our_price
            p2p_filled += take
            remaining -= take
            order[1] -= take
        self.resting[opp] = [o for o in self.resting[opp] if o[1] > 1e-12]

        # 2) AMM backstop for whatever is left
        amm_before = self.amm_price(side)
        amm_cost = 0.0
        amm_filled = 0.0
        if remaining > 1e-12:
            amm_cost = self.amm.buy(side, remaining)
            amm_filled = remaining
            remaining = 0.0
        amm_after = self.amm_price(side)

        filled = p2p_filled + amm_filled
        total = p2p_cost + amm_cost
        return {
            "filled": filled,
            "p2p_filled": p2p_filled,
            "amm_filled": amm_filled,
            "total_cost": total,
            "avg_price": (total / filled) if filled > 0 else float("nan"),
            "amm_price_before": amm_before,
            "amm_price_after": amm_after,
            "p2p_avg_price": (p2p_cost / p2p_filled) if p2p_filled > 0 else float("nan"),
        }
