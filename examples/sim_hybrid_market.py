"""Simulation: a hybrid order book over an AMM removes slippage.

A pure AMM charges slippage on every trade — a large taker walks up the curve and
moves the price against itself. A hybrid market matches the taker against resting
*complementary* limit orders first (a YES buyer and a NO buyer fund a minted pair
at zero slippage) and only routes the unmatched remainder to the AMM backstop. We
run the same large YES buy through (a) the AMM alone and (b) the hybrid with a
resting NO book, and compare the average price paid and how far the AMM price
drifts.

Run:  python examples/sim_hybrid_market.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _viz import section, labeled_bars

from mechanisms.hybrid_market import HybridBinaryMarket, YES, NO


def simulate(qty=150.0, b=100.0):
    section("Same 150-share YES buy: AMM-only vs hybrid")

    pure = HybridBinaryMarket(b=b)
    r_pure = pure.market_buy(YES, qty)

    hyb = HybridBinaryMarket(b=b)
    hyb.rest(NO, price=0.52, size=120.0)   # complementary NO interest rests on the book
    r_hyb = hyb.market_buy(YES, qty)

    print(f"  AMM-only:  avg price {r_pure['avg_price']:.4f}   "
          f"AMM {r_pure['amm_price_before']:.3f} -> {r_pure['amm_price_after']:.3f}  "
          f"(all {r_pure['amm_filled']:.0f} via AMM)")
    print(f"  hybrid:    avg price {r_hyb['avg_price']:.4f}   "
          f"AMM {r_hyb['amm_price_before']:.3f} -> {r_hyb['amm_price_after']:.3f}  "
          f"({r_hyb['p2p_filled']:.0f} P2P @ {r_hyb['p2p_avg_price']:.3f}, "
          f"{r_hyb['amm_filled']:.0f} via AMM)")

    section("Average price paid (lower is better)")
    labeled_bars([("AMM-only", r_pure["avg_price"]),
                  ("hybrid", r_hyb["avg_price"])], vmax=max(r_pure["avg_price"], 1e-9),
                 fmt="{:.4f}")

    section("AMM price drift caused by the trade (slippage footprint)")
    labeled_bars([("AMM-only", r_pure["amm_price_after"] - r_pure["amm_price_before"]),
                  ("hybrid", r_hyb["amm_price_after"] - r_hyb["amm_price_before"])],
                 vmax=(r_pure["amm_price_after"] - r_pure["amm_price_before"]) or 1e-9,
                 fmt="{:+.4f}")
    print("  the resting complementary orders absorb most of the flow at a fixed")
    print("  price, so the hybrid pays less and barely disturbs the AMM.")


if __name__ == "__main__":
    simulate()
