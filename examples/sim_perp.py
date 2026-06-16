"""Simulation: the funding rate tethers a perpetual to its index, and a liquidation.

An index price random-walks; the perpetual trades at a premium/discount that
decays as funding payments flow (longs pay shorts when the perp is rich). We also
follow a 5x leveraged long: each step it pays/receives funding and is liquidated
if the mark price crosses its liquidation price.

Run:  python examples/sim_perp.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, sparkline

from mechanisms.perp import funding_rate, PerpPosition


def simulate(steps=120, vol=0.02, seed=3):
    rng = np.random.default_rng(seed)
    index = 100.0
    premium = 0.01                       # perp starts 1% rich
    idx_path, perp_path, fund_path = [], [], []
    pos = PerpPosition(side=1, size=1.0, entry_price=100.0, collateral=20.0,
                       maintenance_margin_ratio=0.005)   # 5x long
    liquidated_at = None
    for t in range(steps):
        index *= np.exp(rng.normal(0, vol))
        perp = index * (1 + premium)
        rate = funding_rate(premium_index=premium, interest_rate=0.0001, clamp=0.0005)
        # Funding mean-reverts the premium toward zero (the tether), plus shocks.
        premium = 0.7 * premium + rng.normal(0, 0.003)
        idx_path.append(index); perp_path.append(perp); fund_path.append(rate)
        pos.apply_funding(mark_price=index, rate=rate)
        if liquidated_at is None and pos.is_liquidated(index):
            liquidated_at = (t, index)
    return idx_path, perp_path, fund_path, pos, liquidated_at


def main():
    section("Perpetual future — funding tether and a leveraged liquidation")
    idx, perp, fund, pos, liq = simulate()
    basis = [(p - i) / i * 100 for p, i in zip(perp, idx)]
    print(f"index price:    {sparkline(idx)}")
    print(f"perp premium %: {sparkline(basis)}  (start {basis[0]:+.2f}% -> end {basis[-1]:+.2f}%)")
    print(f"funding rate:   {sparkline(fund)}")
    print("  funding payments shrink the perp's premium toward the index — the tether.\n")
    print(f"5x long, entry 100, liquidation price ≈ {pos.liquidation_price():.2f}")
    if liq:
        print(f"  LIQUIDATED at step {liq[0]} when the index hit {liq[1]:.2f} (mark < liq price).")
    else:
        print(f"  survived to the end; final equity {pos.equity(idx[-1]):.2f} "
              f"(remaining collateral {pos.collateral:.2f} after funding).")


if __name__ == "__main__":
    main()
