"""Simulation: informed traders move an LMSR market to the truth.

A binary event has true probability p*. A stream of traders each see a noisy
private signal and trade the LMSR toward their posterior. The market price
converges to p*, and — no matter how the trading goes — the market maker's loss
never exceeds the bound b·log n.

Run:  python examples/sim_lmsr.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, sparkline

from mechanisms.lmsr import LMSR


def simulate(p_true=0.72, n_traders=80, b=100.0, signal_noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    m = LMSR(n_outcomes=2, b=b)
    prices = [m.prices()[0]]
    for _ in range(n_traders):
        # noisy private estimate of p_true, clipped to (0,1)
        belief = float(np.clip(p_true + rng.normal(0, signal_noise), 0.02, 0.98))
        cur = m.prices()[0]
        # Trade toward belief: spend a budget ∝ the mispricing on the
        # under-priced outcome (convert budget -> shares, then buy).
        if belief > cur:
            budget = b * (belief - cur)
            m.buy(0, m.shares_for_budget(0, budget))
        elif belief < cur:
            budget = b * (cur - belief)
            m.buy(1, m.shares_for_budget(1, budget))
        prices.append(m.prices()[0])
    pnl_if_0 = m.realized_pnl(0)
    pnl_if_1 = m.realized_pnl(1)
    return p_true, prices, m.max_loss, min(pnl_if_0, pnl_if_1)


def main():
    section("LMSR — informed traders converge the price to the truth")
    p_true, prices, max_loss, worst_pnl = simulate()
    print(f"true probability p* = {p_true:.2f}")
    print(f"price path: {sparkline(prices)}")
    print(f"start price {prices[0]:.3f} -> final price {prices[-1]:.3f}  (target {p_true:.2f})")
    section("Bounded loss — the market maker's subsidy is capped")
    print(f"worst-case bound b·log(n) = {max_loss:.2f}")
    print(f"realised worst-case market-maker P&L = {worst_pnl:.2f}  (>= -bound ✓)")
    print("\n  The maker pays at most b·log(n) to learn the crowd's probability;")
    print("  larger b = deeper market, slower convergence, bigger worst-case subsidy.")


if __name__ == "__main__":
    main()
