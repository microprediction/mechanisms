"""Simulation: the pm-AMM prices a binary market as a Gaussian probability.

Moallemi & Robinson (2024). A constant-product AMM is a poor fit for prediction
markets: its price can wander anywhere in (0, ∞), but an outcome probability
lives in (0, 1). The pm-AMM uses an invariant whose marginal price is exactly the
standard-normal CDF of the (scaled) reserve difference, ``p = Phi((y - x) / L)``
— the Black-Scholes binary-option probability. So the price is always a genuine
probability, the two outcomes' prices sum to 1, and liquidity ``L`` sets how
much a trade moves the market. We trace the price curve, confirm the
sum-to-one symmetry, and show the invariant vanishes along the trading curve.

Run:  python examples/sim_pm_amm.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from _viz import section, labeled_bars

from mechanisms.pm_amm import implied_price, pm_amm_invariant, pm_amm_price


def solve_y_on_curve(x, L, lo=-50.0, hi=200.0):
    """Find y such that the static invariant is zero (bisection, no scipy)."""
    f = lambda y: pm_amm_invariant(x, y, L)
    a, b = lo, hi
    fa = f(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def simulate(L=100.0):
    section("pm-AMM price = Phi((y - x) / L), always a probability in (0, 1)")
    diffs = np.linspace(-2.5 * L, 2.5 * L, 11)
    labeled_bars(
        [(f"y-x={d:+7.0f}", implied_price(d, L)) for d in diffs],
        vmax=1.0, fmt="{:.3f}",
    )
    print("\n  As the reserve difference sweeps from very negative to very")
    print("  positive, the YES price sweeps smoothly from ~0 to ~1 — an S-curve,")
    print("  never escaping [0, 1] the way a constant-product price would.")

    section("Two outcomes' prices sum to one")
    for d in (-80.0, -20.0, 40.0):
        p_yes = implied_price(d, L)
        p_no = implied_price(-d, L)   # NO market sees the opposite reserve diff
        print(f"  y-x={d:+6.0f}   P(YES)={p_yes:.3f}   P(NO)={p_no:.3f}   sum={p_yes + p_no:.3f}")

    section("The invariant vanishes on the trading curve")
    for x in (20.0, 60.0, 120.0):
        y = solve_y_on_curve(x, L)
        inv = pm_amm_invariant(x, y, L)
        print(f"  x={x:6.1f}  ->  y={y:7.2f}   price={pm_amm_price(x, y, L):.3f}   invariant={inv:+.2e}")


if __name__ == "__main__":
    simulate()
