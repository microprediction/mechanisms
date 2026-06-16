"""Simulation: zero-intelligence traders in a continuous double auction.

Gode & Sunder (1993): even traders who submit *random* (budget-constrained) bids
and asks drive a CDA to near the competitive equilibrium price and capture almost
all the available gains from trade. "The market is a partial substitute for
individual rationality." We reproduce it: buyers hold private values, sellers
private costs; each period a random trader submits a random budget-constrained
order into the order book.

Run:  python examples/sim_cda.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, sparkline

from mechanisms.cda import LimitOrderBook, Order


def competitive_equilibrium(values, costs):
    """Intersect demand (sorted values desc) and supply (costs asc)."""
    v = np.sort(values)[::-1]
    c = np.sort(costs)
    q = int(np.sum(v >= c[:len(v)][np.arange(len(v)) < len(c)])) if len(c) else 0
    q = 0
    for k in range(min(len(v), len(c))):
        if v[k] >= c[k]:
            q = k + 1
        else:
            break
    price = None
    if q > 0:
        price = 0.5 * (v[q - 1] + c[q - 1])
    surplus = float(np.sum(np.maximum(v[:q] - c[:q], 0))) if q else 0.0
    return price, q, surplus


def simulate(n=12, periods=4000, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.uniform(40, 100, size=n)     # buyers' private values
    costs = rng.uniform(20, 80, size=n)       # sellers' private costs
    eq_price, eq_q, eq_surplus = competitive_equilibrium(values, costs)

    # Each trader has one unit. State: idle -> resting (one live order) -> done.
    # The resting lock ensures a trader never trades more than once. The book is
    # cleared after each trade so no stale orders survive (Gode & Sunder reset).
    book = LimitOrderBook()
    trade_prices, realized_surplus = [], 0.0
    bstate = ["idle"] * n
    sstate = ["idle"] * n

    for _ in range(periods):
        buy_side = rng.random() < 0.5
        if buy_side:
            idle = [i for i in range(n) if bstate[i] == "idle"]
            if not idle:
                continue
            i = int(rng.choice(idle))
            price = rng.uniform(1, values[i])      # ZI-C: never bid above value
            trades = book.submit(Order("buy", price, 1, f"B{i}"))
            bstate[i] = "resting"
        else:
            idle = [i for i in range(n) if sstate[i] == "idle"]
            if not idle:
                continue
            i = int(rng.choice(idle))
            price = rng.uniform(costs[i], 100)     # ZI-C: never ask below cost
            trades = book.submit(Order("sell", price, 1, f"S{i}"))
            sstate[i] = "resting"
        for t in trades:
            trade_prices.append(t.price)
            bi, si = int(t.buyer[1:]), int(t.seller[1:])
            realized_surplus += values[bi] - costs[si]
            bstate[bi] = sstate[si] = "done"
            book = LimitOrderBook()                # clear stale resting orders
            for k in range(n):
                if bstate[k] == "resting": bstate[k] = "idle"
                if sstate[k] == "resting": sstate[k] = "idle"
    eff = realized_surplus / eq_surplus if eq_surplus else 0.0
    return eq_price, eq_q, trade_prices, eff


def main():
    section("Zero-intelligence CDA (Gode & Sunder 1993)")
    eq_price, eq_q, prices, eff = simulate()
    print(f"competitive equilibrium price ≈ {eq_price:.1f}, quantity {eq_q}")
    if prices:
        print(f"trade prices: {sparkline(prices)}")
        print(f"mean trade price {np.mean(prices):.1f} (last 5: {[round(p,1) for p in prices[-5:]]})")
    print(f"allocative efficiency = realized / max surplus = {eff*100:.1f}%")
    print("\n  Random, budget-constrained orders still clear near equilibrium and")
    print("  capture nearly all gains from trade — efficiency is in the mechanism.")


if __name__ == "__main__":
    main()
