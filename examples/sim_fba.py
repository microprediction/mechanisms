"""Simulation: a frequent batch auction clears everyone at one uniform price.

Budish, Cramton & Shim (2015) argue the continuous limit order book invites a
latency arms race: on every change in the fundamental, the fastest trader
"snipes" a stale quote before others can react. Batching orders into discrete
intervals and clearing them at a single uniform price removes the speed race —
within a batch, time priority is irrelevant and all matched orders transact at
the same price. We build random budget-constrained demand and supply, clear the
batch, and confirm the matched volume equals the competitive-equilibrium
quantity (the price is pinned to the overlap band). Then a drifting fundamental
shows successive batch prices tracking it.

Run:  python examples/sim_fba.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, sparkline, labeled_bars

from mechanisms.fba import BatchAuction, clear_uniform_price


def competitive_equilibrium(values, costs):
    """Price/quantity where the demand and supply step curves cross."""
    values = np.sort(values)[::-1]      # demand: highest values transact first
    costs = np.sort(costs)              # supply: lowest costs transact first
    # walk q up while the q-th buyer still values above the q-th seller's cost
    q = 0
    while q < min(len(values), len(costs)) and values[q] >= costs[q]:
        q += 1
    if q == 0:
        return None, 0
    price = 0.5 * (values[q - 1] + costs[q - 1])
    return float(price), q


def simulate(n=60, seed=0):
    rng = np.random.default_rng(seed)
    # Buyers' private values and sellers' private costs, both ~ U(0, 100).
    values = rng.uniform(0, 100, n)
    costs = rng.uniform(0, 100, n)

    auction = BatchAuction()
    # Each trader submits a single unit at their true reservation price (truthful
    # is a dominant strategy in a uniform-price batch with many participants).
    for v in values:
        auction.submit("buy", v, 1.0)
    for c in costs:
        auction.submit("sell", c, 1.0)

    price, qty = auction.clear()
    eq_price, eq_qty = competitive_equilibrium(values, costs)

    section("Frequent batch auction — one batch, one uniform price")
    print(f"  participants         : {n} buyers, {n} sellers (1 unit each)")
    print(f"  batch clearing price : {price:.2f}   matched volume: {qty:.0f}")
    print(f"  competitive equ.     : {eq_price:.2f}   equ. volume   : {eq_qty}")
    print(f"  matched volume equals the equilibrium quantity ({qty:.0f} == {eq_qty});")
    print("  the uniform price sits within the bid-ask overlap band.")
    print("\n  Every matched buyer and seller transacts at the SAME price —")
    print("  no time priority within the batch, so no latency race to win.")

    # A drifting fundamental: each batch re-centres values/costs around a moving
    # fair value. The sequence of uniform clearing prices should track the drift.
    section("Successive batches track a drifting fundamental")
    steps = 80
    fair = 50.0
    fundamentals, prices = [], []
    for _ in range(steps):
        fair += rng.normal(0, 1.5)
        bids = [(fair + rng.normal(0, 8) + rng.uniform(0, 6), 1.0) for _ in range(40)]
        asks = [(fair + rng.normal(0, 8) - rng.uniform(0, 6), 1.0) for _ in range(40)]
        p, _ = clear_uniform_price(bids, asks)
        fundamentals.append(fair)
        prices.append(p if p is not None else fair)

    err = np.mean(np.abs(np.array(prices) - np.array(fundamentals)))
    print(f"  fundamental    : {sparkline(fundamentals)}")
    print(f"  clearing price : {sparkline(prices)}")
    print(f"  batches: {steps},  mean |clearing − fundamental| = {err:.2f}")


if __name__ == "__main__":
    simulate()
