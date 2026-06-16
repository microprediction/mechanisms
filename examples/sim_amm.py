"""Simulation: an AMM liquidity provider, fees vs impermanent loss.

An external reference price follows a random walk. Each step an arbitrageur
trades the constant-product pool back to the external price (paying the fee). We
track the LP's pool value against simply holding the initial basket (HODL). The
gap is impermanent loss; the accumulated fees are the LP's compensation. Whether
LPing beats holding depends on how fees stack up against divergence.

Run:  python examples/sim_amm.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, sparkline

from mechanisms.amm import ConstantProductAMM, impermanent_loss


def simulate(steps=400, vol=0.03, fee=0.003, seed=0):
    rng = np.random.default_rng(seed)
    x0, y0 = 1000.0, 1000.0           # 1:1 pool, external price starts at 1
    pool = ConstantProductAMM(x0, y0, fee=fee)
    price = 1.0
    fees = 0.0
    lp_vals, hodl_vals, prices = [], [], []
    for _ in range(steps):
        price *= np.exp(rng.normal(0, vol))            # GBM external price
        # arbitrage the pool spot (y/x) back to `price`
        spot = pool.y / pool.x
        if price > spot:                                # buy X (cheap) from pool
            # solve approx trade size by a few small steps
            dy = pool.y * (np.sqrt(price / spot) - 1)
            if dy > 0:
                fees += fee * dy
                pool.swap(dy, x_in=False)
        elif price < spot:
            dx = pool.x * (np.sqrt(spot / price) - 1)
            if dx > 0:
                fees += fee * dx
                pool.swap(dx, x_in=True)
        lp_value = pool.x * price + pool.y               # value X in units of Y
        hodl_value = x0 * price + y0
        lp_vals.append(lp_value + fees)
        hodl_vals.append(hodl_value)
        prices.append(price)
    return prices, lp_vals, hodl_vals, fees


def main():
    section("AMM liquidity provider — fees vs impermanent loss")
    prices, lp, hodl, fees = simulate()
    print(f"external price: {sparkline(prices)}  (end {prices[-1]:.3f})")
    print(f"LP value (incl. fees): {sparkline(lp)}")
    print(f"HODL value:            {sparkline(hodl)}")
    print(f"\nfinal LP+fees = {lp[-1]:.1f}   HODL = {hodl[-1]:.1f}   fees earned = {fees:.1f}")
    il = impermanent_loss(prices[-1])
    print(f"closed-form impermanent loss at price {prices[-1]:.3f}: {il*100:.2f}%")
    verdict = "LP beat HODL ✓ (fees > divergence)" if lp[-1] >= hodl[-1] else "HODL beat LP (divergence > fees)"
    print(f"=> {verdict}")


if __name__ == "__main__":
    main()
