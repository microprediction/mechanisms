"""Simulation: one convex potential subsumes LMSR — and many other makers.

Abernethy, Chen & Wortman Vaughan (2013). A whole family of automated market
makers is described by a single convex *potential* C(q) over the outstanding
share vector: prices are the gradient ∇C(q), convexity gives no-arbitrage, and a
bounded gradient range gives bounded worst-case loss. LMSR is just the choice
C(q) = b·log Σ exp(q_i/b). We (a) show the generic maker driven by the LMSR
potential reproduces the dedicated LMSR class exactly, (b) confirm the
finite-difference price fallback matches the analytic gradient, and (c) swap in a
quadratic potential to show LMSR's softmax is one option among many.

Run:  python examples/sim_cmm.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars

from mechanisms.cmm import (
    CostFunctionMarketMaker,
    lmsr_potential,
    quadratic_potential,
)
from mechanisms.lmsr import LMSR


def simulate(b=100.0, seed=0):
    rng = np.random.default_rng(seed)
    n = 3
    trades = [(int(rng.integers(n)), float(rng.uniform(-60, 60))) for _ in range(8)]

    section("Generic potential maker reproduces the dedicated LMSR class")
    cost, grad = lmsr_potential(b)
    cmm = CostFunctionMarketMaker(n, cost, grad)
    ref = LMSR(n_outcomes=n, b=b)

    max_price_gap = 0.0
    max_cost_gap = 0.0
    for outcome, shares in trades:
        c_cmm = cmm.buy(outcome, shares)
        c_ref = ref.buy(outcome, shares)
        max_cost_gap = max(max_cost_gap, abs(c_cmm - c_ref))
        max_price_gap = max(max_price_gap, float(np.max(np.abs(cmm.prices() - ref.prices()))))
    print(f"  after {len(trades)} trades:")
    print(f"  max |price difference| vs LMSR : {max_price_gap:.2e}")
    print(f"  max |trade-cost difference|     : {max_cost_gap:.2e}")
    print("  the generic ∇C(q) maker IS the LMSR — same prices, same costs.")

    section("Finite-difference prices match the analytic gradient")
    # Same potential, but withhold the gradient so the maker differentiates C(q).
    cmm_fd = CostFunctionMarketMaker(n, cost, grad=None, q0=cmm.q)
    fd_gap = float(np.max(np.abs(cmm_fd.prices() - cmm.prices())))
    print(f"  max |finite-diff price − analytic price| : {fd_gap:.2e}")
    print("  so a new mechanism needs only a convex C — the gradient is optional.")

    section("A different potential, a different maker (quadratic)")
    qcost, qgrad = quadratic_potential(alpha=0.01)
    quad = CostFunctionMarketMaker(n, qcost, qgrad, q0=[40.0, -10.0, 5.0])
    p = quad.prices()
    labeled_bars(
        [(f"outcome {i}", p[i]) for i in range(n)],
        vmax=float(np.max(np.abs(p))) or 1.0, fmt="{:+.3f}",
    )
    print(f"  prices = α·q (sum = {p.sum():+.3f}, NOT 1): a bounded-budget maker,")
    print("  not a probability market — the softmax is what makes LMSR a market.")


if __name__ == "__main__":
    simulate()
