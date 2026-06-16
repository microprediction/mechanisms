"""Simulation: a combinatorial market keeps conditionals coherent.

One LMSR over the joint states of three binary events prices every logical
combination at once. We read marginals and a conditional "if B then A" off the
joint distribution, then have a trader buy the combinatorial security A∧B and
watch the conditional P(A | B) rise above the marginal P(A) — the market has
learned a dependence no bag of independent binaries could represent — all while
the maker's loss stays bounded by b·log(2^n).

Run:  python examples/sim_combinatorial.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars

from mechanisms.combinatorial import CombinatorialMarket


def simulate(b=50.0):
    m = CombinatorialMarket(n_vars=3, b=b)
    A, B, C = m.var(0), m.var(1), m.var(2)

    section("Three binary events, one joint LMSR (initially independent)")
    labeled_bars([("P(A)", m.marginal(0)), ("P(B)", m.marginal(1)),
                  ("P(C)", m.marginal(2)),
                  ("P(A|B)", m.conditional(A, B))], vmax=1.0, fmt="{:.3f}")
    print(f"  P(A|B) = P(A) at the start — the events are independent.")
    print(f"  maker's worst-case loss is bounded: b·log(2^3) = {m.max_loss:.1f}")

    section("A trader buys the combinatorial security A∧B")
    cost = m.buy_event(A & B, 60.0)
    print(f"  cost of the trade: {cost:.2f}")
    labeled_bars([("P(A)", m.marginal(0)), ("P(B)", m.marginal(1)),
                  ("P(A∧B)", m.prob(A & B)),
                  ("P(A|B)", m.conditional(A, B)),
                  ("P(A|¬B)", m.conditional(A, ~B))], vmax=1.0, fmt="{:.3f}")
    print(f"  P(A|B) = {m.conditional(A, B):.3f}  >  P(A) = {m.marginal(0):.3f}: the market")
    print("  now encodes a dependence — and P(A∧B) = P(A|B)·P(B) stays coherent.")

    section("Conditioning is consistent (Bayes holds on the implied joint)")
    lhs = m.prob(A & B)
    rhs = m.conditional(A, B) * m.prob(B)
    print(f"  P(A∧B) = {lhs:.4f}   P(A|B)·P(B) = {rhs:.4f}   (equal by construction)")
    print(f"  C is untouched, so P(C|A) = {m.conditional(C, A):.3f} ≈ P(C) = {m.marginal(2):.3f}")


if __name__ == "__main__":
    simulate()
