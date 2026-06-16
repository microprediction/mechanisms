"""Simulation: a parimutuel pool aggregates beliefs — and the favourite-longshot bias.

Many bettors each hold a noisy private estimate of the true win probabilities and
stake in proportion to their belief. The pool fractions (implied probabilities)
emerge from the crowd. We then reproduce the favourite-longshot bias: when
bettors systematically overweight longshots, the implied probability on the
favourite is too low and on the longshot too high, so favourites are underbet and
longshots overbet — exactly the racetrack anomaly.

Run:  python examples/sim_parimutuel.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars

from mechanisms.parimutuel import ParimutuelPool


def aggregate(true_p, n_bettors=400, noise=0.15, longshot_bias=0.0, seed=0):
    """Build a pool from noisy bettors; return implied probabilities."""
    rng = np.random.default_rng(seed)
    pool = ParimutuelPool(n_outcomes=len(true_p), takeout=0.0)
    for i in range(n_bettors):
        belief = true_p * np.exp(rng.normal(0, noise, size=len(true_p)))
        # longshot_bias > 0 tilts stakes toward low-probability outcomes
        belief = belief * (true_p ** (-longshot_bias))
        belief /= belief.sum()
        outcome = int(rng.choice(len(true_p), p=belief))
        pool.bet(f"b{i}", outcome, float(rng.uniform(1, 5)))
    return pool.implied_probabilities()


def main():
    true_p = np.array([0.6, 0.25, 0.10, 0.05])  # favourite ... longshot

    section("Wisdom of the crowd — pool recovers the true probabilities")
    implied = aggregate(true_p, noise=0.2, seed=1)
    labeled_bars([(f"outcome {i} (true {true_p[i]:.2f})", implied[i]) for i in range(4)],
                 vmax=1.0, fmt="{:.3f}")

    section("Favourite-longshot bias — bettors overweight longshots")
    biased = aggregate(true_p, noise=0.2, longshot_bias=0.5, seed=1)
    print("  outcome   true     implied    implied/true (>1 = overbet)")
    for i in range(4):
        ratio = biased[i] / true_p[i]
        tag = "  <- longshot overbet" if ratio > 1.15 else ("  <- favourite underbet" if ratio < 0.9 else "")
        print(f"   {i}       {true_p[i]:.3f}    {biased[i]:.3f}      {ratio:.2f}{tag}")
    print("\n  A bettor backing the (underbet) favourite has positive expected value —")
    print("  the empirical signature of the favourite-longshot bias.")


if __name__ == "__main__":
    main()
