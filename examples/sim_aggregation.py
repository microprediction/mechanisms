"""Simulation: opinion pools, and why trimming defends against a bad node.

Genest & Zidek (1986). Combining many categorical forecasts into one: the
**linear** pool is a weighted arithmetic mean (robust, but tends to
under-confidence), the **logarithmic** pool a normalised geometric mean (sharper,
multiplies evidence — but one near-zero vote can veto an outcome), and the
**depth-trimmed** pool drops the most outlying forecasts before averaging. We
take a crowd of well-informed forecasters plus a handful of adversarial nodes
reporting nonsense, and score each pool by Brier loss against the truth. The
trimmed pool, having discarded the outliers, lands closest.

Run:  python examples/sim_aggregation.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars

from mechanisms.aggregation import (
    linear_opinion_pool,
    logarithmic_opinion_pool,
    depth_trimmed_mean,
)
from mechanisms.scoring_rules import brier_score


def simulate(n_good=24, n_bad=6, rounds=2000, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.array([0.6, 0.25, 0.15])

    pools = {"linear": [], "logarithmic": [], "depth-trimmed": []}

    for _ in range(rounds):
        # Good forecasters: Dirichlet samples concentrated around the truth.
        good = rng.dirichlet(40.0 * truth, size=n_good)
        # Bad nodes: confidently WRONG — piling mass on the least-likely outcome.
        bad = rng.dirichlet(np.array([1.0, 2.0, 30.0]), size=n_bad)
        forecasts = np.vstack([good, bad])

        y = int(rng.choice(3, p=truth))
        pools["linear"].append(brier_score(linear_opinion_pool(forecasts), y))
        pools["logarithmic"].append(brier_score(logarithmic_opinion_pool(forecasts), y))
        pools["depth-trimmed"].append(
            brier_score(depth_trimmed_mean(forecasts, trim=float(n_bad) / (n_good + n_bad)), y)
        )

    section(f"Brier loss of each pool ({n_good} good + {n_bad} adversarial nodes)")
    means = [(name, float(np.mean(losses))) for name, losses in pools.items()]
    labeled_bars(means, fmt="{:.4f}")
    best = min(means, key=lambda kv: kv[1])[0]
    print(f"\n  lower is better — '{best}' pool wins by discarding the outlying")
    print("  adversarial forecasts before averaging the rest.")


if __name__ == "__main__":
    simulate()
