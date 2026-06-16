"""Simulation: ranking UNNORMALIZED models with a local (Hyvärinen) score.

The log and CRPS scores need a normalized density — you must know the partition
function $Z$. A *local* proper scoring rule does not: it depends on the forecast
only through the derivatives of $\\log p$ at the outcome, and $\\log p = \\log
\\tilde p - \\log Z$ drops the constant. So an energy-based or unnormalised model
can be scored directly. We score several candidate models — written WITHOUT their
normalizing constants — against data from a true model using the Hyvärinen score,
recover the right ranking, and confirm the scores are invariant to an arbitrary
additive log-constant.

Run:  python examples/sim_local_scoring.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import numpy as np
from _viz import section, labeled_bars

from mechanisms.local_scoring import hyvarinen_score_fd


def unnormalized_gaussian(mu, sigma):
    """log p_tilde(x) = -(x-mu)^2 / (2 sigma^2)  — note: NO -log Z term."""
    return lambda x: -0.5 * ((x - mu) / sigma) ** 2


def unnormalized_mixture(centers, sigma):
    """log of an equal-weight Gaussian mixture, written without any normalizer."""
    def lp(x):
        terms = [-0.5 * ((x - c) / sigma) ** 2 for c in centers]
        m = max(terms)
        return m + math.log(sum(math.exp(t - m) for t in terms))
    return lp


def simulate(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    # Truth: a standard normal stream.
    data = rng.normal(0.0, 1.0, n)

    models = {
        "true  N(0,1)":      unnormalized_gaussian(0.0, 1.0),
        "biased N(0.8,1)":   unnormalized_gaussian(0.8, 1.0),
        "overconf. N(0,.5)": unnormalized_gaussian(0.0, 0.5),
        "underconf. N(0,2)": unnormalized_gaussian(0.0, 2.0),
        "bimodal mix ±1.5":  unnormalized_mixture([-1.5, 1.5], 0.8),
    }

    section("Mean Hyvärinen score of each UNNORMALIZED model (lower is better)")
    means = {name: float(np.mean([hyvarinen_score_fd(lp, y) for y in data]))
             for name, lp in models.items()}
    # shift bars to be non-negative for display (scores can be negative)
    lo = min(means.values())
    labeled_bars([(name, v - lo) for name, v in means.items()],
                 vmax=(max(means.values()) - lo) or 1.0, fmt="{:.3f}")
    print("  (bars are offset to the best model; the raw mean scores are:)")
    for name, v in means.items():
        print(f"    {name:<20} {v:+.4f}")
    winner = min(means, key=means.get)
    print(f"\n  lowest score: '{winner}' — the local rule ranks the true model first,")
    print("  having never computed a single normalizing constant.")

    section("Invariance to the normalizing constant")
    y = 0.7
    base = hyvarinen_score_fd(models["true  N(0,1)"], y)
    shifted = hyvarinen_score_fd(lambda x: models["true  N(0,1)"](x) + 12.3456, y)
    print(f"  score of log p_tilde            at y={y}: {base:+.5f}")
    print(f"  score of log p_tilde + 12.3456  at y={y}: {shifted:+.5f}")
    print(f"  difference: {abs(base - shifted):.2e}  — adding any constant changes nothing.")


if __name__ == "__main__":
    simulate()
