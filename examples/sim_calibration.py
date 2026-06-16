"""Simulation: diagnosing — and fixing — an overconfident forecaster.

Murphy (1973); Gneiting, Balabdaoui & Raftery (2007). Once a market or model
emits probabilities, are they any good? Two properties matter: *calibration* (of
events called p, do ~p occur?) and *sharpness/resolution* (how far forecasts move
from the base rate). We take an overconfident forecaster — one that pushes its
probabilities toward 0 and 1 — and diagnose it with a reliability diagram, the
expected calibration error, and Murphy's reliability/resolution/uncertainty
decomposition of the Brier score. Then a simple temperature recalibration pulls
the forecasts back to the diagonal and the ECE collapses.

Run:  python examples/sim_calibration.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, bar

from mechanisms.calibration import (
    reliability_diagram,
    expected_calibration_error,
    brier_decomposition,
)


def overconfident(p, gamma=2.2):
    """Sharpen probabilities toward 0/1 (logit scaled by gamma) — overconfidence."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) * gamma
    return 1.0 / (1.0 + np.exp(-z))


def recalibrate(p, T):
    """Temperature-scale logits by 1/T (T>1 softens overconfidence)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / T
    return 1.0 / (1.0 + np.exp(-z))


def reliability_table(probs, outcomes, n_bins=10):
    centers, mean_pred, emp_freq, counts = reliability_diagram(probs, outcomes, n_bins)
    print("   pred→actual   n     reliability (| | = miscalibration)")
    for c, mp, ef, k in zip(centers, mean_pred, emp_freq, counts):
        if k == 0:
            continue
        # bar length ~ empirical frequency; marker shows where prediction sits
        line = bar(ef, 1.0, width=24)
        print(f"   {mp:0.2f}→{ef:0.2f}   {k:4d}   {line}")


def simulate(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    # "True" probabilities, then outcomes drawn from them: a perfectly calibrated
    # forecaster would just report true_p.
    true_p = rng.beta(1.4, 1.4, n)
    outcomes = (rng.uniform(size=n) < true_p).astype(int)

    forecasts = overconfident(true_p, gamma=2.2)

    section("Overconfident forecaster — reliability diagram")
    reliability_table(forecasts, outcomes)
    ece = expected_calibration_error(forecasts, outcomes)
    rel, res, unc = brier_decomposition(forecasts, outcomes)
    print(f"\n  ECE = {ece:.4f}   (0 = perfectly calibrated)")
    print(f"  Brier = reliability − resolution + uncertainty")
    print(f"        = {rel:.4f} − {res:.4f} + {unc:.4f} = {rel - res + unc:.4f}")
    print("  reliability is large: the forecasts are systematically off the diagonal.")

    section("After temperature recalibration (T = 2.2)")
    fixed = recalibrate(forecasts, T=2.2)
    reliability_table(fixed, outcomes)
    ece2 = expected_calibration_error(fixed, outcomes)
    rel2, res2, unc2 = brier_decomposition(fixed, outcomes)
    print(f"\n  ECE  {ece:.4f} -> {ece2:.4f}")
    print(f"  reliability {rel:.4f} -> {rel2:.4f}   (resolution preserved: {res:.4f} -> {res2:.4f})")
    print("  recalibration fixes calibration without sacrificing sharpness.")


if __name__ == "__main__":
    simulate()
