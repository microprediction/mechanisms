"""Simulation: why proper scoring rules make honesty pay.

A forecaster has a true belief about a 3-outcome event. We draw many outcomes
from that belief and accumulate the loss of (a) reporting truthfully vs (b) a
confident misreport, under the log, Brier and spherical rules. Honest reporting
wins every time — that is exactly what "strictly proper" means.

Run:  python examples/sim_scoring_rules.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars

from mechanisms import scoring_rules as sr


def simulate(rounds=20000, seed=0):
    rng = np.random.default_rng(seed)
    belief = np.array([0.2, 0.5, 0.3])
    misreport = np.array([0.05, 0.15, 0.80])   # overconfident on outcome 2
    outcomes = rng.choice(3, size=rounds, p=belief)
    rules = {"log": sr.log_score, "brier": sr.brier_score, "spherical": sr.spherical_score}
    out = {}
    for name, fn in rules.items():
        honest = np.mean([fn(belief, y) for y in outcomes])
        lying = np.mean([fn(misreport, y) for y in outcomes])
        out[name] = (honest, lying)
    return belief, misreport, out


def main():
    section("Proper scoring rules — honesty pays (mean loss, lower is better)")
    belief, misreport, out = simulate()
    print(f"true belief = {belief},  misreport = {misreport}\n")
    for name, (honest, lying) in out.items():
        verdict = "honest wins ✓" if honest < lying else "??"
        print(f"  {name:<10} honest {honest:.4f}   misreport {lying:.4f}   {verdict}")

    section("Distributional forecasts — the energy score ranks ensembles")
    rng = np.random.default_rng(1)
    truth = np.array([0.5, -0.3])
    clouds = {
        "well-aimed, calibrated": rng.normal(truth, 1.0, size=(800, 2)),
        "well-aimed, over-confident": rng.normal(truth, 0.2, size=(800, 2)),
        "mis-aimed": rng.normal([4, 4], 1.0, size=(800, 2)),
    }
    labeled_bars([(k, sr.energy_score(v, truth)) for k, v in clouds.items()],
                 fmt="{:.3f}")
    print("  (lower energy score = better; the calibrated, well-aimed cloud wins)")


if __name__ == "__main__":
    main()
