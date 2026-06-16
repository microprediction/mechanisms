"""Simulation: the Bayesian Truth Serum rewards honesty without ground truth.

Prelec (2004). When there is no verifiable answer, how do you reward truthful
reporting? Naive output-agreement (pay people who match others) collapses to
everyone parroting the obvious answer. The BTS asks each respondent for two
things — their own answer and a prediction of how the population will answer —
and rewards answers that are *more common than collectively predicted*. Truthful
reporting is a Bayesian Nash equilibrium. Here a population of honest respondents
draws private signals from a shared truth; one strategic respondent always picks
the popular answer with an overconfident prediction. The honest crowd out-scores
the strategist on average.

Run:  python examples/sim_peer_prediction.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars

from mechanisms.peer_prediction import bayesian_truth_serum


N_ANSWERS = 3


def honest_prediction(rng, truth, concentration=20.0):
    """A calibrated respondent reports a noisy Dirichlet sample around the truth."""
    return rng.dirichlet(concentration * truth)


def simulate(n_honest=40, rounds=400, seed=0):
    rng = np.random.default_rng(seed)
    # The shared "true" population answer distribution (unknown to respondents).
    truth = np.array([0.55, 0.30, 0.15])

    honest_totals = np.zeros(n_honest)
    strategist_total = 0.0

    for _ in range(rounds):
        # Honest respondents: answer ~ their private signal of the truth; their
        # prediction report is a calibrated estimate of the population.
        answers, predictions = [], []
        for _ in range(n_honest):
            answers.append(int(rng.choice(N_ANSWERS, p=truth)))
            predictions.append(honest_prediction(rng, truth))

        # One strategist: always answers the modal option and predicts everyone
        # else does too (overconfident on the popular answer).
        answers.append(0)
        predictions.append(np.array([0.90, 0.05, 0.05]))

        scores = bayesian_truth_serum(answers, predictions, n_answers=N_ANSWERS)
        honest_totals += scores[:n_honest]
        strategist_total += scores[n_honest]

    mean_honest = honest_totals.mean() / rounds
    mean_strategist = strategist_total / rounds

    section("Bayesian Truth Serum — mean score per round (higher is better)")
    labeled_bars(
        [
            ("honest crowd (avg)", mean_honest),
            ("strategist", mean_strategist),
        ],
        vmax=max(mean_honest, mean_strategist, 1e-9),
        fmt="{:+.3f}",
    )
    verdict = "honest crowd wins" if mean_honest > mean_strategist else "strategist wins"
    print(f"\n  {verdict}: the prediction-report term penalises the strategist's")
    print("  miscalibrated forecast, so gaming the answer does not pay.")


if __name__ == "__main__":
    simulate()
