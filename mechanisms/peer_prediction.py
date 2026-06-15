"""Peer prediction — eliciting truth without verifiable ground truth.

Scoring rules and markets need an objectively observed outcome to settle. Many
valuable predictions concern things that are never verified (subjective quality,
unobserved field reports). **Peer prediction** mechanisms instead reward a
report based on its statistical relationship to *other agents'* reports, designed
so that truthful reporting is a (strict) equilibrium.

This module implements:

- :func:`output_agreement` — the naive baseline (reward = agreement). It is
  vulnerable to the *degenerate equilibrium* where everyone reports the same
  thing regardless of their signal; included to show what BTS fixes.
- :func:`bayesian_truth_serum` — Prelec's (2004) Bayesian Truth Serum, which
  asks each respondent both for their answer (*information report*) and for their
  estimate of the population answer distribution (*prediction report*), and
  rewards answers that are "surprisingly common" relative to predictions.

References
----------
- Prelec, D. (2004). "A Bayesian Truth Serum for Subjective Data." Science 306.
- Shnayder, V. et al. (2016). "Informed Truthfulness in Multi-Task Peer
  Prediction" (Correlated Agreement).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

__all__ = ["output_agreement", "bayesian_truth_serum"]

_EPS = 1e-12


def output_agreement(report_a: int, report_b: int) -> float:
    """Naive output-agreement reward: 1 if the two reports match, else 0.

    Demonstrates the degenerate-equilibrium failure mode: agents can collude on a
    constant report and always score 1 without revealing private signals.
    """
    return 1.0 if int(report_a) == int(report_b) else 0.0


def bayesian_truth_serum(information_reports: Sequence[int],
                         prediction_reports: Sequence[Sequence[float]],
                         n_answers: int,
                         alpha: float = 1.0):
    r"""Score respondents with Prelec's Bayesian Truth Serum.

    Each respondent ``i`` supplies an **information report** ``x_i`` (their chosen
    answer, an index in ``0..n_answers-1``) and a **prediction report** ``y_i``
    (a distribution over answers estimating how the population will answer).

    The BTS score for respondent ``i`` choosing answer ``k = x_i`` is

    .. math::
        \\mathrm{score}_i = \\log\\frac{\\bar x_k}{\\bar y_k}
                          + \\alpha \\sum_m \\bar x_m \\log\\frac{y_{i,m}}{\\bar x_m},

    the **information score** (answer is more common than collectively predicted)
    plus the **prediction score** (negative KL between the empirical distribution
    ``\\bar x`` and the respondent's prediction ``y_i``). Truthful reporting is a
    Bayesian Nash equilibrium; honest, well-calibrated respondents score highest.

    Parameters
    ----------
    information_reports : length-``n`` integer answers.
    prediction_reports  : length-``n`` list of length-``n_answers`` distributions.
    n_answers           : number of answer options.
    alpha               : weight on the prediction score (default 1).

    Returns
    -------
    numpy array of per-respondent scores.
    """
    x = np.asarray(information_reports, dtype=int)
    y = np.asarray(prediction_reports, dtype=float)
    n = len(x)
    if y.shape != (n, n_answers):
        raise ValueError("prediction_reports must be (n, n_answers)")

    # Empirical answer frequencies x-bar.
    x_bar = np.bincount(x, minlength=n_answers).astype(float) / n
    # Geometric mean of predicted frequencies y-bar_k = exp(mean_i log y_{i,k}).
    log_y = np.log(np.clip(y, _EPS, 1.0))
    y_bar = np.exp(log_y.mean(axis=0))

    scores = np.empty(n)
    for i in range(n):
        k = x[i]
        info = math.log(max(x_bar[k], _EPS) / max(y_bar[k], _EPS))
        # prediction score: alpha * sum_m x_bar_m log(y_{i,m} / x_bar_m)
        pred = alpha * np.sum(
            x_bar * (np.log(np.clip(y[i], _EPS, 1.0)) - np.log(np.clip(x_bar, _EPS, 1.0)))
        )
        scores[i] = info + pred
    return scores
