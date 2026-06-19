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

__all__ = ["output_agreement", "bayesian_truth_serum", "correlated_agreement"]

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


def _ca_sign_matrix(reports: np.ndarray, n_signals: int) -> np.ndarray:
    r"""Detail-free sign matrix ``S = sign(\Delta)`` for Correlated Agreement.

    ``\Delta[a,b] = P_same(a,b) - P(a)P(b)``, where ``P_same`` is the empirical
    joint of two agents' reports on the *same* task and ``P(a)P(b)`` is the
    independent baseline (peers on *unrelated* tasks). ``S[a,b]`` is +1 where
    co-reporting ``a,b`` is more common than chance, -1 where rarer.
    """
    n_agents, n_tasks = reports.shape
    marg = np.bincount(reports.ravel(), minlength=n_signals).astype(float)
    marg /= marg.sum()
    p_same = np.zeros((n_signals, n_signals))
    pairs = 0
    for t in range(n_tasks):
        col = reports[:, t]
        for a in range(n_agents):
            for b in range(n_agents):
                if a != b:
                    p_same[col[a], col[b]] += 1.0
                    pairs += 1
    p_same /= max(pairs, 1)
    return np.sign(p_same - np.outer(marg, marg))


def correlated_agreement(reports, n_signals: int = None):
    r"""Correlated Agreement (Dasgupta–Ghosh 2013; Shnayder et al. 2016).

    A multi-task peer-prediction mechanism that is **informed-truthful** and needs
    no elicited prediction report (unlike BTS) — only each agent's signal on a
    shared pool of tasks. Each agent is rewarded for agreeing with peers on the
    *same* task, beyond the chance agreement expected from unrelated tasks, scored
    through the sign matrix ``S`` (see :func:`_ca_sign_matrix`):

    .. math::
        \mathrm{score}_i = \operatorname{mean}_{j\neq i,\,t} S[r_{i,t}, r_{j,t}]
        \;-\; \operatorname{mean}_{j\neq i,\,t}\ \mathbb{E}_{b\sim\text{marg}} S[r_{i,t}, b].

    The baseline subtraction is what defeats the degenerate equilibrium of
    :func:`output_agreement`: a constant report agrees with peers but agrees
    *equally* on unrelated tasks, so it nets zero. Truthful reporting is the
    highest-paying strategy whenever the sign matrix captures the true correlation.

    In **binary** settings this is the mechanism for which truthful reporting is
    *stochastically-dominant* truthful (Schoenebeck et al.), i.e. robust to any
    monotone, risk-averse utility — the "Enforced Agreement" property
    (demonstrated in ``examples/sim_correlated_agreement.py``).

    Parameters
    ----------
    reports : integer array, shape ``(n_agents, n_tasks)``; each entry a signal.
    n_signals : number of distinct signals (default: inferred from the data).

    Returns
    -------
    numpy array of per-agent scores.
    """
    reports = np.asarray(reports, dtype=int)
    if reports.ndim != 2:
        raise ValueError("reports must be 2-D (n_agents, n_tasks)")
    n_agents, n_tasks = reports.shape
    if n_signals is None:
        n_signals = int(reports.max()) + 1
    S = _ca_sign_matrix(reports, n_signals)
    marg = np.bincount(reports.ravel(), minlength=n_signals).astype(float)
    marg /= marg.sum()
    baseline = S @ marg  # expected S-score of report a against an independent peer

    scores = np.zeros(n_agents)
    for i in range(n_agents):
        agree = 0.0
        for j in range(n_agents):
            if i == j:
                continue
            agree += float(np.sum(S[reports[i], reports[j]]))
        cnt = (n_agents - 1) * n_tasks
        base = (n_agents - 1) * float(np.sum(baseline[reports[i]]))
        scores[i] = (agree - base) / max(cnt, 1)
    return scores
