"""Peer prediction — eliciting truth without verifiable ground truth.

Scoring rules and markets need an objectively observed outcome to settle. Many
valuable predictions concern things that are never verified (subjective quality,
unobserved field reports). **Peer prediction** mechanisms instead reward a
report based on its statistical relationship to *other agents'* reports, designed
so that truthful reporting is a (strict) equilibrium.

This module implements:

- :func:`output_agreement` — the naive baseline (reward = agreement), truthful
  only under the *self-dominant* condition and subject to an uninformative
  equilibrium besides.
- :func:`bayesian_truth_serum` — Prelec's (2004) Bayesian Truth Serum, which
  asks each respondent both for their answer (*information report*) and for their
  estimate of the population answer distribution (*prediction report*), and
  rewards answers that are "surprisingly common" relative to predictions.
- :func:`correlated_agreement` — the multi-task mechanism of Shnayder et al.
  (2016), informed-truthful for all worlds, and :func:`correlated_agreement_df`,
  its detail-free version.
- :func:`enforced_agreement` — Zhang et al.'s (2025) output agreement with the
  report marginal forced to a target, which is the mechanism that carries the
  stochastic-dominance guarantee. It is not Correlated Agreement.

A note on the names, because this literature uses several and they are not
interchangeable. *Truthful* means truth-telling is a best response when everyone
else is truthful. *Informed-truthful* means no strategy profile beats truthful
and ties only against other informed strategies. *Strongly truthful* tightens the
tie condition to permutation strategies. *Dominantly truthful* means truth-telling
is optimal whatever anyone else does, which is a separate and stronger axis that
none of the mechanisms here have. *SD-truthful* means the truthful payoff
distribution first-order stochastically dominates any deviation's, so it is
preferred under every increasing utility, not merely in expectation.

References
----------
- Prelec, D. (2004). "A Bayesian Truth Serum for Subjective Data." Science 306.
- Witkowski, J. & Parkes, D. (2012). "A Robust Bayesian Truth Serum for Small
  Populations." AAAI.
- Waggoner, B. & Chen, Y. (2014). "Output Agreement Mechanisms and Common
  Knowledge." HCOMP.
- Radanovic, G. & Faltings, B. (2013). "A Robust Bayesian Truth Serum for
  Non-Binary Signals." AAAI. (the self-dominant condition)
- Dasgupta, A. & Ghosh, A. (2013). "Crowdsourced Judgement Elicitation with
  Endogenous Proficiency." WWW. (the binary predecessor of CA)
- Shnayder, V., Agarwal, A., Frongillo, R. & Parkes, D. (2016). "Informed
  Truthfulness in Multi-Task Peer Prediction." EC. (Correlated Agreement)
- Zhang, Y., Xu, S., Pennock, D. & Schoenebeck, G. (2025). "Stochastically
  Dominant Peer Prediction." arXiv:2506.02259. (Enforced Agreement)
- Kong, Y. & Schoenebeck, G. (2019). "An Information Theoretic Framework..."
  ACM TEAC 7(1). (the naming hierarchy above)
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

__all__ = ["output_agreement", "output_agreement_is_truthful",
           "bayesian_truth_serum", "correlated_agreement",
           "correlated_agreement_df", "enforce_marginal",
           "enforced_agreement"]

_EPS = 1e-12


def output_agreement(report_a: int, report_b: int) -> float:
    r"""Naive output-agreement reward: 1 if the two reports match, else 0.

    Two distinct things go wrong with it, and they are worth keeping apart.

    *An uninformative equilibrium exists.* Agents can settle on a constant report
    and always score 1 without consulting their signals. This is not special to
    output agreement: an information-elicitation game without verification has an
    uninformative equilibrium exactly when its reward matrix, read as a
    two-player complete-information game, has one (Waggoner & Chen 2014,
    Theorem 3).

    *Truth-telling need not be an equilibrium at all.* Positive correlation is
    not enough, which is the more surprising failure. Take symmetric binary
    signals with

    .. math::
        P(0,0) = 0.50,\quad P(0,1) = P(1,0) = 0.20,\quad P(1,1) = 0.10.

    The signals are positively correlated, since
    :math:`P(1,1) - P(A{=}1)P(B{=}1) = 0.10 - 0.09 > 0`. But an agent who sees a
    1 computes :math:`P(B{=}1 \mid A{=}1) = 0.10/0.30 = 1/3` and maximises
    expected agreement by reporting 0. Truth is not even a best response on that
    signal. See :func:`output_agreement_is_truthful` for the check.

    The condition that does make it truthful is *self-dominance*:
    :math:`P(S_k = x \mid S_i = x) > P(S_k = \tilde x \mid S_i = x)` for every
    :math:`\tilde x \neq x`, that is, whatever you saw is what you think your peer
    most likely saw (Radanovic & Faltings 2013, Proposition 1). Under it, output
    agreement is the unique truthful mechanism up to positive affine
    transformation (Frongillo & Witkowski 2016, Corollary 7).

    Do not confuse self-dominance with the weaker *self-predicting* condition
    :math:`P(S_k = x \mid S_i = x) > P(S_k = x \mid S_i = x')`, which compares
    down a column rather than across a row, is equivalent to positive correlation
    in the binary case, and is the condition for the Peer Truth Serum rather than
    for output agreement.

    Even where truth is an equilibrium, no mechanism without verification can
    guarantee a strict equilibrium more informative than common knowledge
    (Waggoner & Chen 2014, Theorem 2).
    """
    return 1.0 if int(report_a) == int(report_b) else 0.0


def output_agreement_is_truthful(joint) -> bool:
    r"""Does the self-dominant condition hold for a joint signal distribution?

    ``joint[i, j] = P(S_a = i, S_b = j)``. Returns ``True`` when, for every signal
    ``i`` an agent might observe, ``i`` maximises ``P(S_b = j | S_a = i)`` over
    ``j`` — the condition under which reporting your own signal maximises
    expected agreement. Positive correlation does not imply it; see
    :func:`output_agreement`.
    """
    P = np.asarray(joint, float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("joint must be a square matrix of probabilities")
    if np.any(P < 0) or not np.isclose(P.sum(), 1.0):
        raise ValueError("joint must be non-negative and sum to 1")
    for i in range(P.shape[0]):
        row = P[i]
        if row.sum() <= 0:
            continue                      # signal never observed: nothing to check
        if row[i] < row.max() - 1e-12 or np.sum(row >= row.max() - 1e-12) > 1:
            return False
    return True


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
    plus the **prediction score**, which is :math:`-D_{KL}(\\bar x \\Vert y_i)`
    *provided each prediction row is a probability distribution* — hence the
    validation below, without which the expression is not a KL divergence at all
    and a row like ``(1, 1)`` would return a positive value for a term that can
    only be non-positive.

    What is guaranteed, and what is not. Truthful reporting is a Bayesian Nash
    equilibrium under a common prior, conditionally independent signals, and an
    **infinite** population; Prelec notes it survives for suitably large finite
    ``n`` with a prior-dependent threshold. It is *not* the unique equilibrium.
    What it has beyond existence is payoff dominance: no other equilibrium earns
    a higher expected information score, and uninformative ones earn strictly
    less (Kong & Schoenebeck 2019, Theorem 7.8).

    Small populations break it outright. Witkowski & Parkes (2012, Theorem 2)
    show BTS is neither Bayes-Nash incentive compatible nor interim individually
    rational at ``n = 3``, with an explicit counterexample. Their Robust BTS is
    Bayes-Nash IC for every ``n >= 3`` and every admissible prior and is ex-post
    individually rational, at the cost of binary signals; Radanovic & Faltings
    (2013) extend it to non-binary signals under self-dominance. This function is
    the original, so read its output as a large-population construct.

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
    if n == 0:
        raise ValueError("need at least one respondent")
    if y.shape != (n, n_answers):
        raise ValueError("prediction_reports must be (n, n_answers)")
    if np.any(x < 0) or np.any(x >= n_answers):
        raise ValueError("information_reports must index 0..n_answers-1")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("prediction_reports must be finite and non-negative")
    if not np.allclose(y.sum(axis=1), 1.0):
        raise ValueError("each prediction report must sum to 1")

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
    r"""The score matrix ``S = Sign(\Delta)`` for Correlated Agreement.

    ``\Delta[a,b] = P_same(a,b) - P(a)P(b)``, where ``P_same`` is the empirical
    joint of two agents' reports on the *same* task and ``P(a)P(b)`` is the
    independent baseline (peers on *unrelated* tasks).

    ``Sign`` here is Shnayder et al.'s, which is 1 where ``\Delta > 0`` and **0**
    otherwise, not numpy's three-valued sign (their footnote 8 flags the same
    trap). Returning -1 on negative entries would be a different mechanism: it
    would pay agents to actively disagree on pairs that are rarer than chance,
    rather than merely not paying them to agree.
    """
    n_agents, n_tasks = reports.shape
    marg = np.bincount(reports.ravel(), minlength=n_signals).astype(float)
    total = marg.sum()
    marg = marg / total if total > 0 else marg
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
    return (p_same - np.outer(marg, marg) > 0).astype(float)


def _ca_scores(reports, S, marg, task_slice=None) -> np.ndarray:
    """Score every agent with a given sign matrix, optionally on a subset of tasks."""
    n_agents, n_tasks = reports.shape
    tasks = range(n_tasks) if task_slice is None else task_slice
    tasks = list(tasks)
    baseline = S @ marg      # expected S-score of a report against an independent peer
    scores = np.zeros(n_agents)
    for i in range(n_agents):
        agree = 0.0
        for j in range(n_agents):
            if i == j:
                continue
            agree += float(np.sum(S[reports[i, tasks], reports[j, tasks]]))
        cnt = (n_agents - 1) * len(tasks)
        base = (n_agents - 1) * float(np.sum(baseline[reports[i, tasks]]))
        scores[i] = (agree - base) / max(cnt, 1)
    return scores


def correlated_agreement(reports, n_signals: int = None):
    r"""Correlated Agreement (Shnayder, Agarwal, Frongillo & Parkes 2016).

    A multi-task peer-prediction mechanism that needs no elicited prediction
    report (unlike BTS) — only each agent's signal on a shared pool of tasks.
    Each agent is rewarded for agreeing with peers on the *same* task, beyond the
    chance agreement expected from unrelated tasks, scored through the matrix
    ``S = Sign(Delta)`` (see :func:`_ca_sign_matrix`):

    .. math::
        \mathrm{score}_i = \operatorname{mean}_{j\neq i,\,t} S[r_{i,t}, r_{j,t}]
        \;-\; \operatorname{mean}_{j\neq i,\,t}\ \mathbb{E}_{b\sim\text{marg}} S[r_{i,t}, b].

    The baseline subtraction is what defeats the uninformative equilibrium of
    :func:`output_agreement`: a constant report agrees with peers but agrees
    *equally* on unrelated tasks, so it nets zero. Dasgupta & Ghosh (2013) is the
    binary predecessor CA generalises, with the identity score matrix; that
    special case is strongly truthful exactly when the world is *categorical*,
    and binary plus positive correlation is automatically categorical.

    What CA guarantees is **informed truthfulness**, for all worlds (their
    Theorem 5.2): no strategy profile earns more than truthful, and a profile can
    only tie if both agents play informed strategies. Three qualifications belong
    with that.

    It is not dominant-strategy truthful. The equilibrium notion is a strict
    correlated equilibrium, so the guarantee describes what happens when others
    are truthful, not what happens whatever they do.

    Truthful is only *weakly* highest-paying. Ties are characterised: a profile
    with a non-permutation strategy can tie exactly when signals are *clustered*
    (two identical rows or columns in ``Sign(Delta)``, their Lemma 5.6), and
    asymmetric permutation profiles can tie exactly when the distribution has
    *paired permutations* (Lemma 5.8). Absent both, CA is strongly truthful; the
    authors report that holds in roughly half of their empirical worlds.

    It is not stochastic-dominance truthful. That property belongs to a different
    mechanism, **Enforced Agreement**, which is output agreement plus a forced
    rebalancing of each agent's report marginal (Zhang, Xu, Pennock &
    Schoenebeck 2025, Theorem 6.1, for binary self-predicting signals). Their
    Example 4.1 exhibits a case where a constant strategy is *not* stochastically
    dominated under CA, so CA needs their partition-rounding reduction to get
    there. First-order stochastic dominance means the truthful payoff
    distribution is preferred under every increasing utility, which is stronger
    than, and not implied by, an expectation comparison.

    Hypotheses. Two agents (more reduce by random pairing); at least three tasks;
    tasks a priori identical with the bonus and penalty sets drawn *after*
    assignment so agents cannot tell which is which; a finite, symmetric, commonly
    known signal distribution satisfying stochastic relevance. This function takes
    ``Sign(Delta)`` as estimated from the same reports it scores, which is the
    known-``Delta`` variant; :func:`correlated_agreement_df` is the detail-free
    one.

    Parameters
    ----------
    reports : integer array, shape ``(n_agents, n_tasks)``; each entry a signal.
    n_signals : number of distinct signals (default: inferred from the data).

    Returns
    -------
    numpy array of per-agent scores.
    """
    reports, n_signals = _validate_reports(reports, n_signals)
    S = _ca_sign_matrix(reports, n_signals)
    marg = _marginal(reports, n_signals)
    return _ca_scores(reports, S, marg)


def correlated_agreement_df(reports, n_signals: int = None, rng=None):
    r"""Detail-free Correlated Agreement (Shnayder et al. 2016, Definition 5.11).

    :func:`correlated_agreement` estimates ``Sign(Delta)`` from the very reports
    it then scores, so an agent's own reports help set the matrix that grades
    them. The detail-free version splits the tasks randomly into halves ``A`` and
    ``B``, scores ``A`` with the matrix estimated from ``B`` and ``B`` with the
    matrix from ``A``, and averages. The swap is the whole point.

    The guarantee weakens accordingly, from exact informed truthfulness to an
    ``(epsilon, delta)`` version: their Theorem 5.13 gives
    ``m = O(n^3 log(1/delta) / epsilon^2)`` tasks such that, with probability at
    least ``1 - delta``, no strategy profile earns more than ``epsilon`` above
    truthful and every uninformed strategy earns strictly less.
    """
    reports, n_signals = _validate_reports(reports, n_signals)
    n_agents, n_tasks = reports.shape
    if n_tasks < 2:
        raise ValueError("the detail-free split needs at least 2 tasks")
    rng = np.random.default_rng() if rng is None else rng
    order = rng.permutation(n_tasks)
    half_a, half_b = order[: n_tasks // 2], order[n_tasks // 2:]

    marg = _marginal(reports, n_signals)
    s_from_b = _ca_sign_matrix(reports[:, half_b], n_signals)
    s_from_a = _ca_sign_matrix(reports[:, half_a], n_signals)
    scores_a = _ca_scores(reports, s_from_b, marg, task_slice=half_a)
    scores_b = _ca_scores(reports, s_from_a, marg, task_slice=half_b)
    wa, wb = len(half_a) / n_tasks, len(half_b) / n_tasks
    return wa * scores_a + wb * scores_b


def _validate_reports(reports, n_signals):
    reports = np.asarray(reports, dtype=int)
    if reports.ndim != 2:
        raise ValueError("reports must be 2-D (n_agents, n_tasks)")
    if reports.shape[0] < 2:
        raise ValueError("need at least 2 agents")
    if np.any(reports < 0):
        raise ValueError("signals must be non-negative integer indices")
    if n_signals is None:
        n_signals = int(reports.max()) + 1
    elif np.any(reports >= n_signals):
        raise ValueError("a report exceeds n_signals")
    return reports, int(n_signals)


def _marginal(reports, n_signals) -> np.ndarray:
    marg = np.bincount(reports.ravel(), minlength=n_signals).astype(float)
    total = marg.sum()
    return marg / total if total > 0 else marg


def enforce_marginal(agent_reports, target_ones: int, rng=None) -> np.ndarray:
    r"""Flip an agent's excess reports so their marginal matches a target.

    The enforcement step of Enforced Agreement, for binary signals. If the agent
    reported more 1s than the target ``target_ones``, a uniformly random subset of
    the excess is flipped to 0, and symmetrically for 0s. The agent's *count* of
    each report is therefore fixed by the mechanism, and only *which* tasks carry
    which report is left to them.
    """
    r = np.asarray(agent_reports, dtype=int).copy()
    if np.any((r != 0) & (r != 1)):
        raise ValueError("enforcement is defined here for binary signals")
    if not (0 <= target_ones <= r.size):
        raise ValueError("target_ones must lie in [0, n_tasks]")
    rng = np.random.default_rng() if rng is None else rng
    excess = int(r.sum()) - int(target_ones)
    if excess > 0:
        idx = np.flatnonzero(r == 1)
        r[rng.choice(idx, size=excess, replace=False)] = 0
    elif excess < 0:
        idx = np.flatnonzero(r == 0)
        r[rng.choice(idx, size=-excess, replace=False)] = 1
    return r


def enforced_agreement(reports, target_ones: int = None, rng=None) -> np.ndarray:
    r"""Enforced Agreement (Zhang, Xu, Pennock & Schoenebeck 2025).

    A *different* mechanism from :func:`correlated_agreement`, and the one that
    actually carries the stochastic-dominance guarantee. It is plain output
    agreement preceded by :func:`enforce_marginal`: each agent's report counts are
    forced to a target ``Phi``, so over-reporting the majority signal buys
    nothing, because the mechanism simply flips the surplus back.

    Their Theorem 6.1: for binary, *self-predicting* signals, Enforced Agreement
    is SD-truthful for any target ``Phi``. First-order stochastic dominance means
    the truthful score distribution is preferred under every increasing utility,
    not merely in expectation — a genuinely stronger claim than the
    informed-truthfulness of CA, and one CA does not have (their Example 4.1
    exhibits a constant strategy that CA fails to stochastically dominate).

    It does not extend freely: for more than two signals SD-truthfulness needs the
    minimal enforcement rule and *uniformly* self-predicting signals.

    ``target_ones`` defaults to the rounded average number of 1s across agents.
    Returns each agent's mean output-agreement score against the other agents.
    """
    reports, _ = _validate_reports(reports, 2)
    n_agents, n_tasks = reports.shape
    rng = np.random.default_rng() if rng is None else rng
    if target_ones is None:
        target_ones = int(round(float(reports.sum(axis=1).mean())))
    enforced = np.array([enforce_marginal(reports[i], target_ones, rng)
                         for i in range(n_agents)])
    scores = np.zeros(n_agents)
    for i in range(n_agents):
        peers = [j for j in range(n_agents) if j != i]
        scores[i] = np.mean([np.mean(enforced[i] == enforced[j]) for j in peers])
    return scores
