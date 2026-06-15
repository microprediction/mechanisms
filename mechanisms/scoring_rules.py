"""Strictly proper scoring rules and distributional scores.

A scoring rule ``S(p, y)`` assigns a reward (here, by convention, a *loss* — lower
is better) to a probabilistic forecast ``p`` once the outcome ``y`` is observed.
A rule is **proper** if reporting one's true belief maximises expected reward
(minimises expected loss), and **strictly proper** if the true belief is the
*unique* optimiser. Properness is what makes a scoring rule incentive-compatible:
an expert with private information cannot do better than to report it honestly.

References
----------
- Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper Scoring Rules,
  Prediction, and Estimation." J. Amer. Statist. Assoc. 102(477), 359-378.
- Brier, G. W. (1950). "Verification of forecasts expressed in terms of
  probability." Monthly Weather Review 78(1).
- Good, I. J. (1952). "Rational Decisions." JRSS-B 14(1).  (logarithmic score)
- Savage, L. J. (1971). "Elicitation of personal probabilities and
  expectations." JASA 66(336).

All forecasts ``p`` are categorical distributions over a finite outcome set, given
as a 1-D array summing to 1. Outcomes ``y`` are integer class indices.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "log_score",
    "brier_score",
    "spherical_score",
    "expected_score",
    "pinball_loss",
    "interval_score",
    "crps_ensemble",
    "energy_score",
]

_EPS = 1e-15


def _as_prob(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p must be a 1-D probability vector")
    if np.any(p < -1e-9):
        raise ValueError("probabilities must be non-negative")
    s = p.sum()
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"probabilities must sum to 1 (got {s:.6f})")
    return np.clip(p, 0.0, 1.0)


def log_score(p, y: int) -> float:
    r"""Logarithmic score (loss form): ``-log p[y]``.

    The unique *local* strictly proper scoring rule (it depends on the forecast
    only through the probability assigned to the realised outcome). This is the
    score underlying Hanson's LMSR and the cross-entropy / negative-log-likelihood
    loss in machine learning.
    """
    p = _as_prob(p)
    return float(-np.log(max(p[int(y)], _EPS)))


def brier_score(p, y: int) -> float:
    r"""Brier (quadratic) score, loss form: ``sum_i (p_i - e_i)^2``.

    where ``e`` is the one-hot indicator of the outcome ``y``. Strictly proper,
    bounded in ``[0, 2]``, and the most widely used score for calibration studies.
    """
    p = _as_prob(p)
    e = np.zeros_like(p)
    e[int(y)] = 1.0
    return float(np.sum((p - e) ** 2))


def spherical_score(p, y: int) -> float:
    r"""Spherical score, loss form: ``1 - p[y] / ||p||_2``.

    Strictly proper. The reward form is ``p[y] / ||p||_2``; we return ``1 - reward``
    so that, like the others here, lower is better and the minimum is 0.
    """
    p = _as_prob(p)
    norm = np.linalg.norm(p)
    return float(1.0 - p[int(y)] / max(norm, _EPS))


def expected_score(score_fn, belief, p) -> float:
    """Expected score, under distribution ``belief``, of reporting forecast ``p``.

    Useful for *demonstrating* properness: for a strictly proper ``score_fn`` the
    map ``p -> expected_score(score_fn, belief, p)`` is minimised uniquely at
    ``p == belief``.
    """
    belief = _as_prob(belief)
    return float(sum(belief[k] * score_fn(p, k) for k in range(len(belief))))


def pinball_loss(z: float, y: float, tau: float) -> float:
    r"""Pinball (tick / quantile) loss for eliciting the ``tau``-quantile.

    .. math::
        L_\\tau(z, y) = (y - z)\\,(\\tau - \\mathbf{1}\\{y < z\\}).

    The unique minimiser of expected pinball loss is the true ``tau``-quantile,
    so it is the *consistent scoring function* for quantile elicitation. ``tau``
    in (0, 1); ``tau = 0.5`` recovers (half) the absolute error and elicits the
    median.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0, 1)")
    z = float(z)
    y = float(y)
    return float((y - z) * (tau - (1.0 if y < z else 0.0)))


def interval_score(lower: float, upper: float, y: float, alpha: float) -> float:
    r"""Negatively-oriented interval score for a central ``(1-alpha)`` interval.

    .. math::
        S^{\\text{int}}_\\alpha(l, u; y) = (u - l)
            + \\tfrac{2}{\\alpha}(l - y)\\mathbf{1}\\{y < l\\}
            + \\tfrac{2}{\\alpha}(y - u)\\mathbf{1}\\{y > u\\}.

    Rewards narrow intervals but penalises (scaled by ``2/alpha``) realisations
    that fall outside ``[l, u]``. A proper scoring rule for the pair of
    predictive quantiles at levels ``alpha/2`` and ``1 - alpha/2``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    l = float(lower)
    u = float(upper)
    y = float(y)
    if u < l:
        raise ValueError("upper must be >= lower")
    s = u - l
    if y < l:
        s += (2.0 / alpha) * (l - y)
    elif y > u:
        s += (2.0 / alpha) * (y - u)
    return float(s)


def crps_ensemble(samples, y: float) -> float:
    r"""Continuous Ranked Probability Score from an ensemble (sample) forecast.

    For a univariate forecast represented by samples ``x_1..x_m`` and observation
    ``y``, the energy-form estimator of the CRPS is

    .. math::
        \mathrm{CRPS} = \frac{1}{m}\sum_i |x_i - y|
                        - \frac{1}{2m^2}\sum_{i,j} |x_i - x_j|.

    The CRPS is a strictly proper scoring rule for the full predictive
    distribution and reduces to the absolute error for a point forecast. This is
    the one-dimensional special case of the :func:`energy_score`.
    """
    x = np.asarray(samples, dtype=float).ravel()
    m = x.size
    if m == 0:
        raise ValueError("need at least one sample")
    term1 = np.mean(np.abs(x - float(y)))
    # E|X - X'| via all pairwise distances (O(m^2); fine for reference use).
    term2 = np.abs(x[:, None] - x[None, :]).mean()
    return float(term1 - 0.5 * term2)


def energy_score(samples, y, beta: float = 1.0) -> float:
    r"""Energy score for a multivariate sample (Monte-Carlo) forecast.

    Given an ensemble of ``m`` points ``x_i`` in ``R^d`` and an observation ``y``,

    .. math::
        \mathrm{ES} = \frac{1}{m}\sum_i \|x_i - y\|^\beta
                      - \frac{1}{2m^2}\sum_{i,j} \|x_i - x_j\|^\beta,
        \qquad \beta \in (0, 2).

    Strictly proper for ``beta in (0, 2)`` (Gneiting & Raftery 2007, §4.3). With
    ``beta = 1`` this is the multivariate generalisation of the CRPS and is the
    scoring rule behind sample-based distributional-forecasting contests such as
    monteprediction.com, where participants submit a cloud of Monte-Carlo points.

    Parameters
    ----------
    samples : (m, d) array — the ensemble.
    y       : (d,) array   — the realised outcome.
    beta    : exponent in (0, 2); 1.0 by default.
    """
    if not (0.0 < beta < 2.0):
        raise ValueError("energy score is strictly proper only for beta in (0, 2)")
    x = np.atleast_2d(np.asarray(samples, dtype=float))
    if x.shape[0] == 1 and x.shape[1] != 1 and np.ndim(samples) == 1:
        # a single 1-D sample vector was passed as the ensemble
        x = x.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[1] != y.size:
        x = x.reshape(-1, y.size)
    m = x.shape[0]
    term1 = np.mean(np.linalg.norm(x - y[None, :], axis=1) ** beta)
    diff = x[:, None, :] - x[None, :, :]
    term2 = (np.linalg.norm(diff, axis=2) ** beta).mean()
    return float(term1 - 0.5 * term2)
