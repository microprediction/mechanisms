"""Local (derivative-based) proper scoring rules.

A scoring rule is **m-local** if it depends on the forecast density only through
its value and its first ``m`` derivatives at the realised outcome. The decisive
practical consequence: such a rule can be evaluated **without the normalizing
constant** of the density. Because ``log p = log p_tilde - log Z``, every
derivative of ``log p`` drops the constant ``log Z`` — so an *unnormalized*
density (an energy-based model, an un-normalised mixture) can be scored directly,
with no partition function in sight.

Genuinely local proper scoring rules on the real line exist iff ``m`` is even
(Parry, Dawid & Lauritzen, 2012). The ``m = 2`` case is the **Hyvärinen score**,

    S(y, p) = Δ log p(y) + ½ ‖∇ log p(y)‖²        (loss form; lower is better),

whose population minimiser is **score matching** (Hyvärinen, 2005) and whose
associated divergence is the **Fisher divergence**
``½ ∫ p_data ‖∇log p − ∇log p_data‖²``. It is strictly proper relative to that
divergence.

Conventions match :mod:`mechanisms.scoring_rules`: loss form, lower is better.

References
----------
- Hyvärinen, A. (2005). "Estimation of Non-Normalized Statistical Models by Score
  Matching." JMLR 6, 695-709.
- Parry, M., Dawid, A. P. & Lauritzen, S. (2012). "Proper local scoring rules."
  Annals of Statistics 40(1), 561-592.

Connections
-----------
The marginal score ``∇log p`` that this module evaluates is also the object in
**Tweedie's formula** ``E[θ | y] = y + σ²·∇log p(y)`` (Efron, 2011): the exact
posterior mean of a latent ``θ`` written in terms of the marginal density alone.
Hansen & Tong (2026, arXiv:2605.15902) use this to show that the *score-driven*
filters of econometrics (Creal, Koopman & Lucas, 2013) — which step a latent
parameter along the conditional likelihood score — are exact Bayesian updates
under local precision discounting. That bridges this scoring substrate to
time-series filtering; it is not itself a market mechanism. See
``docs/connections.html``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hyvarinen_score",
    "hyvarinen_score_fd",
    "gaussian_hyvarinen_score",
    "fisher_divergence_gaussian",
]


def hyvarinen_score(grad_log_p, second_log_p) -> float:
    r"""Hyvärinen score from the analytic derivatives of ``log p`` at the outcome.

    ``S = \sum_i \big[\partial_i^2 \log p(y) + \tfrac12 (\partial_i \log p(y))^2\big]``.

    Parameters
    ----------
    grad_log_p : the score function ``∇ log p(y)`` — scalar (1-D) or length-``d``.
    second_log_p : the per-coordinate second derivatives ``∂_i^2 log p(y)``
        (their sum is the Laplacian) — scalar or length-``d``.
    """
    g = np.atleast_1d(np.asarray(grad_log_p, float))
    lap = float(np.sum(np.asarray(second_log_p, float)))
    return float(lap + 0.5 * np.sum(g * g))


def hyvarinen_score_fd(log_pdf, y: float, h: float = 1e-4) -> float:
    """Hyvärinen score of a (possibly UNNORMALIZED) 1-D log-density, by finite differences.

    ``log_pdf`` is a callable ``x -> log p_tilde(x)``; any additive normalizing
    constant is irrelevant because only derivatives of ``log_pdf`` enter. This is
    the point of a local rule — score an energy-based model without its partition
    function.
    """
    y = float(y)
    f0, fp, fm = log_pdf(y), log_pdf(y + h), log_pdf(y - h)
    d1 = (fp - fm) / (2.0 * h)
    d2 = (fp - 2.0 * f0 + fm) / (h * h)
    return float(d2 + 0.5 * d1 * d1)


def gaussian_hyvarinen_score(y: float, mu: float, sigma: float) -> float:
    r"""Closed-form Hyvärinen score for a 1-D Gaussian ``N(mu, sigma^2)``.

    With ``∂ log p = -(y-mu)/σ²`` and ``∂² log p = -1/σ²``,
    ``S = (y-mu)^2 / (2σ^4) - 1/σ^2``.
    """
    s2 = float(sigma) ** 2
    d1 = -(float(y) - float(mu)) / s2
    d2 = -1.0 / s2
    return float(d2 + 0.5 * d1 * d1)


def fisher_divergence_gaussian(mu_p: float, sigma_p: float,
                               mu_q: float, sigma_q: float) -> float:
    r"""Fisher divergence ``½ E_p\|∇log p - ∇log q\|^2`` between two Gaussians.

    This is the divergence the Hyvärinen score induces: non-negative, and zero iff
    ``p = q``. Equivalently ``E_p[S(\cdot,q)] - E_p[S(\cdot,p)]``.
    """
    sp2, sq2 = float(sigma_p) ** 2, float(sigma_q) ** 2
    # ∇log p - ∇log q = a·y + b  with
    a = 1.0 / sq2 - 1.0 / sp2
    b = mu_p / sp2 - mu_q / sq2  # note ∇log p = -(y-mu)/σ² = -y/σ² + mu/σ²
    # E_p[(a y + b)^2] = a^2 Var_p + (a·E_p[y] + b)^2
    return float(0.5 * (a * a * sp2 + (a * mu_p + b) ** 2))
