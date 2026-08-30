"""Aggregating probabilistic forecasts — opinion pools.

Once many forecasts have been elicited (by markets, scoring rules, or peer
mechanisms), they must be combined into a single consensus distribution. The two
canonical pooling operators:

- :func:`linear_opinion_pool` — a weighted arithmetic mean. It preserves the
  support and is characterised by the marginalization property (McConway 1981),
  which needs at least three outcomes.
- :func:`logarithmic_opinion_pool` — a normalised weighted geometric mean. It is
  the *externally Bayesian* one (Genest 1984), and it zeroes out any outcome that
  some forecaster rules out.

Three pieces of folklore about these are false, and each is corrected on the
function it belongs to.

*The log pool does not multiply independent evidence.* Combining conditionally
independent evidence under a common prior ``pi`` gives
``p ∝ pi^{1-m} prod_i p_i`` (Genest & Schervish 1985, Theorem 4.1; Dietrich 2010,
Corollary 1). The identity ``pi prod_i (p_i/pi)^{w_i} = pi^{1-sum w_i} prod_i
p_i^{w_i}`` shows the prior correction vanishes *exactly* when the weights sum to
one, which is the case implemented here. So the averaging log pool and the
evidence-multiplying product are different operators, and the latter, with
``sum w_i = m``, is not externally Bayesian (Genest & Zidek 1986, eq. 4.4).

*The log pool is not sharper than its members.* See
:func:`logarithmic_opinion_pool` for a counterexample where it is seven times
more diffuse than either member and more diffuse than the linear pool too.

*"Underconfident" is not the entropy fact.* See :func:`linear_opinion_pool`.

Also provided: :func:`depth_trimmed_mean`, a robust pool that drops the most
extreme forecasts (by distance to the coordinate-wise median) before averaging —
a simple stand-in for the statistical-depth trimming used in robust forecast
aggregation.

References
----------
- Genest, C. & Zidek, J. (1986). "Combining Probability Distributions: A Critique
  and an Annotated Bibliography." Statistical Science 1(1).
- Genest, C. & Schervish, M. (1985). "Modeling Expert Judgments for Bayesian
  Updating." Annals of Statistics 13(3).
- Genest, C., McConway, K. & Schervish, M. (1986). "Characterization of
  Externally Bayesian Pooling Operators." Annals of Statistics 14(2).
- Ranjan, R. & Gneiting, T. (2010). "Combining Probability Forecasts." JRSS-B
  72(1).
- Gneiting, T. & Ranjan, R. (2013). "Combining predictive distributions."
  Electronic Journal of Statistics 7.
"""

from __future__ import annotations

import numpy as np

__all__ = ["linear_opinion_pool", "logarithmic_opinion_pool", "depth_trimmed_mean"]


def _as_forecast_matrix(forecasts) -> np.ndarray:
    P = np.atleast_2d(np.asarray(forecasts, dtype=float))
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] == 0:
        raise ValueError("forecasts must be a non-empty 2-D array")
    if not np.all(np.isfinite(P)) or np.any(P < 0):
        raise ValueError("forecast components must be finite and non-negative")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each forecast must be a probability vector summing to 1")
    return P


def _stack_and_weights(forecasts, weights):
    P = _as_forecast_matrix(forecasts)
    m = P.shape[0]
    if weights is None:
        return P, np.full(m, 1.0 / m)
    w = np.asarray(weights, dtype=float)
    if w.shape != (m,):
        raise ValueError("weights must align with forecasts")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("pooling weights must be finite and non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("pooling weights must have a positive sum")
    return P, w / total


def linear_opinion_pool(forecasts, weights=None) -> np.ndarray:
    r"""Weighted arithmetic mean ``sum_i w_i p_i`` of categorical forecasts.

    Two facts about its dispersion are often run together, and they are separate
    theorems about different objects.

    *Entropy.* Concavity of entropy gives
    ``sum_i w_i H(p_i) <= H(sum_i w_i p_i) <= sum_i w_i H(p_i) + H(w)``, the gap
    being the generalized Jensen-Shannon divergence (Cover & Thomas, Theorem
    2.7.3; Lin 1991). So the pool is at least as diffuse as the *average* member.
    It is not always at least as diffuse as the sharpest: a uniform forecast on
    four atoms mixed with a point mass at weights ``(0.01, 0.99)`` has entropy
    0.076 bits, below both. And entropy is a property of the forecast alone.

    *Underconfidence* is a property of the joint law of forecast and outcome, so
    the entropy inequality cannot establish it. The actual results: for binary
    outcomes, if the components are calibrated and at least two differ with
    positive probability, the pool is *not* calibrated and not sharp, and its
    recalibration beats it under every strictly proper score (Ranjan & Gneiting
    2010, Theorem 2.1). In the general case, neutrally dispersed regular
    components pool to something overdispersed, which is underconfident in the
    PIT sense ``var(PIT) < 1/12`` (Gneiting & Ranjan 2013, Definition 2.6(c) and
    Theorem 3.1(c)).

    "Intrinsically underconfident" overstates it. Linear pooling pushes
    dispersion *up*, so it overshoots when its inputs are already calibrated and
    helps when they are overconfident, which is the usual empirical case and part
    of why linear pooling works as well as it does. Extremizing is the standard
    repair, and it is not guaranteed to help; see Lichtendahl et al. on when
    anti-extremizing is optimal.
    """
    P, w = _stack_and_weights(forecasts, weights)
    return w @ P


def logarithmic_opinion_pool(forecasts, weights=None) -> np.ndarray:
    r"""Normalised weighted geometric mean ``prop_to prod_i p_i^{w_i}``.

    Weights are normalised to sum to one, which makes this the externally
    Bayesian operator (Genest 1984). Note it is not the *unique* one: without a
    locality axiom the externally Bayesian operators are pinned down only within
    likelihood-ratio equivalence classes (Genest, McConway & Schervish 1986,
    Theorem 2.5), and the logarithmic form is recovered only on a sample space
    partitionable into at least four non-negligible sets. Three atoms was open as
    of 1986; for two there is an explicit non-logarithmic counterexample.

    It is *not* generally sharper than its members, despite the folklore. With
    ``p1 = (0.98, 0.01, 0.01)`` and ``p2 = (0.01, 0.98, 0.01)`` at equal weights
    the pool is ``(0.476, 0.476, 0.048)``, whose entropy 0.853 nats is seven
    times either member's 0.112, and larger than the linear pool's 0.742 as well.
    The Gaussian case says why: pooling ``N(mu_i, sigma_i^2)`` gives precision
    ``sum_i w_i / sigma_i^2``, so the pooled variance is the weighted *harmonic
    mean* of the members' and therefore always lies between the smallest and
    largest. The log pool is never sharper than its sharpest member. Sharpening
    is what the *unnormalised* product does, and that needs ``sum w_i = m``,
    which is exactly the choice that breaks external Bayesianity.

    Zeros do annihilate in exact arithmetic: an outcome any forecaster with
    positive weight rules out gets probability zero. In floating point the
    logarithm is floored at ``1e-15``, so a ruled-out outcome comes back as a
    denormal rather than an exact zero, and a forecast of ``0`` is treated as one
    of ``1e-15``. That is a deliberate guard against ``-inf``, not exact
    annihilation, and code that branches on ``== 0`` should not rely on it.
    """
    P, w = _stack_and_weights(forecasts, weights)
    log_pool = w @ np.log(np.clip(P, 1e-15, 1.0))
    pool = np.exp(log_pool - log_pool.max())
    return pool / pool.sum()


def depth_trimmed_mean(forecasts, trim: float = 0.2) -> np.ndarray:
    r"""Robust linear pool: drop the ``trim`` fraction of most-outlying forecasts.

    "Outlyingness" is the L1 distance from the coordinate-wise median forecast (a
    simple proxy for low statistical depth). The deepest ``1 - trim`` fraction is
    averaged. With ``trim = 0`` this is the plain linear pool.
    """
    if not (0.0 <= trim < 1.0):
        raise ValueError("trim must be in [0, 1)")
    P = _as_forecast_matrix(forecasts)
    m = P.shape[0]
    med = np.median(P, axis=0)
    dist = np.abs(P - med).sum(axis=1)
    keep = max(1, int(round(m * (1.0 - trim))))
    idx = np.argsort(dist)[:keep]
    pooled = P[idx].mean(axis=0)
    return pooled / pooled.sum()
