"""The nearest-the-pin parimutuel — a continuous, density-scored pool.

A parimutuel pool over a *continuum* of outcomes. Instead of betting tickets on a
finite set of outcomes, each participant submits a predictive density (in
practice, a cloud of Monte-Carlo samples). When the outcome ``z`` is revealed, the
pot is split in proportion to the **density each participant placed at ``z``** —
the closer your probability mass landed to the realised point, the larger your
share. This is the reward rule behind monteprediction.com (a "splitting of the pot
in proportion to the density you ascribe to the truth, ... [which] also depends on
the density that others ascribe to it").

Two ways to turn a sample cloud into a score at ``z`` are provided:

1. :func:`kde_density` — a kernel density estimate, used directly in the
   density pot-split :func:`pot_split`.
2. :func:`energy_score_via_projection` — the *projection* (sliced) version:
   project the cloud and ``z`` onto random directions and average the 1-D CRPS.
   By the energy-distance projection identity this recovers the multivariate
   energy score up to a dimension constant, and is the high-dimensional-robust
   way to score the 11-dimensional monteprediction clouds.

See ``papers/nearest-the-pin-parimutuel.md`` for the derivation and the links to
the random-projections literature and the Schur pseudo-likelihood.
"""

from __future__ import annotations

import math

import numpy as np

from .scoring_rules import crps_ensemble

__all__ = [
    "kde_density",
    "pot_split",
    "projection_constant",
    "energy_score_via_projection",
    "random_directions",
]


def kde_density(samples, z, bandwidth: float = None) -> float:
    r"""Gaussian-KDE density at ``z`` implied by a cloud of ``samples``.

    ``samples`` is ``(m, d)``; ``z`` is ``(d,)``. With ``m`` points and an
    isotropic Gaussian kernel of bandwidth ``h`` (Scott's rule by default),

    .. math:: \hat q(z) = \frac1m \sum_i \frac{1}{(2\pi h^2)^{d/2}}
                          \exp\!\Big(-\frac{\|z - x_i\|^2}{2h^2}\Big).
    """
    x = np.atleast_2d(np.asarray(samples, float))
    z = np.asarray(z, float).ravel()
    m, d = x.shape
    if bandwidth is None:
        # Scott's rule, with a floor so a degenerate cloud doesn't divide by 0.
        bandwidth = max(m ** (-1.0 / (d + 4)) * np.std(x, axis=0).mean(), 1e-6)
    h2 = bandwidth ** 2
    norm = 1.0 / ((2 * math.pi * h2) ** (d / 2.0))
    sq = np.sum((x - z[None, :]) ** 2, axis=1)
    return float(norm * np.mean(np.exp(-sq / (2 * h2))))


def pot_split(densities, stakes, b: float = 0.1):
    r"""Density-weighted parimutuel pot split; returns each player's wealth change.

    Each of ``n`` participants risks a stake ``s_i = b * W_i`` (a fraction ``b`` of
    wealth ``W_i``). The collected pot ``S = sum_i s_i`` is redistributed in
    proportion to ``s_i * q_i(z)`` — your stake times the density you placed at the
    realised outcome. The net change for participant ``i`` is

    .. math:: \Delta W_i = S\,\frac{s_i\,q_i(z)}{\sum_j s_j\,q_j(z)} - s_i,

    a zero-sum transfer ("nearest the pin"): players whose density beat the
    stake-weighted crowd average gain, the rest lose. The pool is self-funding —
    no operator subsidy. A log-wealth (Kelly) maximiser's unique optimum is to
    report its *true* predictive density (see the accompanying paper).

    Parameters
    ----------
    densities : ``q_i(z)``, the density each participant placed at the outcome.
    stakes    : either the wealths ``W_i`` (then stake ``= b * W_i``) or explicit
                stakes if ``b`` is set to 1.
    b         : fraction of wealth at risk per round (default 0.1).
    """
    q = np.asarray(densities, float)
    s = b * np.asarray(stakes, float)
    pot = s.sum()
    denom = float(np.sum(s * q))
    if denom <= 0:
        # No one placed mass at z: stakes are returned (no transfer).
        return np.zeros_like(q)
    payout = pot * (s * q) / denom
    return payout - s


def projection_constant(d: int) -> float:
    r"""Constant ``c_d`` in the energy-distance projection identity.

    For a uniformly random unit vector ``u`` on the sphere ``S^{d-1}`` and any
    ``x``, ``E_u|<u, x>| = c_d * ||x||`` with

    .. math:: c_d = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d+1)/2)}.

    Hence ``||x|| = E_u|<u,x>| / c_d``, which slices the energy score into 1-D
    CRPS terms. (``c_1 = 1``; ``c_2 = 2/\pi``.)
    """
    return math.gamma(d / 2.0) / (math.sqrt(math.pi) * math.gamma((d + 1) / 2.0))


def random_directions(d: int, k: int, rng=None) -> np.ndarray:
    """``k`` uniformly random unit vectors in ``R^d`` (rows)."""
    rng = np.random.default_rng() if rng is None else rng
    u = rng.standard_normal((k, d))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def energy_score_via_projection(samples, y, n_proj: int = 200, rng=None) -> float:
    r"""Projection (sliced) estimate of the energy score of a sample forecast.

    Projects the ensemble ``samples`` (``m, d``) and the outcome ``y`` (``d``) onto
    ``n_proj`` random unit directions, scores each 1-D projection with the CRPS,
    averages, and rescales by ``1/c_d``. By the projection identity this is an
    unbiased estimator (over directions) of the multivariate energy score
    ``E||X-y|| - 1/2 E||X-X'||`` — the same quantity as
    :func:`mechanisms.scoring_rules.energy_score`, but built from cheap 1-D CRPS
    evaluations, which is how one robustly scores high-dimensional clouds.
    """
    x = np.atleast_2d(np.asarray(samples, float))
    y = np.asarray(y, float).ravel()
    d = x.shape[1]
    U = random_directions(d, n_proj, rng=rng)         # (k, d)
    proj_x = x @ U.T                                   # (m, k)
    proj_y = U @ y                                     # (k,)
    crps = np.array([crps_ensemble(proj_x[:, j], proj_y[j]) for j in range(n_proj)])
    return float(crps.mean() / projection_constant(d))
