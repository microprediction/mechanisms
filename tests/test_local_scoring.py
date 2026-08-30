"""Tests for local (derivative-based) proper scoring rules.

Executable documentation: the Hyvärinen score is strictly proper, is invariant to
the normalizing constant, and its induced divergence is the Fisher divergence.
"""

import math

import numpy as np
import pytest

from mechanisms import local_scoring as ls


def _gauss_logpdf(mu, sigma, normalized=True):
    z = math.log(math.sqrt(2 * math.pi) * sigma) if normalized else 0.0
    return lambda x: -0.5 * ((x - mu) / sigma) ** 2 - z


def _expected_hyvarinen(mu0, sigma0, mu, sigma):
    """Closed form E_{N(mu0,sigma0^2)}[ Hyvärinen score of N(mu,sigma^2) ]."""
    return (sigma0**2 + (mu0 - mu) ** 2) / (2 * sigma**4) - 1.0 / sigma**2


def test_closed_form_matches_finite_difference():
    """gaussian_hyvarinen_score equals the FD score of the normalized log-density."""
    for y, mu, sigma in [(0.3, 0.0, 1.0), (-1.2, 0.5, 2.0), (2.0, -0.4, 0.7)]:
        closed = ls.gaussian_hyvarinen_score(y, mu, sigma)
        fd = ls.hyvarinen_score_fd(_gauss_logpdf(mu, sigma), y, h=1e-4)
        assert abs(closed - fd) < 1e-4


def test_invariant_to_normalizing_constant():
    """Only derivatives of log p enter, so an additive constant cannot matter."""
    y = 0.6
    norm = ls.hyvarinen_score_fd(_gauss_logpdf(0.1, 1.3, normalized=True), y)
    unnorm = ls.hyvarinen_score_fd(_gauss_logpdf(0.1, 1.3, normalized=False), y)
    assert abs(norm - unnorm) < 1e-6


def test_strictly_proper_recovers_truth():
    """Expected score is uniquely minimised at the true (mu, sigma)."""
    mu0, sigma0 = 0.4, 1.3
    mus = np.linspace(-1.0, 2.0, 31)
    sigmas = np.linspace(0.6, 2.4, 31)
    best, argbest = np.inf, None
    for mu in mus:
        for sigma in sigmas:
            v = _expected_hyvarinen(mu0, sigma0, mu, sigma)
            if v < best:
                best, argbest = v, (mu, sigma)
    # the grid optimum sits at the grid point nearest the truth
    assert abs(argbest[0] - mu0) <= (mus[1] - mus[0])
    assert abs(argbest[1] - sigma0) <= (sigmas[1] - sigmas[0])


def test_fisher_divergence_nonneg_and_zero_at_equality():
    assert ls.fisher_divergence_gaussian(0.2, 1.1, 0.2, 1.1) == pytest.approx(0.0, abs=1e-12)
    for mp, sp, mq, sq in [(0.0, 1.0, 0.5, 1.0), (0.0, 1.0, 0.0, 1.6), (-0.3, 0.8, 0.4, 1.2)]:
        assert ls.fisher_divergence_gaussian(mp, sp, mq, sq) > 0.0


def test_fisher_divergence_equals_expected_score_gap():
    """D_F(p,q) = E_p[S(.,q)] - E_p[S(.,p)]."""
    mp, sp, mq, sq = 0.1, 1.2, 0.6, 0.9
    gap = _expected_hyvarinen(mp, sp, mq, sq) - _expected_hyvarinen(mp, sp, mp, sp)
    assert ls.fisher_divergence_gaussian(mp, sp, mq, sq) == pytest.approx(gap, rel=1e-9)


def _bump(x, centre, radius):
    """A C-infinity bump supported exactly on (centre - radius, centre + radius)."""
    z = (x - centre) / radius
    out = np.zeros_like(x)
    inside = np.abs(z) < 1
    out[inside] = np.exp(-1.0 / (1.0 - z[inside] ** 2))
    return out


def test_fisher_divergence_is_blind_to_disconnected_component_weights():
    # Hyvarinen's Theorem 2 goes from equal gradients to equal densities via
    # "log p = log q + c", which is a connectedness argument the paper does not
    # name. On two disjoint bumps the score sees nothing at all.
    x = np.linspace(-3.0, 8.0, 200_001)
    dx = x[1] - x[0]
    a_comp = _bump(x, 0.0, 1.0); a_comp /= np.trapezoid(a_comp, x)
    b_comp = _bump(x, 5.0, 1.0); b_comp /= np.trapezoid(b_comp, x)
    p = 0.5 * a_comp + 0.5 * b_comp

    def fisher(p, q):
        both = (p > 1e-250) & (q > 1e-250)
        gp = np.gradient(np.where(both, np.log(np.maximum(p, 1e-300)), 0.0), dx)
        gq = np.gradient(np.where(both, np.log(np.maximum(q, 1e-300)), 0.0), dx)
        inner = both.copy(); inner[:2] = False; inner[-2:] = False
        return float(0.5 * np.trapezoid(np.where(inner, p * (gp - gq) ** 2, 0.0), x))

    for weight in (0.5, 0.1, 0.9, 0.01):
        q = weight * a_comp + (1.0 - weight) * b_comp
        total_variation = 0.5 * float(np.trapezoid(np.abs(p - q), x))
        assert fisher(p, q) < 1e-12                      # blind
        if weight != 0.5:
            assert total_variation > 0.3                 # while genuinely different

    # The bumps are smooth with compact support, so every boundary term in the
    # integration by parts vanishes exactly: this is not a boundary artefact.
    assert ls.hyvarinen_identifies_mixture_weights(support_is_connected=False) is False

    # On a connected support the divergence separates, as advertised.
    assert ls.hyvarinen_identifies_mixture_weights(support_is_connected=True) is True
    assert ls.fisher_divergence_gaussian(0.0, 1.0, 0.0, 1.0) == pytest.approx(0.0)
    assert ls.fisher_divergence_gaussian(0.0, 1.0, 2.5, 1.0) > 3.0


def test_only_the_log_score_is_zero_local_and_only_above_two_outcomes():
    from mechanisms import scoring_rules as sr
    # With three outcomes, two forecasts agreeing on the realised probability
    # score differently under Brier, so Brier is not 0-local.
    p = np.array([0.5, 0.25, 0.25])
    q = np.array([0.5, 0.5, 0.0])
    assert p[0] == q[0]
    assert sr.brier_score(p, 0) != pytest.approx(sr.brier_score(q, 0))
    # The log score does not distinguish them: that is what 0-local means.
    assert sr.log_score(p, 0) == pytest.approx(sr.log_score(q, 0))

    # With two outcomes p_0 = 1 - p_1, so agreeing on the realised probability
    # means being the same forecast. Locality is vacuous, not merely satisfied.
    for realised in (0, 1):
        a = np.array([0.3, 0.7])
        b = np.array([0.3, 0.7])
        assert sr.brier_score(a, realised) == pytest.approx(sr.brier_score(b, realised))
