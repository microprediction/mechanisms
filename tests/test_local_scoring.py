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
