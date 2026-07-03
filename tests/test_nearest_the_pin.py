"""Tests for the nearest-the-pin parimutuel.

Verifies the self-funding (zero-sum) pot split, the log-wealth honesty incentive,
and the projection identity that the sliced energy score recovers the
multivariate energy score.
"""

import numpy as np
import pytest

from mechanisms import nearest_the_pin as ntp
from mechanisms import scoring_rules as sr


# ----------------------- pot split ----------------------- #
def test_pot_split_is_zero_sum():
    # Self-funding: total wealth is conserved (a pure transfer).
    densities = [0.3, 0.1, 0.05, 0.2]
    wealth = [100.0, 100.0, 100.0, 100.0]
    dW = ntp.pot_split(densities, wealth, b=0.1)
    assert dW.sum() == pytest.approx(0.0, abs=1e-9)


def test_pot_split_rewards_higher_density():
    # Equal wealth -> the player with the highest density at z gains the most.
    dW = ntp.pot_split([0.5, 0.1, 0.1], [100.0, 100.0, 100.0], b=0.1)
    assert dW[0] > 0 and dW[1] < 0 and dW[2] < 0
    assert np.argmax(dW) == 0


def test_pot_split_no_mass_returns_stakes():
    dW = ntp.pot_split([0.0, 0.0], [100.0, 100.0], b=0.1)
    np.testing.assert_allclose(dW, [0.0, 0.0])


def test_log_wealth_honesty_incentive():
    # A log-wealth maximiser should prefer reporting the true density.
    # Outcomes drawn from N(0,1); compare expected log-wealth growth of a player
    # reporting the truth vs a biased report, against a fixed honest crowd.
    rng = np.random.default_rng(0)
    grid_true = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    grid_bias = lambda z: np.exp(-0.5 * (z - 1.0) ** 2) / np.sqrt(2 * np.pi)
    draws = rng.normal(size=4000)

    def growth(report):
        g = 0.0
        for z in draws:
            # crowd: two honest players + our player `report`
            q = [grid_true(z), grid_true(z), report(z)]
            dW = ntp.pot_split(q, [100.0, 100.0, 100.0], b=0.1)
            g += np.log1p(dW[2] / 100.0)
        return g / len(draws)

    assert growth(grid_true) > growth(grid_bias)


# ----------------------- projection identity ----------------------- #
def test_projection_constant_known_values():
    assert ntp.projection_constant(1) == pytest.approx(1.0)
    assert ntp.projection_constant(2) == pytest.approx(2.0 / np.pi)


def test_sliced_energy_matches_energy_score():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(300, 4))
    y = np.array([0.2, -0.1, 0.5, 0.0])
    exact = sr.energy_score(samples, y, beta=1.0)
    sliced = ntp.energy_score_via_projection(samples, y, n_proj=4000, rng=rng)
    # Monte-Carlo over directions -> a few percent agreement.
    assert sliced == pytest.approx(exact, rel=0.05)


def test_sliced_energy_1d_is_crps():
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(200, 1))
    y = [0.3]
    # In 1-D the projection is +/-1, c_1 = 1, so it equals the CRPS exactly.
    sliced = ntp.energy_score_via_projection(samples, y, n_proj=50, rng=rng)
    crps = sr.crps_ensemble(samples.ravel(), 0.3)
    assert sliced == pytest.approx(crps, rel=1e-9)


# --------------- mollified scoring (deconvolution incentive & repair) --------------- #
# Analytic setting: truth N(0,1), Gaussian KDE bandwidth h, report a cloud from
# N(0,v). The population-smoothed report is N(0, v+h^2), so
#   raw KDE log score:  E log rho_h(z), z~N(0,1)      -> f_raw(v)
#   mollified score:    int p*_h log rho_h            -> f_moll(v)
# with closed forms below. The theorem: f_raw peaks at v = 1 - h^2 (shave the
# bandwidth: improper), f_moll peaks at v = 1 (truthful: strictly proper).

def _f_raw(v, h2):
    s = v + h2
    return -0.5 * np.log(2 * np.pi * s) - 1.0 / (2 * s)


def _f_moll(v, h2):
    s = v + h2
    return -0.5 * np.log(2 * np.pi * s) - (1.0 + h2) / (2 * s)


def test_raw_kde_log_score_prefers_deconvolved_report():
    # Improperness, with the closed-form optimum v* = tau^2 - h^2.
    h2 = 0.25
    grid = np.linspace(0.4, 1.6, 1201)
    v_star = grid[np.argmax(_f_raw(grid, h2))]
    assert v_star == pytest.approx(1.0 - h2, abs=2e-3)   # shave exactly h^2
    assert _f_raw(1.0 - h2, h2) > _f_raw(1.0, h2)        # beats truth-telling


def test_mollified_score_prefers_truthful_report():
    # The jittered-pin repair restores the truthful optimum v* = tau^2.
    h2 = 0.25
    grid = np.linspace(0.4, 1.6, 1201)
    v_star = grid[np.argmax(_f_moll(grid, h2))]
    assert v_star == pytest.approx(1.0, abs=2e-3)
    assert _f_moll(1.0, h2) > _f_moll(1.0 - h2, h2)      # truth-telling beats shaving


def test_mollified_log_score_matches_analytic_value():
    # Function-level check: on a large cloud from N(0,1) the mollified score at
    # z=0 approaches (phi_h * log N(0,1+h^2))(0) = -log(2*pi*(1+h^2))/2 - h^2/(2*(1+h^2)).
    rng = np.random.default_rng(7)
    h = 0.5
    cloud = rng.standard_normal((20000, 1))
    got = ntp.mollified_log_score(cloud, [0.0], bandwidth=h)
    want = -0.5 * np.log(2 * np.pi * (1 + h * h)) - h * h / (2 * (1 + h * h))
    assert got == pytest.approx(want, abs=0.02)


def test_mollified_log_score_montecarlo_agrees_with_quadrature():
    # The d>1 (seeded MC) path is consistent with the 1-D quadrature path via an
    # exact separability identity: embed the cloud as (x_i, 0). The product
    # Gaussian kernel factorizes, so KDE_2d((a, b)) = KDE_1d(a) * phi_h(b), and
    # the mollified score splits as
    #   S_2d = S_1d + E[log phi_h(h*eps)] = S_1d - log(2*pi*h^2)/2 - 1/2.
    rng = np.random.default_rng(11)
    h = 0.4
    cloud = rng.standard_normal((4000, 1))
    quad = ntp.mollified_log_score(cloud, [0.3], bandwidth=h)
    cloud2 = np.hstack([cloud, np.zeros_like(cloud)])
    mc2 = ntp.mollified_log_score(cloud2, [0.3, 0.0], bandwidth=h,
                                  n_nodes=20000, rng=3)
    offset = -0.5 * np.log(2 * np.pi * h * h) - 0.5
    assert mc2 - quad == pytest.approx(offset, abs=0.05)


def test_jitter_calibration_peak():
    # v* = tau^2 + j^2 - h^2: j=0 recovers the raw shave, j=h and only j=h is
    # truthful, j>h pays padding. "Jitter exactly as much as you smooth."
    grid = np.linspace(0.05, 3.0, 5901)
    for h, j, tau in [(0.5, 0.0, 1.0), (0.5, 0.5, 1.0), (0.5, 0.8, 1.0), (0.4, 0.4, 1.2)]:
        scores = [ntp.gaussian_expected_mollified_log_score(v, h, tau, jitter=j)
                  for v in grid]
        v_star = grid[int(np.argmax(scores))]
        assert v_star == pytest.approx(tau * tau + j * j - h * h, abs=2e-3)
