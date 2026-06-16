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
