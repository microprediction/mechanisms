"""Tests for the Kelly sizing primitive and its ties to the log score."""

import numpy as np
import pytest

from mechanisms import kelly
from mechanisms import scoring_rules as sr


def test_binary_fraction_and_edge():
    # Even odds (b=1), 60% edge: f* = 0.6 - 0.4 = 0.2.
    assert kelly.kelly_fraction(0.6, 1.0) == pytest.approx(0.2)
    # 3:1 odds, p=0.4: f* = 0.4 - 0.6/3 = 0.2.
    assert kelly.kelly_fraction(0.4, 3.0) == pytest.approx(0.2)
    # No edge -> do not bet (clamped to 0).
    assert kelly.kelly_fraction(0.4, 1.0) == 0.0
    with pytest.raises(ValueError):
        kelly.kelly_fraction(0.5, 0.0)


def test_growth_is_maximised_at_fstar():
    p, b = 0.6, 1.0
    fstar = kelly.kelly_fraction(p, b)
    g_star = kelly.kelly_growth_binary(p, b)            # uses f*
    for f in (0.5 * fstar, 0.8 * fstar, 1.5 * fstar, 0.0):
        assert kelly.kelly_growth_binary(p, b, f) <= g_star + 1e-12
    # A no-edge bet should be sized at zero with zero growth.
    assert kelly.kelly_growth_binary(0.5, 1.0) == pytest.approx(0.0)


def test_allocation_is_normalised_beliefs():
    p = [2.0, 1.0, 1.0]
    np.testing.assert_allclose(kelly.kelly_allocation(p), [0.5, 0.25, 0.25])


def test_growth_rate_is_kl_and_nonnegative():
    p = np.array([0.6, 0.3, 0.1])
    pi = np.array([0.4, 0.4, 0.2])
    kl = float(np.sum(p * np.log(p / pi)))
    assert kelly.kelly_growth_rate(p, pi) == pytest.approx(kl)
    # No edge over the market => zero growth.
    assert kelly.kelly_growth_rate(p, p) == pytest.approx(0.0, abs=1e-12)
    # Non-negative in general.
    assert kelly.kelly_growth_rate([0.5, 0.5], [0.9, 0.1]) >= 0.0


def test_growth_rate_equals_log_score_regret():
    # KL(p||pi) = E_{y~p}[ logloss(pi, y) - logloss(p, y) ], the log-score regret.
    p = np.array([0.6, 0.3, 0.1])
    pi = np.array([0.4, 0.4, 0.2])
    regret = sum(
        p[y] * (sr.log_score(pi, y) - sr.log_score(p, y)) for y in range(len(p))
    )
    assert kelly.kelly_growth_rate(p, pi) == pytest.approx(regret)


def test_fractional_kelly():
    assert float(kelly.fractional_kelly(0.2, 0.5)) == pytest.approx(0.1)
    np.testing.assert_allclose(kelly.fractional_kelly([0.2, 0.4], 0.5), [0.1, 0.2])


def test_wealth_weighted_update_selects_the_accurate():
    w = np.array([0.5, 0.5])
    # Member 0 gave the outcome higher probability -> gains wealth share.
    nw = kelly.wealth_weighted_update(w, [0.8, 0.2])
    assert nw.sum() == pytest.approx(1.0)
    assert nw[0] > 0.5 and nw[1] < 0.5
    np.testing.assert_allclose(nw, [0.8, 0.2])
    # Iterating compounds the advantage.
    nw2 = kelly.wealth_weighted_update(nw, [0.8, 0.2])
    assert nw2[0] > nw[0]
    with pytest.raises(ValueError):
        kelly.wealth_weighted_update([0.5, 0.5], [0.0, 0.0])
