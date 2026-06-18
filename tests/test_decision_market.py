"""Tests for the decision-market meta-mechanism and its decision rules."""

import numpy as np
import pytest

from mechanisms import decision_market as dm


def test_conditional_expected_values():
    P = [[0.8, 0.2], [0.3, 0.7]]   # two actions, outcomes valued [0, 1]
    np.testing.assert_allclose(dm.conditional_expected_values(P, [0.0, 1.0]), [0.2, 0.7])
    with pytest.raises(ValueError):
        dm.conditional_expected_values(P, [1.0, 2.0, 3.0])


def test_argmax_decision_and_ties():
    np.testing.assert_allclose(dm.argmax_decision([0.2, 0.7, 0.1]), [0, 1, 0])
    np.testing.assert_allclose(dm.argmax_decision([1.0, 1.0, 0.0]), [0.5, 0.5, 0.0])


def test_softmax_full_support_and_monotone():
    d = dm.softmax_decision([0.2, 0.7, 0.1], temperature=0.5)
    assert d.sum() == pytest.approx(1.0)
    assert np.all(d > 0.0)                 # full support
    assert np.argmax(d) == 1               # most weight on the best action
    # Lower temperature concentrates toward argmax.
    hot = dm.softmax_decision([0.2, 0.7, 0.1], temperature=2.0)
    cold = dm.softmax_decision([0.2, 0.7, 0.1], temperature=0.1)
    assert cold[1] > hot[1]
    with pytest.raises(ValueError):
        dm.softmax_decision([0.1, 0.2], temperature=0.0)


def test_epsilon_greedy_full_support():
    d = dm.epsilon_greedy_decision([0.2, 0.7, 0.1], epsilon=0.3)
    assert d.sum() == pytest.approx(1.0)
    assert np.all(d > 0.0)
    assert np.argmax(d) == 1
    # eps=0 collapses to argmax (no full support).
    np.testing.assert_allclose(dm.epsilon_greedy_decision([0.2, 0.7, 0.1], epsilon=0.0), [0, 1, 0])


def test_decision_market_conditionals_and_recommend():
    m = dm.DecisionMarket(n_actions=3, n_outcomes=2, b=50.0)
    # Fresh markets are uniform over outcomes.
    np.testing.assert_allclose(m.conditional_matrix(), np.full((3, 2), 0.5))
    # Push action 2's conditional toward the high-value outcome (index 1).
    m.trade(action=2, outcome=1, shares=80.0)
    vals = m.values([0.0, 1.0])
    assert m.recommend([0.0, 1.0]) == 2
    assert vals[2] > 0.5 and vals[0] == pytest.approx(0.5)


def test_full_support_is_the_incentive_fix():
    m = dm.DecisionMarket(n_actions=3, n_outcomes=2, b=50.0)
    m.trade(action=1, outcome=1, shares=120.0)   # make action 1 clearly best
    values = [0.0, 1.0]
    # Deterministic argmax starves the unchosen markets (zero settlement prob).
    argmax_d = m.decision(values, rule="argmax")
    assert np.any(argmax_d == 0.0)
    # Stochastic full-support rules keep every conditional market live.
    assert np.all(m.decision(values, rule="softmax", temperature=0.5) > 0.0)
    assert np.all(m.decision(values, rule="epsilon_greedy", epsilon=0.1) > 0.0)
