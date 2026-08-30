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


def _unscaled_score(reports, truth, rule_probs, score):
    """Expected score when only the taken action is scored, with no 1/R factor."""
    return sum(rule_probs[a] * sum(truth[a, k] * score(reports[a], k)
                                   for k in range(reports.shape[1]))
               for a in range(reports.shape[0]))


def test_inverse_selection_factor_is_what_makes_reporting_truthful():
    from mechanisms.scoring_rules import log_score
    truth = np.array([[0.8, 0.2], [0.3, 0.7]])
    values = np.array([1.0, 0.0])

    def scores(p0, p1):
        reports = np.array([[p0, 1 - p0], [p1, 1 - p1]])
        probs = dm.softmax_decision(reports @ values, temperature=0.15)
        return (_unscaled_score(reports, truth, probs, log_score),
                dm.expected_decision_score(reports, truth, score=log_score))

    grid = np.linspace(0.02, 0.98, 97)
    unscaled = np.array([[scores(a, b)[0] for b in grid] for a in grid])
    scaled = np.array([[scores(a, b)[1] for b in grid] for a in grid])

    # These are losses, so the agent minimises. With the factor the best report
    # is the truth, to within the grid step.
    i, j = np.unravel_index(np.argmin(scaled), scaled.shape)
    assert grid[i] == pytest.approx(truth[0, 0], abs=0.011)
    assert grid[j] == pytest.approx(truth[1, 0], abs=0.011)

    # Without it, truthful reporting is strictly beaten: only the taken action
    # is scored, so the agent shades reports to steer which one that is.
    i, j = np.unravel_index(np.argmin(unscaled), unscaled.shape)
    truthful = scores(truth[0, 0], truth[1, 0])[0]
    assert unscaled[i, j] < truthful - 1e-6
    assert abs(grid[i] - truth[0, 0]) > 0.05 or abs(grid[j] - truth[1, 0]) > 0.05


def test_inverse_selection_score_matches_its_expectation():
    from mechanisms.scoring_rules import log_score
    rng = np.random.default_rng(21)
    reports = np.array([[0.6, 0.4], [0.25, 0.75]])
    truth = np.array([[0.55, 0.45], [0.3, 0.7]])
    probs = dm.softmax_decision(reports @ np.array([1.0, 0.0]), temperature=0.4)

    n = 400_000
    actions = rng.choice(2, size=n, p=probs)
    draws = np.empty(n)
    for idx, a in enumerate(actions):
        k = int(rng.choice(2, p=truth[a]))
        draws[idx] = dm.inverse_selection_score(log_score(reports[a], k), probs[a])
    assert float(draws.mean()) == pytest.approx(
        dm.expected_decision_score(reports, truth, score=log_score), abs=0.02)


def test_settle_scales_the_makers_payoff_and_its_bound():
    market = dm.DecisionMarket(n_actions=2, n_outcomes=2, b=10.0)
    market.trade(action=0, outcome=0, shares=5.0, trader="alice")
    market.trade(action=1, outcome=1, shares=3.0, trader="bob")
    decision = np.array([0.25, 0.75])
    raw = market.markets[0].realized_pnl(0)
    maker, payouts = market.settle(0, 0, decision)
    assert maker == pytest.approx(raw / 0.25)
    # Alice's five shares of the realised outcome pay 5 / R(P, 0).
    assert payouts == {"alice": pytest.approx(5.0 / 0.25)}
    # Bob's position sits in the counterfactual market and is void.
    assert "bob" not in payouts
    with pytest.raises(ValueError):
        market.settle(0, 0, np.array([0.0, 1.0]))


def test_select_draws_from_the_rule_and_refuses_zero_support():
    market = dm.DecisionMarket(n_actions=3, n_outcomes=2, b=50.0)
    market.trade(action=1, outcome=0, shares=40.0, trader="alice")
    rng = np.random.default_rng(3)
    counts = np.zeros(3)
    for _ in range(4000):
        action, decision = market.select([1.0, 0.0], rule="softmax",
                                         temperature=0.3, rng=rng)
        counts[action] += 1
    empirical = counts / counts.sum()
    target = market.decision([1.0, 0.0], rule="softmax", temperature=0.3)
    np.testing.assert_allclose(empirical, target, atol=0.03)
    assert empirical[1] > empirical[0]              # the market that was bought
    with pytest.raises(ValueError, match="never settles"):
        market.select([1.0, 0.0], rule="argmax", rng=rng)


def test_settlement_is_zero_sum_between_maker_and_traders():
    rng = np.random.default_rng(17)
    for _ in range(200):
        market = dm.DecisionMarket(n_actions=3, n_outcomes=3, b=40.0)
        paid = 0.0
        for _ in range(8):
            paid += market.trade(int(rng.integers(0, 3)), int(rng.integers(0, 3)),
                                 float(rng.uniform(1, 20)), trader="alice")
        action, decision = market.select([1.0, 0.5, 0.0], rule="epsilon_greedy",
                                         epsilon=0.4, rng=rng)
        outcome = int(rng.integers(0, 3))
        maker, payouts = market.settle(action, outcome, decision)
        # The maker's scaled P&L is the negative of what settles out to traders,
        # net of the cash the maker already collected on the settled market.
        collected = market.markets[action].cost() - market.markets[action].cost(
            np.zeros(3))
        held = market.positions["alice"][action, outcome]
        assert maker == pytest.approx(
            (collected - held) / decision[action], rel=1e-9)
        assert sum(payouts.values()) == pytest.approx(
            held / decision[action], rel=1e-9)
