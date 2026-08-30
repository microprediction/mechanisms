"""Tests for the advanced mechanisms: quantile/interval scores, pm-AMM,
frequent batch auctions, peer prediction, and opinion-pool aggregation.
"""

import numpy as np
import pytest

from mechanisms import (
    aggregation,
    fba,
    peer_prediction as pp,
    pm_amm,
    scoring_rules as sr,
)


# --------------------------------------------------------------------------- #
# pinball / interval scores
# --------------------------------------------------------------------------- #
def test_pinball_elicits_quantile():
    rng = np.random.default_rng(0)
    data = rng.normal(size=20000)
    tau = 0.75
    true_q = np.quantile(data, tau)
    # Expected pinball loss is minimised at the true quantile.
    grid = np.linspace(true_q - 1.0, true_q + 1.0, 41)
    losses = [np.mean([sr.pinball_loss(z, d, tau) for d in data[:2000]]) for z in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(true_q, abs=0.15)


def test_pinball_median_is_half_abs_error():
    assert sr.pinball_loss(0.0, 4.0, 0.5) == pytest.approx(0.5 * 4.0)


def test_interval_score_penalises_misses():
    inside = sr.interval_score(0, 10, y=5, alpha=0.1)
    outside = sr.interval_score(0, 10, y=20, alpha=0.1)
    assert inside == pytest.approx(10)            # just the width
    assert outside > inside                       # plus a scaled miss penalty


# --------------------------------------------------------------------------- #
# pm-AMM
# --------------------------------------------------------------------------- #
def test_pm_amm_price_is_half_at_balance():
    assert pm_amm.pm_amm_price(x=100, y=100, L=50) == pytest.approx(0.5)


def test_pm_amm_prices_sum_to_one():
    x, y, L = 80.0, 120.0, 40.0
    p_yes = pm_amm.pm_amm_price(x, y, L)
    p_no = pm_amm.pm_amm_price(y, x, L)  # opposite token
    assert p_yes + p_no == pytest.approx(1.0)


def test_pm_amm_price_monotone_in_reserve_diff():
    assert pm_amm.implied_price(10, 50) > pm_amm.implied_price(-10, 50)


# --------------------------------------------------------------------------- #
# frequent batch auction
# --------------------------------------------------------------------------- #
def test_fba_uniform_clearing_price():
    a = fba.BatchAuction()
    a.submit("buy", 105, 10)
    a.submit("buy", 102, 10)
    a.submit("sell", 100, 10)
    a.submit("sell", 103, 10)
    bids, asks = a.book
    assert a.peek() == a.peek()                  # peeking leaves the batch alone
    price, qty, bid_fills, ask_fills = a.clear()
    # Max matchable volume is 10: getting the 103 seller in needs price >= 103,
    # which drops the 102 buyer, so the two sides never both reach 20.
    assert qty == pytest.approx(10)
    assert 100 <= price <= 105
    # Fills balance, and every filled order is willing to trade at the price.
    assert sum(bid_fills) == pytest.approx(qty)
    assert sum(ask_fills) == pytest.approx(qty)
    for (bp, _), f in zip(bids, bid_fills):
        assert f == 0 or bp >= price - 1e-9
    for (ap, _), f in zip(asks, ask_fills):
        assert f == 0 or ap <= price + 1e-9
    # Clearing consumes the batch, so nothing trades twice.
    assert a.book == ([], [])
    assert a.clear()[1] == 0


def test_fba_keep_unfilled_carries_residuals_forward():
    a = fba.BatchAuction()
    a.submit("buy", 105, 10)
    a.submit("sell", 100, 4)
    _, qty, _, _ = a.clear(keep_unfilled=True)
    assert qty == pytest.approx(4)
    bids, asks = a.book
    assert bids == [(105.0, 6.0)]                # residual buy interest survives
    assert asks == []                            # the seller filled completely


def test_fba_no_cross_no_trade():
    price, qty = fba.clear_uniform_price([(99, 5)], [(101, 5)])
    assert qty == 0


def test_fba_allocations_balance_and_are_feasible():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        nb, na = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        bids = [(round(float(p), 2), round(float(q), 2)) for p, q in
                zip(rng.uniform(90, 110, nb), rng.uniform(0.5, 6, nb))]
        asks = [(round(float(p), 2), round(float(q), 2)) for p, q in
                zip(rng.uniform(90, 110, na), rng.uniform(0.5, 6, na))]
        price, vol, bid_fills, ask_fills = fba.clear_with_allocations(bids, asks)
        if price is None:
            assert vol == 0 and not any(bid_fills) and not any(ask_fills)
            continue
        # Both sides move exactly the cleared volume.
        assert sum(bid_fills) == pytest.approx(vol, abs=1e-9)
        assert sum(ask_fills) == pytest.approx(vol, abs=1e-9)
        # No order over-fills, and nobody trades at a price they refused.
        for (p, q), f in zip(bids, bid_fills):
            assert f <= q + 1e-9 and (f < 1e-12 or p >= price - 1e-9)
        for (p, q), f in zip(asks, ask_fills):
            assert f <= q + 1e-9 and (f < 1e-12 or p <= price + 1e-9)


def test_fba_rations_pro_rata_at_the_cut_level():
    # Two bids tied at 100 chase 2 units of supply: 3:1 sizes split 1.5 : 0.5.
    price, vol, bid_fills, ask_fills = fba.clear_with_allocations(
        [(100.0, 3.0), (100.0, 1.0)], [(99.0, 2.0)])
    assert vol == pytest.approx(2.0)
    assert bid_fills == pytest.approx([1.5, 0.5])
    assert ask_fills == pytest.approx([2.0])


def test_fba_short_side_cuts_a_strictly_crossing_order():
    # A lone bid of 10 against a lone ask of 4 clears 4. Filling every strictly
    # crossing order completely would hand the bid 10 against 4 units of supply.
    _, vol, bid_fills, ask_fills = fba.clear_with_allocations(
        [(105.0, 10.0)], [(100.0, 4.0)])
    assert vol == pytest.approx(4.0)
    assert bid_fills == pytest.approx([4.0])
    assert ask_fills == pytest.approx([4.0])



# --------------------------------------------------------------------------- #
# peer prediction
# --------------------------------------------------------------------------- #
def test_bts_rewards_surprisingly_common_answer():
    # 30% choose answer 1, but everyone predicted only ~10% would.
    n = 100
    info = [1] * 30 + [0] * 70
    preds = [[0.9, 0.1]] * n
    scores = pp.bayesian_truth_serum(info, preds, n_answers=2)
    answer1_score = np.mean([scores[i] for i in range(n) if info[i] == 1])
    answer0_score = np.mean([scores[i] for i in range(n) if info[i] == 0])
    # The "surprisingly common" answer 1 earns a higher information score.
    assert answer1_score > answer0_score


def test_output_agreement_degenerate():
    # Colluding constant reports always score 1 — the failure BTS fixes.
    assert pp.output_agreement(1, 1) == 1.0
    assert pp.output_agreement(0, 1) == 0.0


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_linear_pool_is_mean():
    p = aggregation.linear_opinion_pool([[0.8, 0.2], [0.4, 0.6]])
    np.testing.assert_allclose(p, [0.6, 0.4])


def test_log_pool_sharper_than_linear():
    forecasts = [[0.7, 0.3], [0.6, 0.4]]
    lin = aggregation.linear_opinion_pool(forecasts)
    log = aggregation.logarithmic_opinion_pool(forecasts)
    # The log pool is more confident in the agreed-upon leading outcome.
    assert log[0] > lin[0]
    assert log.sum() == pytest.approx(1.0)


def test_depth_trimmed_mean_drops_outlier():
    forecasts = [[0.5, 0.5], [0.55, 0.45], [0.45, 0.55], [0.99, 0.01]]
    trimmed = aggregation.depth_trimmed_mean(forecasts, trim=0.25)
    plain = aggregation.linear_opinion_pool(forecasts)
    # Dropping the [0.99, 0.01] outlier pulls the pool back toward 0.5.
    assert abs(trimmed[0] - 0.5) < abs(plain[0] - 0.5)


# --------------------------------------------------------------------------- #
# peer prediction: the conditions the guarantees actually need
# --------------------------------------------------------------------------- #
def test_output_agreement_truthfulness_needs_more_than_correlation():
    # Positively correlated, yet an agent seeing 1 maximises agreement with 0.
    joint = np.array([[0.50, 0.20], [0.20, 0.10]])
    p_a1, p_b1 = joint[1].sum(), joint[:, 1].sum()
    assert joint[1, 1] - p_a1 * p_b1 > 0                 # positive correlation
    assert joint[1, 1] / joint[1].sum() == pytest.approx(1 / 3)
    assert not pp.output_agreement_is_truthful(joint)
    # Best response after seeing 1 is to report 0, so truth is not an equilibrium.
    expected = [joint[1, j] / joint[1].sum() for j in (0, 1)]
    assert int(np.argmax(expected)) == 0

    # Self-dominant distributions do pass.
    assert pp.output_agreement_is_truthful(np.array([[0.4, 0.1], [0.1, 0.4]]))


def test_ca_sign_matrix_is_zero_one_not_three_valued():
    # Shnayder et al.'s Sign is 1 for positive and 0 otherwise. A -1 would pay
    # agents to disagree on pairs that are rarer than chance.
    rng = np.random.default_rng(0)
    state = rng.integers(0, 2, 400)
    reports = state[None, :] ^ (rng.random((4, 400)) > 0.8).astype(int)
    S = pp._ca_sign_matrix(reports, 2)
    assert set(np.unique(S)) <= {0.0, 1.0}
    np.testing.assert_allclose(S, np.eye(2))     # positively correlated: identity


def test_ca_punishes_the_uninformative_strategy():
    rng = np.random.default_rng(1)
    state = rng.integers(0, 2, 400)
    reports = state[None, :] ^ (rng.random((6, 400)) > 0.82).astype(int)
    truthful = pp.correlated_agreement(reports, n_signals=2)[0]
    deviated = reports.copy(); deviated[0, :] = 0
    constant = pp.correlated_agreement(deviated, n_signals=2)[0]
    assert truthful > constant
    assert abs(constant) < 0.1                   # the constant exploit nets ~0


def test_ca_sign_matrix_is_manipulable_when_one_agent_is_half_the_data():
    # CA assumes Sign(Delta) is fixed. Estimated from two agents' reports, a
    # constant reporter inverts it and is then paid by the inversion.
    rng = np.random.default_rng(3)
    state = rng.integers(0, 2, 400)
    reports = state[None, :] ^ (rng.random((2, 400)) > 0.82).astype(int)
    deviated = reports.copy(); deviated[0, :] = 0
    np.testing.assert_allclose(pp._ca_sign_matrix(reports, 2), np.eye(2))
    np.testing.assert_allclose(pp._ca_sign_matrix(deviated, 2), 1 - np.eye(2))
    assert (pp.correlated_agreement(deviated, n_signals=2)[0]
            > pp.correlated_agreement(reports, n_signals=2)[0])


def test_ca_detail_free_matches_ca_on_a_truthful_crowd():
    rng = np.random.default_rng(4)
    state = rng.integers(0, 2, 400)
    reports = state[None, :] ^ (rng.random((6, 400)) > 0.82).astype(int)
    plain = pp.correlated_agreement(reports, n_signals=2)
    df = pp.correlated_agreement_df(reports, n_signals=2,
                                    rng=np.random.default_rng(0))
    np.testing.assert_allclose(df, plain, atol=0.05)


def test_enforced_agreement_fixes_the_report_marginal():
    rng = np.random.default_rng(5)
    reports = (rng.random(100) < 0.8).astype(int)     # ~80 ones
    forced = pp.enforce_marginal(reports, target_ones=50, rng=rng)
    assert int(forced.sum()) == 50
    # Only 1s are flipped down, never 0s flipped up, when there is a surplus.
    assert np.all(forced[reports == 0] == 0)


def test_enforced_agreement_beats_the_constant_strategy():
    rng = np.random.default_rng(6)
    state = rng.integers(0, 2, 300)
    reports = state[None, :] ^ (rng.random((5, 300)) > 0.82).astype(int)
    truthful = pp.enforced_agreement(reports, rng=np.random.default_rng(0))[0]
    deviated = reports.copy(); deviated[0, :] = 0
    constant = pp.enforced_agreement(deviated, rng=np.random.default_rng(0))[0]
    assert truthful > constant


def test_bts_rejects_prediction_rows_that_are_not_distributions():
    # A row of (1, 1) would make the "negative KL" term positive.
    with pytest.raises(ValueError, match="sum to 1"):
        pp.bayesian_truth_serum([0, 1], [[1.0, 1.0], [0.5, 0.5]], n_answers=2)
    with pytest.raises(ValueError, match="non-negative"):
        pp.bayesian_truth_serum([0, 1], [[1.5, -0.5], [0.5, 0.5]], n_answers=2)
    with pytest.raises(ValueError, match="index"):
        pp.bayesian_truth_serum([0, 5], [[0.5, 0.5], [0.5, 0.5]], n_answers=2)


def test_bts_prediction_score_is_a_negative_kl_on_valid_input():
    info = [0, 1, 0, 0]
    preds = [[0.75, 0.25]] * 4
    scores = pp.bayesian_truth_serum(info, preds, n_answers=2, alpha=1.0)
    x_bar = np.array([0.75, 0.25])
    kl = float(np.sum(x_bar * np.log(x_bar / np.array([0.75, 0.25]))))
    assert kl == pytest.approx(0.0)              # prediction matches the crowd
    # With a perfect prediction the prediction score contributes nothing.
    y_bar = np.exp(np.mean(np.log(np.array(preds)), axis=0))
    for i, k in enumerate(info):
        assert scores[i] == pytest.approx(np.log(x_bar[k] / y_bar[k]))
