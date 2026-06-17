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
    price, qty = a.clear()
    # Max matchable volume is 10: getting the 103 seller in needs price >= 103,
    # which drops the 102 buyer, so the two sides never both reach 20.
    assert qty == pytest.approx(10)
    assert 100 <= price <= 105


def test_fba_no_cross_no_trade():
    price, qty = fba.clear_uniform_price([(99, 5)], [(101, 5)])
    assert qty == 0


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
