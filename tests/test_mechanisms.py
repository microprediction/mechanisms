"""Tests for the reference mechanism implementations.

These double as executable documentation: properness of the scoring rules, the
LMSR bounded-loss bound, the constant-product invariant, parimutuel
self-funding, CDA matching, and the perp funding tether.
"""

import numpy as np
import pytest

from mechanisms import (
    amm,
    calibration,
    cda,
    cmm,
    lmsr,
    parimutuel,
    perp,
    scoring_rules as sr,
)


# --------------------------------------------------------------------------- #
# scoring rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("score_fn", [sr.log_score, sr.brier_score, sr.spherical_score])
def test_scoring_rules_are_strictly_proper(score_fn):
    """Truthful reporting uniquely minimises expected loss."""
    belief = np.array([0.2, 0.5, 0.3])
    truthful = sr.expected_score(score_fn, belief, belief)
    rng = np.random.default_rng(0)
    for _ in range(200):
        misreport = rng.dirichlet(np.ones(3))
        if np.allclose(misreport, belief, atol=1e-3):
            continue
        assert sr.expected_score(score_fn, belief, misreport) >= truthful - 1e-9


def test_log_score_values():
    assert sr.log_score([0.5, 0.5], 0) == pytest.approx(np.log(2))
    assert sr.brier_score([1.0, 0.0], 0) == pytest.approx(0.0)


def test_crps_reduces_to_abs_error_for_point_forecast():
    # A degenerate ensemble (all same point) gives |x - y|.
    assert sr.crps_ensemble([3.0, 3.0, 3.0], 5.0) == pytest.approx(2.0)


def test_energy_score_matches_crps_in_1d():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=50)
    y = 0.3
    es = sr.energy_score(samples.reshape(-1, 1), [y], beta=1.0)
    crps = sr.crps_ensemble(samples, y)
    assert es == pytest.approx(crps, rel=1e-6)


def test_energy_score_rewards_truth():
    rng = np.random.default_rng(2)
    truth_center = np.array([0.0, 0.0])
    good = rng.normal(loc=truth_center, scale=1.0, size=(500, 2))
    bad = rng.normal(loc=[5.0, 5.0], scale=1.0, size=(500, 2))
    y = np.array([0.1, -0.2])
    assert sr.energy_score(good, y) < sr.energy_score(bad, y)


# --------------------------------------------------------------------------- #
# parimutuel
# --------------------------------------------------------------------------- #
def test_parimutuel_self_funding_and_payout():
    pool = parimutuel.ParimutuelPool(n_outcomes=3, takeout=0.0)
    pool.bet("alice", 0, 100)
    pool.bet("bob", 0, 50)
    pool.bet("carol", 1, 30)
    assert pool.total_pool == pytest.approx(180)
    np.testing.assert_allclose(pool.implied_probabilities(), [150/180, 30/180, 0])
    payouts = pool.settle(winning_outcome=0)
    # With zero takeout the total paid equals the pool (self-funding).
    assert sum(payouts.values()) == pytest.approx(180)
    # Alice staked twice Bob's amount -> twice the payout.
    assert payouts["alice"] == pytest.approx(2 * payouts["bob"])


def test_parimutuel_takeout_reduces_payout():
    pool = parimutuel.ParimutuelPool(n_outcomes=2, takeout=0.2)
    pool.bet("a", 0, 100)
    pool.bet("b", 1, 100)
    payouts = pool.settle(0)
    assert sum(payouts.values()) == pytest.approx(0.8 * 200)


def test_dynamic_parimutuel_early_buyer_advantage():
    m = parimutuel.DynamicParimutuelMarket(n_outcomes=2)
    early = m.buy("early", 0, spend=10)
    late = m.buy("late", 0, spend=10)
    # Buying the same outcome later costs more per share (price rose).
    assert early > late


# --------------------------------------------------------------------------- #
# LMSR
# --------------------------------------------------------------------------- #
def test_lmsr_prices_are_probabilities():
    m = lmsr.LMSR(n_outcomes=3, b=50)
    p = m.prices()
    assert p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p, np.ones(3) / 3)  # symmetric start


def test_lmsr_buy_raises_price():
    m = lmsr.LMSR(n_outcomes=2, b=100)
    before = m.prices()[0]
    m.buy(0, 50)
    assert m.prices()[0] > before


def test_lmsr_bounded_loss():
    m = lmsr.LMSR(n_outcomes=4, b=10)
    rng = np.random.default_rng(3)
    for _ in range(100):
        m.buy(rng.integers(4), rng.uniform(1, 20))
    worst = min(m.realized_pnl(k) for k in range(4))
    assert worst >= -m.max_loss - 1e-6


def test_lmsr_shares_for_budget_inverts_cost():
    m = lmsr.LMSR(n_outcomes=3, b=75)
    budget = 40.0
    shares = m.shares_for_budget(1, budget)
    assert m.cost_to_trade(np.eye(3)[1] * shares) == pytest.approx(budget, rel=1e-9)


def test_cmm_lmsr_matches_lmsr_module():
    cost, grad = cmm.lmsr_potential(b=60)
    maker = cmm.CostFunctionMarketMaker(3, cost, grad)
    ref = lmsr.LMSR(3, b=60)
    np.testing.assert_allclose(maker.prices(), ref.prices())
    d = np.array([5.0, 0.0, 0.0])
    assert maker.cost_to_trade(d) == pytest.approx(ref.cost_to_trade(d))


# --------------------------------------------------------------------------- #
# AMM
# --------------------------------------------------------------------------- #
def test_constant_product_invariant_grows_with_fee():
    pool = amm.ConstantProductAMM(x=1000, y=1000, fee=0.003)
    k0 = pool.k
    pool.swap(100, x_in=True)
    # Fees mean k weakly increases after a trade.
    assert pool.k >= k0


def test_constant_product_no_fee_preserves_k():
    pool = amm.ConstantProductAMM(x=1000, y=1000, fee=0.0)
    k0 = pool.k
    pool.swap(100, x_in=True)
    assert pool.k == pytest.approx(k0, rel=1e-9)


def test_impermanent_loss_zero_at_parity_and_negative_otherwise():
    assert amm.impermanent_loss(1.0) == pytest.approx(0.0)
    assert amm.impermanent_loss(2.0) < 0
    assert amm.impermanent_loss(0.5) < 0
    # 4x price move -> classic 20% IL.
    assert amm.impermanent_loss(4.0) == pytest.approx(2*2/5 - 1)


def test_constant_mean_reduces_to_constant_product():
    cm = amm.ConstantMeanAMM([1000, 1000], [0.5, 0.5], fee=0.0)
    cp = amm.ConstantProductAMM(1000, 1000, fee=0.0)
    assert cm.amount_out(0, 1, 100) == pytest.approx(cp.amount_out(100, x_in=True))


# --------------------------------------------------------------------------- #
# CDA
# --------------------------------------------------------------------------- #
def test_cda_crossing_orders_match():
    book = cda.LimitOrderBook()
    assert book.submit(cda.Order("sell", price=101, qty=5, trader="mm")) == []
    trades = book.submit(cda.Order("buy", price=102, qty=3, trader="taker"))
    assert len(trades) == 1
    t = trades[0]
    assert t.price == 101  # executes at resting (maker) price
    assert t.qty == 3
    assert t.buyer == "taker" and t.seller == "mm"
    # 2 units of the sell remain.
    assert book.best_ask() == 101


def test_cda_price_time_priority():
    book = cda.LimitOrderBook()
    book.submit(cda.Order("buy", 100, 5, "first"))
    book.submit(cda.Order("buy", 100, 5, "second"))
    trades = book.submit(cda.Order("sell", 100, 5, "seller"))
    # First-in at the best price trades first.
    assert trades[0].buyer == "first"


def test_cda_spread_and_mid():
    book = cda.LimitOrderBook()
    book.submit(cda.Order("buy", 99, 1))
    book.submit(cda.Order("sell", 101, 1))
    assert book.spread() == pytest.approx(2)
    assert book.mid() == pytest.approx(100)


# --------------------------------------------------------------------------- #
# perp
# --------------------------------------------------------------------------- #
def test_funding_rate_clamp():
    # Large positive premium -> funding ~ premium (clamp binds on (I - P)).
    f = perp.funding_rate(premium_index=0.01, interest_rate=0.0001, clamp=0.0005)
    assert f == pytest.approx(0.01 - 0.0005)


def test_funding_transfers_long_to_short():
    long = perp.PerpPosition(side=1, size=1, entry_price=100, collateral=10)
    short = perp.PerpPosition(side=-1, size=1, entry_price=100, collateral=10)
    rate = 0.001
    lf = long.apply_funding(mark_price=100, rate=rate)
    sf = short.apply_funding(mark_price=100, rate=rate)
    assert lf < 0 and sf > 0          # long pays, short receives
    assert lf == pytest.approx(-sf)   # zero-sum


def test_perp_liquidation_price_long():
    pos = perp.PerpPosition(side=1, size=1, entry_price=100, collateral=10,
                            maintenance_margin_ratio=0.0)
    # No maintenance margin -> liquidates when equity hits 0 -> price 90.
    assert pos.liquidation_price() == pytest.approx(90)
    assert pos.is_liquidated(89)
    assert not pos.is_liquidated(95)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_perfect_calibration_has_zero_ece():
    rng = np.random.default_rng(7)
    probs = rng.uniform(size=20000)
    outcomes = (rng.uniform(size=20000) < probs).astype(int)
    assert calibration.expected_calibration_error(probs, outcomes, n_bins=10) < 0.02


def test_brier_decomposition_identity():
    rng = np.random.default_rng(8)
    probs = rng.uniform(size=5000)
    outcomes = (rng.uniform(size=5000) < probs).astype(int)
    rel, res, unc = calibration.brier_decomposition(probs, outcomes, n_bins=10)
    bs = np.mean((probs - outcomes) ** 2)
    assert rel - res + unc == pytest.approx(bs, abs=0.02)
