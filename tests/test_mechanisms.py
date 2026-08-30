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


def test_dynamic_parimutuel_money_is_the_cost_function():
    rng = np.random.default_rng(11)
    m = parimutuel.DynamicParimutuelMarket(n_outcomes=4, kappa=1.7)
    for _ in range(40):
        m.buy("t", int(rng.integers(0, 4)), float(rng.uniform(0.1, 5)))
        assert m.money == pytest.approx(m.cost(), rel=1e-12)


def test_dynamic_parimutuel_purchases_are_path_independent():
    one = parimutuel.DynamicParimutuelMarket(n_outcomes=3)
    bulk = one.buy("a", 0, spend=6.0)
    sliced = parimutuel.DynamicParimutuelMarket(n_outcomes=3)
    total = sum(sliced.buy("a", 0, spend=6.0 / 100) for _ in range(100))
    assert total == pytest.approx(bulk, rel=1e-9)
    assert sliced.shares[0] == pytest.approx(one.shares[0], rel=1e-9)


def test_dynamic_parimutuel_settlement_conserves_cash():
    rng = np.random.default_rng(12)
    for _ in range(200):
        n = int(rng.integers(2, 6))
        m = parimutuel.DynamicParimutuelMarket(n_outcomes=n)
        paid = 0.0
        for _ in range(15):
            spend = float(rng.uniform(0.01, 8))
            m.buy(f"t{rng.integers(0, 4)}", int(rng.integers(0, n)), spend)
            paid += spend
        assert m.money == pytest.approx(m.subsidy + paid, rel=1e-10)
        w = int(rng.integers(0, n))
        out = sum(m.settle(w).values()) + m.house_settlement(w)
        assert out == pytest.approx(m.money, rel=1e-10)
        # The house can lose at most what it staked to seed the market.
        assert m.subsidy - m.house_settlement(w) <= m.subsidy + 1e-12


def test_dynamic_parimutuel_share_ratio_and_payoff():
    m = parimutuel.DynamicParimutuelMarket(n_outcomes=3)
    m.buy("a", 0, 3.0)
    m.buy("b", 2, 1.0)
    p = m.prices()
    assert p[0] / p[1] == pytest.approx(m.shares[0] / m.shares[1])
    assert float(np.sum(p ** 2)) == pytest.approx(1.0)
    # A DPM share pays 1/p_i, not $1: it is not an Arrow security.
    assert m.payoff_per_share(0) == pytest.approx(1.0 / p[0])


def test_dynamic_parimutuel_round_trip_sale():
    m = parimutuel.DynamicParimutuelMarket(n_outcomes=2)
    bought = m.buy("a", 0, spend=4.0)
    back = m.sell("a", 0, bought)
    assert back == pytest.approx(4.0, rel=1e-12)
    assert m.shares[0] == pytest.approx(1.0)


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


def test_lmsr_pnl_distinguishes_prior_issuance_from_a_seeded_position():
    rng = np.random.default_rng(0)
    session_shortfall = 0.0
    for _ in range(500):
        n = int(rng.integers(2, 6))
        b = float(rng.uniform(5, 200))
        m = lmsr.LMSR(n, b=b, q0=rng.normal(0, 50, n))
        for _ in range(int(rng.integers(1, 10))):
            m.buy(int(rng.integers(0, n)), float(rng.normal(0, 60)))
        for w in range(n):
            # Read q0 as prior issuance whose revenue was banked: bound holds.
            assert m.bounded_loss_holds(w)
            # Read it as a seeded grant: the maker owes shares it was never paid
            # for, and the b log n bound simply does not apply.
            session_shortfall = min(session_shortfall,
                                    m.session_pnl(w) + m.max_loss)
    assert session_shortfall < -1.0

    # With no prior state the two readings coincide.
    m = lmsr.LMSR(3, b=50.0)
    m.buy(0, 30.0)
    assert m.realized_pnl(0) == pytest.approx(m.session_pnl(0))


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
    bs = np.mean((probs - outcomes) ** 2)
    for n_bins in (1, 2, 3, 10, 50, None):
        d = calibration.brier_decomposition(probs, outcomes, n_bins=n_bins)
        assert d.brier == pytest.approx(bs, rel=1e-12, abs=1e-12)


def test_brier_within_bin_vanishes_on_exact_grouping():
    rng = np.random.default_rng(9)
    probs = rng.choice([0.1, 0.4, 0.9], size=500)
    outcomes = (rng.uniform(size=500) < probs).astype(int)
    d = calibration.brier_decomposition(probs, outcomes, n_bins=None)
    assert d.within_bin == pytest.approx(0.0, abs=1e-12)


def test_brier_within_bin_is_not_negligible_when_coarse():
    # Two forecasts in one bin: the three-term identity implies 0.25, actual 0.01.
    d = calibration.brier_decomposition([0.1, 0.9], [0, 1], n_bins=1)
    assert d.reliability - d.resolution + d.uncertainty == pytest.approx(0.25)
    assert d.brier == pytest.approx(0.01)
