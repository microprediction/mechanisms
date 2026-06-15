"""Tests for the Perpetual Demand Lending Pool model (arXiv:2502.06028).

Where possible these check against the paper's own worked numbers.
"""

import numpy as np
import pytest

from mechanisms import pdlp


# ----------------------- §2.1 exchange ----------------------- #
def test_liquidation_price_eth_example():
    # Min-collateral long, p0=2000, eta=4 -> liquidation at p0(1-1/eta)=1500.
    p0, eta, delta = 2000.0, 4.0, 4.0
    c = pdlp.min_collateral(delta, eta, p0)        # saturates eq (1)
    assert pdlp.liquidation_price(delta, eta, p0, c) == pytest.approx(1500.0)
    assert pdlp.liquidation_price(delta, eta, p0) == pytest.approx(p0 * (1 - 1/eta))


def test_collateral_condition():
    assert pdlp.collateral_ok(c=2000, delta=4, eta=4, p0=2000)       # saturated
    assert not pdlp.collateral_ok(c=2000, delta=5, eta=4, p0=2000)   # over-levered


def test_short_liquidation_is_above_mark():
    liq = pdlp.liquidation_price(delta=1, eta=-4, p0=2000.0)
    assert liq > 2000.0


def test_funding_rate_example():
    # L=1000, S=250, p=p0 -> gamma_L = kappa(4 - 1) = 3 kappa.
    g = pdlp.linear_funding_rate(1000, 250, p=2000, p0=2000, kappa=0.01)
    assert g == pytest.approx(3 * 0.01)


# ----------------------- §2.3.1 funding arb ----------------------- #
def test_funding_arb_zeroes_the_rate():
    L0, p, p0, kappa = 1000.0, 2200.0, 2000.0, 0.01
    ell = pdlp.funding_arb_size(L0, p, p0)
    assert ell == pytest.approx(L0 * (p / p0 - 1))
    # After opening ell, long OI = L0 + ell, short = L0; funding should be 0.
    g = pdlp.linear_funding_rate(L0 + ell, L0, p=p, p0=p0, kappa=kappa)
    assert g == pytest.approx(0.0, abs=1e-9)


def test_funding_fee_bound_with_price_cap():
    kappa, L0, B = 0.01, 1000.0, 1.10
    assert pdlp.funding_fee_upper_bound(kappa, L0, 2200, 2000, B=B) == pytest.approx(
        kappa * (1 - 1 / B) / L0)


# ----------------------- §2.3.2-3 PDLP price-impact arb ----------------------- #
def test_uniswap_v2_optimal_x_and_loss():
    p0, R1 = 2000.0, 1_000_000.0
    R2 = R1 / p0
    fwd = pdlp.UniswapV2Forward(R1, R2)
    assert fwd.initial_price == pytest.approx(p0)
    p = 2200.0
    x_star = fwd.optimal_x(p)
    # G'(x*) should equal 1/p.
    assert fwd.G_prime(x_star) == pytest.approx(1.0 / p, rel=1e-9)
    # generic bisection matches the closed form
    assert pdlp.pdlp_arb_optimal_x(fwd, p) == pytest.approx(x_star, rel=1e-6)
    # LP loss equals the net PDLP value change magnitude: x* - p G(x*).
    net = x_star - p * fwd.G(x_star)
    assert fwd.lp_loss(p) == pytest.approx(-net, rel=1e-9)
    assert fwd.lp_loss(p) >= 0


def test_uniswap_v2_loss_zero_at_no_move():
    fwd = pdlp.UniswapV2Forward(1_000_000.0, 500.0)
    assert fwd.lp_loss(fwd.initial_price) == pytest.approx(0.0, abs=1e-6)


# ----------------------- §3 target weight mechanism ----------------------- #
def test_weights_sum_to_one():
    w = pdlp.weights([1.0, 2000.0, 60000.0], [500_000.0, 150.0, 5.0])
    assert w.sum() == pytest.approx(1.0)


def test_target_weight_trade_hits_target():
    prices = np.array([1.0, 2000.0, 60000.0])
    R = np.array([500_000.0, 150.0, 5.0])
    target = np.array([1/3, 1/3, 1/3])
    delta = pdlp.target_weight_trade(prices, R, available=R, target=target)
    w_after = pdlp.weights(prices, R + delta)
    np.testing.assert_allclose(w_after, target, atol=1e-9)


def test_gmx_discount_rewards_rebalancing():
    target = np.array([0.5, 0.5])
    w_before = np.array([0.3, 0.7])          # underweight asset 0
    closer = np.array([0.45, 0.55])          # moved toward target
    farther = np.array([0.2, 0.8])           # moved away
    F_closer = pdlp.gmx_glp_discount(w_before, closer, target, gamma_t=0.1)
    F_farther = pdlp.gmx_glp_discount(w_before, farther, target, gamma_t=0.1)
    assert F_closer > 0          # subsidy for improving balance
    assert F_farther == 0        # max(0, negative) -> no discount


def test_gmx_discount_zero_for_no_trade():
    target = np.array([0.5, 0.5])
    w = np.array([0.4, 0.6])
    assert pdlp.gmx_glp_discount(w, w, target) == 0.0


def test_portfolio_dilution_reduces_existing_value():
    prices = np.array([1.0, 2000.0])
    R = np.array([100_000.0, 50.0])
    base = float(prices @ R)
    v = pdlp.portfolio_dilution_value(prices, R, ell=0.0, fee=0.001, F=0.05)
    assert v == pytest.approx(base / 1.05)


# ----------------------- §4 hedged PDLP ----------------------- #
def test_delta_hedge_closed_form():
    Sigma = np.array([[0.04, 0.0], [0.0, 0.05]])
    ell = np.array([40.0, 1.0])
    Delta = np.array([150.0, 5.0])
    fee, gamma = 0.001, 2.0
    pi = pdlp.delta_hedge(fee, gamma, Sigma, ell, Delta)
    expected = (fee / gamma) * np.linalg.solve(Sigma, ell) - Delta
    np.testing.assert_allclose(pi, expected)


def test_sharpe_conditions_return_bools():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.05]])
    ell = np.array([40.0, 1.0])
    Delta = np.array([1.0, 0.1])
    exp_ok, var_ok = pdlp.sharpe_improves(0.01, ell, Delta, Sigma, [2000.0, 60000.0], 1.0)
    assert isinstance(exp_ok, (bool, np.bool_)) and isinstance(var_ok, (bool, np.bool_))
