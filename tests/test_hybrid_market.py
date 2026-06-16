"""Tests for the hybrid CLOB + AMM binary market."""

import numpy as np
import pytest

from mechanisms.hybrid_market import HybridBinaryMarket, YES, NO


def test_complementary_match_is_zero_slippage():
    """A YES taker matched against a resting NO bid pays 1-b and never touches the AMM."""
    m = HybridBinaryMarket(b=100.0)
    m.rest(NO, price=0.6, size=100.0)
    before = m.amm_price(YES)
    r = m.market_buy(YES, 50.0)
    assert r["p2p_filled"] == pytest.approx(50.0)
    assert r["amm_filled"] == pytest.approx(0.0)
    assert r["avg_price"] == pytest.approx(0.4)            # 1 - 0.6
    assert r["amm_price_after"] == pytest.approx(before)   # AMM untouched


def test_amm_backstop_when_book_empty():
    """With no resting orders the whole order routes through the AMM and moves price."""
    m = HybridBinaryMarket(b=100.0)
    before = m.amm_price(YES)
    r = m.market_buy(YES, 80.0)
    assert r["amm_filled"] == pytest.approx(80.0)
    assert r["p2p_filled"] == pytest.approx(0.0)
    assert r["amm_price_after"] > before                   # slippage: price rose
    assert r["avg_price"] > before                         # paid more than the pre-trade price


def test_hybrid_split_and_conservation():
    m = HybridBinaryMarket(b=100.0)
    m.rest(NO, price=0.55, size=30.0)
    r = m.market_buy(YES, 100.0)
    assert r["p2p_filled"] == pytest.approx(30.0)
    assert r["amm_filled"] == pytest.approx(70.0)
    assert r["filled"] == pytest.approx(100.0)
    assert r["p2p_avg_price"] == pytest.approx(0.45)


def test_hybrid_beats_amm_only():
    """For the same size, complementary book yields a lower average price and less AMM drift."""
    qty = 120.0
    pure = HybridBinaryMarket(b=100.0)
    r_pure = pure.market_buy(YES, qty)

    hyb = HybridBinaryMarket(b=100.0)
    hyb.rest(NO, price=0.55, size=100.0)
    r_hyb = hyb.market_buy(YES, qty)

    assert r_hyb["avg_price"] < r_pure["avg_price"]
    drift_pure = r_pure["amm_price_after"] - r_pure["amm_price_before"]
    drift_hyb = r_hyb["amm_price_after"] - r_hyb["amm_price_before"]
    assert drift_hyb < drift_pure


def test_rest_validates_price():
    m = HybridBinaryMarket()
    with pytest.raises(ValueError):
        m.rest(YES, price=1.2, size=10.0)
