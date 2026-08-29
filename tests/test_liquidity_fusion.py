"""Theorem tests for research/liquidity-is-precision.md.

Merging quadratic makers is Gaussian precision fusion (exactly); a diagonal
quoter turns the aggregate into a Ledoit-Wolf blend with intensity equal to
its capital share; selection reweights capital toward the better model; and
inventory shifts a maker's quote without changing its contributed weight.
"""

import numpy as np
import pytest

from mechanisms.fee_routing import LogCoshMaker
from mechanisms.liquidity_fusion import QuadraticMaker, fuse


def _spd(rng, n, scale=1.0):
    A = rng.normal(size=(n, n))
    return scale * (A @ A.T + n * np.eye(n))


def test_fusion_is_gaussian_posterior():
    rng = np.random.default_rng(3)
    n = 4
    makers = [QuadraticMaker(rng.normal(size=n), _spd(rng, n)) for _ in range(3)]
    p, precision, fills = fuse(makers)
    # precision-weighted mean of the quotes, precisions added
    expected = np.linalg.solve(
        np.sum([m.L for m in makers], axis=0),
        np.sum([m.L @ m.price for m in makers], axis=0),
    )
    assert np.allclose(p, expected)
    assert np.allclose(precision, np.sum([m.L for m in makers], axis=0))
    # fills clear (zero net demand) and each maker reprices to the consensus
    assert np.allclose(np.sum(fills, axis=0), 0.0, atol=1e-10)
    for m, s in zip(makers, fills):
        m.apply_fill(s)
        assert np.allclose(m.price, p)


def test_price_impact_is_posterior_covariance():
    rng = np.random.default_rng(4)
    n = 3
    makers = [QuadraticMaker(rng.normal(size=n), _spd(rng, n)) for _ in range(2)]
    delta = rng.normal(size=n)
    p0, precision, _ = fuse(makers)
    p1, _, _ = fuse(makers, demand=delta)
    assert np.allclose(p1 - p0, np.linalg.solve(precision, delta))


def test_diagonal_quoter_is_ledoit_wolf_with_capital_share_intensity():
    rng = np.random.default_rng(5)
    n = 6  # e.g. flattened covariance entries
    sample = rng.normal(size=n)          # the correlated maker's estimate S
    target = np.full(n, sample.mean())   # the naive maker's structured target F
    w_s, w_f = 3.0, 1.0                  # capital
    full_mm = QuadraticMaker(sample, w_s * np.eye(n))
    naive_mm = QuadraticMaker(target, w_f * np.eye(n))
    p, _, _ = fuse([full_mm, naive_mm])
    delta = w_f / (w_s + w_f)
    assert np.allclose(p, (1 - delta) * sample + delta * target)


def test_selection_tunes_the_shrinkage_intensity():
    # Two makers quote; an informed trader clears the merged market to the
    # true price each epoch and positions settle there. Each maker's loss is
    # (1/2) e' Lambda e with e its quote error, so capital drains from the
    # worse model and the naive quoter's share (the shrinkage intensity)
    # decays when the correlated model is real.
    rng = np.random.default_rng(6)
    n = 4
    base = rng.normal(size=n)
    w_good, w_bad = 1.0, 1.0
    for _ in range(40):
        truth = base + 0.1 * rng.normal(size=n)
        good = QuadraticMaker(base, w_good * np.eye(n))                 # right prior
        bad = QuadraticMaker(base + 0.5, w_bad * np.eye(n))             # biased prior
        makers = [good, bad]
        # demand that clears the merged market exactly at the true price
        demand = sum(m.L for m in makers) @ truth - sum(m.L @ m.price for m in makers)
        p, _, fills = fuse(makers, demand=demand)
        assert np.allclose(p, truth)
        pnl = [m.trade_cost(s) - truth @ s for m, s in zip(makers, fills)]
        w_good += pnl[0]
        w_bad += pnl[1]
    assert w_good > w_bad
    assert w_bad / (w_good + w_bad) < 0.5  # intensity decayed


def test_inventory_shifts_the_quote_not_the_weight():
    # Quadratic: liquidity is constant; loading moves the price only.
    m = QuadraticMaker([0.0, 0.0], np.diag([2.0, 3.0]))
    before = m.L.copy()
    m.apply_fill(np.array([1.0, -2.0]))
    assert np.allclose(m.L, before)
    assert not np.allclose(m.price, m.m0)
    # Log-cosh: the supply slope at a fixed price is inventory-independent
    # ((d/dp) supply = b / (1 - p^2) regardless of q); only the level moves.
    p, h = 0.4, 1e-6
    light = LogCoshMaker(b=5.0, fee=0.0, q0=0.0)
    heavy = LogCoshMaker(b=5.0, fee=0.0, q0=4.0)
    slope_light = (light.supply(p + h) - light.supply(p - h)) / (2 * h)
    slope_heavy = (heavy.supply(p + h) - heavy.supply(p - h)) / (2 * h)
    assert slope_light == pytest.approx(slope_heavy, rel=1e-6)
    assert light.supply(p) != pytest.approx(heavy.supply(p))


def test_invalid_liquidity_rejected():
    with pytest.raises(ValueError):
        QuadraticMaker([0.0, 0.0], np.diag([1.0, -1.0]))
