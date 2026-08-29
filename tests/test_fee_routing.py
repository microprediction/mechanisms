"""Theorem tests for fee-bearing makers and clearing-price routing.

Each test checks a claim from research/proportional-fees-and-the-order-book.md:
round trips cost exactly the fee, routing solves the infimal convolution,
liquidity adds when fees vanish, the L1 term makes splits sparse, and the
aggregate supply curve is a consolidated order book (flat on quote bands).
"""

import numpy as np
import pytest

from mechanisms.fee_routing import LogCoshMaker, consolidated_book, route


def test_round_trip_costs_exactly_the_fee():
    m = LogCoshMaker(b=10.0, fee=0.02, q0=3.0)
    s = 1.7
    c_buy = m.apply_fill(s)
    c_sell = m.apply_fill(-s)
    assert m.q == pytest.approx(3.0)
    assert c_buy + c_sell == pytest.approx(2 * m.fee * s, abs=1e-12)
    # and with the fee off, the round trip is free (path independence)
    m0 = LogCoshMaker(b=10.0, fee=0.0, q0=3.0)
    assert m0.apply_fill(s) + m0.apply_fill(-s) == pytest.approx(0.0, abs=1e-12)


def test_routing_minimises_total_effective_cost():
    rng = np.random.default_rng(7)
    makers = [
        LogCoshMaker(b=5.0, fee=0.01, q0=1.0),
        LogCoshMaker(b=12.0, fee=0.03, q0=-2.0),
        LogCoshMaker(b=3.0, fee=0.00, q0=0.5),
    ]
    size = 4.0
    fills, _ = route(makers, size)
    assert fills.sum() == pytest.approx(size, abs=1e-9)
    best = sum(m.trade_cost(s) for m, s in zip(makers, fills))
    # no random feasible split does better
    for _ in range(2000):
        perturbation = rng.normal(scale=0.5, size=len(makers))
        perturbation -= perturbation.mean()  # stays feasible
        alt = fills + perturbation
        alt_cost = sum(m.trade_cost(s) for m, s in zip(makers, alt))
        assert alt_cost >= best - 1e-9


def test_active_makers_equalise_fee_adjusted_marginals():
    makers = [
        LogCoshMaker(b=5.0, fee=0.01),
        LogCoshMaker(b=8.0, fee=0.02),
    ]
    fills, p_star = route(makers, 6.0)
    for m, s in zip(makers, fills):
        if s > 1e-9:  # active seller: marginal price + fee = clearing price
            assert np.tanh((m.q + s) / m.b) + m.fee == pytest.approx(p_star, abs=1e-6)
        else:  # dead zone: clearing price inside the quote band
            assert m.bid - 1e-9 <= p_star <= m.ask + 1e-9


def test_zero_fee_routing_is_inf_convolution_and_liquidity_adds():
    # For perspective families, C_{b1} box C_{b2} = C_{b1+b2}: merging log-cosh
    # makers of depths b1, b2 is the log-cosh maker of depth b1 + b2.
    b1, b2, size = 4.0, 9.0, 2.5
    makers = [LogCoshMaker(b=b1, fee=0.0), LogCoshMaker(b=b2, fee=0.0)]
    fills, _ = route(makers, size)
    total = sum(m.trade_cost(s) for m, s in zip(makers, fills))
    merged = LogCoshMaker(b=b1 + b2, fee=0.0)
    assert total == pytest.approx(merged.trade_cost(size), abs=1e-9)
    # and the split is proportional to depth
    assert fills[0] / fills[1] == pytest.approx(b1 / b2, rel=1e-6)


def test_small_trades_route_sparsely_to_the_cheapest_quote():
    cheap = LogCoshMaker(b=10.0, fee=0.01)
    dear = LogCoshMaker(b=10.0, fee=0.05)
    fills, p_star = route([cheap, dear], 0.2)
    assert fills[0] == pytest.approx(0.2, abs=1e-9)
    assert fills[1] == 0.0
    assert p_star < dear.ask  # clearing price never entered the dear band
    # a large enough trade walks through the dear maker's quote too
    fills_big, _ = route([cheap, dear], 8.0)
    assert fills_big[1] > 0.0


def test_bounded_worst_case_loss():
    rng = np.random.default_rng(0)
    b = 6.0
    for _ in range(200):
        m = LogCoshMaker(b=b, fee=0.0)
        collected = sum(m.apply_fill(s) for s in rng.normal(scale=3.0, size=50))
        for settle in (-1.0, 1.0):
            loss = m.q * settle - collected
            assert loss <= m.worst_case_loss() + 1e-9


def test_consolidated_book_is_monotone_and_flat_on_quote_bands():
    makers = [
        LogCoshMaker(b=5.0, fee=0.02, q0=0.0),
        LogCoshMaker(b=8.0, fee=0.04, q0=0.0),
        LogCoshMaker(b=6.0, fee=0.03, q0=0.1),
    ]
    grid = np.linspace(-0.9, 0.9, 3001)
    book = consolidated_book(makers, grid)
    assert np.all(np.diff(book) >= -1e-12)  # monotone supply
    # where every band covers p, the book is exactly flat at zero net supply
    inside_all = (grid > max(m.bid for m in makers)) & (grid < min(m.ask for m in makers))
    assert inside_all.any()
    assert np.allclose(book[inside_all], 0.0)


def test_capacity_error():
    with pytest.raises(ValueError):
        route([LogCoshMaker(b=1.0, fee=0.0)], 1e6)
