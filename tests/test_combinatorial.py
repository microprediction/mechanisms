"""Tests for combinatorial / conditional LMSR markets."""

import numpy as np
import pytest

from mechanisms.combinatorial import CombinatorialMarket


def test_init_uniform_and_independent():
    m = CombinatorialMarket(n_vars=3, b=50.0)
    assert np.allclose(m.prices(), 1.0 / 8)
    for i in range(3):
        assert m.marginal(i) == pytest.approx(0.5)
    # independent at init: P(X0=1 | X1=1) == P(X0=1)
    assert m.conditional(m.var(0), m.var(1)) == pytest.approx(0.5)


def test_event_probabilities_are_coherent():
    m = CombinatorialMarket(n_vars=3, b=40.0)
    m.buy_event(m.var(0) & m.var(1), 30.0)
    a, b = m.var(0), m.var(1)
    # P(A and B) = P(A|B) P(B)
    assert m.prob(a & b) == pytest.approx(m.conditional(a, b) * m.prob(b), rel=1e-9)
    # complementary events sum to one
    assert m.prob(a) + m.prob(~a) == pytest.approx(1.0)


def test_buying_an_event_raises_its_probability():
    m = CombinatorialMarket(n_vars=2, b=30.0)
    before = m.prob(m.var(0) & m.var(1))
    m.buy_event(m.var(0) & m.var(1), 25.0)
    assert m.prob(m.var(0) & m.var(1)) > before


def test_combinatorial_security_induces_correlation():
    """Betting on the joint event makes the conditional exceed the marginal."""
    m = CombinatorialMarket(n_vars=2, b=30.0)
    m.buy_event(m.var(0) & m.var(1), 40.0)
    # X0 and X1 are now positively correlated
    assert m.conditional(m.var(0), m.var(1)) > m.marginal(0)


def test_bounded_loss():
    m = CombinatorialMarket(n_vars=3, b=100.0)
    assert m.max_loss == pytest.approx(100.0 * np.log(8))
