"""Propositions 1, 3, 10 and 11 against one exact linear program.

On a finite outcome space "is there a portfolio priced below its worst payoff"
is a linear program, so it can be *answered* rather than searched for. The same
LP settles the chord condition (Prop 1), the fee that prices a violation
(Prop 3), the PSD-versus-binary gap (Prop 10) and separation (Prop 11), and its
optimal ``y`` is the separating functional the propositions are about.

Every threshold here is bracketed at ``eps - d``, ``eps`` and ``eps + d`` rather
than sampled on one side, because sampling one side of a threshold is how the
``C(q) = 100q`` error survived the previous suite.
"""

import itertools

import numpy as np
import pytest

from mechanisms import nonconvex_maker as nm
from oracles.finite_lp import arbitrage_depth_lp, arbitrage_lp

pytest.importorskip("scipy")


# --------------------------------------------------------------------------- #
# Proposition 11: the LP's optimum is a separating functional
# --------------------------------------------------------------------------- #
def test_lp_optimum_is_a_checkable_separating_portfolio():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_out = int(rng.integers(2, 6))
        n_sec = int(rng.integers(1, 4))
        payoffs = rng.normal(0.0, 1.0, (n_out, n_sec))
        prices = rng.normal(0.0, 1.0, n_sec)
        cert = arbitrage_lp(prices, payoffs)
        # The certificate must re-price from scratch to what it claims.
        assert cert.verify(prices, payoffs)
        if cert.exists:
            worst = float(np.min(payoffs @ cert.portfolio))
            assert worst - float(prices @ cert.portfolio) > 1e-9
        # Prices inside the payoff hull never admit arbitrage: a convex
        # combination of the payoff rows is priced at its own expectation.
        weights = rng.dirichlet(np.ones(n_out))
        inside = weights @ payoffs
        assert arbitrage_lp(inside, payoffs).margin <= 1e-7


# --------------------------------------------------------------------------- #
# Propositions 1 and 3: the chord condition, and the fee that prices it
# --------------------------------------------------------------------------- #
def test_scalar_chord_condition_agrees_with_the_lp():
    # One security settling in [-1, 1]. A fill of s from q is the portfolio
    # y = s, priced at the chord cost. Prop 1 says arbitrage-free iff every
    # chord slope lies in [-1, 1].
    payoffs = np.array([[-1.0], [1.0]])
    rng = np.random.default_rng(1)
    for _ in range(400):
        slope = float(rng.uniform(-2.5, 2.5))
        cert = arbitrage_lp([slope], payoffs)
        assert cert.exists == (abs(slope) > 1.0 + 1e-9)
        if cert.exists:
            assert cert.margin == pytest.approx(abs(slope) - 1.0, abs=1e-7)


def test_fee_threshold_is_exactly_the_arbitrage_depth_bracketed():
    payoffs = np.array([[-1.0], [1.0]])
    for slope in (1.4, 2.0, 5.0):
        depth = arbitrage_depth_lp([slope], payoffs)
        assert depth == pytest.approx(slope - 1.0, abs=1e-9)
        for delta in (1e-3, 1e-5):
            # Below the depth: arbitrage survives.
            assert arbitrage_lp([slope], payoffs, fee=depth - delta).exists
            # Above it: gone. At it: the margin is exactly zero, the boundary
            # case the strong-arbitrage convention counts as coherent.
            assert not arbitrage_lp([slope], payoffs, fee=depth + delta).exists
        assert arbitrage_lp([slope], payoffs, fee=depth).margin == pytest.approx(
            0.0, abs=1e-7)


def test_grid_maker_coherence_agrees_with_the_lp_at_every_chord():
    # Cross-check the library's O(n) scan against the LP, chord by chord.
    payoffs = np.array([[-1.0], [1.0]])
    rng = np.random.default_rng(2)
    for _ in range(150):
        xs = np.unique(np.sort(rng.uniform(-3.0, 3.0, 8)))
        if xs.size < 3:
            continue
        ys = rng.normal(0.0, 1.5, xs.size)
        fee = float(rng.uniform(0.0, 0.8))
        library = nm.max_sure_profit(xs, ys, fee)
        worst = -np.inf
        for i in range(xs.size):
            for j in range(xs.size):
                s = xs[j] - xs[i]
                if s == 0.0:
                    continue
                # Price of the fill, per unit of position, as the LP sees it.
                unit_price = (ys[j] - ys[i]) / s
                cert = arbitrage_lp([unit_price], payoffs, fee=fee)
                worst = max(worst, cert.margin * abs(s))
        assert library == pytest.approx(worst, abs=1e-6)


# --------------------------------------------------------------------------- #
# Proposition 10: PSD is necessary but not sufficient for binary coherence
# --------------------------------------------------------------------------- #
def _pairwise_payoffs():
    """Payoffs of the three products X_i X_j over all +-1 assignments."""
    outcomes = np.array(list(itertools.product([-1, 1], repeat=3)))
    return np.array([[o[0] * o[1], o[0] * o[2], o[1] * o[2]] for o in outcomes],
                    float)


def test_psd_correlations_can_still_be_arbitraged_by_a_binary_trader():
    # The cut polytope is strictly inside the PSD cone. Equicorrelation rho on
    # three +-1 variables is PSD for rho >= -1/2, but binary-realizable only
    # for rho >= -1/3, where the triangle inequality rho12+rho13+rho23 >= -1
    # binds. Between the two, the quote is PSD and arbitrageable.
    payoffs = _pairwise_payoffs()
    for rho in (-0.49, -0.45, -0.40, -0.35):
        matrix = np.full((3, 3), rho)
        np.fill_diagonal(matrix, 1.0)
        assert np.linalg.eigvalsh(matrix).min() > -1e-12          # PSD
        cert = arbitrage_lp([rho] * 3, payoffs)
        assert cert.exists                                        # yet arbitraged
        assert cert.margin == pytest.approx(-1.0 / 3.0 - rho, abs=1e-7)
        # The separating functional is the triangle inequality itself.
        np.testing.assert_allclose(cert.portfolio, np.full(3, 1 / 3), atol=1e-6)
        assert cert.verify([rho] * 3, payoffs)


def test_the_cut_polytope_boundary_is_bracketed():
    payoffs = _pairwise_payoffs()
    threshold = -1.0 / 3.0
    for delta in (1e-3, 1e-5):
        assert arbitrage_lp([threshold - delta] * 3, payoffs).exists
        assert not arbitrage_lp([threshold + delta] * 3, payoffs).exists
    assert arbitrage_lp([threshold] * 3, payoffs).margin == pytest.approx(
        0.0, abs=1e-7)


def test_correlations_inside_the_cut_polytope_are_coherent():
    payoffs = _pairwise_payoffs()
    rng = np.random.default_rng(3)
    for _ in range(300):
        # Sample an actual binary distribution, so the quote is realizable by
        # construction and no arbitrage can exist.
        weights = rng.dirichlet(np.ones(payoffs.shape[0]))
        quote = weights @ payoffs
        assert arbitrage_lp(quote, payoffs).margin <= 1e-7
