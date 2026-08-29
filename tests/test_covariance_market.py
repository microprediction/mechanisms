"""Theorem tests for research/covariance-market.md.

The Gaussian-family maker quotes PSD prices by construction, its gradient is
the quoted moment matrix, the myopic optimum is truthful with Stein's loss as
the collected edge, liquidity adds across makers, a coverage-restricted maker
quotes the Dempster (max-entropy) completion, and a nuclear-norm fee creates
a spectral no-trade band.
"""

import numpy as np
import pytest

from mechanisms.covariance_market import GaussianCovarianceMaker


def _spd(rng, d, scale=1.0):
    A = rng.normal(size=(d, d))
    return scale * (A @ A.T + d * np.eye(d))


def _rand_sym(rng, d, scale=1.0):
    S = rng.normal(size=(d, d)) * scale
    return 0.5 * (S + S.T)


def test_prices_stay_positive_definite_along_random_paths():
    rng = np.random.default_rng(0)
    d = 4
    mm = GaussianCovarianceMaker(_spd(rng, d), b=5.0)
    for _ in range(50):
        S = _rand_sym(rng, d, scale=0.3)
        if np.isfinite(mm.trade_cost(S)):
            mm.apply_fill(S)
        vals = np.linalg.eigvalsh(mm.price)
        assert np.all(vals > 0)


def test_incoherent_prices_are_unreachable():
    mm = GaussianCovarianceMaker(np.eye(2), b=1.0)
    # a fill large enough to push the precision matrix indefinite is refused
    with pytest.raises(ValueError):
        mm.apply_fill(np.diag([10.0, 0.0]))


def test_gradient_of_cost_is_the_quoted_matrix():
    rng = np.random.default_rng(1)
    d = 3
    mm = GaussianCovarianceMaker(_spd(rng, d), b=2.0)
    mm.apply_fill(_rand_sym(rng, d, scale=0.1))
    S, h = _rand_sym(rng, d), 1e-6
    numeric = (mm.cost(mm.Q + h * S) - mm.cost(mm.Q - h * S)) / (2 * h)
    assert numeric == pytest.approx(float(np.sum(mm.price * S)), rel=1e-5)


def test_truthful_quote_is_myopically_optimal_and_earns_stein_loss():
    rng = np.random.default_rng(2)
    d = 3
    sigma0 = _spd(rng, d)
    truth = _spd(rng, d)  # believed second-moment matrix
    b = 3.0
    mm = GaussianCovarianceMaker(sigma0, b=b)

    def expected_profit(sigma_quote):
        S = mm.fill_to(sigma_quote)
        return float(np.sum(truth * S)) - mm.trade_cost(S)

    best = expected_profit(truth)
    # moving the quote to the belief beats nearby alternatives
    for _ in range(200):
        alt = truth + _rand_sym(rng, d, scale=0.2)
        if np.all(np.linalg.eigvalsh(alt) > 0):
            assert expected_profit(alt) <= best + 1e-9
    # and the collected edge is b times Stein's loss of the standing quote
    assert best == pytest.approx(
        b * GaussianCovarianceMaker.stein_divergence(truth, sigma0), rel=1e-9
    )


def test_liquidity_adds_across_makers():
    rng = np.random.default_rng(3)
    d = 3
    sigma0 = _spd(rng, d)
    b1, b2 = 2.0, 5.0
    S = _rand_sym(rng, d, scale=0.4)
    split = GaussianCovarianceMaker(sigma0, b=b1).trade_cost(b1 / (b1 + b2) * S) + \
        GaussianCovarianceMaker(sigma0, b=b2).trade_cost(b2 / (b1 + b2) * S)
    merged = GaussianCovarianceMaker(sigma0, b=b1 + b2).trade_cost(S)
    assert split == pytest.approx(merged, rel=1e-10)
    # the depth-proportional split is optimal among random splits
    for _ in range(200):
        T = _rand_sym(rng, d, scale=0.2)
        alt = GaussianCovarianceMaker(sigma0, b=b1).trade_cost(b1 / (b1 + b2) * S + T) + \
            GaussianCovarianceMaker(sigma0, b=b2).trade_cost(b2 / (b1 + b2) * S - T)
        assert alt >= merged - 1e-9


def test_restricted_coverage_quotes_the_dempster_completion():
    # A maker starting diagonal and trading only on a subgraph of entries has
    # precision supported on that subgraph: unquoted partial correlations are
    # exactly zero — the max-entropy (covariance selection) completion.
    rng = np.random.default_rng(4)
    d = 4
    mm = GaussianCovarianceMaker(np.diag(rng.uniform(0.5, 2.0, size=d)), b=2.0)
    edges = [(0, 1), (1, 2)]  # quoted off-diagonal pairs
    for _ in range(20):
        S = np.zeros((d, d))
        for i in range(d):
            S[i, i] = rng.normal() * 0.1
        for i, j in edges:
            S[i, j] = S[j, i] = rng.normal() * 0.1
        if np.isfinite(mm.trade_cost(S)):
            mm.apply_fill(S)
    precision = mm.precision
    off_graph = [(0, 2), (0, 3), (1, 3), (2, 3)]
    for i, j in off_graph:
        assert precision[i, j] == pytest.approx(0.0, abs=1e-12)
    # the quoted covariance is generally dense there regardless
    assert np.any(np.abs(mm.price[0, 2]) > 1e-8) or np.any(np.abs(mm.price[1, 3]) > 1e-8)


def test_nuclear_fee_creates_a_spectral_no_trade_band():
    rng = np.random.default_rng(5)
    d = 3
    sigma0 = _spd(rng, d)
    fee = 0.05
    mm = GaussianCovarianceMaker(sigma0, b=4.0, fee=fee)

    def profitable(truth):
        vals, vecs = np.linalg.eigh(truth - mm.price)
        k = int(np.argmax(np.abs(vals)))
        best = -np.inf
        for t in np.linspace(-0.5, 0.5, 41):
            S = t * np.outer(vecs[:, k], vecs[:, k])
            c = mm.trade_cost(S)
            if np.isfinite(c):
                best = max(best, float(np.sum(truth * S)) - c)
        for _ in range(200):
            S = _rand_sym(rng, d, scale=0.1)
            c = mm.trade_cost(S)
            if np.isfinite(c):
                best = max(best, float(np.sum(truth * S)) - c)
        return best > 1e-10

    # mispricing inside the spectral band: no profitable trade
    G = _rand_sym(rng, d)
    G *= 0.5 * fee / max(np.abs(np.linalg.eigvalsh(G)))
    assert not profitable(mm.price + G)
    # mispricing outside the band: a rank-one trade in the top eigendirection pays
    G2 = _rand_sym(rng, d)
    G2 *= 3.0 * fee / max(np.abs(np.linalg.eigvalsh(G2)))
    assert profitable(mm.price + G2)
