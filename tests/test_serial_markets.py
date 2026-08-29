"""Theorem tests for research/serial-markets-and-belief-propagation.md.

Alternating model and market steps is the Kalman filter exactly; market
operations on a Gaussian chain reproduce exact posterior marginals (belief
propagation with trades as messages); inconsistent pairwise quotes around a
cycle are a sure-arbitrage certificate, consistent ones are not.
"""

import numpy as np
import pytest

from mechanisms.serial_markets import (
    chain_posterior,
    cycle_arbitrage,
    market_kalman_filter,
)


def test_market_filter_is_the_kalman_filter():
    rng = np.random.default_rng(0)
    a, q, r = 0.9, 0.4, 0.6
    mean0, var0 = 1.0, 2.0
    x = mean0
    ys = []
    for _ in range(60):
        x = a * x + rng.normal(scale=np.sqrt(q))
        ys.append(x + rng.normal(scale=np.sqrt(r)))
    means_m, vars_m = market_kalman_filter(ys, a, q, r, mean0, var0)
    # classical covariance-form Kalman recursion
    m, P = mean0, var0
    for t, y in enumerate(ys):
        m, P = a * m, a * a * P + q
        K = P / (P + r)
        m, P = m + K * (y - m), (1 - K) * P
        assert means_m[t] == pytest.approx(m, rel=1e-12)
        assert vars_m[t] == pytest.approx(P, rel=1e-12)


def test_chain_marginals_match_joint_conditioning():
    rng = np.random.default_rng(1)
    n = 5
    m0, tau0 = 0.5, 2.0
    coeffs = rng.uniform(0.5, 1.2, size=n - 1)
    noises = rng.uniform(0.2, 0.8, size=n - 1)
    rs = rng.uniform(0.3, 0.9, size=n)
    # joint distribution of the chain states
    mean = np.empty(n)
    mean[0] = m0
    Sigma = np.zeros((n, n))
    Sigma[0, 0] = 1.0 / tau0
    for i in range(n - 1):
        mean[i + 1] = coeffs[i] * mean[i]
        Sigma[i + 1, :i + 1] = coeffs[i] * Sigma[i, :i + 1]
        Sigma[:i + 1, i + 1] = Sigma[i + 1, :i + 1]
        Sigma[i + 1, i + 1] = coeffs[i] ** 2 * Sigma[i, i] + noises[i]
    ys = mean + rng.normal(size=n)  # any observation vector works
    S_yy = Sigma + np.diag(rs)
    post_mean = mean + Sigma @ np.linalg.solve(S_yy, ys - mean)
    post_cov = Sigma - Sigma @ np.linalg.solve(S_yy, Sigma)
    for node in range(n):
        mu, var = chain_posterior(node, m0, tau0, coeffs, noises, ys, rs)
        assert mu == pytest.approx(post_mean[node], rel=1e-10)
        assert var == pytest.approx(post_cov[node, node], rel=1e-10)


def test_cycle_inconsistency_is_arbitrage():
    # quotes around a 3-cycle admitting no joint distribution
    bad = np.array([[1.0, 0.9, -0.5], [0.9, 1.0, 0.9], [-0.5, 0.9, 1.0]])
    w, profit = cycle_arbitrage(bad)
    assert profit > 0.1
    assert float(w @ bad @ w) == pytest.approx(-profit, abs=1e-12)  # negative price
    x = np.random.default_rng(2).normal(size=(500, 3))
    assert np.all((x @ w) ** 2 >= 0)  # nonnegative payoff, trivially but explicitly
    # a consistent triple offers no such bundle
    good = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.9], [0.7, 0.9, 1.0]])
    assert np.all(np.linalg.eigvalsh(good) >= 0)
    w2, profit2 = cycle_arbitrage(good)
    assert w2 is None and profit2 == 0.0
