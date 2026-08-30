"""Check the library against independent exact oracles, not against itself.

The errors that survived the earlier suite were of one kind: a test compared a
discretisation with itself, or with a refinement of itself, and so could not see
an error in what the discretisation was discretising. These tests compare each
routine with a separate implementation in ``tests/oracles`` that computes the
same quantity from the definition by an unrelated algorithm.

Where the library is deliberately a grid approximation, the test says so and
checks the *direction* and *rate* of the discretisation error rather than
pretending the two should agree.
"""

import numpy as np
import pytest

from mechanisms import nonconvex_maker as nm
from oracles.piecewise_linear import (
    exact_chord_bounds,
    exact_lower_convex_envelope,
    exact_max_sure_profit,
    exact_moreau_envelope,
)


def _random_grid(rng, lo=-4.0, hi=4.0, min_n=3, max_n=25):
    n = int(rng.integers(min_n, max_n))
    xs = np.unique(np.sort(rng.uniform(lo, hi, n)))
    while xs.size < min_n:
        xs = np.unique(np.sort(rng.uniform(lo, hi, n + 5)))
    return xs, rng.normal(0.0, 2.0, xs.size)


def test_lower_convex_envelope_matches_the_linear_program():
    # The library walks a monotone chain; the oracle enumerates the pairs of
    # knots that can carry an optimal basic solution of the defining LP.
    rng = np.random.default_rng(0)
    for _ in range(300):
        xs, ys = _random_grid(rng)
        np.testing.assert_allclose(nm.lower_convex_envelope(xs, ys),
                                   exact_lower_convex_envelope(xs, ys), atol=1e-12)


def test_chord_bounds_match_exhaustive_enumeration():
    # On a piecewise-linear cost d(s) = b + a/s is monotone on each piece, so
    # every extremum sits at a knot and the enumeration is exact, not a sample.
    rng = np.random.default_rng(1)
    for _ in range(200):
        xs, ys = _random_grid(rng, max_n=20)
        for i in range(xs.size):
            got = nm.chord_bounds(xs, ys, i)
            want = exact_chord_bounds(xs, ys, i)
            for a, b in zip(got, want):
                if np.isfinite(a) or np.isfinite(b):
                    assert a == pytest.approx(b, abs=1e-12, rel=1e-12)
                else:
                    assert a == b               # both infinite, same sign


def test_max_sure_profit_matches_brute_force():
    # O(n) running scan against O(n^2) enumeration of every ordered pair.
    rng = np.random.default_rng(2)
    for _ in range(500):
        xs, ys = _random_grid(rng, max_n=20)
        fee = float(rng.uniform(0.0, 1.0))
        assert nm.max_sure_profit(xs, ys, fee) == pytest.approx(
            exact_max_sure_profit(xs, ys, fee), abs=1e-12)


def test_sure_profit_certificate_witness_reproduces_its_own_claim():
    # A certificate that names a chord must be checkable by re-pricing that
    # chord. This is what distinguishes a certificate from a boolean.
    rng = np.random.default_rng(3)
    for _ in range(300):
        xs, ys = _random_grid(rng, max_n=15)
        fee = float(rng.uniform(0.0, 0.6))
        cert = nm.sure_profit_certificate(xs, ys, fee)
        s = xs[cert.end_index] - xs[cert.start_index]
        worst_payoff = (-1.0 if s > 0 else 1.0) * s
        realised = (worst_payoff - (ys[cert.end_index] - ys[cert.start_index])
                    - fee * abs(s))
        assert realised == pytest.approx(cert.max_profit, abs=1e-12)


def test_moreau_envelope_is_a_grid_approximation_from_above():
    # The library minimises over sampled points, the oracle over the continuum,
    # so the library's answer is an upper bound. Asserting equality here would
    # be the self-comparison this file exists to avoid.
    xs = np.linspace(-6.0, 6.0, 241)
    ys = np.log(np.cosh(xs)) + 0.4 * np.sin(3.0 * xs)
    for lam in (0.05, 0.5, 2.0):
        approx = nm.moreau_envelope(xs, ys, lam)
        exact = exact_moreau_envelope(xs, ys, lam)
        assert np.all(approx >= exact - 1e-12)          # minimising over less
        assert np.max(approx - exact) < 1e-2


def test_moreau_grid_error_vanishes_as_the_mesh_refines():
    # And the gap closes at the rate a grid minimisation should: the error is
    # O(dx^2 / lam), since the minimand is locally quadratic with curvature
    # 1/lam. Halving the mesh should quarter it.
    lam = 0.5
    errors = []
    for n in (121, 241, 481, 961):
        xs = np.linspace(-6.0, 6.0, n)
        ys = np.log(np.cosh(xs)) + 0.4 * np.sin(3.0 * xs)
        errors.append(float(np.max(nm.moreau_envelope(xs, ys, lam)
                                   - exact_moreau_envelope(xs, ys, lam))))
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse
        assert 2.5 < coarse / fine < 6.0                # quadratic, near enough


def test_oracles_disagree_with_a_deliberately_broken_implementation():
    # A test suite that cannot fail proves nothing. Perturb the envelope by a
    # single knot and check the oracle notices, so we know the comparison has
    # teeth rather than both sides being trivially equal.
    xs = np.linspace(-3.0, 3.0, 25)
    ys = xs ** 2 - 2.0 * np.cos(3.0 * xs)
    good = nm.lower_convex_envelope(xs, ys)
    np.testing.assert_allclose(good, exact_lower_convex_envelope(xs, ys), atol=1e-12)
    broken = good.copy()
    broken[12] += 0.5
    assert not np.allclose(broken, exact_lower_convex_envelope(xs, ys), atol=1e-12)
