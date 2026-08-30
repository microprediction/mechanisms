"""Metamorphic tests: relations between runs, not values from one run.

A test that asserts a number can only be as right as the number someone wrote
down. A metamorphic test asserts a *relation* between two runs of the same code
on related inputs, so it needs no ground truth and still fails when the code is
wrong. These cover the transformations under which the maker's constructions are
equivariant, and the two composition laws the paper leans on.
"""

import numpy as np
import pytest

from mechanisms import nonconvex_maker as nm
from mechanisms.fee_routing import LogCoshMaker, route


def _wiggly(xs):
    return np.log(np.cosh(xs)) + 0.5 * np.sin(3.0 * xs)


GRID = np.linspace(-4.0, 4.0, 401)
COST = _wiggly(GRID)


# --------------------------------------------------------------------------- #
# equivariance of the convex envelope
# --------------------------------------------------------------------------- #
def test_envelope_commutes_with_translation():
    for shift in (-2.0, 0.4, 1.3):
        np.testing.assert_allclose(
            nm.lower_convex_envelope(GRID, COST),
            nm.lower_convex_envelope(GRID + shift, COST), atol=1e-12)


def test_envelope_commutes_with_tilt():
    # Adding an affine function adds it to the envelope: conjugation shifts the
    # dual variable and the contact set is unchanged.
    for slope in (-1.5, 0.7, 3.0):
        np.testing.assert_allclose(
            nm.lower_convex_envelope(GRID, COST) + slope * GRID,
            nm.lower_convex_envelope(GRID, COST + slope * GRID), atol=1e-12)


def test_envelope_commutes_with_reflection():
    reflected = nm.lower_convex_envelope(-GRID[::-1], COST[::-1])
    np.testing.assert_allclose(nm.lower_convex_envelope(GRID, COST),
                               reflected[::-1], atol=1e-12)


def test_envelope_is_positively_homogeneous_in_the_cost():
    for scale in (0.1, 2.5, 100.0):
        np.testing.assert_allclose(
            scale * nm.lower_convex_envelope(GRID, COST),
            nm.lower_convex_envelope(GRID, scale * COST), rtol=1e-12)


def test_chord_bounds_are_equivariant():
    rng = np.random.default_rng(0)
    xs = np.unique(np.sort(rng.uniform(-3.0, 3.0, 12)))
    ys = rng.normal(0.0, 1.0, xs.size)
    for i in range(xs.size):
        lo, hi = nm.chord_bounds(xs, ys, i)
        # A tilt by b moves both one-sided bounds by exactly b.
        b = 0.9
        lo_t, hi_t = nm.chord_bounds(xs, ys + b * xs, i)
        for a, c in ((lo, lo_t), (hi, hi_t)):
            if np.isfinite(a):
                assert c == pytest.approx(a + b, abs=1e-12)
        # A rescale by c > 0 scales both.
        c_scale = 3.0
        lo_s, hi_s = nm.chord_bounds(xs, c_scale * ys, i)
        for a, c in ((lo, lo_s), (hi, hi_s)):
            if np.isfinite(a):
                assert c == pytest.approx(c_scale * a, abs=1e-12)


# --------------------------------------------------------------------------- #
# the Moreau semigroup
# --------------------------------------------------------------------------- #
def test_moreau_semigroup_error_vanishes_with_the_mesh():
    # e_lam(e_mu C) = e_{lam+mu} C exactly on the continuum. The library
    # minimises over a grid, so the composed version compounds two
    # discretisation errors; the test is that the gap closes with the mesh,
    # not that it is zero at any one mesh.
    lam, mu = 0.3, 0.5
    errors = []
    for n in (201, 401, 801, 1601):
        xs = np.linspace(-4.0, 4.0, n)
        ys = _wiggly(xs)
        composed = nm.moreau_envelope(xs, nm.moreau_envelope(xs, ys, mu), lam)
        direct = nm.moreau_envelope(xs, ys, lam + mu)
        errors.append(float(np.max(np.abs(composed - direct))))
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse
    assert errors[-1] < errors[0] / 8.0


def test_moreau_envelope_is_monotone_in_depth():
    # A deeper co-quoter smooths more, so the envelope rises toward C and the
    # gap to the convex envelope narrows but never closes.
    previous = None
    for lam in (0.05, 0.2, 1.0, 5.0):
        current = nm.moreau_envelope(GRID, COST, lam)
        assert np.all(current <= COST + 1e-12)
        if previous is not None:
            assert np.all(current <= previous + 1e-9)
        previous = current


# --------------------------------------------------------------------------- #
# composition laws for the router
# --------------------------------------------------------------------------- #
def test_routing_is_invariant_to_the_order_of_makers():
    specs = [(1.0, 0.10), (3.0, 0.05), (0.5, 0.30)]
    for size in (0.3, 2.0, 6.0):
        base, price = route([LogCoshMaker(b, f) for b, f in specs], size)
        for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
            permuted, p2 = route([LogCoshMaker(*specs[i]) for i in perm], size)
            np.testing.assert_allclose(base[perm], permuted, atol=1e-9)
            assert p2 == pytest.approx(price, abs=1e-9)


def test_splitting_a_maker_in_half_preserves_the_aggregate():
    # Liquidity adds under infimal convolution: the log-cosh conjugate is
    # b * h(p), so C_b conv C_b = C_{2b} and splitting a maker into two of half
    # the depth must leave the total cost of any fill unchanged.
    for size in (0.5, 2.0, 5.0, 12.0):
        whole = LogCoshMaker(b=2.0)
        fills_whole, _ = route([whole], size)
        cost_whole = whole.trade_cost(fills_whole[0])

        halves = [LogCoshMaker(b=1.0), LogCoshMaker(b=1.0)]
        fills_halves, _ = route(halves, size)
        cost_halves = sum(m.trade_cost(s) for m, s in zip(halves, fills_halves))

        assert cost_halves == pytest.approx(cost_whole, rel=1e-9, abs=1e-12)
        assert float(np.sum(fills_halves)) == pytest.approx(size, abs=1e-9)


def test_routing_is_odd_under_reflecting_the_demand():
    # Symmetric makers at zero inventory: reversing the sign of the demand
    # reverses every fill and the clearing price.
    makers = [LogCoshMaker(1.0, 0.1), LogCoshMaker(2.0, 0.05)]
    for size in (0.4, 1.5, 4.0):
        up, p_up = route(makers, size)
        down, p_down = route(makers, -size)
        np.testing.assert_allclose(up, -down, atol=1e-8)
        assert p_up == pytest.approx(-p_down, abs=1e-8)


def test_routing_scales_with_a_common_rescaling_of_depth():
    # Scaling every maker's depth by c scales the fills by c at the same
    # clearing price, since supply is b * arctanh(...) and b enters linearly.
    specs = [(1.0, 0.1), (2.5, 0.1)]
    for c in (0.5, 3.0):
        base, p_base = route([LogCoshMaker(b, f) for b, f in specs], 2.0)
        scaled, p_scaled = route([LogCoshMaker(c * b, f) for b, f in specs], 2.0 * c)
        np.testing.assert_allclose(c * base, scaled, atol=1e-7)
        assert p_scaled == pytest.approx(p_base, abs=1e-8)
