"""Tests over structured cases, and certificates rather than booleans.

Uniform sampling concentrates on a problem's easy interior. Every error the
audits found lived where a uniform sampler almost never goes: the far side of a
threshold, a violation of exactly zero, a tie, a matrix on a cone's boundary.
These sweep those places on purpose.

They also check the certificates. A routine that returns a boolean asks to be
trusted; one that returns a witness, a duality gap and a KKT residual can be
checked, which is the difference the review asked for.
"""

import numpy as np
import pytest

from mechanisms.fee_routing import route, route_certificate
from mechanisms.nonconvex_maker import (
    chord_bounds,
    lower_convex_envelope,
    max_sure_profit,
    sure_profit_certificate,
)
from oracles.generators import (
    binary_support,
    controlled_violation_cost,
    multi_well_cost,
    near_singular_correlation,
    non_attained_threshold_cost,
    tied_makers,
)

GRID = np.linspace(-6.0, 6.0, 2401)


# --------------------------------------------------------------------------- #
# controlled violation: sweep the coherence boundary rather than sample near it
# --------------------------------------------------------------------------- #
def test_coherence_boundary_is_crossed_not_approached():
    for excess in (-0.5, -0.05, -1e-4):
        assert max_sure_profit(GRID, controlled_violation_cost(GRID, excess)) < 0
    # Exactly on the boundary the profit is zero, which the strong-arbitrage
    # convention counts as coherent. Random costs never land here.
    assert max_sure_profit(GRID, controlled_violation_cost(GRID, 0.0)) == pytest.approx(
        0.0, abs=1e-9)
    for excess in (1e-4, 0.05, 0.5):
        assert max_sure_profit(GRID, controlled_violation_cost(GRID, excess)) > 0


def test_the_fee_that_restores_coherence_is_the_excess():
    # The violating slope exceeds the hull by `excess`, so a fee of `excess`
    # exactly cancels it and anything less leaves a profit.
    for excess in (0.05, 0.2, 0.75):
        cost = controlled_violation_cost(GRID, excess)
        assert max_sure_profit(GRID, cost, fee=excess * 0.99) > 0
        assert max_sure_profit(GRID, cost, fee=excess) == pytest.approx(0.0, abs=1e-6)
        assert max_sure_profit(GRID, cost, fee=excess * 1.01) <= 1e-9


# --------------------------------------------------------------------------- #
# multi-well costs: several components of {C > C-hat}
# --------------------------------------------------------------------------- #
def test_multi_well_costs_have_several_envelope_components():
    cost = multi_well_cost(GRID, n_wells=3, depth=1.0)
    gap = cost - lower_convex_envelope(GRID, cost)
    above = gap > 1e-9
    # Count maximal runs of "strictly above the envelope".
    boundaries = int(np.sum(np.diff(above.astype(int)) == 1))
    assert boundaries >= 2, "expected more than one excluded inventory range"
    # Each component is bounded by contact points, and within a component the
    # envelope is affine, which is the per-component form M4 restricts to.
    starts = np.flatnonzero(np.diff(above.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(above.astype(int)) == -1)
    for lo, hi in zip(starts, ends):
        if hi - lo < 10:
            continue
        segment = lower_convex_envelope(GRID, cost)[lo:hi + 2]
        second = np.diff(segment, 2)
        assert np.max(np.abs(second)) < 1e-9      # affine on the component


# --------------------------------------------------------------------------- #
# non-attained thresholds
# --------------------------------------------------------------------------- #
def test_a_non_attained_supremum_is_visible_as_window_dependence():
    # C' = 0.6 tanh x approaches +-0.6 without reaching it, so at mu = 0.6 the
    # supremum is finite but attained nowhere and the grid answer chases the
    # window. Inside the domain it does not.
    widest = 160.0
    for mu, chases_the_window in ((0.6, True), (0.3, False)):
        located = []
        for half in (10.0, 40.0, widest):
            xs = np.arange(-half, half + 0.005, 0.005)
            profit = mu * xs - non_attained_threshold_cost(xs)
            located.append(float(xs[int(np.argmax(profit))]))
        drift = max(located) - min(located)
        # Either the maximiser moves with the window (no attained optimum) or
        # it sits still (an attained one). Measure against the window itself.
        assert (drift > 0.5 * widest) == chases_the_window
        if not chases_the_window:
            assert drift < 1e-6


# --------------------------------------------------------------------------- #
# ties
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_tied_makers_split_evenly_and_carry_no_residual(n):
    makers = tied_makers(n=n, b=1.0, fee=0.1)
    for size in (0.4, 3.0, 9.0):
        fills, _ = route(makers, size)
        np.testing.assert_allclose(fills, np.full(n, size / n), atol=1e-7)
        # No maker absorbs the whole rounding residual.
        assert float(np.max(fills) - np.min(fills)) < 1e-7
        assert float(np.sum(fills)) == pytest.approx(size, abs=1e-9)


# --------------------------------------------------------------------------- #
# near-singular PSD, and exhaustive binary support
# --------------------------------------------------------------------------- #
def test_near_singular_correlations_stay_on_the_right_side_of_the_cone():
    for slack in (1e-2, 1e-4, 1e-6, 1e-9):
        matrix, rho = near_singular_correlation(3, slack)
        assert np.linalg.eigvalsh(matrix).min() == pytest.approx(slack, rel=1e-6)
        assert rho == pytest.approx((slack - 1.0) / 2.0)


def test_binary_support_is_enumerated_not_sampled():
    outcomes, payoffs, pairs = binary_support(3)
    assert outcomes.shape == (8, 3)               # every assignment, not a sample
    assert payoffs.shape == (8, 3)
    # The products take only two distinct values in aggregate: 3 when all three
    # signs agree, -1 otherwise. That is the triangle inequality's content.
    sums = payoffs.sum(axis=1)
    assert set(np.unique(sums)) == {-1.0, 3.0}
    assert pairs == [(0, 1), (0, 2), (1, 2)]


# --------------------------------------------------------------------------- #
# certificates rather than booleans
# --------------------------------------------------------------------------- #
def test_routing_certificate_carries_a_checkable_optimality_proof():
    cases = [
        ([(1.0, 0.1), (3.0, 0.05), (0.5, 0.3)], 2.0),
        ([(1.0, 0.2)] * 3, 1.0),
        ([(2.0, 0.0)], 5.0),
        ([(1.0, 0.5), (1.0, 0.5)], 0.0),
    ]
    from mechanisms.fee_routing import LogCoshMaker
    for specs, size in cases:
        cert = route_certificate([LogCoshMaker(b, f) for b, f in specs], size)
        assert cert.is_optimal()
        # The three certificates say different things, so check each.
        assert abs(cert.duality_gap) < 1e-9       # no primal-dual slack
        assert cert.kkt_residual < 1e-9           # p* is in every subdifferential
        assert abs(cert.feasibility_error) < 1e-9
        # And the primal cost is really the cost of the fills it reports.
        recomputed = sum(LogCoshMaker(b, f).trade_cost(s)
                         for (b, f), s in zip(specs, cert.fills))
        assert recomputed == pytest.approx(cert.primal_cost, abs=1e-12)


def test_sure_profit_certificate_reports_the_fee_that_would_close_it():
    for excess in (0.05, 0.2, 0.75):
        cost = controlled_violation_cost(GRID, excess)
        cert = sure_profit_certificate(GRID, cost)
        assert cert.max_profit > 0
        # required_fee is the per-unit excess, so charging it removes the profit.
        assert cert.required_fee == pytest.approx(excess, abs=1e-6)
        assert max_sure_profit(GRID, cost, fee=cert.required_fee) <= 1e-6


def test_chord_bounds_certify_the_contact_test():
    # L <= U is the contact condition. Check it agrees with the envelope gap
    # rather than trusting either alone.
    cost = multi_well_cost(GRID, n_wells=3)
    gap = cost - lower_convex_envelope(GRID, cost)
    for i in range(50, GRID.size - 50, 137):
        lower, upper = chord_bounds(GRID, cost, i)
        assert (lower <= upper + 1e-9) == bool(gap[i] <= 1e-9)
