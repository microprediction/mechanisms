"""Split the tests by dual domain, and flag optima that ran out of window.

A finite grid silently bounds a quantifier. Where the true supremum is attained
outside the window, or not attained at all, a grid search still returns *some*
index, and an assertion about that index can hold for reasons that have nothing
to do with the claim under test.

The wiggly cost used throughout the suite is a case in point. With
``C' = 0.6 tanh(x) + 0.35 sin(x)`` the *maximum* slope is 0.95, but the
*asymptotic* slope is 0.6: as ``x -> +inf`` the derivative oscillates in
``[0.25, 0.95]`` about 0.6, so ``mu x - C(x) ~ (mu - 0.6) x``. The dual domain
is therefore ``|mu| < 0.6``, not ``|mu| < 0.95``, and for ``mu`` between the two
the supremum is ``+inf`` with no maximiser anywhere. These tests split on that
boundary instead of sampling across it.
"""

import numpy as np
import pytest

from mechanisms.nonconvex_maker import lower_convex_envelope, moreau_envelope

ASYMPTOTIC_SLOPE = 0.6      # lim of C' at infinity: the dual domain's edge
MAX_SLOPE = 0.95            # sup of C': larger, and not the relevant bound


def wiggly_cost(xs):
    """``C`` with ``C'(x) = 0.6 tanh x + 0.35 sin x``, on the given grid."""
    dx = xs[1] - xs[0]
    slope = 0.6 * np.tanh(xs) + 0.35 * np.sin(xs)
    values = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * dx)])
    return values - values[len(xs) // 2]


def _argmax_profit(xs, cost, mu):
    profit = mu * xs - cost
    j = int(np.argmax(profit))
    at_edge = j < 3 or j > xs.size - 4
    return j, float(profit[j]), at_edge


# --------------------------------------------------------------------------- #
# inside the dual domain: the optimum is stable as the window grows
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mu", [0.0, 0.2, 0.4, 0.55, -0.55])
def test_inside_the_dual_domain_the_optimum_does_not_move_with_the_window(mu):
    assert abs(mu) < ASYMPTOTIC_SLOPE - 1e-9      # strictly inside
    # Fixed mesh, expanding window: only the window varies, so any drift in
    # the answer is the window's doing and not the mesh's.
    dx = 0.005
    located, values = [], []
    for half in (12.0, 40.0, 200.0):
        xs = np.arange(-half, half + dx / 2, dx)
        j, value, at_edge = _argmax_profit(xs, wiggly_cost(xs), mu)
        assert not at_edge, "optimum ran out of window inside the dual domain"
        located.append(float(xs[j]))
        values.append(value)
    # The maximiser and the value are properties of C, not of the window.
    assert max(located) - min(located) < 1e-6
    assert max(values) - min(values) < 1e-9


@pytest.mark.parametrize("mu", [0.0, 0.3, 0.55, -0.4])
def test_inside_the_dual_domain_the_optimum_lands_on_the_contact_set(mu):
    xs = np.linspace(-40.0, 40.0, 8001)
    cost = wiggly_cost(xs)
    gap = cost - lower_convex_envelope(xs, cost)
    j, _, at_edge = _argmax_profit(xs, cost, mu)
    assert not at_edge
    assert gap[j] == pytest.approx(0.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# outside it: the supremum diverges, and a grid hides that
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mu", [0.62, 0.8, 0.9])
def test_outside_the_dual_domain_the_supremum_diverges(mu):
    # mu sits above the asymptotic slope but below the maximum slope, so C'
    # still equals mu at arbitrarily large x while the profit runs away.
    assert ASYMPTOTIC_SLOPE < mu < MAX_SLOPE
    dx = 0.01
    values = []
    for half in (12.0, 60.0, 300.0, 1500.0):
        xs = np.arange(-half, half + dx / 2, dx)
        _, value, _ = _argmax_profit(xs, wiggly_cost(xs), mu)
        values.append(value)
    for smaller, larger in zip(values, values[1:]):
        assert larger > smaller                  # no convergence, ever
    # It grows like (mu - 0.6) * half, so widening 125-fold multiplies it.
    assert values[-1] > 20 * values[0]

    # And this is exactly what a fixed window conceals: at +-12 the "optimum"
    # is pinned to the boundary, which a test that only checks the index would
    # read as a perfectly good answer.
    xs = np.arange(-12.0, 12.0 + dx / 2, dx)
    _, _, at_edge = _argmax_profit(xs, wiggly_cost(xs), mu)
    assert at_edge


def test_the_dual_domain_edge_is_the_asymptotic_not_the_maximum_slope():
    # Sampling up to the maximum slope, as an earlier version of the suite did,
    # puts most draws outside the dual domain. Quantify that rather than
    # assert it.
    xs = np.linspace(-12.0, 12.0, 6001)
    cost = wiggly_cost(xs)
    rng = np.random.default_rng(0)
    naive = [_argmax_profit(xs, cost, rng.uniform(-0.9, 0.9))[2] for _ in range(400)]
    careful = [_argmax_profit(xs, cost, rng.uniform(-0.55, 0.55))[2]
               for _ in range(400)]
    assert sum(naive) > 100          # a third of the draws hit the boundary
    assert sum(careful) == 0         # none do inside the dual domain


# --------------------------------------------------------------------------- #
# the Moreau curvature test, with the argmin recorded
# --------------------------------------------------------------------------- #
def _moreau_argmin(xs, cost, lam):
    """Where the grid minimisation actually put the minimiser, per point."""
    block = xs[:, None] - xs[None, :]
    objective = cost[None, :] + block * block / (2.0 * lam)
    return np.argmin(objective, axis=1)


def test_moreau_curvature_is_measured_on_a_fixed_mesh_with_boundary_flagged():
    # Curvature must be normalised by dx^2 to be comparable across windows, and
    # any point whose minimiser sits at the window edge must be excluded: its
    # value is a truncation artifact, not a property of the envelope.
    lam = 0.5
    dx = 0.02                                   # fixed mesh, expanding window
    verdicts = []
    for half in (6.0, 12.0, 24.0):
        xs = np.arange(-half, half + dx / 2, dx)
        cost = wiggly_cost(xs)
        env = moreau_envelope(xs, cost, lam)
        argmin = _moreau_argmin(xs, cost, lam)
        interior = (argmin > 2) & (argmin < xs.size - 3)
        interior[:2] = interior[-2:] = False
        curvature = np.full(xs.size, np.nan)
        curvature[1:-1] = np.diff(env, 2) / dx ** 2
        usable = interior & np.isfinite(curvature)
        verdicts.append(float(np.nanmin(curvature[usable])))
        # Every point we trust must have found an interior minimiser.
        assert usable.sum() > 0.5 * xs.size

    # The interior verdict is stable across windows, which is the whole point:
    # an earlier claim that the Moreau envelope "convexifies" this cost came
    # from reading a +3.6e-8 curvature on a +-12 window as a positive number.
    assert max(verdicts) - min(verdicts) < 0.05
    assert all(v < -1e-3 for v in verdicts)     # genuinely still non-convex
