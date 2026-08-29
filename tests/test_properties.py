"""Property tests for the maker lemmas, as opposed to worked examples.

The other modules pin specific counterexamples. These randomize over cost
families and parameters and check the statements themselves against
brute-force oracles, concentrating on three fault lines: whether the
one-sided bounds are attained and whether the no-trade set is closed at
its endpoints; costs with infinite one-sided slopes, which no finite fee
can stabilize; and the bounded/unrestricted payoff-regime distinction,
which decides whether a quadratic maker is admissible at all.
"""

import numpy as np
import pytest

from mechanisms.fee_routing import LogCoshMaker, route
from mechanisms.nonconvex_maker import max_sure_profit, moreau_envelope

# Grid shared by the scalar properties. Coarse enough to keep the suite fast,
# fine enough that chord slopes are resolved.
XS = np.linspace(-6.0, 6.0, 2401)
DX = XS[1] - XS[0]


def _random_lipschitz_cost(rng, lo=-0.9, hi=0.9, n_knots=12):
    """Piecewise-linear cost whose slopes lie in [lo, hi].

    A piecewise-linear function with slopes in [lo, hi] has every chord
    slope in [lo, hi] too, since a chord slope is a convex combination of
    the slopes it spans. So these are chord-coherent by construction for a
    payoff hull containing [lo, hi], and generally non-convex.
    """
    knots = np.sort(rng.choice(np.arange(1, len(XS) - 1), n_knots, replace=False))
    slopes = rng.uniform(lo, hi, size=n_knots + 1)
    seg = np.searchsorted(knots, np.arange(len(XS)))
    C = np.concatenate([[0.0], np.cumsum(slopes[seg[:-1]] * DX)])
    return C - C[len(XS) // 2]


def _no_trade_bounds(C, i0):
    """(sup_{s<0} d_q, inf_{s>0} d_q) on the grid, as in Lemma 5."""
    s = XS - XS[i0]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (C - C[i0]) / s
    return np.nanmax(d[s < -1e-12]), np.nanmin(d[s > 1e-12])


def _max_profit(C, i0, mu, f):
    return float((mu * (XS - XS[i0]) - (C - C[i0]) - f * np.abs(XS - XS[i0])).max())


def test_lemma5_interval_matches_brute_force_over_random_costs():
    # Property: for random coherent costs, states and fees, the set of
    # beliefs admitting no profitable trade is exactly the fee-widened
    # one-sided chord interval intersected with the hull.
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(40):
        C = _random_lipschitz_cost(rng)
        i0 = int(rng.integers(300, len(XS) - 300))
        f = float(rng.choice([0.0, 0.05, 0.2, 0.6]))
        lo, hi = _no_trade_bounds(C, i0)
        a, b = lo - f, hi + f                      # predicted interval
        for mu in np.linspace(-1.0, 1.0, 41):
            predicted = (a - 1e-9) <= mu <= (b + 1e-9)
            actual = _max_profit(C, i0, mu, f) <= 1e-9
            assert predicted == actual, (mu, a, b, f)
            checked += 1
    assert checked > 1000


def test_lemma5_endpoints_are_included_and_just_outside_is_not():
    # Closedness: at an endpoint the best trade breaks even, which under the
    # strong-arbitrage convention is no arbitrage, so the endpoint belongs to
    # the no-trade set; just beyond it there is strict profit.
    rng = np.random.default_rng(12)
    for _ in range(25):
        C = _random_lipschitz_cost(rng)
        i0 = int(rng.integers(300, len(XS) - 300))
        f = float(rng.choice([0.0, 0.1, 0.35]))
        lo, hi = _no_trade_bounds(C, i0)
        a, b = lo - f, hi + f
        if not (a <= b):                            # empty interval: nothing to check
            continue
        assert _max_profit(C, i0, a, f) <= 1e-9     # left endpoint: break even
        assert _max_profit(C, i0, b, f) <= 1e-9     # right endpoint: break even
        assert _max_profit(C, i0, b + 0.02, f) > 1e-6
        assert _max_profit(C, i0, a - 0.02, f) > 1e-6


def test_infinite_one_sided_slope_defeats_every_finite_fee():
    # C(q) = -sqrt(|q|) has d_0(s) = -1/sqrt(s) -> -inf as s -> 0+, so it is
    # not chord-coherent, Proposition 3's excursion is infinite, and Lemma
    # 5's upper bound at q = 0 is -inf: the no-trade interval is empty for
    # every finite fee. At belief mu = 0 the profit from a fill s > 0 is
    # sqrt(s) - f s, maximized at s = 1/(4f^2) with value 1/(4f) > 0, so the
    # profitable trade shrinks as the fee grows but never disappears. A test
    # on a fixed grid would wrongly report stabilization once f exceeds the
    # largest slope the grid resolves, which is why the scale is chosen from
    # f here.
    xs = np.linspace(-4.0, 4.0, 40001)
    C = -np.sqrt(np.abs(xs))
    s_pos = xs[xs > 1e-12]
    assert ((-np.sqrt(s_pos)) / s_pos).min() < -50         # d_0 diverging
    assert max_sure_profit(xs, C, stride=40) > 0           # incoherent bare

    for f in (0.5, 5.0, 50.0, 500.0):
        s_star = 1.0 / (4.0 * f ** 2)                      # analytic optimum
        assert np.sqrt(s_star) - f * s_star == pytest.approx(1.0 / (4 * f), rel=1e-12)
        grid = np.linspace(s_star / 8, 8 * s_star, 2001)   # a grid that resolves it
        assert (np.sqrt(grid) - f * grid).max() > 0.9 / (4 * f)


def test_convex_kink_is_an_intrinsic_spread():
    # Lemma 4 needs C'(q), which does not exist at a kink. Lemma 5 does not:
    # at the kink of C = alpha|q| the one-sided bounds are -alpha and
    # +alpha, so the no-trade interval is [-alpha - f, alpha + f] and the
    # kink acts as a built-in half-spread alpha. Away from the kink the
    # bounds coincide and the interval collapses to Lemma 4's band.
    alpha = 0.5
    C = alpha * np.abs(XS)
    i_kink = int(np.argmin(np.abs(XS)))
    for f in (0.0, 0.1, 0.3):
        lo, hi = _no_trade_bounds(C, i_kink)
        assert lo == pytest.approx(-alpha, abs=1e-9)
        assert hi == pytest.approx(alpha, abs=1e-9)
        for mu in np.linspace(-1.0, 1.0, 41):
            predicted = (-alpha - f - 1e-9) <= mu <= (alpha + f + 1e-9)
            assert predicted == (_max_profit(C, i_kink, mu, f) <= 1e-9)
    # a differentiable state: the interval is the symmetric band m -+ f
    i_smooth = int(np.argmin(np.abs(XS - 2.0)))
    lo, hi = _no_trade_bounds(C, i_smooth)
    assert lo == pytest.approx(alpha, abs=1e-9) and hi == pytest.approx(alpha, abs=1e-9)


def test_payoff_regime_decides_whether_a_quadratic_maker_is_admissible():
    # A quadratic maker has unbounded chord slopes, so it is inadmissible in
    # the bounded regime (hull [-1,1]) and fine in the unrestricted regime
    # (hull R), where the sure-profit test is vacuous for finite positions.
    xs = np.linspace(-6.0, 6.0, 4001)
    lam = 2.0
    C = xs ** 2 / (2 * lam)
    assert max_sure_profit(xs, C, stride=8) > 0            # bounded regime: arbitraged
    # unrestricted regime: worst-case payoff of a position is unbounded below,
    # so no finite position is a sure profit; the chord test is not binding.
    s = xs[xs != 0]
    chords = (C[xs != 0] - C[len(xs) // 2]) / s
    assert np.abs(chords).max() > 1.0                      # slopes leave [-1,1]
    assert np.isfinite(chords).all()                       # but are finite throughout


def test_moreau_preserves_asymmetric_chord_bounds_over_random_costs():
    # Proposition 9(i) as a property, with an asymmetric slope range: if the
    # chord slopes of C lie in [a, b], so do those of its Moreau envelope,
    # at every depth.
    rng = np.random.default_rng(13)
    xs = XS[::3]
    for _ in range(12):
        a = float(rng.uniform(-0.9, -0.1))
        b = float(rng.uniform(0.1, 0.9))
        C = _random_lipschitz_cost(rng, lo=a, hi=b)[::3]
        for lam in (0.5, 3.0, 20.0):
            e = moreau_envelope(xs, C, lam)
            slopes = np.diff(e) / (xs[1] - xs[0])
            interior = slice(5, len(slopes) - 5)
            assert slopes[interior].min() >= a - 1e-6
            assert slopes[interior].max() <= b + 1e-6


def test_routing_properties_over_random_maker_configurations():
    # Lemma 6 as a property: fills clear the demand, every maker whose quote
    # band strictly contains the clearing price sits out, and the split is
    # cost-optimal against random feasible perturbations.
    rng = np.random.default_rng(14)
    for _ in range(30):
        n = int(rng.integers(2, 5))
        makers = [
            LogCoshMaker(b=float(rng.uniform(2.0, 12.0)),
                         fee=float(rng.choice([0.0, 0.01, 0.05, 0.15])),
                         q0=float(rng.normal(scale=1.5)))
            for _ in range(n)
        ]
        demand = float(rng.uniform(-4.0, 4.0))
        fills, p = route(makers, demand)
        assert fills.sum() == pytest.approx(demand, abs=1e-8)
        for m, s in zip(makers, fills):
            if m.bid + 1e-9 < p < m.ask - 1e-9:
                assert s == pytest.approx(0.0, abs=1e-9)   # sparsity
        best = sum(m.trade_cost(s) for m, s in zip(makers, fills))
        for _ in range(20):
            pert = rng.normal(scale=0.3, size=n)
            pert -= pert.mean()
            alt = sum(m.trade_cost(s) for m, s in zip(makers, fills + pert))
            assert alt >= best - 1e-8


def test_routing_capacity_boundary_is_the_bounded_slope_range():
    # Log-cosh slopes live in (-1, 1), so aggregate capacity is finite: a
    # demand beyond what the makers can supply at the extreme price has no
    # clearing price and must be refused rather than silently clamped.
    makers = [LogCoshMaker(b=1.0, fee=0.0), LogCoshMaker(b=2.0, fee=0.0)]
    reachable = sum(m.supply(1.0 - 1e-9) for m in makers)
    fills, _ = route(makers, 0.9 * reachable)
    assert fills.sum() == pytest.approx(0.9 * reachable, abs=1e-6)
    with pytest.raises(ValueError):
        route(makers, 1.5 * reachable)
