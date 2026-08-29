"""Theorem tests for research/predictors-as-markets.md (non-convex makers).

Coherence is a chord condition, not convexity; rational flow reads the
biconjugate (lands on the contact set, with the start gap as a pass-through);
a proportional fee prices bounded slope excursions; and a deep quadratic
co-quoter convexifies the merged venue (Moreau).
"""

import numpy as np
import pytest

from mechanisms.nonconvex_maker import (
    NonconvexMaker,
    lower_convex_envelope,
    max_sure_profit,
    moreau_envelope,
)

XS = np.linspace(-12.0, 12.0, 6001)


def _wiggly_cost():
    # C' = 0.6 tanh + 0.35 sin: slopes within [-0.95, 0.95] (coherent for
    # settlement in [-1,1]) but non-monotone, so C is non-convex.
    dx = XS[1] - XS[0]
    slope = 0.6 * np.tanh(XS) + 0.35 * np.sin(XS)
    C = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * dx)])
    return C - C[len(XS) // 2]


def test_coherence_without_convexity():
    C = _wiggly_cost()
    assert np.diff(C, 2).min() < -1e-8            # genuinely non-convex
    assert max_sure_profit(XS, C, stride=10) <= 0  # yet arbitrage-free


def test_rational_flow_reads_the_biconjugate():
    C = _wiggly_cost()
    env = lower_convex_envelope(XS, C)
    gap = C - env
    assert gap.max() > 0.3  # the book has real holes
    rng = np.random.default_rng(0)
    for _ in range(100):
        mu = rng.uniform(-0.9, 0.9)
        i0 = rng.integers(500, len(XS) - 500)
        prof_c = mu * (XS - XS[i0]) - (C - C[i0])
        prof_e = mu * (XS - XS[i0]) - (env - env[i0])
        j = int(np.argmax(prof_c))
        # optimal fill lands on the contact set: the maker trades C**
        assert gap[j] == pytest.approx(0.0, abs=1e-9)
        # profit identity: envelope profit plus the gap at the start state
        assert prof_c.max() == pytest.approx(prof_e.max() + gap[i0], abs=1e-9)


def test_off_contact_states_are_a_pass_through():
    # Whoever lands off-contact overpays the landing gap relative to the
    # envelope; the start-gap term of the profit identity hands exactly that
    # amount to the next rational trader. Charges relative to the envelope
    # telescope through the gap.
    C = _wiggly_cost()
    env = lower_convex_envelope(XS, C)
    gap = C - env
    i_on = int(np.argmin(gap))          # a contact state
    j_off = int(np.argmax(gap))         # deep in a hole
    overpay = (C[j_off] - C[i_on]) - (env[j_off] - env[i_on])
    assert overpay == pytest.approx(gap[j_off] - gap[i_on], abs=1e-12)
    assert overpay > 0.3


def test_slope_excursions_are_priced_by_the_fee():
    C = 1.25 * _wiggly_cost()  # slopes reach ~1.19: excursion ~0.19 beyond the hull
    assert max_sure_profit(XS, C, stride=10) > 0.1          # arbitrageable bare
    assert max_sure_profit(XS, C, fee=0.20, stride=10) <= 0  # priced by f >= excursion
    m = NonconvexMaker(lambda q: 1.25 * float(np.interp(q, XS, _wiggly_cost())), fee=0.20)
    s = 2.0
    assert m.apply_fill(s) + m.apply_fill(-s) == pytest.approx(2 * 0.20 * s, abs=1e-9)


def test_moreau_nonconvexity_is_not_only_crossings():
    # C(y) = a sin y with lam*a < 1: the prox is unique everywhere (no
    # branch crossing), yet the envelope is smoothly non-convex with
    # curvature C''/(1 + lam C''), and coherence is still preserved
    # (slopes stay within a). The crossing-jump bound therefore measures
    # crossing defects only, not total non-convexity.
    a, lam = 0.5, 1.0
    xs = np.linspace(-15.0, 15.0, 12001)
    dx = xs[1] - xs[0]
    e = moreau_envelope(xs, a * np.sin(xs), lam)
    interior = slice(200, len(xs) - 200)
    d2 = np.diff(e, 2)[interior]
    assert d2.min() == pytest.approx(-1.0 * dx * dx, rel=1e-4)  # C''/(1+lam C'') = -1 at trough
    slopes = np.diff(e)[interior] / dx
    assert np.abs(slopes).max() <= a + 1e-6                      # coherence preserved


def test_no_trade_interval_is_fee_adjusted_chord_bounds():
    # The no-trade beliefs at state q are exactly
    # [sup_{s<0} d_q(s) - f, inf_{s>0} d_q(s) + f] with d_q the chord slope.
    # Convex case: both bounds equal C'(q), recovering the band m -+ f. Off
    # the contact set the frictionless interval is empty (chord gap
    # Delta_q > 0) and the fee fills the hole exactly when 2f >= Delta_q:
    # friction can stabilize a state inside a quote hole.
    C = _wiggly_cost()
    env = lower_convex_envelope(XS, C)
    gap = C - env

    def max_profit(i0, mu, f):
        return (mu * (XS - XS[i0]) - (C - C[i0]) - f * np.abs(XS - XS[i0])).max()

    # contact state: interval is the convex band around the envelope slope
    f = 0.10
    contacts = np.flatnonzero(gap[1000:5000] < 1e-10) + 1000
    i_c = int(contacts[len(contacts) // 2])
    dx = XS[1] - XS[0]
    m = (env[i_c + 5] - env[i_c - 5]) / (10 * dx)
    for mu in np.linspace(m - f + 0.02, m + f - 0.02, 9):
        assert max_profit(i_c, mu, f) <= 1e-9
    for mu in (m - f - 0.05, m + f + 0.05):
        assert max_profit(i_c, mu, f) > 1e-4

    # off-contact state: compute the one-sided chord bounds and the gap
    i_o = int(np.argmax(gap))
    s = XS - XS[i_o]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (C - C[i_o]) / s
    lo = np.nanmax(d[s < -1e-9])
    hi = np.nanmin(d[s > 1e-9])
    delta = lo - hi
    assert delta > 0.4  # genuinely off contact
    # small fee (2f < Delta): every belief still profits; the state is transient
    for mu in np.linspace(-0.9, 0.9, 19):
        assert max_profit(i_o, mu, 0.10) > 0.3
    # large fee (2f > Delta): the interval opens up exactly as predicted
    f_big = delta / 2 + 0.05
    mu_mid = (lo + hi) / 2
    assert max_profit(i_o, mu_mid, f_big) <= 1e-9
    assert max_profit(i_o, lo - f_big - 0.03, f_big) > 1e-3
    assert max_profit(i_o, hi + f_big + 0.03, f_big) > 1e-3


def test_contact_set_can_be_empty_without_attainment():
    # C = arctan is 1-Lipschitz (chord-coherent for hull [-1,1]); on R its
    # convex envelope is the constant -pi/2, never attained, so the contact
    # set is empty and the mu = 0 supremum pi/2 is only approached as
    # x -> -inf. Numerically: the optimal fill sits at the left boundary for
    # every truncation and the value keeps rising with the domain, so no
    # interior optimum exists to land on a contact point.
    prev = -np.inf
    for M in (10.0, 100.0, 1000.0, 10000.0):
        xs = np.linspace(-M, M, 20001)
        prof = -(np.arctan(xs) - np.arctan(0.0))   # mu = 0, from state q = 0
        assert int(np.argmax(prof)) == 0           # optimum at the left edge
        assert prof.max() < np.pi / 2              # never attains the supremum
        assert prof.max() > prev                   # and keeps improving
        prev = prof.max()
    assert prev > np.pi / 2 - 1e-3                 # approaching pi/2


def test_chord_coherence_does_not_give_a_proper_envelope():
    # C = -alpha|q| is alpha-Lipschitz, hence chord-coherent for hull
    # [-1,1], but has no affine minorant: C* = +inf everywhere, so the
    # convex envelope is -inf and Proposition 2's gap is undefined. The
    # speculative supremum at q = 0, mu = 0 is already infinite.
    alpha = 0.5
    for M in (10.0, 100.0, 1000.0):
        xs = np.linspace(-M, M, 20001)
        C = -alpha * np.abs(xs)
        # chord-coherent: every chord slope within [-1, 1]
        s = xs[xs != 0.0]
        chords = (-alpha * np.abs(s) - 0.0) / s
        assert np.abs(chords).max() <= 1.0 + 1e-12
        # yet mu = 0 profit from q = 0 grows without bound
        assert (-(C - 0.0)).max() == pytest.approx(alpha * M, rel=1e-6)


def test_depth_attenuates_but_never_fills_the_double_well():
    # e_lam C(0) > 0 for every finite lam while e_lam C(+-c) = 0, so the
    # midpoint stays strictly off contact at every depth; the gap decays
    # like c^2/(2 lam).
    alpha, c = 0.5, 3.0
    xs = np.linspace(-30.0, 30.0, 6001)
    C = alpha * np.minimum(np.abs(xs - c), np.abs(xs + c))
    mid = len(xs) // 2
    for lam in (1.0, 6.0, 30.0, 200.0):
        e = moreau_envelope(xs, C, lam)
        expected = alpha * c - lam * alpha ** 2 / 2 if lam * alpha < c else c ** 2 / (2 * lam)
        assert e[mid] == pytest.approx(expected, abs=1e-3)
        assert e[mid] > 0                                  # never filled
        assert (e - lower_convex_envelope(xs, e))[mid] > 0  # still off contact


def test_nonconvexity_shows_as_a_block_not_a_gap():
    # Across an excluded inventory range the envelope is affine, so the
    # marginal price is constant there: the book shows a block of size
    # q2 - q1 at one price, not a missing price range.
    xs = np.linspace(-3.0, 3.0, 12001)
    dx = xs[1] - xs[0]
    C = xs ** 4 / 4 - xs ** 2 / 2          # coercive double well, minima at +-1
    env = lower_convex_envelope(xs, C)
    idx = np.flatnonzero(C - env > 1e-9)
    slopes = np.diff(env)[idx[:-1]] / dx
    assert slopes.max() - slopes.min() < 1e-6      # one supporting price
    assert xs[idx[-1]] - xs[idx[0]] == pytest.approx(2.0, abs=1e-2)  # block size


def test_no_trade_interval_must_meet_the_payoff_hull():
    # C(q) = 100q has Delta_q = 0, so the untruncated interval {100} is
    # nonempty for every fee, yet no belief in the hull [-1,1] declines to
    # trade: non-emptiness needs the intersection, i.e. chord coherence.
    xs = np.linspace(-5.0, 5.0, 4001)
    C = 100.0 * xs
    for f in (0.0, 0.5, 5.0):
        for mu in np.linspace(-1.0, 1.0, 21):
            prof = (mu * (xs - 0.0) - (C - 0.0) - f * np.abs(xs)).max()
            assert prof > 0                        # every admissible belief trades


def test_double_well_jump_is_constant_until_the_coquoter_reaches_across():
    # Proposition 9(iii): jump = 2*alpha while lam*alpha < c, then 2c/lam.
    alpha, c = 0.5, 3.0
    xs = np.linspace(-20.0, 20.0, 4001)
    dx = xs[1] - xs[0]
    C = alpha * np.minimum(np.abs(xs - c), np.abs(xs + c))
    mid = len(xs) // 2
    for lam in (1.0, 3.0, 12.0, 30.0):
        sl = np.diff(moreau_envelope(xs, C, lam)) / dx
        jump = sl[mid - 2] - sl[mid + 2]
        expected = 2 * alpha if lam * alpha < c else 2 * c / lam
        assert jump == pytest.approx(expected, abs=2e-2)
        assert np.abs(sl).max() <= alpha + 1e-9    # coherence preserved


def test_fee_can_stabilize_the_sine_hole():
    # C(q) = a sin q at q = 0 sits a above its flat envelope; belief mu = 0
    # admits no profitable trade exactly when f >= a.
    a = 0.5
    ys = np.linspace(-40.0, 40.0, 40001)

    def profit(mu, f):
        return (mu * ys - a * np.sin(ys) - f * np.abs(ys)).max()

    assert profit(0.0, a) == pytest.approx(0.0, abs=1e-9)
    assert profit(0.0, 0.5 * a) > 0.1


def test_deep_quadratic_coquoter_fills_the_gaps():
    # Merging with a quadratic co-quoter of liquidity lam is the Moreau
    # envelope. A deep co-quoter (large lam) convexifies the merged venue; a
    # shallow one leaves it non-convex. The threshold is global (gap
    # geometry): lam near the local weak-convexity scale 1/0.35 is NOT yet
    # enough, which is part of the point.
    C = _wiggly_cost()
    xs = XS[::4]
    c = C[::4]
    interior = slice(20, len(xs) - 20)
    deep = np.diff(moreau_envelope(xs, c, lam=25.0), 2)[interior]
    local_scale = np.diff(moreau_envelope(xs, c, lam=3.0), 2)[interior]
    shallow = np.diff(moreau_envelope(xs, c, lam=1.0), 2)[interior]
    assert deep.min() >= -1e-6           # deep co-quoter: convex venue
    assert local_scale.min() < -1e-3     # 1/rho depth: still non-convex
    assert shallow.min() < -1e-4         # shallow: still non-convex
    # coherence is preserved at every depth: the envelope's slopes are a
    # selection of C's slopes, so the merged venue stays 1-Lipschitz
    dx = xs[1] - xs[0]
    for lam in (1.0, 3.0, 25.0):
        slopes = np.diff(moreau_envelope(xs, c, lam)) / dx
        assert np.abs(slopes[interior]).max() <= 0.95 + 1e-6
