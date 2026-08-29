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


def test_generalized_fee_lemma():
    # With a fee, the dead zone sits around the ENVELOPE's marginal price and
    # exists exactly on the contact set; at off-contact states every belief
    # yields profit at least the gap, so such states are transient. The
    # linear-fee paper's Lemma 1 is the convex case (contact set everywhere).
    C = _wiggly_cost()
    env = lower_convex_envelope(XS, C)
    gap = C - env
    f = 0.10

    def max_profit(i0, mu):
        return (mu * (XS - XS[i0]) - (C - C[i0]) - f * np.abs(XS - XS[i0])).max()

    contacts = np.flatnonzero(gap[1000:5000] < 1e-10) + 1000
    i_c = int(contacts[len(contacts) // 2])
    dx = XS[1] - XS[0]
    m = (env[i_c + 5] - env[i_c - 5]) / (10 * dx)  # envelope slope
    for mu in np.linspace(m - f + 0.02, m + f - 0.02, 9):
        assert max_profit(i_c, mu) <= 1e-9          # inside the band: no trade
    for mu in (m - f - 0.05, m + f + 0.05):
        assert max_profit(i_c, mu) > 1e-4           # outside: profitable
    i_o = int(np.argmax(gap))
    for mu in np.linspace(-0.9, 0.9, 19):
        assert max_profit(i_o, mu) > 0.3            # off-contact: no band at all


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
