"""Every worked example in *Non-Convex Market Makers*, with its stated numbers.

The paper makes its claims concrete with named examples — the affine maker at
100, the double well, the arctangent, the three-fee split, the equicorrelated
triple. This module checks each of them against the number the paper prints,
so that editing a claim without editing its example breaks the build.

These are deliberately *not* property tests. A property test over random inputs
answers "does the theorem hold in general"; these answer "is the sentence in the
paper true as written". Both matter, and the second is what a reader checks
first. Where a claim has a closed form, the example is evaluated analytically
rather than by grid search, because a grid can only be as fine as the feature
it is trying to see.

Section numbers refer to the paper.
"""

import itertools

import numpy as np
import pytest

from mechanisms.fee_routing import LogCoshMaker, route
from mechanisms.nonconvex_maker import lower_convex_envelope, max_sure_profit


# --------------------------------------------------------------------------- #
# §2  Proposition 1: no-arbitrage without convexity
# --------------------------------------------------------------------------- #
def test_prop1_compactness_pointwise_strict_is_not_a_uniform_margin():
    """A fill strictly profitable at every outcome, with worst-case margin zero.

    Outcomes 1, 2, 3, ... paying phi(w) = 1/w, so payoffs lie in (0, 1] and the
    infimum 0 is never attained. A fill of one unit costing nothing profits at
    every single outcome and still has no positive margin, which is why
    Proposition 1 assumes a compact payoff range or the uniform-margin
    definition.
    """
    outcomes = np.arange(1, 2_000_001)
    payoffs = 1.0 / outcomes
    profit = payoffs * 1.0 - 0.0

    assert np.all(profit > 0)                 # strictly profitable pointwise
    assert profit.min() < 1e-6                # yet the infimum is 0
    # Refining the outcome set only drives the margin down, never to a positive
    # floor: there is no epsilon that works.
    finer = 1.0 / np.arange(1, 20_000_001)
    assert finer.min() < profit.min()


# --------------------------------------------------------------------------- #
# §3  Proposition 2: contact and pass-through
# --------------------------------------------------------------------------- #
def _wiggly(xs):
    """C with C' = 0.6 tanh + 0.35 sin: chord slopes in [-0.95, 0.95]."""
    slope = 0.6 * np.tanh(xs) + 0.35 * np.sin(xs)
    values = np.concatenate([[0.0],
                             np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(xs))])
    return values - values[xs.size // 2]


def test_section3_chord_coherence_alone_does_not_give_an_affine_minorant():
    """C(q) = -alpha|q| with alpha < 1 is coherent but has C* identically +inf.

    The paper's point: coherence bounds what one fill can extract for sure, and
    says nothing about speculative profit against a belief.
    """
    alpha = 0.6
    qs = np.linspace(-50.0, 50.0, 200_001)
    cost = -alpha * np.abs(qs)

    slopes = np.diff(cost) / np.diff(qs)
    assert np.max(np.abs(slopes)) == pytest.approx(alpha)     # alpha-Lipschitz
    assert np.max(np.abs(slopes)) <= 1.0                      # coherent for [-1, 1]

    # C*(p) = sup_q [pq - C(q)] diverges for every p, because an affine minorant
    # would need slope <= -alpha and >= +alpha at once. Divergence shows up as
    # the supremum growing in exact proportion to the window: widening tenfold
    # multiplies it by ten, so there is no limit to converge to.
    for p in (-1.0, 0.0, 0.5, 1.0):
        values = []
        for reach in (50.0, 500.0, 5000.0):
            grid = np.linspace(-reach, reach, 200_001)
            values.append(float(np.max(p * grid + alpha * np.abs(grid))))
        assert values[1] == pytest.approx(10 * values[0], rel=1e-9)
        assert values[2] == pytest.approx(100 * values[0], rel=1e-9)

    # And at q = 0, mu = 0 the trader's own supremum is already infinite.
    assert np.max(alpha * np.abs(qs)) == pytest.approx(alpha * 50.0)


def test_section3_arctan_has_an_empty_contact_set():
    """C = arctan: envelope is the constant -pi/2, and the optimum is at -inf."""
    xs = np.linspace(-1e6, 1e6, 2_000_001)
    cost = np.arctan(xs)
    assert cost.min() == pytest.approx(-np.pi / 2, abs=1e-5)

    profit = 0.0 * xs - cost                  # mu = 0
    assert profit.max() == pytest.approx(np.pi / 2, abs=1e-5)
    assert int(np.argmax(profit)) == 0        # attained only at the window edge
    # Widening the window moves the argmax with it: the contact is at infinity.
    wider = np.linspace(-1e8, 1e8, 200_001)
    assert int(np.argmax(-np.arctan(wider))) == 0


def test_section3_coercive_is_not_enough_for_attainment():
    """h(x) = x^2 off zero, 1 at zero: coercive, infimum 0, no minimiser."""
    xs = np.linspace(-2.0, 2.0, 400_001)
    h = xs ** 2
    h[np.isclose(xs, 0.0)] = 1.0

    assert h[0] > 3.0                                  # coercive
    assert h.min() < 1e-9                              # infimum is 0
    assert not np.any(h == 0.0)                        # but never attained
    assert h[np.isclose(xs, 0.0)][0] == 1.0            # the puncture


def test_prop2_profit_identity_holds_pointwise():
    """Pi_C(q, mu) = Pi_Chat(q, mu) + g(q), the pass-through identity."""
    xs = np.linspace(-12.0, 12.0, 6001)
    cost = _wiggly(xs)
    env = lower_convex_envelope(xs, cost)
    gap = cost - env

    rng = np.random.default_rng(0)
    for _ in range(300):
        # Inside the dual domain (asymptotic slope 0.6), so the optimum exists.
        mu = rng.uniform(-0.55, 0.55)
        i0 = int(rng.integers(500, xs.size - 500))
        pi_c = float(np.max(mu * (xs - xs[i0]) - (cost - cost[i0])))
        pi_env = float(np.max(mu * (xs - xs[i0]) - (env - env[i0])))
        assert pi_c == pytest.approx(pi_env + gap[i0], abs=1e-12)


def test_section3_the_conjugate_is_convex_so_exp_minus_it_is_log_concave():
    """Non-convex C still has convex C*, so no bimodal implied density follows."""
    xs = np.linspace(-12.0, 12.0, 6001)
    cost = _wiggly(xs)
    assert float(np.min(np.diff(cost, 2))) < -1e-9       # C genuinely non-convex

    ps = np.linspace(-0.55, 0.55, 2001)
    conjugate = np.array([np.max(p * xs - cost) for p in ps])
    assert float(np.min(np.diff(conjugate, 2))) >= -1e-12   # C* convex
    # log exp(-C*) = -C*, which is concave.
    assert float(np.max(np.diff(-conjugate, 2))) <= 1e-12


def test_section3_double_well_block_is_indivisible():
    """Full trade -c -> c costs nothing; the half trade -c -> 0 costs alpha*c."""
    alpha, c = 0.4, 2.0
    well = lambda q: alpha * min(abs(q - c), abs(q + c))

    assert well(c) - well(-c) == pytest.approx(0.0)          # the whole block
    assert well(0.0) - well(-c) == pytest.approx(alpha * c)  # half of it is not free
    assert alpha * c == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# §4 / §5  Propositions 3 and Lemma 5: the affine maker at 100
# --------------------------------------------------------------------------- #
AFFINE_TABLE = [          # (fee, untruncated interval, intersection with [-1, 1])
    (0.0, (100.0, 100.0), None),
    (50.0, (50.0, 150.0), None),
    (98.9, (1.1, 198.9), None),
    (99.0, (1.0, 199.0), (1.0, 1.0)),
    (100.0, (0.0, 200.0), (0.0, 1.0)),
    (101.0, (-1.0, 201.0), (-1.0, 1.0)),
]


@pytest.mark.parametrize("fee,untruncated,truncated", AFFINE_TABLE)
def test_lemma5_affine_maker_table(fee, untruncated, truncated):
    """C(q) = 100q: every row of the paper's table, exactly as printed."""
    # Every chord slope is 100, so L = U = 100 and the interval is [100-f, 100+f].
    qs = np.linspace(-5.0, 5.0, 2001)
    cost = 100.0 * qs
    i0 = qs.size // 2
    s = np.delete(qs - qs[i0], i0)
    chords = np.delete(cost - cost[i0], i0) / s
    assert np.allclose(chords, 100.0)

    lo, hi = 100.0 - fee, 100.0 + fee
    assert (lo, hi) == pytest.approx(untruncated)

    band_lo, band_hi = max(lo, -1.0), min(hi, 1.0)
    if truncated is None:
        assert band_lo > band_hi                       # empty
    else:
        assert (band_lo, band_hi) == pytest.approx(truncated)


def test_lemma5_affine_maker_threshold_is_the_arbitrage_depth():
    """The threshold is f = 99, which is how far the quote 100 exits [-1, 1]."""
    for fee in (98.0, 98.9, 98.999):
        assert max(100.0 - fee, -1.0) > min(100.0 + fee, 1.0)     # still empty
    assert max(100.0 - 99.0, -1.0) == min(100.0 + 99.0, 1.0) == 1.0
    for fee in (99.001, 100.0, 150.0):
        assert max(100.0 - fee, -1.0) <= min(100.0 + fee, 1.0)    # non-empty
    assert 100.0 - 1.0 == 99.0        # the depth is the excursion beyond the hull


def test_section5_sine_maker_needs_a_fee_of_exactly_a():
    """C(q) = a sin q at q = 0: mu = 0 declines to trade exactly when f >= a."""
    fills = np.concatenate([np.linspace(-40.0, -1e-9, 200_001),
                            np.linspace(1e-9, 40.0, 200_001)])
    for a in (0.2, 0.5, 0.9):
        required = float(np.max(-a * np.sin(fills) / np.abs(fills)))
        assert required == pytest.approx(a, abs=1e-6)
        # Below a, some fill is profitable; at or above it, none is.
        assert not np.all(a * np.sin(fills) + 0.99 * a * np.abs(fills) >= -1e-12)
        assert np.all(a * np.sin(fills) + a * np.abs(fills) >= -1e-12)


# --------------------------------------------------------------------------- #
# §5  Lemma 4 (fee-spread duality) and Lemma 5 at a kink
# --------------------------------------------------------------------------- #
def test_lemma4_conjugate_is_the_soft_thresholded_profit():
    """Ct*(p) = g*(m + T_f(p - m)), and is zero exactly on the quote band."""
    lam, m, fee = 2.0, 0.3, 0.25
    fills = np.arange(-200.0, 200.0, 1e-4)
    g_star = lambda p: lam * (p - m) ** 2 / 2.0
    soft = lambda x: np.sign(x) * max(abs(x) - fee, 0.0)

    for p in (0.05, 0.20, 0.30, 0.40, 0.55, 0.80, -0.40):
        direct = float(np.max(p * fills
                              - (m * fills + fills ** 2 / (2 * lam)
                                 + fee * np.abs(fills))))
        assert direct == pytest.approx(g_star(m + soft(p - m)), abs=1e-6)
        # Zero exactly on the closed band [m - f, m + f]; p = 0.55 is its edge.
        assert (direct < 1e-9) == (abs(p - m) <= fee + 1e-12)


def test_lemma5_at_a_convex_kink_the_spread_survives_a_zero_fee():
    """C = max(-2q, 3q): the interval is [-2 - f, 3 + f], width 5 even at f = 0."""
    cost = lambda q: np.maximum(-2.0 * q, 3.0 * q)
    fills = np.concatenate([np.linspace(-30.0, -1e-6, 300_001),
                            np.linspace(1e-6, 30.0, 300_001)])
    chords = (cost(fills) - cost(0.0)) / fills
    lower = float(np.max(chords[fills < 0]))
    upper = float(np.min(chords[fills > 0]))
    assert lower == pytest.approx(-2.0, abs=1e-4)      # left derivative
    assert upper == pytest.approx(3.0, abs=1e-4)       # right derivative
    for fee in (0.0, 0.5, 1.5):
        assert (lower - fee, upper + fee) == pytest.approx((-2.0 - fee, 3.0 + fee),
                                                           abs=1e-4)
    assert upper - lower == pytest.approx(5.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# §6  Lemma 6, Remark 7, Corollary 8
# --------------------------------------------------------------------------- #
def test_section6_infimal_convolution_at_zero_is_the_internal_cross():
    """(g1 [] g2)(0) = -(m1 - m2)^2 / (2(1/lam1 + 1/lam2)), the paper's formula."""
    fills = np.linspace(-50.0, 50.0, 2_000_001)
    for m1, m2, l1, l2 in [(0.3, -0.2, 1.0, 2.0), (0.5, 0.5, 1.0, 1.0),
                           (1.0, -1.0, 0.5, 4.0)]:
        g = lambda s, m, lam: m * s + s * s / (2 * lam)
        numeric = float(np.min(g(fills, m1, l1) + g(-fills, m2, l2)))
        formula = -(m1 - m2) ** 2 / (2 * (1 / l1 + 1 / l2))
        assert numeric == pytest.approx(formula, abs=1e-7)
        # Zero only when the quotes agree: otherwise there is money on the table.
        assert (abs(numeric) < 1e-12) == (m1 == m2)


def test_remark7_liquidity_adds_for_the_log_cosh_family():
    """C_b1 [] C_b2 = C_{b1+b2}: the perspective family's aggregation law."""
    for b1, b2 in [(1.0, 1.0), (0.5, 2.5), (3.0, 0.25)]:
        for size in (0.7, 3.0):
            merged = LogCoshMaker(b1 + b2)
            whole_fill, _ = route([merged], size)
            parts = [LogCoshMaker(b1), LogCoshMaker(b2)]
            part_fills, _ = route(parts, size)
            assert (sum(m.trade_cost(s) for m, s in zip(parts, part_fills))
                    == pytest.approx(merged.trade_cost(whole_fill[0]), abs=1e-9))


def test_remark7_quadratic_merge_is_gaussian_fusion():
    """The merged quote is the precision-weighted mean; precisions add."""
    m1, m2, l1, l2 = 0.4, -0.1, 2.0, 3.0
    fills = np.linspace(-30.0, 30.0, 1_200_001)
    g = lambda s, m, lam: m * s + s * s / (2 * lam)
    inf_conv = lambda d: float(np.min(g(fills, m1, l1) + g(d - fills, m2, l2)))

    h = 1e-4
    merged_quote = (inf_conv(h) - inf_conv(-h)) / (2 * h)
    assert merged_quote == pytest.approx((l1 * m1 + l2 * m2) / (l1 + l2), abs=1e-4)


def test_lemma6_three_fee_split_matches_the_papers_table():
    """The split at demands 0.05, 0.5 and 1.0, to the two decimals printed."""
    makers = lambda: [LogCoshMaker(1.0, 0.05), LogCoshMaker(1.0, 0.20),
                      LogCoshMaker(1.0, 0.50)]
    for size, printed in [(0.05, (0.05, 0.00, 0.00)),
                          (0.5, (0.33, 0.17, 0.00)),
                          (1.0, (0.56, 0.38, 0.06))]:
        fills, _ = route(makers(), size)
        assert tuple(round(float(x), 2) for x in fills) == printed


def test_corollary8_quartic_maker_is_active_but_not_differentiable_at_zero():
    """C(q) = q^4 at q_i = -1 has S(p) = (p/4)^(1/3) + 1, active at p = 0."""
    cbrt = lambda z: np.sign(z) * np.abs(z) ** (1.0 / 3.0)
    supply = lambda p: cbrt(p / 4.0) + 1.0

    # The supply really inverts the marginal price: C'(q_i + S(p)) = p.
    for p in (-4.0, -0.5, 0.0, 0.5, 4.0):
        new_q = supply(p) - 1.0
        assert 4.0 * new_q ** 3 == pytest.approx(p, abs=1e-9)

    assert supply(0.0) == pytest.approx(1.0)      # active at p = 0
    h = 1e-9
    assert (supply(h) - supply(0.0)) / h > 1e4    # infinite slope from the right
    assert (supply(0.0) - supply(-h)) / h > 1e4   # and from the left


def test_corollary8_local_depth_is_the_sum_of_reciprocal_curvatures():
    """S'(p) = sum over active makers of 1 / C''(q_i + s_i(p))."""
    makers = [LogCoshMaker(b, 0.0) for b in (1.0, 2.5, 0.7)]
    p0, h = 0.3, 1e-6
    total = lambda p: sum(m.supply(p) for m in makers)
    numeric = (total(p0 + h) - total(p0 - h)) / (2 * h)
    # C(q) = b log cosh(q/b) has C'' = (1/b) sech^2(q/b).
    analytic = sum(1.0 / ((1 / m.b) / np.cosh((m.q + m.supply(p0)) / m.b) ** 2)
                   for m in makers)
    assert numeric == pytest.approx(analytic, abs=1e-4)


# --------------------------------------------------------------------------- #
# §6  Proposition 9: the double well under Moreau smoothing
# --------------------------------------------------------------------------- #
ALPHA, WELL_C = 0.4, 2.0            # so the regime switches at lam = c/alpha = 5


def _double_well_prox(x, lam):
    """Exact proximal set of alpha*min(|y-c|, |y+c|), by per-piece minimisation."""
    pieces = [(-np.inf, -WELL_C, -ALPHA), (-WELL_C, 0.0, ALPHA),
              (0.0, WELL_C, -ALPHA), (WELL_C, np.inf, ALPHA)]
    best, argmins = np.inf, []
    for lo, hi, slope in pieces:
        y = np.clip(x - lam * slope,
                    lo if np.isfinite(lo) else -1e9,
                    hi if np.isfinite(hi) else 1e9)
        value = ALPHA * min(abs(y - WELL_C), abs(y + WELL_C)) + (y - x) ** 2 / (2 * lam)
        if value < best - 1e-12:
            best, argmins = value, [y]
        elif abs(value - best) <= 1e-12:
            argmins.append(y)
    return best, sorted({round(y, 10) for y in argmins})


@pytest.mark.parametrize("lam", [1.0, 3.0, 4.99, 5.0, 5.01, 8.0, 20.0])
def test_prop9_double_well_envelope_value_at_the_midpoint(lam):
    """e_lam C(0) = alpha*c - lam*alpha^2/2 while lam*alpha < c, then c^2/(2 lam)."""
    value, _ = _double_well_prox(0.0, lam)
    shallow = ALPHA * WELL_C - lam * ALPHA ** 2 / 2.0
    deep = WELL_C ** 2 / (2.0 * lam)
    assert value == pytest.approx(shallow if lam * ALPHA < WELL_C else deep, abs=1e-9)
    # And the wells themselves stay at zero, so the midpoint is strictly off
    # contact at every depth: smoothing never closes the gap.
    assert _double_well_prox(WELL_C, lam)[0] == pytest.approx(0.0, abs=1e-9)
    assert value > 0.0


@pytest.mark.parametrize("lam", [1.0, 3.0, 4.99, 5.0, 5.01, 8.0, 20.0])
def test_prop9_jump_is_the_spread_of_the_proximal_set_over_lambda(lam):
    """The jump is (max P - min P)/lam: 2*alpha shallow, 2c/lam deep."""
    _, prox = _double_well_prox(0.0, lam)
    assert len(prox) == 2                       # a genuine crossing
    jump = (max(prox) - min(prox)) / lam
    claim = 2 * ALPHA if lam * ALPHA < WELL_C else 2 * WELL_C / lam
    assert jump == pytest.approx(claim, abs=1e-9)
    # Shallow: the competing minimisers are +-lam*alpha. Deep: they are +-c.
    expected = lam * ALPHA if lam * ALPHA < WELL_C else WELL_C
    assert max(prox) == pytest.approx(expected, abs=1e-9)


def test_prop9_the_jump_is_constant_until_the_co_quoter_reaches_the_wells():
    """2*alpha holds for every lam below c/alpha, then decays as 2c/lam."""
    switch = WELL_C / ALPHA
    assert switch == 5.0
    for lam in (0.1, 1.0, 3.0, 4.9, 4.999):
        _, prox = _double_well_prox(0.0, lam)
        assert (max(prox) - min(prox)) / lam == pytest.approx(2 * ALPHA, abs=1e-9)
    previous = np.inf
    for lam in (6.0, 10.0, 40.0, 200.0):
        _, prox = _double_well_prox(0.0, lam)
        jump = (max(prox) - min(prox)) / lam
        assert jump == pytest.approx(2 * WELL_C / lam, abs=1e-9)
        assert jump < previous
        previous = jump


def test_prop9_sine_curvature_formula():
    """C = a sin y with lam*a < 1: (e_lam C)'' = C''(y*) / (1 + lam C''(y*))."""
    for a, lam in ((0.5, 1.0), (0.3, 2.0), (0.9, 1.0)):
        assert lam * a < 1.0
        ys = np.arange(-30.0, 30.0, 1e-5)
        cost = a * np.sin(ys)
        env = lambda x: float(np.min(cost + (ys - x) ** 2 / (2 * lam)))
        h = 1e-3
        for x0 in (0.4, 1.7, 3.0, 5.2):
            numeric = (env(x0 + h) - 2 * env(x0) + env(x0 - h)) / h ** 2
            minimiser = ys[int(np.argmin(cost + (ys - x0) ** 2 / (2 * lam)))]
            second = -a * np.sin(minimiser)
            assert numeric == pytest.approx(second / (1 + lam * second), abs=2e-3)


def test_prop9_coherence_is_preserved_at_every_depth():
    """Chord slopes of C in [-1, 1] implies the same for e_lam C, for all lam."""
    step = 1e-3
    ys = np.arange(-400.0, 400.0, step)
    slope = 0.6 * np.tanh(ys) + 0.35 * np.sin(ys)
    cost = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * step)])
    cost -= cost[ys.size // 2]
    assert slope.min() >= -1.0 and slope.max() <= 1.0

    probe = np.linspace(-30.0, 30.0, 121)
    for lam in (0.05, 0.5, 5.0, 50.0, 500.0):
        env = np.array([np.min(cost + (ys - x) ** 2 / (2 * lam)) for x in probe])
        chords = np.diff(env) / np.diff(probe)
        assert chords.min() >= -1.0 - 1e-9
        assert chords.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# §8  Propositions 10 and 11
# --------------------------------------------------------------------------- #
def test_prop10_eigenvector_bundle_costs_the_negative_eigenvalue():
    """The bundle costs w'Pw = lambda_min < 0 and pays (w'x)^2 >= 0."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(20):
        a = rng.normal(size=(3, 3))
        quotes = (a + a.T) / 2
        np.fill_diagonal(quotes, 1.0)
        eigenvalues, vectors = np.linalg.eigh(quotes)
        if eigenvalues.min() >= 0:
            continue
        w = vectors[:, 0]
        assert float(w @ quotes @ w) == pytest.approx(eigenvalues[0], abs=1e-12)
        draws = rng.normal(size=(20_000, 3))
        payoff = (draws @ w) ** 2
        assert payoff.min() >= 0.0
        assert payoff.min() - float(w @ quotes @ w) >= abs(eigenvalues[0]) - 1e-9
        checked += 1
    assert checked >= 3


def test_prop10_psd_is_not_enough_for_binary_variables():
    """Equicorrelation: PSD for rho >= -1/2, realizable only for rho >= -1/3."""
    outcomes = np.array(list(itertools.product([-1, 1], repeat=3)))
    payoffs = np.array([[o[0] * o[1], o[0] * o[2], o[1] * o[2]] for o in outcomes],
                       float)
    portfolio = np.full(3, 1 / 3)              # the triangle inequality

    for rho in (-0.49, -0.45, -0.40, -0.35):
        matrix = np.full((3, 3), rho)
        np.fill_diagonal(matrix, 1.0)
        assert np.linalg.eigvalsh(matrix).min() > 0          # strictly PSD
        worst = float(np.min(payoffs @ portfolio))
        assert worst == pytest.approx(-1 / 3)                # one sign disagrees
        profit = worst - float(np.array([rho] * 3) @ portfolio)
        assert profit == pytest.approx(-1 / 3 - rho, abs=1e-12)
        assert profit > 0
    # The paper's headline number.
    assert -1 / 3 - (-0.49) == pytest.approx(0.157, abs=5e-4)
    # PSD bound and cut-polytope bound, and the gap between them.
    assert np.linalg.eigvalsh(np.full((3, 3), -0.5) + 1.5 * np.eye(3)).min() \
        == pytest.approx(0.0, abs=1e-12)
    assert -1 / 3 - (-1 / 3) == pytest.approx(0.0)           # profit vanishes there


def test_section4_the_arbitrage_depth_is_the_fee_that_closes_it():
    """Proposition 3 as an iff, on a cost whose excess is set by construction."""
    xs = np.linspace(-6.0, 6.0, 2401)
    for excess in (0.05, 0.2, 0.75):
        slope = np.full(xs.size, 0.0)
        slope[(xs > -1.0) & (xs < 1.0)] = 1.0 + excess
        cost = np.concatenate([[0.0],
                               np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(xs))])
        assert max_sure_profit(xs, cost, fee=0.99 * excess) > 0
        assert max_sure_profit(xs, cost, fee=excess) == pytest.approx(0.0, abs=1e-6)
        assert max_sure_profit(xs, cost, fee=1.01 * excess) <= 1e-9


# --------------------------------------------------------------------------- #
# §1  The dictionary: proximal steps are trades against a fee-bearing maker
# --------------------------------------------------------------------------- #
def test_section1_prox_is_the_response_to_a_unit_quadratic_maker():
    """prox_g(x) = argmax_s {x s - s^2/2 - g(s)}, for any convex g."""
    fills = np.arange(-10.0, 10.0, 1e-5)
    for penalty in (lambda s: np.abs(s),            # lasso
                    lambda s: s ** 2 / 2,           # ridge
                    lambda s: 0.3 * np.abs(s)):
        for x in (-1.5, -0.2, 0.4, 2.0):
            prox = fills[int(np.argmin(0.5 * (fills - x) ** 2 + penalty(fills)))]
            trade = fills[int(np.argmax(x * fills - fills ** 2 / 2 - penalty(fills)))]
            assert prox == pytest.approx(trade, abs=1e-4)


def test_section1_prox_of_the_absolute_value_is_the_soft_threshold():
    """The trader's optimal response to a proportional fee, in closed form."""
    fills = np.arange(-10.0, 10.0, 1e-5)
    for fee in (0.2, 0.7):
        soft = lambda x: np.sign(x) * max(abs(x) - fee, 0.0)
        for x in (-1.5, -0.1, 0.0, 0.3, 2.0):
            prox = fills[int(np.argmin(0.5 * (fills - x) ** 2 + fee * np.abs(fills)))]
            assert prox == pytest.approx(soft(x), abs=1e-4)


def test_section1_without_the_makers_curvature_the_l1_problem_is_unbounded():
    """A bare conjugate diverges exactly when |mu| > f, which is why Lemma 4
    carries the base cost g_q."""
    fee = 0.3
    for mu in (0.2, 0.3, 0.5, 1.0):
        values = []
        for reach in (10.0, 100.0, 1000.0):
            grid = np.arange(-reach, reach, 1e-3)
            values.append(float(np.max(mu * grid - fee * np.abs(grid))))
        diverges = values[-1] > 10.0
        assert diverges == (abs(mu) > fee)
        if diverges:
            # And it diverges linearly in the reach, so there is no limit.
            assert values[2] == pytest.approx(10 * values[1], rel=1e-3)


# --------------------------------------------------------------------------- #
# §7  Routing disciplines the posted menu
# --------------------------------------------------------------------------- #
def test_section7_quoting_inside_the_spread_takes_the_flow_first():
    """An entrant inside the prevailing spread takes all of a small trade."""
    def book(entrant_fee):
        return [LogCoshMaker(1.0, 0.30), LogCoshMaker(1.0, 0.30),
                LogCoshMaker(1.0, entrant_fee)]

    inside, _ = route(book(0.10), 0.15)
    assert inside[2] == pytest.approx(0.15)          # the entrant takes all of it
    assert inside[0] == inside[1] == 0.0             # incumbents get exactly zero

    tied, _ = route(book(0.30), 0.15)
    np.testing.assert_allclose(tied, np.full(3, 0.05), atol=1e-9)

    outside, _ = route(book(0.35), 0.15)
    assert outside[2] < outside[0]                   # quoting wide earns less


def test_section7_quoting_behind_earns_nothing_until_the_front_is_consumed():
    """The back of the book activates exactly at arctanh(0.35), not before."""
    book = lambda: [LogCoshMaker(1.0, 0.05), LogCoshMaker(1.0, 0.40)]
    # The back's ask is 0.40; the front alone supplies arctanh(0.40 - 0.05) by
    # the time the clearing price gets there.
    threshold = float(np.arctanh(0.40 - 0.05))
    assert threshold == pytest.approx(0.365443754, abs=1e-9)

    for demand in (0.05, 0.3, threshold * 0.999):
        fills, _ = route(book(), demand)
        assert fills[1] == 0.0                       # exactly zero, not small
        assert fills[0] == pytest.approx(demand, abs=1e-9)

    # Exactly at the threshold the clearing price sits on the back's ask, so the
    # bisection leaves a rounding residual rather than a clean zero.
    at_edge, _ = route(book(), threshold)
    assert abs(at_edge[1]) < 1e-12

    for demand in (threshold * 1.001, 0.6, 1.5):
        fills, _ = route(book(), demand)
        assert fills[1] > 1e-9                       # now it genuinely participates


def test_section7_quoting_too_tight_is_picked_off_by_informed_flow():
    """A maker's P&L against 30% informed flow turns on the fee it posts."""
    rng = np.random.default_rng(0)
    results = {}
    for fee in (0.02, 0.10, 0.30):
        maker = LogCoshMaker(1.0, fee)
        pnl = 0.0
        for _ in range(20_000):
            settlement = rng.choice([-1.0, 1.0])
            informed = rng.uniform() < 0.3
            fill = 0.25 * (settlement if informed else rng.choice([-1.0, 1.0]))
            pnl += maker.trade_cost(fill) - settlement * fill
        results[fee] = pnl
    assert results[0.02] < 0                         # too tight: picked off
    assert results[0.10] < 0
    assert results[0.30] > 0                         # wide enough to survive
    assert results[0.02] < results[0.10] < results[0.30]


# --------------------------------------------------------------------------- #
# §9  The argmax identity behind "the market trades the biconjugate"
# --------------------------------------------------------------------------- #
def test_section9_argmax_set_identity():
    """argmax{mu x - C} = argmax{mu x - Chat} intersected with the contact set."""
    xs = np.linspace(-12.0, 12.0, 24001)
    cost = _wiggly(xs)
    env = lower_convex_envelope(xs, cost)
    contact = set(np.flatnonzero(np.abs(cost - env) <= 1e-9).tolist())

    for mu in np.linspace(-0.55, 0.55, 45):
        raw = mu * xs - cost
        smoothed = mu * xs - env
        left = set(np.flatnonzero(raw >= raw.max() - 1e-9).tolist())
        right = set(np.flatnonzero(smoothed >= smoothed.max() - 1e-9).tolist()) & contact
        assert left == right
        assert left                                  # non-empty inside the domain


# --------------------------------------------------------------------------- #
# Abstract-level claims that deserve their own example
# --------------------------------------------------------------------------- #
def test_corollary8_aggregate_is_a_consolidated_limit_order_book():
    """Non-decreasing, zero on the common band, strictly rising where active."""
    from mechanisms.fee_routing import consolidated_book

    makers = [LogCoshMaker(1.0, 0.10), LogCoshMaker(2.0, 0.25),
              LogCoshMaker(0.5, 0.40)]
    best_bid = max(m.bid for m in makers)
    best_ask = min(m.ask for m in makers)
    assert (best_bid, best_ask) == pytest.approx((-0.10, 0.10))

    prices = np.linspace(-0.9, 0.9, 20001)
    supply = consolidated_book(makers, prices)

    assert np.all(np.diff(supply) >= -1e-12)                 # non-decreasing
    inside = (prices > best_bid) & (prices < best_ask)
    assert inside.sum() > 1000
    assert np.all(np.abs(supply[inside]) < 1e-15)            # identically zero

    active = np.array([sum(1 for m in makers if not (m.bid < p < m.ask))
                       for p in prices])
    both_active = (active[:-1] > 0) & (active[1:] > 0)
    assert np.all(np.diff(supply)[both_active] > 0)          # strictly increasing
    both_idle = (active[:-1] == 0) & (active[1:] == 0)
    assert np.all(np.abs(np.diff(supply)[both_idle]) < 1e-15)   # flat

    # Six price levels, one per band edge: the book a large order walks.
    levels = sorted({round(m.bid, 6) for m in makers} | {round(m.ask, 6) for m in makers})
    assert levels == [-0.4, -0.25, -0.1, 0.1, 0.25, 0.4]


def test_abstract_moreau_attenuates_the_gap_without_ever_closing_it():
    """The double well's midpoint gap shrinks over five decades of lambda, and
    stays strictly positive at every one."""
    previous = np.inf
    for lam in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1000.0, 10000.0):
        value, _ = _double_well_prox(0.0, lam)
        # The wells stay at zero, so the envelope of e_lam C is zero at the
        # midpoint and the gap is exactly the value there.
        assert _double_well_prox(WELL_C, lam)[0] == pytest.approx(0.0, abs=1e-9)
        assert value > 0.0                       # never closes
        assert value < previous                  # but always attenuating
        previous = value
    assert previous == pytest.approx(WELL_C ** 2 / (2 * 10000.0), abs=1e-9)


def test_lemma5_holds_for_a_genuinely_non_convex_cost():
    """"For any cost C (no convexity assumed)" — checked on a wiggly one.

    The belief perturbation has to clear the grid: the binding fill can be a
    single step of width dx, so a perturbation delta shows up as a profit of
    only delta*dx. Bracketing closer than that measures the mesh, not the claim.
    """
    from mechanisms.nonconvex_maker import chord_bounds

    xs = np.linspace(-12.0, 12.0, 24001)
    step = xs[1] - xs[0]
    cost = _wiggly(xs)
    assert float(np.min(np.diff(cost, 2))) < -1e-9          # genuinely non-convex

    delta, slack = 1e-3, 1e-12
    assert delta >= step                                     # resolvable

    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(400):
        i = int(rng.integers(2000, xs.size - 2000))
        lower, upper = chord_bounds(xs, cost, i)
        fee = float(rng.uniform(0.0, 0.6))
        lo, hi = lower - fee, upper + fee
        if lo > hi:
            continue
        fills = np.delete(xs - xs[i], i)
        deltas = np.delete(cost - cost[i], i)
        profit = lambda mu: float(np.max(mu * fills - deltas - fee * np.abs(fills)))
        for mu, no_trade in ((lo - delta, False), (lo, True),
                             (0.5 * (lo + hi), True), (hi, True),
                             (hi + delta, False)):
            assert (profit(mu) <= slack) == no_trade
            checked += 1
    assert checked > 1000


def test_lemma5_non_emptiness_criterion_and_the_off_contact_claim():
    """Non-empty iff 2f >= Delta_q; and off the contact set Delta_q > 0."""
    from mechanisms.nonconvex_maker import chord_bounds

    xs = np.linspace(-12.0, 12.0, 24001)
    cost = _wiggly(xs)
    gap = cost - lower_convex_envelope(xs, cost)

    rng = np.random.default_rng(1)
    for _ in range(400):
        i = int(rng.integers(2000, xs.size - 2000))
        lower, upper = chord_bounds(xs, cost, i)
        fee = float(rng.uniform(0.0, 0.6))
        assert ((lower - fee) <= (upper + fee)) == (2 * fee >= lower - upper - 1e-12)

    off_contact = np.flatnonzero(gap > 1e-7)
    assert off_contact.size > 100
    for i in off_contact[::200]:
        lower, upper = chord_bounds(xs, cost, int(i))
        assert lower - upper > 0                 # Delta_q > 0 off the contact set
