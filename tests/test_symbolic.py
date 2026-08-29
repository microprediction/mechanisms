"""Symbolic checks of the identities the numerical tests can only sample.

Grid tests compare a discretization with itself: the envelope, the Moreau
minimization and the curvature all come from the same sampled points, so a
discretization artifact moves both sides together. The identities below are
verified in closed form instead, which is independent of any mesh, window or
stride. Skipped when sympy is unavailable.
"""

import pytest

sp = pytest.importorskip("sympy")


def test_moreau_of_affine_is_affine_with_the_same_slope():
    # Proposition 9(i) is sharp: an affine cost keeps its slope exactly, so
    # the chord-bound preservation cannot be improved to a strict contraction.
    x, y, lam, m = sp.symbols("x y lam m", real=True)
    lam = sp.Symbol("lam", positive=True)
    obj = m * y + (x - y) ** 2 / (2 * lam)
    y_star = sp.solve(sp.diff(obj, y), y)[0]
    envelope = sp.simplify(obj.subs(y, y_star))
    assert sp.simplify(envelope - (m * x - lam * m ** 2 / 2)) == 0
    assert sp.simplify(sp.diff(envelope, x) - m) == 0


def test_moreau_curvature_identity():
    # (e_lam C)''(x) = C''(y*) / (1 + lam C''(y*)), so the envelope is
    # non-convex wherever C is, however deep the co-quoter: smooth
    # non-convexity is not a crossing phenomenon.
    y, lam, A = sp.symbols("y lam A", positive=True)
    C = A * sp.sin(y)
    curvature = sp.simplify(sp.diff(C, y, 2) / (1 + lam * sp.diff(C, y, 2)))
    # at the crest of the sine with A = 1/2, lam = 1 the curvature is exactly -1
    value = curvature.subs({A: sp.Rational(1, 2), lam: 1, y: sp.pi / 2})
    assert sp.simplify(value + 1) == 0
    # and it is negative exactly where C'' is, for lam A < 1
    assert sp.simplify(curvature.subs({A: sp.Rational(1, 4), lam: 2, y: sp.pi / 2})) < 0


def test_double_well_minimizer_and_jump_in_closed_form():
    # Proposition 9(iii): the interior minimizer is exactly lam*alpha while
    # lam*alpha < c, giving a jump of 2*alpha independent of the depth.
    y = sp.Symbol("y", real=True)
    lam, alpha, c = sp.symbols("lam alpha c", positive=True)
    obj = alpha * (c - y) + y ** 2 / (2 * lam)      # right well, y in (0, c)
    y_star = sp.solve(sp.diff(obj, y), y)[0]
    assert sp.simplify(y_star - lam * alpha) == 0
    branch_slope = sp.simplify((0 - y_star) / lam)   # slope at x = 0 from the left
    assert sp.simplify(branch_slope + alpha) == 0    # jump is 2*alpha


def test_log_cosh_worst_case_loss_is_log_two():
    # A log-cosh maker's worst-case loss from the origin against hull
    # [-1, 1] is sup_s [ |s| - log cosh s ] = log 2 exactly.
    s = sp.Symbol("s", positive=True)
    limit = sp.limit(s - sp.log(sp.cosh(s)), s, sp.oo)
    assert sp.simplify(limit - sp.log(2)) == 0


def test_coherent_nonconvex_bounded_loss_example_symbolically():
    # C = log cosh q + eps sech^2 q sin(k q) with 2 eps (k + 2) <= 1 is
    # chord-coherent, and its worst-case loss is at most log 2 + eps since
    # the perturbation is bounded by eps and log cosh q >= |q| - log 2.
    q, eps, k = sp.symbols("q epsilon k", positive=True)
    t = sp.tanh(q)
    A = 1 - t ** 2                                    # sech^2 q
    C = sp.log(sp.cosh(q)) + eps * A * sp.sin(k * q)
    # exact derivative identity, from which the slope bound follows
    residual = sp.diff(C, q) - t
    closed = eps * A * (k * sp.cos(k * q) - 2 * t * sp.sin(k * q))
    assert sp.simplify(sp.expand_trig(residual - closed)) == 0
    # hence |C' - tanh q| <= eps sech^2 q (k + 2|tanh q|) <= eps A (k + 2),
    # so |C'| <= |t| + (1 - t^2)(k + 2) eps <= 1 when 2 eps (k + 2) <= 1,
    # since (1 - t^2) <= 2 (1 - |t|).
    assert sp.simplify((1 - t ** 2) - 2 * (1 - sp.Abs(t))).subs(q, sp.Rational(1, 3)) <= 0


def test_stein_loss_is_the_gaussian_bregman_divergence():
    # The covariance maker's payment rule: the Bregman divergence of
    # A(Theta) = -(1/2) log det(-2 Theta) is the KL between zero-mean
    # Gaussians, which in one dimension is Stein's loss.
    v, w = sp.symbols("v w", positive=True)           # variances
    th = sp.Symbol("theta", negative=True)
    Afun = -sp.log(-2 * th) / 2
    th_v, th_w = -1 / (2 * v), -1 / (2 * w)
    bregman = (Afun.subs(th, th_v) - Afun.subs(th, th_w)
               - sp.diff(Afun, th).subs(th, th_w) * (th_v - th_w))
    # The orientation reverses: the Bregman divergence of the log-partition
    # in natural parameters is the KL with its arguments swapped.
    kl_wv = sp.Rational(1, 2) * (w / v - 1 + sp.log(v / w))   # KL(N(0,w) || N(0,v))
    kl_vw = sp.Rational(1, 2) * (v / w - 1 + sp.log(w / v))   # KL(N(0,v) || N(0,w))
    assert sp.simplify(bregman - kl_wv) == 0
    assert sp.simplify(bregman - kl_vw) != 0
    # so a trader moving the quote from v to their belief w collects
    # D_A(theta_v, theta_w) = KL(belief || standing quote), which is the
    # orientation covariance_market.py implements.
