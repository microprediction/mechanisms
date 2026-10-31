#!/usr/bin/env python3
"""Brute-force numerical verification of every statement in 'Bounded Market Makers from Convex Duality'.
Each check prints PASS/FAIL with the worst observed discrepancy. Exit status 1 if any check fails."""
import numpy as np
from scipy.optimize import minimize, brentq

rng = np.random.default_rng(20261031)
LOG2 = np.log(2.0)
FAILS = []

def report(name, ok, worst):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  (worst {worst:.2e})")
    if not ok: FAILS.append(name)

lc = lambda x: np.logaddexp(x, -x) - LOG2
def lcs(y):
    y = np.asarray(y, float)
    out = np.full(y.shape, np.inf)
    m = np.abs(y) <= 1
    yy = y[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(1 + yy > 0, (1 + yy) * np.log(1 + yy), 0.0)
        c = np.where(1 - yy > 0, (1 - yy) * np.log(1 - yy), 0.0)
    out[m] = 0.5 * (a + c)
    return out

class Maker:
    def __init__(self, V, b): self.V, self.b = np.asarray(V, float), float(b)
    def C(self, q): return self.b * lc(self.V.T @ q / self.b).sum()
    def p(self, q): return self.V @ np.tanh(self.V.T @ q / self.b)
    def H(self, q): return (self.V * (1 / np.cosh(self.V.T @ q / self.b) ** 2)) @ self.V.T / self.b
    def conj_num(self, s):   # sup_q <q,s> - C(q) by numerical maximisation (concave objective)
        n = self.V.shape[0]
        best = -np.inf
        for x0 in [np.zeros(n), rng.normal(size=n), 3 * rng.normal(size=n)]:
            r = minimize(lambda q: -(q @ s - self.C(q)), x0, jac=lambda q: -(s - self.p(q)), method="BFGS", options={"gtol": 1e-12, "maxiter": 2000})
            best = max(best, -r.fun)
        return best

# 1. conjugate formula ------------------------------------------------------------
worst = 0.0
for b in (0.5, 1.0, 3.0):
    M = Maker(np.eye(3), b)
    for _ in range(200):
        s = rng.uniform(-0.98, 0.98, 3)
        worst = max(worst, abs(M.conj_num(s) - b * lcs(s).sum()))
report("1a conjugate = b*sum lcs(s) for V=I", worst < 1e-6, worst)
V23 = np.array([[1.0, 0.0, 0.8], [0.0, 1.0, 0.6]]); M = Maker(V23, 1.0)
worst = 0.0
for _ in range(60):
    y0 = rng.uniform(-0.9, 0.9, 3); s = V23 @ y0
    # constrained min of sum lcs(y) s.t. V y = s, |y|<=1 : parametrise null space
    N = np.linalg.svd(V23)[2][-1]            # null vector of V (2x3)
    lo = max((-1 - y0[i]) / N[i] if N[i] > 0 else (1 - y0[i]) / N[i] for i in range(3) if N[i] != 0)
    hi = min((1 - y0[i]) / N[i] if N[i] > 0 else (-1 - y0[i]) / N[i] for i in range(3) if N[i] != 0)
    ts = np.linspace(lo, hi, 20001)
    vals = lcs(y0[None, :] + ts[:, None] * N[None, :]).sum(1)
    worst = max(worst, abs(M.conj_num(s) - vals.min()))
report("1b conjugate = constrained min over representations (n=2, D=3)", worst < 1e-5, worst)

# 2. loss bound + attainment --------------------------------------------------------
M = Maker(V23, 1.0); D = 3
worst = -np.inf
for _ in range(2000):
    q = rng.normal(size=2) * rng.choice([0.3, 3, 30])
    for y in np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T:
        s = V23 @ y
        worst = max(worst, q @ s - M.C(q) - D * LOG2)
report("2a loss <= b D log2 at every vertex settlement", worst <= 1e-9, worst)
# attainment: need D <= n for the ray formula; use V = I_2 with vertex y
M2 = Maker(np.eye(2), 1.0); y = np.array([1.0, -1.0]); t = 60.0; q = t * y
att = q @ y - M2.C(q)
report("2b loss attains b D log2 along the vertex ray", abs(att - 2 * LOG2) < 1e-6, abs(att - 2 * LOG2))
s_out = np.array([1.5, 0.0]); losses = [t * (s_out @ np.array([1, 0])) - M2.C(t * np.array([1, 0])) for t in (10, 20, 40)]
report("2c settlement outside the zonotope: loss grows without bound", losses[2] - losses[1] > 0.99 * (losses[1] - losses[0]) > 0, losses[2] - losses[1])

# 3. price set / inverse -------------------------------------------------------------
Vs = np.array([[1.0, 0.3], [-0.2, 0.9]]); M = Maker(Vs, 2.0); worst = 0.0; inside = True
for _ in range(500):
    q = rng.normal(size=2) * 5
    p = M.p(q); yv = np.linalg.solve(Vs, p); inside &= bool(np.all(np.abs(yv) < 1))
    qb = M.b * np.linalg.solve(Vs.T, np.arctanh(yv)); worst = max(worst, np.abs(qb - q).max())
report("3  V^-1 p in (-1,1)^n and q(p(q)) = q", inside and worst < 1e-9, worst)

# 4. dead band / lasso ------------------------------------------------------------
def trader_opt(M, q, pi, f, iters=20000, step=None):
    """maximise <d,pi> - [C(q+d)-C(q)] - <f,|d|> by proximal gradient."""
    n = len(q); d = np.zeros(n)
    L = np.linalg.eigvalsh(M.H(np.zeros(n))).max() + 1e-9   # Lipschitz bound of grad C (sech^2 <= 1 at origin is the max)
    eta = 1.0 / L if step is None else step
    for _ in range(iters):
        g = pi - M.p(q + d)
        z = d + eta * g
        d_new = np.sign(z) * np.maximum(np.abs(z) - eta * f, 0.0)
        if np.abs(d_new - d).max() < 1e-13: d = d_new; break
        d = d_new
    return d
M = Maker(np.array([[1.0, 0.4], [0.0, 1.0]]), 1.5); ok = True; worst = 0.0
for _ in range(300):
    q = rng.normal(size=2); f = rng.uniform(0.01, 0.2, 2); p = M.p(q)
    pi = M.p(q + rng.normal(size=2) * 0.6)   # a belief inside the price set (outside it the trader's optimum is unbounded)
    d = trader_opt(M, q, pi, f)
    inband = np.all(np.abs(pi - p) <= f)
    zero = np.abs(d).max() < 1e-9
    ok &= (inband == zero)
    pn = M.p(q + d)
    for i in range(2):
        if abs(d[i]) > 1e-9: worst = max(worst, abs(pn[i] - (pi[i] - f[i] * np.sign(d[i]))))
        else: worst = max(worst, max(0.0, abs(pi[i] - pn[i]) - f[i]))
report("4  trade is zero iff pi in p(q)+[-f,f]^n; active-set conditions hold", ok and worst < 1e-7, worst)

# 5. conjugates add under inf-convolution (1-D, two fee-bearing makers) --------------
def feecost(b, q0, f):
    return lambda d: b * lc((q0 + d) / b) - b * lc(q0 / b) + f * abs(d)
def conj1d(fn, pi, grid):
    return np.max(pi * grid - np.array([fn(d) for d in grid]))
grid = np.linspace(-40, 40, 80001)
A, B = feecost(1.0, 0.3, 0.05), feecost(2.5, -0.4, 0.02)
def infconv(d, k=4001):
    ds = np.linspace(-40, 40, k); return np.min([A(x) + B(d - x) for x in ds])
worst = 0.0
for pi in rng.uniform(-0.8, 0.8, 12):
    lhs = np.max(pi * grid[::20] - np.array([infconv(d) for d in grid[::20]]))
    rhs = conj1d(A, pi, grid) + conj1d(B, pi, grid)
    worst = max(worst, abs(lhs - rhs))
report("5  (A box B)* = A* + B* for fee-bearing 1-D makers", worst < 1e-3, worst)   # coarse grids: 1e-3 is the grid resolution

# 6. one clearing price, horizontal summation, bisection vs direct minimisation ------
MAKERS = [(1.0, 0.02, 0.00), (3.0, 0.05, 0.02), (10.0, 0.10, -0.01)]
def supply_k(pi, b, f, p):
    q = b * np.arctanh(p)
    if pi > p + f: return b * np.arctanh(pi - f) - q
    if pi < p - f: return b * np.arctanh(pi + f) - q
    return 0.0
def clear(delta):
    fmin = min(m[1] for m in MAKERS)
    pi = brentq(lambda x: sum(supply_k(x, *m) for m in MAKERS) - delta, -1 + fmin + 1e-9, 1 - fmin - 1e-9, xtol=1e-13)
    return pi, np.array([supply_k(pi, *m) for m in MAKERS])
def direct(delta):
    bs = np.array([m[0] for m in MAKERS]); fs = np.array([m[1] for m in MAKERS]); qs = bs * np.arctanh([m[2] for m in MAKERS])
    def total(d):
        return float(np.sum(bs * lc((qs + d) / bs) - bs * lc(qs / bs) + fs * np.abs(d)))
    # proximal gradient on the simplex-free problem via penalty-free projection: parametrise d_K = delta - sum(others)
    d = np.full(3, delta / 3); eta = 0.02
    for _ in range(400000):
        g = np.tanh((qs + d) / bs)
        # projected gradient with L1 via prox, then re-project onto sum = delta (alternating; converges for this small problem)
        z = d - eta * g
        z = np.sign(z) * np.maximum(np.abs(z) - eta * fs, 0.0)
        # projection onto {sum = delta} that keeps exact zeros where the band is not crossed is the KKT fixed point; use Lagrange shift on active set
        act = np.abs(z) > 1e-12
        if act.any(): z[act] += (delta - z.sum()) / act.sum()
        else: z[np.argmin(fs)] = delta
        if np.abs(z - d).max() < 1e-14: d = z; break
        d = z
    return d, total(d)
worst = 0.0; zeros_exact = True; mult_ok = True
for delta in rng.uniform(-5, 5, 200):
    pi, fills = clear(delta)
    d_dir, _ = direct(delta)
    worst = max(worst, np.abs(fills - d_dir).max())
    for (b, f, p), x in zip(MAKERS, fills):
        if abs(pi - p) <= f: zeros_exact &= (x == 0.0)
        else: mult_ok &= abs(np.tanh((b * np.arctanh(p) + x) / b) - (pi - f * np.sign(x))) < 1e-9
report("6  bisection on closed-form supplies = direct minimisation; inactive fills exactly 0; one clearing price", worst < 1e-6 and zeros_exact and mult_ok, worst)

# 7. liquidity adds near the origin (fee-free, at origin) ----------------------------
Ms = [Maker(np.eye(2), 1.0), Maker(np.array([[1.0, 0.5], [0.0, 1.0]]), 2.0), Maker(np.array([[0.7, 0.0], [0.7, 1.2]]), 0.8)]
def infconv_val(d):
    # minimise sum C_k(d_k) s.t. sum d_k = d  (smooth, use BFGS on the first K-1 blocks)
    def obj(x):
        parts = [x[0:2], x[2:4], d - x[0:2] - x[2:4]]
        return sum(Mk.C(pk) for Mk, pk in zip(Ms, parts))
    r = minimize(obj, np.zeros(4), method="BFGS", options={"gtol": 1e-12}); return r.fun
h = 1e-3; Hn = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        e_i, e_j = np.eye(2)[i], np.eye(2)[j]
        Hn[i, j] = (infconv_val(h * e_i + h * e_j) - infconv_val(h * e_i - h * e_j) - infconv_val(-h * e_i + h * e_j) + infconv_val(-h * e_i - h * e_j)) / (4 * h * h)
Hinv_sum = sum(np.linalg.inv(Mk.H(np.zeros(2))) for Mk in Ms)
worst = np.abs(np.linalg.inv(Hn) - Hinv_sum).max()
report("7  inverse Hessian of the inf-convolution at 0 = sum of inverse Hessians", worst < 1e-4, worst)

# 8. Bregman identity ----------------------------------------------------------------
M = Maker(np.eye(2), 1.3); worst = 0.0
for _ in range(500):
    q = rng.normal(size=2) * 2; s = rng.uniform(-1, 1, 2); p = M.p(q)
    lhs = q @ s - M.C(q)
    Cstar = M.b * lcs(p).sum(); grad_Cstar = M.b * np.arctanh(p)   # q = grad C*(p)
    rhs = Cstar + grad_Cstar @ (s - p)
    worst = max(worst, abs(lhs - rhs))
report("8  <q,s> - C(q) = C*(p) + <grad C*(p), s - p>", worst < 1e-9, worst)

# 9. re-centring multiplies the bound -------------------------------------------------
b = 1.0; m = 3; t = 40.0; total = 0.0
for _ in range(m + 1):
    total += t - b * lc(t / b)        # each epoch: sold t units, settled at s = +1, cost b*lc(t/b)
worst = abs(total - (m + 1) * b * LOG2)
report("9  re-centred m times: loss -> (m+1) b log2", worst < 1e-4, worst)

print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
raise SystemExit(1 if FAILS else 0)
