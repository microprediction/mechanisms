#!/usr/bin/env python3
"""Figures for 'Bounded Market Makers from Convex Duality'. Writes figures/*.pdf. Pure numpy/scipy/matplotlib."""
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "figure.dpi": 150})
LOG2 = np.log(2.0)

lc = lambda x: np.logaddexp(x, -x) - LOG2          # log cosh, stable
def lcs(y):                                          # conjugate of log cosh on [-1,1]
    y = np.clip(y, -1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(1 + y > 0, (1 + y) * np.log(1 + y), 0.0)
        c = np.where(1 - y > 0, (1 - y) * np.log(1 - y), 0.0)
    return 0.5 * (a + c)

# ---------------------------------------------------------------- fig 1: one contract
def fig1():
    q = np.linspace(-4, 4, 801)
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    ax[0].plot(q, lc(q), label=r"$\log\cosh q$")
    ax[0].plot(q, q**2 / 2, "--", label=r"$q^2/2$")
    ax[0].plot(q, np.abs(q) - LOG2, ":", label=r"$|q|-\log 2$")
    ax[0].set_ylim(-0.8, 4); ax[0].set_xlabel("position $q$"); ax[0].set_title("cost"); ax[0].legend()
    ax[1].plot(q, np.tanh(q)); ax[1].axhline(1, color="k", lw=0.5); ax[1].axhline(-1, color="k", lw=0.5)
    ax[1].set_xlabel("position $q$"); ax[1].set_title(r"price $p=\tanh q$")
    ax[2].plot(q, np.abs(q) - lc(q), label=r"loss $|q|-C(q)$ at $s=\mathrm{sgn}\,q$")
    ax[2].axhline(LOG2, color="r", lw=0.8, label=r"bound $b\log 2$")
    ax[2].set_xlabel("position $q$"); ax[2].set_title("worst-case loss"); ax[2].legend(loc="lower right")
    fig.tight_layout(); fig.savefig("figures/fig1_one_contract.pdf"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: the book
MAKERS = [(1.0, 0.02, 0.00), (3.0, 0.05, 0.02), (10.0, 0.10, -0.01)]   # (b, f, current price)

def supply_k(pi, b, f, p):
    q = b * np.arctanh(p)
    pi = np.asarray(pi, dtype=float)
    out = np.zeros_like(pi)
    hi = pi > p + f; lo = pi < p - f
    out[hi] = b * np.arctanh(np.clip(pi[hi] - f, -1 + 1e-12, 1 - 1e-12)) - q
    out[lo] = b * np.arctanh(np.clip(pi[lo] + f, -1 + 1e-12, 1 - 1e-12)) - q
    return out

def supply(pi, makers=MAKERS):
    return sum(supply_k(pi, *m) for m in makers)

def clear(delta, makers=MAKERS):
    fmin = min(m[1] for m in makers)
    lo, hi = -1 + fmin + 1e-9, 1 - fmin - 1e-9
    pi = brentq(lambda x: supply(np.array([x]), makers)[0] - delta, lo, hi, xtol=1e-12)
    return pi, [supply_k(np.array([pi]), *m)[0] for m in makers]

def fig2():
    pi = np.linspace(-0.6, 0.6, 2401)
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    for (b, f, p), ls in zip(MAKERS, ["-", "--", "-."]):
        ax[0].plot(pi, supply_k(pi, b, f, p), ls, lw=1, label=rf"$b={b:g},\ f={f:g}$")
    ax[0].plot(pi, supply(pi), "k", lw=1.6, label="aggregate")
    ax[0].set_xlabel(r"clearing price $\pi$"); ax[0].set_ylabel("supply $S(\\pi)$"); ax[0].set_title("supplies add horizontally")
    ax[0].set_ylim(-6, 6); ax[0].legend(loc="upper left")
    pz = np.linspace(-0.3, 0.3, 4801); depth = np.gradient(supply(pz), pz)
    ax[1].plot(pz, depth, "k", lw=1); ax[1].set_ylim(0, 16)
    for (b, f, p) in MAKERS:
        ax[1].axvline(p - f, color="C0", lw=0.5, alpha=0.6); ax[1].axvline(p + f, color="C3", lw=0.5, alpha=0.6)
    ax[1].set_xlabel(r"price $\pi$"); ax[1].set_title(r"depth $S'(\pi)$: a ladder of quotes")
    deltas = np.linspace(0, 6, 241)
    fills = np.array([clear(d)[1] for d in deltas])
    for k, (b, f, p) in enumerate(MAKERS):
        ax[2].plot(deltas, fills[:, k], label=rf"maker {k+1} ($f={f:g}$)")
    ax[2].set_xlabel(r"trade size $\delta$"); ax[2].set_ylabel("fill"); ax[2].set_title("sparse routing in fee order"); ax[2].legend()
    fig.tight_layout(); fig.savefig("figures/fig2_book.pdf"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: curvature
V = np.array([[1.0, 0.0, 0.8], [0.0, 1.0, 0.6]])
def price(q, b=1.0): return V @ np.tanh(V.T @ q / b)
def hess(q, b=1.0): return (V * (1 / np.cosh(V.T @ q / b) ** 2)) @ V.T / b

def zonotope_vertices(V):
    pts = np.array([V @ y for y in np.array(np.meshgrid(*[[-1, 1]] * V.shape[1])).reshape(V.shape[1], -1).T])
    c = pts.mean(0); ang = np.arctan2(*(pts - c).T[::-1])
    from scipy.spatial import ConvexHull
    h = ConvexHull(pts); return pts[h.vertices]

def fig3():
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    Z = zonotope_vertices(V); Z = np.vstack([Z, Z[:1]])
    ax[0].plot(Z[:, 0], Z[:, 1], "k", lw=1, label=r"$\mathcal{Z}_V$")
    ax[0].plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], ":", color="gray", lw=0.8, label=r"$[-1,1]^2$")
    ts = np.linspace(0, 6, 300); path = np.array([price(t * V[:, 2]) for t in ts])
    ax[0].plot(path[:, 0], path[:, 1], "C3", lw=1.5, label=r"price path, buying along $v_3$")
    ax[0].set_aspect("equal"); ax[0].set_title("price set = interior of the zonotope"); ax[0].legend(loc="lower right", fontsize=7)
    eig = np.array([np.linalg.eigvalsh(hess(t * V[:, 2])) for t in ts])
    ax[1].plot(ts, eig[:, 1], label="largest eigenvalue (loaded direction)")
    ax[1].plot(ts, eig[:, 0], label="smallest eigenvalue")
    ax[1].set_xlabel("inventory $t$ along $v_3$"); ax[1].set_title("curvature depletes in the loaded direction"); ax[1].legend(fontsize=7)
    H0, H1 = hess(np.zeros(2)), hess(3.0 * V[:, 2])
    M = np.hstack([H0, np.full((2, 1), np.nan), H1])
    im = ax[2].imshow(M, cmap="viridis", vmin=0, vmax=H0.max())
    for (i, j), v in np.ndenumerate(H0): ax[2].text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
    for (i, j), v in np.ndenumerate(H1): ax[2].text(j + 3, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax[2].set_xticks([0.5, 3.5]); ax[2].set_xticklabels([r"$H(0)=VV^\top$", r"$H(3v_3)$"]); ax[2].set_yticks([])
    ax[2].set_title("cross-impact: quoted, then loaded"); fig.colorbar(im, ax=ax[2], fraction=0.04, pad=0.04)
    fig.tight_layout(); fig.savefig("figures/fig3_curvature.pdf"); plt.close(fig)

if __name__ == "__main__":
    import os; os.makedirs("figures", exist_ok=True)
    fig1(); fig2(); fig3(); print("figures written")
