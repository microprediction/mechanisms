"""Figure for 'Likelihood versus CRPS: A New Perspective'.

Fit a Gaussian N(0, sigma^2) to Student-t(3) truth and plot, against the forecast
width sigma, the population mean CRPS and mean negative log-likelihood. The two
proper scores are minimized at different widths: CRPS prefers a narrower forecast
(the tail squished in) than likelihood. Curves are computed by numerical
integration, so the figure is deterministic. Writes crps-tail-blindness.{svg,pdf}.
"""
import os
import numpy as np
from scipy import stats, integrate
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DF = 3
SQRT_PI = np.sqrt(np.pi)
f = lambda y: stats.t.pdf(y, DF)


def gauss_crps_at(y, s):
    w = y / s
    return s * (w * (2 * stats.norm.cdf(w) - 1) + 2 * stats.norm.pdf(w) - 1 / SQRT_PI)


def mean_crps(s):
    val, _ = integrate.quad(lambda y: gauss_crps_at(y, s) * f(y), -np.inf, np.inf, limit=200)
    return val


def mean_nll(s):
    return 0.5 * np.log(2 * np.pi * s * s) + 3.0 / (2 * s * s)  # E[Y^2] = Var(t3) = 3


sC = minimize_scalar(mean_crps, bounds=(0.8, 2.0), method="bounded").x
sL = float(np.sqrt(3.0))

sig = np.linspace(0.6, 3.2, 140)
crps = np.array([mean_crps(s) for s in sig])
nll = np.array([mean_nll(s) for s in sig])
c_crps, c_nll = "#1a9850", "#762a83"

fig, ax1 = plt.subplots(figsize=(7.4, 4.3))
fig.patch.set_facecolor("white")
ax1.plot(sig, crps, color=c_crps, lw=2.3)
ax1.set_xlabel(r"Gaussian forecast width  $\sigma$   (smaller = tail squished in $\rightarrow$)", fontsize=10.5)
ax1.set_ylabel("mean CRPS", color=c_crps, fontsize=10.5)
ax1.tick_params(axis="y", labelcolor=c_crps)
ax1.axvline(sC, color=c_crps, ls=":", lw=1.3)
ax2 = ax1.twinx()
ax2.plot(sig, nll, color=c_nll, lw=2.3)
ax2.set_ylabel("mean negative log-likelihood", color=c_nll, fontsize=10.5)
ax2.tick_params(axis="y", labelcolor=c_nll)
ax2.axvline(sL, color=c_nll, ls=":", lw=1.3)
ax1.scatter([sC], [crps.min()], color=c_crps, s=85, zorder=5, edgecolor="white", linewidth=1.2)
ax2.scatter([sL], [nll.min()], color=c_nll, s=85, zorder=5, edgecolor="white", linewidth=1.2)
ax1.annotate(f"CRPS wants $\\sigma$={sC:.2f}\n(narrow tail)", (sC, crps.min()),
             xytext=(sC + 0.02, crps.min() + 0.05), color=c_crps, fontsize=10,
             fontweight="bold", ha="center", va="bottom")
ax2.annotate(f"likelihood wants $\\sigma$={sL:.2f}\n(covers the tail)", (sL, nll.min()),
             xytext=(sL + 0.66, nll.min() + 0.13), color=c_nll, fontsize=10,
             fontweight="bold", ha="center", va="bottom")
ax1.axvspan(sC, sL, color="#bbbbbb", alpha=0.22)
ax1.set_xlim(sig.min(), sig.max())
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
fig.tight_layout()
here = os.path.dirname(os.path.abspath(__file__))
for ext in ("svg", "pdf"):
    fig.savefig(os.path.join(here, f"crps-tail-blindness.{ext}"), bbox_inches="tight")
print(f"sigma_CRPS={sC:.4f}  sigma_NLL={sL:.4f}")
print("wrote crps-tail-blindness.svg and .pdf")
