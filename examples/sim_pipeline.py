"""Simulation: composing two mechanisms — an elicitation market and a calibration critic.

This is the "algebra of mechanisms" made concrete (see
research/composition-and-the-algebra-of-mechanisms.md). Two mechanisms are
chained the way a `skaters` pipeline chains a transform with a model:

  1. ELICITATION MARKET (the transformation).  A pool of forecasters each
     report a predictive distribution. Their wealth-weighted aggregate is a
     linear opinion pool — a single predictive distribution F_t. F_t *is* the
     transformation: the coordinate change that should turn outcomes into noise.

  2. CALIBRATION CRITIC (test for uniformity).  Apply the probability-integral
     transform u_t = F_t(x_t); if F_t is right the u_t are Uniform(0,1) and
     z_t = Phi^{-1}(u_t) is standard normal. A critic measures how far the
     z-stream departs from uniform — its detectable edge is exactly the
     aggregate's miscalibration.

The two compose through a feedback loop: each forecaster's wealth is updated by
its log score (the proper-scoring "gradient"), so wealth flows to the calibrated
forecasters, the aggregate F_t sharpens to the truth, and the critic's edge
collapses. That loop is the nearest-the-pin / z-stream mechanism of the
microprediction vision (Cotton, *Microprediction*, MIT Press, 2022, where
z-streams ran on a live platform), seen as a *composition* of elicitation and
calibration.

Run:  python examples/sim_pipeline.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import numpy as np
from _viz import section, sparkline, labeled_bars

from mechanisms.aggregation import linear_opinion_pool
from mechanisms.scoring_rules import log_score

# A shared outcome grid; a categorical pmf over it is a discrete `Dist` whose
# CDF is a cumulative sum — the lingua franca every stage consumes and emits.
GRID = np.arange(-6.0, 6.0001, 0.05)


def gauss_pmf(mu, sd):
    p = np.exp(-0.5 * ((GRID - mu) / sd) ** 2)
    return p / p.sum()


def norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation, no scipy)."""
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    p = min(max(p, 1e-12), 1 - 1e-12)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5; r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def uniformity_error(u):
    """Total-variation distance of the PIT histogram from Uniform(0,1)."""
    if len(u) == 0:
        return 0.0
    h, _ = np.histogram(u, bins=10, range=(0.0, 1.0))
    return 0.5 * float(np.sum(np.abs(h / len(u) - 0.1)))


def pit_stream(P, weights, ys):
    """Replay outcomes ys through the weight-weighted aggregate; return PIT values."""
    agg = linear_opinion_pool(P, weights=weights)
    cdf = np.cumsum(agg)
    # randomisation-free mid-PIT: P(X < grid[y]) + 0.5 P(X = grid[y]); uniform if correct
    return np.array([float(cdf[y] - 0.5 * agg[y]) for y in ys])


def simulate(steps=600, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)

    # The truth: outcomes are iid N(0, 1). The best forecast is therefore N(0,1);
    # the only thing to "discover" is calibration.
    truth = gauss_pmf(0.0, 1.0)

    # The pool: two calibrated reports, drowned out at the start by a bull, a
    # bear, and an overconfident report. Equal-weighted, that aggregate is badly
    # miscalibrated — the spread the critic detects and arbitrages away.
    forecasters = [
        ("calibrated-A", gauss_pmf(0.0, 1.0)),
        ("calibrated-B", gauss_pmf(0.0, 1.0)),
        ("bull (biased +)", gauss_pmf(1.8, 0.9)),
        ("bear (biased -)", gauss_pmf(-1.8, 0.9)),
        ("overconfident", gauss_pmf(0.0, 0.5)),
    ]
    names = [n for n, _ in forecasters]
    P = np.array([p for _, p in forecasters])          # (K, G) the reports
    K = len(forecasters)

    w0 = np.ones(K) / K
    wealth = w0.copy()
    ys, cal_share = [], []

    for t in range(steps):
        # STAGE 1 — elicitation market: wealth-weighted linear opinion pool (F_t).

        # Nature reveals an outcome.
        y = int(rng.choice(len(GRID), p=truth))
        ys.append(y)

        # FEEDBACK — proper score is the gradient: wealth flows by log score.
        rewards = np.array([-log_score(P[i], y) for i in range(K)])  # = log p_i(y)
        wealth = wealth * np.exp(lr * (rewards - rewards.max()))
        wealth /= wealth.sum()
        cal_share.append(float(wealth[0] + wealth[1]))   # share on the calibrated two

    # Clean before/after: replay the SAME outcomes through the initial-weight and
    # final-weight aggregate, isolating what the feedback loop changed.
    pit_before = pit_stream(P, w0, ys)
    pit_after = pit_stream(P, wealth, ys)
    z_before = np.array([norm_ppf(u) for u in pit_before])
    z_after = np.array([norm_ppf(u) for u in pit_after])

    section("The pipeline: elicitation market  →  calibration critic")
    print("  forecaster pool (each reports a predictive distribution):")
    for n, p in forecasters:
        mu = float(GRID @ p)
        sd = float(np.sqrt(GRID**2 @ p - mu**2))
        print(f"    {n:<16}  N(mu={mu:+.2f}, sd={sd:.2f})")
    print("  truth: outcomes are iid N(0.00, 1.00); the calibrated reports are right.")

    section("Before vs after the loop (same outcomes, initial vs final weights)")
    print("  uniformity error of the z-stream (TV from uniform; lower = calibrated):")
    print(f"    equal-weighted aggregate : {uniformity_error(pit_before):.3f}")
    print(f"    converged aggregate      : {uniformity_error(pit_after):.3f}")
    print("  z-stream bias (mean z; target 0):")
    print(f"    equal-weighted aggregate : {z_before.mean():+.3f}")
    print(f"    converged aggregate      : {z_after.mean():+.3f}")

    print("  (bias is ~0 throughout — the symmetric bull/bear cancel — so the MEAN")
    print("  never reveals the problem; the PIT shape does. That is why the critic")
    print("  tests uniformity of the whole z-stream, not a moment or two.)")

    section("The feedback loop: wealth flows to the calibrated reports")
    print("  share of wealth on the two calibrated forecasters over the run:")
    print(f"    {sparkline(cal_share)}")
    print(f"    start = {cal_share[0]:.2f}   →   end = {cal_share[-1]:.2f}")

    section("Where the wealth ended up (the market selected the calibrated reports)")
    labeled_bars(list(zip(names, wealth)), vmax=1.0, fmt="{:.3f}")


if __name__ == "__main__":
    simulate()
