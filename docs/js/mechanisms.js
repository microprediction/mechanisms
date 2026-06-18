/**
 * mechanisms.js — JavaScript ports of the core market mechanisms.
 *
 * These are faithful ports of the Python reference implementations in
 * `mechanisms/` (scoring_rules, lmsr, amm, pm_amm, parimutuel). They are tested
 * for numerical parity against Python by `tests/test_js_parity.py`, which runs
 * this module in Node via `tests/js_parity_runner.js` and compares results.
 *
 * The same file is loaded by the docs site for the interactive demos, so the
 * code the website runs is exactly the code the test suite checks against
 * Python. (Pattern borrowed from microprediction/humpday.)
 *
 * Dual export: `module.exports` under Node, `globalThis.Mechanisms` in a browser.
 */
(function (root) {
  "use strict";

  const SQRT2 = Math.sqrt(2);
  const SQRT2PI = Math.sqrt(2 * Math.PI);
  const EPS = 1e-15;

  // ---- scoring rules -----------------------------------------------------
  function logScore(p, y) {
    return -Math.log(Math.max(p[y], EPS));
  }

  function brierScore(p, y) {
    let s = 0;
    for (let i = 0; i < p.length; i++) {
      const e = i === y ? 1 : 0;
      s += (p[i] - e) * (p[i] - e);
    }
    return s;
  }

  function sphericalScore(p, y) {
    let norm = 0;
    for (let i = 0; i < p.length; i++) norm += p[i] * p[i];
    norm = Math.sqrt(norm);
    return 1 - p[y] / Math.max(norm, EPS);
  }

  function pinballLoss(z, y, tau) {
    return (y - z) * (tau - (y < z ? 1 : 0));
  }

  function intervalScore(lower, upper, y, alpha) {
    let s = upper - lower;
    if (y < lower) s += (2 / alpha) * (lower - y);
    else if (y > upper) s += (2 / alpha) * (y - upper);
    return s;
  }

  function crpsEnsemble(samples, y) {
    const m = samples.length;
    let t1 = 0;
    for (let i = 0; i < m; i++) t1 += Math.abs(samples[i] - y);
    t1 /= m;
    let t2 = 0;
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++) t2 += Math.abs(samples[i] - samples[j]);
    t2 /= m * m;
    return t1 - 0.5 * t2;
  }

  function energyScore(samples, y, beta) {
    beta = beta === undefined ? 1.0 : beta;
    const m = samples.length;
    const d = y.length;
    const dist = (a, b) => {
      let s = 0;
      for (let k = 0; k < d; k++) s += (a[k] - b[k]) * (a[k] - b[k]);
      return Math.pow(Math.sqrt(s), beta);
    };
    let t1 = 0;
    for (let i = 0; i < m; i++) t1 += dist(samples[i], y);
    t1 /= m;
    let t2 = 0;
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++) t2 += dist(samples[i], samples[j]);
    t2 /= m * m;
    return t1 - 0.5 * t2;
  }

  // ---- LMSR --------------------------------------------------------------
  function lmsrCost(q, b) {
    let mx = -Infinity;
    for (const qi of q) mx = Math.max(mx, qi / b);
    let s = 0;
    for (const qi of q) s += Math.exp(qi / b - mx);
    return b * (mx + Math.log(s));
  }

  function lmsrPrices(q, b) {
    let mx = -Infinity;
    for (const qi of q) mx = Math.max(mx, qi / b);
    const e = q.map((qi) => Math.exp(qi / b - mx));
    const tot = e.reduce((a, v) => a + v, 0);
    return e.map((v) => v / tot);
  }

  function lmsrCostToTrade(q, delta, b) {
    const q2 = q.map((qi, i) => qi + delta[i]);
    return lmsrCost(q2, b) - lmsrCost(q, b);
  }

  function lmsrMaxLoss(n, b) {
    return b * Math.log(n);
  }

  // ---- constant-product AMM ---------------------------------------------
  function cpammAmountOut(x, y, dx, fee) {
    const gamma = 1 - fee;
    const dxEff = dx * gamma;
    return (y * dxEff) / (x + dxEff);
  }

  function impermanentLoss(r) {
    return (2 * Math.sqrt(r)) / (1 + r) - 1;
  }

  // ---- pm-AMM ------------------------------------------------------------
  function normPdf(x) {
    return Math.exp(-0.5 * x * x) / SQRT2PI;
  }

  // erf via Abramowitz & Stegun 7.1.26 (matches Python math.erf to ~1e-7).
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y =
      1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
        0.284496736) *
        t +
        0.254829592) *
        t *
        Math.exp(-x * x);
    return sign * y;
  }

  function normCdf(x) {
    return 0.5 * (1 + erf(x / SQRT2));
  }

  function pmAmmPrice(x, y, L) {
    return normCdf((y - x) / L);
  }

  // ---- parimutuel --------------------------------------------------------
  function parimutuelProbabilities(stakes) {
    const tot = stakes.reduce((a, v) => a + v, 0);
    if (tot === 0) return stakes.map(() => 0);
    return stakes.map((w) => w / tot);
  }

  function parimutuelPayoutPerUnit(stakes, winner, takeout) {
    const tot = stakes.reduce((a, v) => a + v, 0);
    if (stakes[winner] === 0) return Infinity;
    return ((1 - takeout) * tot) / stakes[winner];
  }

  // --- aggregation (opinion pools) ----------------------------------------
  function _normWeights(m, weights) {
    let w = weights ? weights.slice() : new Array(m).fill(1 / m);
    const s = w.reduce((a, v) => a + v, 0);
    return w.map((v) => v / s);
  }
  function linearOpinionPool(forecasts, weights) {
    const m = forecasts.length, n = forecasts[0].length;
    const w = _normWeights(m, weights);
    const out = new Array(n).fill(0);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) out[j] += w[i] * forecasts[i][j];
    return out;
  }
  function logarithmicOpinionPool(forecasts, weights) {
    const m = forecasts.length, n = forecasts[0].length;
    const w = _normWeights(m, weights);
    const lp = new Array(n).fill(0);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        lp[j] += w[i] * Math.log(Math.min(Math.max(forecasts[i][j], 1e-15), 1.0));
    const mx = Math.max.apply(null, lp);
    const pool = lp.map((v) => Math.exp(v - mx));
    const s = pool.reduce((a, v) => a + v, 0);
    return pool.map((v) => v / s);
  }

  // --- perpetual futures funding ------------------------------------------
  function clampVal(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function fundingRate(premiumIndex, interestRate, clamp) {
    if (interestRate === undefined) interestRate = 0.0001;
    if (clamp === undefined) clamp = 0.0005;
    return premiumIndex + clampVal(interestRate - premiumIndex, -clamp, clamp);
  }
  function fundingPayment(positionNotional, rate) {
    return positionNotional * rate;
  }

  // --- frequent batch auction (uniform-price clearing) --------------------
  function clearUniformPrice(bids, asks) {
    if (!bids.length || !asks.length) return [null, 0.0];
    const cand = Array.from(
      new Set(bids.map((b) => b[0]).concat(asks.map((a) => a[0]))),
    ).sort((x, y) => x - y);
    let bestP = null, bestVol = 0.0;
    for (const p of cand) {
      let demand = 0, supply = 0;
      for (const [bp, q] of bids) if (bp >= p) demand += q;
      for (const [ap, q] of asks) if (ap <= p) supply += q;
      const vol = Math.min(demand, supply);
      if (vol > bestVol + 1e-12) { bestP = p; bestVol = vol; }
    }
    if (bestVol <= 0) return [null, 0.0];
    const hi = Math.max.apply(null, bids.filter((b) => b[0] >= bestP).map((b) => b[0]));
    const lo = Math.min.apply(null, asks.filter((a) => a[0] <= bestP).map((a) => a[0]));
    return [0.5 * (lo + hi), bestVol];
  }

  // --- local (Hyvärinen) scoring ------------------------------------------
  function gaussianHyvarinenScore(y, mu, sigma) {
    const s2 = sigma * sigma;
    const d1 = -(y - mu) / s2;
    const d2 = -1.0 / s2;
    return d2 + 0.5 * d1 * d1;
  }
  function fisherDivergenceGaussian(muP, sigmaP, muQ, sigmaQ) {
    const sp2 = sigmaP * sigmaP, sq2 = sigmaQ * sigmaQ;
    const a = 1.0 / sq2 - 1.0 / sp2;
    const b = muP / sp2 - muQ / sq2;
    return 0.5 * (a * a * sp2 + Math.pow(a * muP + b, 2));
  }

  // --- cost-function maker: quadratic potential ---------------------------
  function quadraticCost(q, alpha) {
    if (alpha === undefined) alpha = 1.0;
    let s = 0; for (const x of q) s += x * x;
    return 0.5 * alpha * s;
  }
  function quadraticPrices(q, alpha) {
    if (alpha === undefined) alpha = 1.0;
    return q.map((x) => alpha * x);
  }

  // --- combinatorial market (LMSR over joint binary states) ---------------
  function combinatorialPrices(q, b) { return lmsrPrices(q, b); }
  function combinatorialMarginal(q, b, nVars, i) {
    const p = lmsrPrices(q, b);
    let s = 0;
    for (let st = 0; st < p.length; st++) if ((st >> i) & 1) s += p[st];
    return s;
  }

  // --- perpetual demand lending pools -------------------------------------
  function linearFundingRate(L, S, p, p0, kappa) {
    return kappa * (L / S - p / p0);
  }
  function minCollateral(delta, eta, p0) {
    return (p0 * Math.abs(delta)) / Math.abs(eta);
  }
  function liquidationPrice(delta, eta, p0, c) {
    if (c === undefined || c === null) c = minCollateral(delta, eta, p0);
    return eta > 0 ? p0 - c / Math.abs(delta) : p0 + c / Math.abs(delta);
  }

  // --- continuous double auction: sweep a resting ask book ----------------
  function cdaSweepBuy(asks, takerPrice, qty) {
    const book = asks.map((a) => [a[0], a[1]]).sort((x, y) => x[0] - y[0]);
    let remaining = qty, filled = 0, cost = 0;
    for (const lvl of book) {
      if (remaining <= 1e-12 || takerPrice < lvl[0]) break;
      const fill = Math.min(remaining, lvl[1]);
      filled += fill; cost += fill * lvl[0]; remaining -= fill;
    }
    return [filled, cost];
  }

  // --- peer prediction: output agreement ----------------------------------
  function outputAgreement(reportA, reportB) {
    return reportA === reportB ? 1.0 : 0.0;
  }

  // --- hybrid CLOB+AMM binary market: one buy from a fresh LMSR -----------
  function hybridMarketBuy(side, qty, b, oppBook) {
    let remaining = qty, p2pCost = 0, p2pFilled = 0;
    const book = oppBook.map((o) => o.slice()).sort((a, c) => c[0] - a[0]);
    for (const lvl of book) {
      if (remaining <= 1e-12) break;
      const take = Math.min(remaining, lvl[1]);
      p2pCost += take * (1 - lvl[0]); p2pFilled += take; remaining -= take;
    }
    let ammCost = 0, ammFilled = 0;
    if (remaining > 1e-12) {
      const delta = side === 0 ? [remaining, 0] : [0, remaining];
      ammCost = lmsrCostToTrade([0, 0], delta, b);
      ammFilled = remaining;
    }
    return [p2pFilled, ammFilled, p2pCost + ammCost];
  }

  // --- decision market: action-conditional values and decision rules ------
  function conditionalExpectedValues(conditionalProbs, values) {
    return conditionalProbs.map((row) => row.reduce((s, p, o) => s + p * values[o], 0));
  }
  function argmaxDecision(values) {
    const mx = Math.max.apply(null, values);
    const best = values.map((v) => (v === mx ? 1 : 0));
    const k = best.reduce((a, b) => a + b, 0);
    return best.map((x) => x / k);
  }
  function softmaxDecision(values, temperature) {
    if (temperature === undefined) temperature = 1.0;
    const z = values.map((v) => v / temperature);
    const m = Math.max.apply(null, z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  function epsilonGreedyDecision(values, epsilon) {
    if (epsilon === undefined) epsilon = 0.1;
    const n = values.length, mx = Math.max.apply(null, values);
    const isBest = values.map((v) => v === mx);
    const k = isBest.filter(Boolean).length;
    return values.map((v, i) => epsilon / n + (isBest[i] ? (1 - epsilon) / k : 0));
  }

  const Mechanisms = {
    logScore,
    brierScore,
    sphericalScore,
    pinballLoss,
    intervalScore,
    crpsEnsemble,
    energyScore,
    lmsrCost,
    lmsrPrices,
    lmsrCostToTrade,
    lmsrMaxLoss,
    cpammAmountOut,
    impermanentLoss,
    normPdf,
    normCdf,
    pmAmmPrice,
    parimutuelProbabilities,
    parimutuelPayoutPerUnit,
    linearOpinionPool,
    logarithmicOpinionPool,
    fundingRate,
    fundingPayment,
    clearUniformPrice,
    gaussianHyvarinenScore,
    fisherDivergenceGaussian,
    quadraticCost,
    quadraticPrices,
    combinatorialPrices,
    combinatorialMarginal,
    linearFundingRate,
    minCollateral,
    liquidationPrice,
    cdaSweepBuy,
    outputAgreement,
    hybridMarketBuy,
    conditionalExpectedValues,
    argmaxDecision,
    softmaxDecision,
    epsilonGreedyDecision,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = Mechanisms;
  } else {
    root.Mechanisms = Mechanisms;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
