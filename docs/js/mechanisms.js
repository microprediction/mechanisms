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
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = Mechanisms;
  } else {
    root.Mechanisms = Mechanisms;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
