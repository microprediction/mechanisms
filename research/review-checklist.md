# Review checklist (adversarial review at eeb63ae)

Working list for the mathematical review of *Non-Convex Market Makers* and
its reference implementations. Each item is marked DONE when both the claim
and its verification are fixed, and cross-checked at the end.

## Mandatory corrections

- [ ] M1. `C(q)=100q` paragraph in the paper is false: the untruncated
  interval is `[100-f, 100+f]`, not `{100}`. Give the exact table and the
  `f = 99` threshold.
- [ ] M2. Proposition 1 needs compact payoffs (min attained) or a
  uniform-margin definition of arbitrage. Pointwise-strict differs from
  positive worst-case margin without compactness.
- [ ] M3. Proposition 2's attainment example needs lower semicontinuity;
  coercive alone is not enough.
- [ ] M4. "Between contact points the envelope is affine" is too broad;
  restrict to a connected component of `{C > C**}`, and qualify the jump.
- [ ] M5. Sparsity is not guaranteed: exact dead zones yes, sparse
  allocation no (identical makers split evenly). Fix abstract, Lemma 6(iii),
  the lasso analogy, and the module's "fee order" claim.
- [ ] M6. `route` bypasses internal clearing at zero demand; the special
  case must go and the root-find must run.
- [ ] M7. `supply` silently clips outside its dual domain; distinguish
  finite optimizer, finite value with no finite optimizer, and unbounded.
- [ ] M8. Proposition 11's "below the fee is suppressed by Proposition 3"
  overreaches; a below-fee local margin only kills first-order local trades.
- [ ] M9. Independent exact oracles instead of comparing a discretization
  with itself.

## Strengthenings offered by the review

- [x] S1. Lemma 5: `N_0(q) = ∂C**(q)` on contact, empty off it; contact iff
  `L(q) <= U(q)`, with no attainment hypothesis.
- [x] S2. Lemma 5: exact statewise minimum fee
  `f_min(q) = max{0, L-b, a-U, (L-U)/2}`.
- [ ] S3. Proposition 3 as an iff in the arbitrage depth.
- [ ] S4. Coherence does not imply bounded loss; add the worst-case-loss
  formula and a coherent, non-convex, bounded-loss example.
- [ ] S5. Moreau identity `(e_λ C)** = e_λ(C**)`, and the theorem that for
  continuous `C` with proper envelope, `e_λ C` is convex iff `C` is.
- [ ] S6. Proposition 9(ii) for more than two branches: the jump is
  `(max P_λ - min P_λ)/λ`.
- [x] S7. Proposition 9(i) is sharp: affine costs attain the chord bound.
- [ ] S8. Proposition 10: signed positions, "profit at least", and a
  PSD-but-binary-incoherent example (cut polytope).
- [ ] S9. Proposition 11: quantitative local statement under an
  `L`-Lipschitz gradient, with the certified step size.
- [ ] S10. Corollary 8: "strictly increasing on every open interval on which
  at least one maker is active".
- [ ] S11. Proposition 9 hypotheses: `λ, c, α > 0`, finite and lsc `C`.
- [ ] S12. Accessible coherence: make the search randomness explicit and fix
  units.
- [ ] S13. Do not claim a universal `1/λ` gap rate; scope it.

## Verification architecture

- [ ] V1. Exact piecewise-linear oracles (envelope, Moreau, chord) returning
  certificates.
- [x] V2. Exact `O(n)` max-sure-profit scan with witness; retire `stride`.
- [ ] V3. Certificates rather than booleans (witness chord, argmin set, KKT
  residual, primal-dual gap, separating portfolio).
- [ ] V4. LP oracle for finite outcome spaces (Props 1, 3, 11 at once).
- [x] V5. Input-domain validation (lengths, sortedness, finiteness,
  `lam > 0`, stride, fees).
- [ ] V6. Wiggly contact test splits by dual domain (`|mu| <= 0.55` vs
  `|mu| > 0.6` with expanding domain).
- [ ] V7. Fixed-mesh expanding-window Moreau test with argmin recorded and
  boundary minimizers flagged; curvature normalized by `dx^2`.
- [ ] V8. Threshold bracketing everywhere: `eps - d`, `eps`, `eps + d`.
- [ ] V9. Metamorphic tests (shift, tilt, reflect, permute, split a maker,
  rescale) and the Moreau semigroup `e_λ(e_μ C) = e_{λ+μ} C`.
- [ ] V10. Structured generators: controlled-violation costs, non-attained
  thresholds, multi-well costs, routing ties, near-singular PSD, binary
  support enumeration.

## Site review (second report)

- [ ] P1. Proper-scoring page: reward/loss sign convention is backwards
  (`G = S(q;q)` convex in reward form; `H = L(q;q)` concave in loss form).
- [ ] P2. Cost-function-maker page: convexity is not path independence, is
  not sufficient for no-arbitrage (quadratic counterexample), and the
  worst-case-loss formula `sup C - inf C` is false (LMSR is `b log n`).
  Replace with `C = R^*` on the payoff polytope.
- [ ] P3. `b` is the inverse learning rate, not the learning rate (also on
  the Connections page).
- [ ] P4. CFMM correspondence is a level-set/perspective duality, not the
  bare Fenchel conjugate.
- [ ] P5. Decision markets need the inverse-selection-probability factor,
  not merely full support.
- [ ] P6. Finite-ensemble CRPS is not truthful for iid sampling; the fair
  U-statistic form is, and `m = 1` is the counterexample.
- [ ] P7. Opinion pooling: log pool does not multiply independent evidence
  without a prior correction, is not "sharper than its members", and the
  linear pool is not intrinsically underconfident.
- [ ] P8. Local scoring: CRPS needs no density; the log score, not "ordinary
  proper rules", is the 0-local one; Fisher-divergence strictness needs
  regularity and connected support.
- [ ] P9. Likelihood-vs-CRPS: score differences telescope for any score, so
  the path-independence uniqueness claim is false; state the narrower
  residual-decomposition claim.
- [ ] P10. Label every connection: exact equivalence, equivalence under
  hypotheses, composition identity, equilibrium result, or analogy; and
  qualify "truthful" with its equilibrium concept.
- [ ] P11. Smaller: LMSR worst-case not expected loss; margining does not
  restore ex-post IR; batching does not remove all MEV; pinball minimizer
  unique only for unique quantiles.
