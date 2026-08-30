# Review checklist (adversarial review at eeb63ae)

Working list for the mathematical review of *Non-Convex Market Makers* and
its reference implementations. Each item is marked DONE when both the claim
and its verification are fixed, and cross-checked at the end.

## Mandatory corrections

- [x] M1. `C(q)=100q` paragraph in the paper is false: the untruncated
  interval is `[100-f, 100+f]`, not `{100}`. Give the exact table and the
  `f = 99` threshold. Table added; the threshold is the arbitrage depth, and
  the old text confused the untruncated interval's non-emptiness with the
  truncated one's.
- [x] M2. Proposition 1 needs compact payoffs (min attained) or a
  uniform-margin definition of arbitrage. Pointwise-strict differs from
  positive worst-case margin without compactness. Compactness added as an
  explicit hypothesis, with the uniform-margin substitute spelled out for a
  non-compact outcome set.
- [x] M3. Proposition 2's attainment example needs lower semicontinuity;
  coercive alone is not enough. Both hypotheses now stated, with the punctured
  parabola `h(x) = x^2` off zero and `1` at zero as the counterexample, and the
  two failure modes (escape to infinity, escape at a point) separated.
- [x] M4. "Between contact points the envelope is affine" is too broad;
  restrict to a connected component of `{C > C**}`, and qualify the jump.
  Restated for one component with its endpoints, noting that two contact
  points need not bound a gap and that straddling several components gives
  several affine pieces with different slopes.
- [x] M5. Sparsity is not guaranteed: exact dead zones yes, sparse
  allocation no (identical makers split evenly). Fix abstract, Lemma 6(iii),
  the lasso analogy, and the module's "fee order" claim. All four fixed. The
  claim is now the exact dead zone, with `n` identical makers splitting
  `size/n` at every demand as the counterexample, and a three-fee example
  showing the split filling in from 1 to 3 makers as demand grows.
- [x] M6. `route` bypasses internal clearing at zero demand; the special
  case must go and the root-find must run. Removed. Two symmetric makers at
  `q = +-1.5` with `f = 0.05` quote `[0.855, 0.955]` and `[-0.955, -0.855]`;
  the root-find crosses them at 1.45 units for a total effective cost of
  `-1.56` against `0` for doing nothing. Consistent quotes still give exact
  zeros.
- [x] M7. `supply` silently clips outside its dual domain; distinguish
  finite optimizer, finite value with no finite optimizer, and unbounded.
  Added `dual_domain(p)` returning that trichotomy, and `supply` now returns
  `+-inf` outside rather than the arctanh of `1 - eps`, which looked like a
  capacity and was only a statement about floating point.
- [x] M8. Proposition 11's "below the fee is suppressed by Proposition 3"
  overreaches; a below-fee local margin only kills first-order local trades.
  The clause is out; the paragraph now says what is actually ruled out (the
  infinitesimal trade in that one direction) and what is not (other
  directions, and trades large enough for curvature to repay the fee).
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

- [x] P1. Proper-scoring page: reward/loss sign convention is backwards
  (`G = S(q;q)` convex in reward form; `H = L(q;q)` concave in loss form).
- [x] P2. Cost-function-maker page: convexity is not path independence, is
  not sufficient for no-arbitrage (quadratic counterexample), and the
  worst-case-loss formula `sup C - inf C` is false (LMSR is `b log n`).
  Replace with `C = R^*` on the payoff polytope.
- [x] P3. `b` is the inverse learning rate, not the learning rate (also on
  the Connections page).
- [x] P4. CFMM correspondence is a level-set/perspective duality, not the
  bare Fenchel conjugate.
- [x] P5. Decision markets need the inverse-selection-probability factor,
  not merely full support.
- [x] P6. Finite-ensemble CRPS is not truthful for iid sampling; the fair
  U-statistic form is, and `m = 1` is the counterexample.
- [x] P7. Opinion pooling: log pool does not multiply independent evidence
  without a prior correction, is not "sharper than its members", and the
  linear pool is not intrinsically underconfident. Verified against primary
  sources. The identity `pi prod (p_i/pi)^{w_i} = pi^{1-sum w_i} prod p_i^{w_i}`
  shows the correction vanishes exactly when weights sum to one, so the
  averaging pool and the evidence-multiplying product are different operators.
  Sharpness counterexample: `(0.98,.01,.01)` and `(.01,0.98,.01)` pool to
  entropy 0.853 against members' 0.112, and above the linear pool's 0.742; the
  Gaussian case is the harmonic mean of variances, so never sharper than the
  sharpest member. Separated the entropy theorem from the calibration one, and
  fixed the docstring that called the linear pool externally Bayesian.
- [x] P8. Local scoring: CRPS needs no density; the log score, not "ordinary
  proper rules", is the 0-local one; Fisher-divergence strictness needs
  regularity and connected support. All three corrected. Added that the
  discrete locality theorem needs more than two outcomes (binary is vacuous,
  since `p_0 = 1 - p_1`), that PDL's rules exist for even orders only, and a
  numerical counterexample where the Fisher divergence between two disjoint
  smooth bumps is 1e-24 while total variation is 0.49.
- [x] P9. Likelihood-vs-CRPS: score differences telescope for any score, so
  the path-independence uniqueness claim is false; state the narrower
  residual-decomposition claim. The note said in one paragraph that
  telescoping is generic and in the next that path independence singles out
  the log score. Every proper score's market form has a path-independent cost
  (Abernethy, Chen & Wortman Vaughan 2013), so the surviving claim is that the
  log score's *increment* is a proper score of the residual, which follows
  from locality; uniqueness under composition is left open.
- [ ] P10. Label every connection: exact equivalence, equivalence under
  hypotheses, composition identity, equilibrium result, or analogy; and
  qualify "truthful" with its equilibrium concept.
- [x] P11. Smaller: LMSR worst-case not expected loss; margining does not
  restore ex-post IR; batching does not remove all MEV; pinball minimizer
  unique only for unique quantiles. All four corrected in place, with the
  batching one rewritten as ordering-game-to-inclusion-game rather than
  removal, and the margining one as a probabilistic enforcement bound.

## Implementation audit (third report, at 0c4b878)

- [x] A1. FBA `clear_uniform_price` returns a price at which the reported
  volume cannot trade (bids 100/1000 against asks 1/99 gave 500.5). Use the
  marginal accepted orders for the band, and add allocations with pro-rata
  rationing at the margin. Second pass: the first allocation fix filled every
  strictly crossing order completely, so a bid of 10 against an ask of 4 took
  10 against 4 units of supply. Replaced with a price-priority walk that cuts
  at the level where volume runs out. `BatchAuction.clear` now consumes the
  batch and returns fills, with `keep_unfilled` to carry residuals forward.
- [x] A2. Dynamic parimutuel does not conserve cash (one \$1 buy settles to
  \$0.67) and purchases are path dependent (\$1 buys 2.0 shares, two 50c
  buys 1.75). Replace the pre-trade linear price with the integrated
  share-ratio cost `C(s) = kappa ||s||_2`.
- [x] A3. `brier_decomposition` is not an identity when a bin holds distinct
  forecasts: `([0.1, 0.9], [0, 1], n_bins=1)` implies 0.25 against an actual
  0.01. Return the within-bin residual and offer exact grouping.
- [x] A4. `target_weight_trade` clips coordinatewise, which satisfies neither
  the weights nor the value: `R = (100, 0)` fully locked answered `(2/3, 1/3)`
  at value 150 when asked for `(1/2, 1/2)` at 100. Solve the constrained
  program; report infeasibility with the minimum feasible value.
- [x] A5. `decision_market.py` does not implement the `1/R(P, a)`
  inverse-selection settlement the page now describes. Added
  `inverse_selection_score`, `expected_decision_score` and
  `DecisionMarket.settle`, and a grid search that exhibits a strictly
  profitable misreport (0.30 reported as 0.15) when the factor is dropped.
  Second pass added `select` (samples an action, refuses a rule with a
  zero-probability action) and recorded trader positions, so settlement pays
  holders of the realised outcome in the taken market and voids the rest.
- [x] A6. Peer prediction: output agreement is not truthful in general;
  distinguish Correlated Agreement from Enforced Agreement; BTS validation
  claims need their assumptions. Verified against primary sources. Output
  agreement needs self-dominance, not correlation, and the counterexample is
  now a test. Enforced Agreement is a separate mechanism (Zhang, Xu, Pennock &
  Schoenebeck 2025) and is implemented; CA is informed-truthful, a correlated
  equilibrium, not SD-truthful. `_ca_sign_matrix` returned numpy's three-valued
  sign where Shnayder et al.'s is 1-or-0. Added `correlated_agreement_df`, BTS
  input validation, and BTS's infinite-population and non-uniqueness caveats.
- [x] A7. Nearest-the-pin truthfulness needs its equilibrium concept
  (price-taking, Kelly, symmetric) stated wherever it is asserted.
- [x] A8. PDLP delta-hedge section should say which results are the paper's
  and which are standard mean-variance. Also added the Jiang & Wen (2026,
  arXiv:2604.16467) challenge, verified against the arXiv record: they prove
  the sufficient condition behind Claim 3.4 is self-contradictory and that no
  target weight mechanism lowers the delta uniformly. And validated the
  module's degenerate inputs (zero leverage, zero position, zero gamma, zero
  open interest, zero reserves), each of which divided by zero.
- [x] A9. Homepage and Connections slogans disagree with the corrected
  mechanism pages. The CFMM page itself had been missed too.
- [x] A10. Domain validation across the library: pools accepted negative
  weights and negative forecast components and normalised zero-sum weights by
  zero; `depth_trimmed_mean` validated nothing; `kelly_fraction` accepted
  probabilities outside `[0, 1]`; `kelly_allocation` normalised negative
  beliefs; `energy_score` reshaped a mismatched ensemble instead of erroring;
  six PDLP entry points divided by zero. Calibration was done earlier with A3.
