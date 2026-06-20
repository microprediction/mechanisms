# A normalization-free wagering pool (local-score parimutuel)

*Status: a worked generalization with an honest novelty assessment. Likely novel
as a combination; every ingredient is published. See §7.*

## 1. The gap

Two pieces already live in this repository but have never been bolted together:

- **Self-financing wagering mechanisms** (Lambert, Langford, Wortman Vaughan et
  al. 2008, 2015): a budget-balanced pool that pays each participant their
  wager-weighted **proper score** minus the wager-weighted average score of the
  field. Truthful for immutable, risk-neutral beliefs; the operator bears no
  risk. The construction works for *any* proper scoring rule.
- **Local proper scoring rules** (Parry, Dawid & Lauritzen 2012), implemented in
  [`local_scoring.py`](../mechanisms/local_scoring.py): rules that depend on the
  quoted density only through `log p` and a finite number of its derivatives at
  the outcome, the canonical case being the **Hyvärinen score**. Because they
  see only derivatives of `log p`, they can score an **unnormalized** density —
  the partition function `Z` cancels.

The wagering literature always instantiates its mechanism with a *global* score
(log, Brier), which forces every participant to submit a normalized density. The
local-scoring literature is about estimation, not markets. Their composition is a
wagering pool in which **participants may submit unnormalized densities — energy
based models — and no one, ever, computes a partition function.** Call it the
*local-score pool*.

## 2. The mechanism

Each of `n` participants holds wealth `Wᵢ`, risks a stake `sᵢ = b·Wᵢ` for a fixed
fraction `b ∈ (0,1)`, and submits an **unnormalized** log-density `log p̃ᵢ`
over `ℝ^d` (an energy function; a score network; an un-normalised mixture). The
outcome `z` is revealed. Write the local score in **reward** form (higher is
better),

```
R(z, p) = −Δ log p(z) − ½‖∇ log p(z)‖²        (Hyvärinen, m = 2).
```

Two budget-balanced payout rules, with different trade-offs (§6):

- **Additive (WSWM form).** `ΔWᵢ = sᵢ·( R(z,p̃ᵢ) − R̄_s )`, where `R̄_s = Σⱼ sⱼ Rⱼ / Σⱼ sⱼ`
  is the stake-weighted average score. `Σᵢ ΔWᵢ = 0` identically.
- **Multiplicative (parimutuel form).** Split the pot `S = Σ sⱼ` as
  `ΔWᵢ = S · sᵢ e^{βRᵢ} / Σⱼ sⱼ e^{βRⱼ} − sᵢ`, again `Σᵢ ΔWᵢ = 0`. This is exactly
  [`nearest_the_pin`](../mechanisms/nearest_the_pin.py) with the density value
  `qᵢ(z)` replaced by `e^{βRᵢ}`.

Either way the pool is a pure transfer: the operator holds no inventory and bears
no risk, exactly as in the racetrack tote.

## 3. Why an unnormalized submission cannot "hide mass"

The obvious objection: if I can submit `p̃` without normalizing, can't I inflate
my mass and steal the pot? For a **density-value** payout (the log score, which
is what nearest-the-pin uses, `pot ∝ qᵢ(z)`), yes — sending `q̃ = c·q` with
`c → ∞` wins everything, which is *why* that mechanism must normalize.

The local score is immune, for one structural reason: **it is scale-invariant.**

```
log(c·p̃) = log c + log p̃   ⟹   ∇log(c·p̃) = ∇log p̃,   Δlog(c·p̃) = Δlog p̃
```

so `R(z, c·p̃) = R(z, p̃)` for every `c > 0`. The normalizer is not concealed; it
is **not an argument of the payout**. There is nothing to game by misreporting
total mass, because total mass is never read.

## 4. What keeps participants honest

Honesty is enforced not by a normalization constraint but by the **derivative
penalty** `½‖∇log p‖²` (and the Laplacian `Δlog p`). This is the *local surrogate*
for "probability mass must come from somewhere": you cannot spike your log-density
at the outcome for free, because a steep, peaked log-density is charged for its own
slope and curvature.

Formally, the Hyvärinen score is strictly proper **relative to the Fisher
divergence** `½ ∫ p_true ‖∇log q − ∇log p_true‖²`, which vanishes iff
`∇log q = ∇log p_true`, i.e. `q ∝ p_true`. It pins the submitted density down *up
to exactly the multiplicative constant the mechanism ignores.* Shape is elicited;
level is free; the two are consistent.

**Worked example (Gaussian).** For `p = N(μ, σ²)` the normalizer `−½log(2πσ²)`
cancels and

```
R(z, p) = 1/σ² − ½ (z−μ)² / σ⁴.
```

Try to fake confidence by shrinking `σ`: the `1/σ²` reward is exactly clawed back
by the `(z−μ)²/σ⁴` penalty. Maximising `E_{z∼N(μ,τ²)}[R]` over `1/σ² =: a` gives
`a − ½τ²a²`, maximised at `a = 1/τ²`, so `σ² = τ²` — truthful reporting of the
variance, with the absolute level (`log(2πσ²)`, the term you would "pay for" under
the log score) never appearing.

## 5. The genuine limitation (and the hook)

Fisher-divergence properness only pins the shape **on connected support**. If the
truth is multimodal with **well-separated modes**, `∇log q = ∇log p` *within* each
mode constrains nothing about the *relative mass between* modes — so a participant
can shift probability from one mode to another and the local score will not see
it. On disconnected support **you genuinely can hide mass between modes.**

This is not a flaw peculiar to this mechanism; it is the well-known blind spot of
score matching (the reason annealed and diffusion-based score matching exist:
Song & Ermon 2019). It transfers directly: the honest fix here is the same one —
**anneal**. Score the submission at several noise scales (convolve the outcome,
or the densities, with Gaussians of decreasing width) and pool the scores; the
coarse scales connect the modes and restore identifiability of the mixing
proportions. That makes the pool's failure mode a *parameter*, not a surprise —
and ties it to an active ML literature.

## 6. Construction subtleties (honest)

- **Unbounded score / limited liability.** Local scores are unbounded (`Δlog p`,
  `‖∇log p‖²` can be arbitrarily large), so the *additive* WSWM payout can in
  principle charge a participant more than their stake. The *multiplicative* form
  bounds every share to `[0, S]` regardless of score magnitude and is the safer
  default — at the cost that its truthfulness needs its own argument (the
  nearest-the-pin small-stake Gibbs argument gives it for the log score; the
  general `e^{βR}` pot-split is only truthful in the `b → 0` limit and should be
  stated as such).
- **Smoothness / boundary terms.** The `m = 2` rule needs a twice-differentiable
  density on (an open subset of) `ℝ^d`; point masses and hard support boundaries
  require the boundary-corrected local rules of Parry–Dawid–Lauritzen.
- **What is elicited.** The pool elicits the *score field* `∇log p` — the shape of
  beliefs — not absolute event probabilities. It answers "where is the outcome,
  relatively" rather than "what is `P(A)`." That is a feature for high-dimensional
  continuous forecasting (no normalization, no KDE) and a non-feature for pricing
  a binary event; the two pools are complements, not substitutes.

## 7. Prior art and novelty

Calibrated verdict: **likely novel as a combination, narrow gap.** Searched
thoroughly; the two literatures are disjoint. Closest neighbours:

- Lambert, Langford, Wortman Vaughan, Chen, Reeves, Shoham & Pennock, *An
  Axiomatic Characterization of Wagering Mechanisms*, J. Econ. Theory 156 (2015)
  (conf. EC 2008): the WSWM template this instantiates. Allows any proper score;
  never instantiated with a local rule or unnormalized reports.
- Parry, Dawid & Lauritzen, *Proper Local Scoring Rules*, Ann. Statist. 40 (2012):
  the local rules; pure scoring theory, no market content.
- Freeman, Lahaie & Pennock, *Crowdsourced Outcome Determination in Prediction
  Markets* (AAAI 2017), and Freeman & Pennock, *An Axiomatic View of the Parimutuel
  Consensus Mechanism* (IJCAI 2018): proper-score-driven pools over finite
  outcomes, not continuous, not local, not unnormalized.
- Kilgour & Gerchak (2004) and **Johnstone (2007)**, *The Parimutuel Kelly
  Probability Scoring Rule*: competitive/parimutuel scoring, with the warning that
  the parimutuel Kelly score is honest only as a Nash equilibrium, not strictly
  proper — the relevant caution for §6's `b → 0` claim.

The bridge that makes the keystone legitimate — that the Hyvärinen score *is* a
proper local scoring rule usable for inference — is Dawid & Musio (2014, 2015).
The mode-mass limitation of §5 is documented directly by Wenliang & Kanagawa
(2020), Zhang et al. (2022), and Koehler, Heckett & Risteski (2023).

The defensible claim is precise: *the first self-financing wagering mechanism that
elicits unnormalized (energy-based) density forecasts, via a local proper scoring
rule, so that no partition function is ever computed.* No new truthfulness theory
is claimed — properness-implies-truthfulness is inherited from the wagering
literature and holds in the small-stake, risk-neutral, immutable-belief limit.

## 8. Open questions

1. **Multiplicative truthfulness.** Is the `e^{βR}` pot-split truthful beyond the
   `b → 0` limit for a local `R`, or only the additive form? (Cf. the finite-`b`
   caveats in the nearest-the-pin paper.)
2. **Annealed properness.** Does the multi-noise-scale pool of §5 recover strict
   properness relative to a *total* (across-scale) divergence, restoring
   mode-mass identifiability? What is the analogue of the Fisher divergence there?
3. **Sequential twin.** Is there a bounded-loss cost-function *market maker*
   priced by a local rule — a "score-matching market maker" — or do the unbounded
   continuous-outcome loss results (Gao–Pennock) obstruct it? (Separate note.)

## References

*Wagering / self-financing forecasting mechanisms.*
- Lambert, N. S., Langford, J., Wortman, J., Chen, Y., Reeves, D. M., Shoham, Y. &
  Pennock, D. M. (2008). "Self-Financed Wagering Mechanisms for Forecasting." *EC*,
  170–179. doi:10.1145/1386790.1386820.
- Lambert, N. S., Langford, J., Wortman Vaughan, J., Chen, Y., Reeves, D. M.,
  Shoham, Y. & Pennock, D. M. (2015). "An Axiomatic Characterization of Wagering
  Mechanisms." *J. Econ. Theory* 156, 389–416. doi:10.1016/j.jet.2014.03.012.
- Witkowski, J., Freeman, R., Wortman Vaughan, J., Pennock, D. M. & Krause, A.
  (2018). "Incentive-Compatible Forecasting Competitions." *AAAI* 32(1), 1282–1289;
  *Management Science* 69(3), 1354–1374 (2023). arXiv:2101.01816.
- Freeman, R., Lahaie, S. & Pennock, D. M. (2017). "Crowdsourced Outcome
  Determination in Prediction Markets." *AAAI* 31(1), 523–529. arXiv:1612.04885.
- Freeman, R. & Pennock, D. M. (2018). "An Axiomatic View of the Parimutuel
  Consensus Mechanism." *IJCAI*, 254–260. doi:10.24963/ijcai.2018/35.
- Raja, A. A., Pinson, P., Kazempour, J. & Grammatico, S. (2022). "A Market for
  Trading Forecasts: A Wagering Mechanism." arXiv:2205.02668.
- Kilgour, D. M. & Gerchak, Y. (2004). "Elicitation of Probabilities Using
  Competitive Scoring Rules." *Decision Analysis* 1(2), 108–113.
  doi:10.1287/deca.1030.0003.
- Johnstone, D. J. (2007). "The Parimutuel Kelly Probability Scoring Rule."
  *Decision Analysis* 4(2), 66–75. doi:10.1287/deca.1070.0091.

*Proper scoring rules: foundations.*
- Brier, G. W. (1950). "Verification of Forecasts Expressed in Terms of
  Probability." *Monthly Weather Review* 78(1), 1–3.
- Good, I. J. (1952). "Rational Decisions." *JRSS B* 14(1), 107–114.
- McCarthy, J. (1956). "Measures of the Value of Information." *PNAS* 42(9), 654–655.
- Savage, L. J. (1971). "Elicitation of Personal Probabilities and Expectations."
  *JASA* 66(336), 783–801.
- Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction,
  and Estimation." *JASA* 102(477), 359–378.
- Waghmare, K. & Ziegel, J. (2026). "Proper Scoring Rules for Estimation and
  Forecast Evaluation." *Annual Review of Statistics* 13, 271–296. arXiv:2504.01781.

*Local / m-local proper scoring rules.*
- Parry, M., Dawid, A. P. & Lauritzen, S. (2012). "Proper Local Scoring Rules."
  *Ann. Statist.* 40(1), 561–592. arXiv:1101.5011.
- Ehm, W. & Gneiting, T. (2012). "Local Proper Scoring Rules of Order Two."
  *Ann. Statist.* 40(1), 609–637. arXiv:1102.5031.
- Dawid, A. P., Lauritzen, S. & Parry, M. (2012). "Proper Local Scoring Rules on
  Discrete Sample Spaces." *Ann. Statist.* 40(1), 593–608. arXiv:1104.2224.
- Dawid, A. P. & Musio, M. (2014). "Theory and Applications of Proper Scoring
  Rules." *METRON* 72(2), 169–183. arXiv:1401.0398.
- Dawid, A. P. & Musio, M. (2015). "Bayesian Model Selection Based on Proper
  Scoring Rules." *Bayesian Analysis* 10(2), 479–499. arXiv:1409.5291.

*Score matching & score-based generative models.*
- Hyvärinen, A. (2005). "Estimation of Non-Normalized Statistical Models by Score
  Matching." *JMLR* 6, 695–709.
- Hyvärinen, A. (2007). "Some Extensions of Score Matching." *Comput. Statist. Data
  Anal.* 51(5), 2499–2512.
- Vincent, P. (2011). "A Connection Between Score Matching and Denoising
  Autoencoders." *Neural Computation* 23(7), 1661–1674.
- Lyu, S. (2009). "Interpretation and Generalization of Score Matching." *UAI*,
  359–366. arXiv:1205.2629.
- Song, Y., Garg, S., Shi, J. & Ermon, S. (2019). "Sliced Score Matching." *UAI*,
  PMLR 115, 574–584.
- Song, Y. & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the
  Data Distribution." *NeurIPS* 32. arXiv:1907.05600.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. & Poole, B.
  (2021). "Score-Based Generative Modeling through Stochastic Differential
  Equations." *ICLR*. arXiv:2011.13456.

*Mode-mass / mixture-weight blindness (the §5 limitation).*
- Wenliang, L. K. & Kanagawa, H. (2020). "Blindness of Score-Based Methods to
  Isolated Components and Mixing Proportions." arXiv:2008.10087.
- Zhang, M., Key, O., Hayes, P., Barber, D., Paige, B. & Briol, F.-X. (2022).
  "Towards Healing the Blindness of Score Matching." arXiv:2209.07396.
- Koehler, F., Heckett, A. & Risteski, A. (2023). "Statistical Efficiency of Score
  Matching: The View from Isoperimetry." *ICLR*. arXiv:2210.00726.

*Continuous / measurable-space prediction markets.*
- Gao, X. A., Chen, Y. & Pennock, D. M. (2009). "Betting on the Real Line." *WINE*,
  LNCS 5929, 553–560.
- Chen, Y., Ruberry, M. & Wortman Vaughan, J. (2013). "Cost Function Market Makers
  for Measurable Spaces." *EC*, 785–802. doi:10.1145/2482540.2482608.
- Dudík, M., Wang, X., Pennock, D. M. & Rothschild, D. M. (2021). "Log-time
  Prediction Markets for Interval Securities." *AAMAS*. arXiv:2102.07308.
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). "Efficient Market Making
  via Convex Optimization, and a Connection to Online Learning." *ACM TEAC* 1(2).

*Parimutuel theory & the Fisher-market connection.*
- Eisenberg, E. & Gale, D. (1959). "Consensus of Subjective Probabilities: The
  Pari-Mutuel Method." *Ann. Math. Statist.* 30(1), 165–168.
- Pennock, D. M. (2004). "A Dynamic Pari-Mutuel Market for Hedging, Wagering, and
  Information Aggregation." *EC*, 170–179.
- Zhang, L. (2011). "Proportional Response Dynamics in the Fisher Market." *Theor.
  Comput. Sci.* 412(24), 2691–2698.
- Agrawal, S., Delage, E., Peters, M., Wang, Z. & Ye, Y. (2011). "A Unified
  Framework for Dynamic Prediction Market Design." *Oper. Res.* 59(3), 550–568.
