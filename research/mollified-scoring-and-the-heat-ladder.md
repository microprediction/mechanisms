# Mollified scoring and the heat-ladder pool

*Status: a candidate-original mechanism with a calibrated novelty assessment (§6).
Three claims of increasing ambition: a closed-form improperness (§1), its
symmetric repair (§2), and a scale-decomposed wagering mechanism (§3–4). The
ingredients are published; the closed form, the strictness closure, and the
assembly appear to be new. Every identity below is verified by direct
computation. **Full write-up with proofs:**
[papers/scoring-point-cloud-distributional-submissions.md](../papers/scoring-point-cloud-distributional-submissions.md).*

## 1. The flaw: score a KDE at the raw outcome and you elicit a deconvolution

The density pot-split of the [nearest-the-pin pool](../mechanisms/nearest_the_pin.py)
(and the reward rule of monteprediction-style contests) scores a participant by
the kernel density estimate of their submitted cloud, evaluated at the raw
outcome `z`. Write `ρ` for the density the participant samples their cloud
from, `φ_h` for the KDE kernel, and `ρ_h = ρ * φ_h` for the smoothed submission.
Under the log score the expected payoff is

```
E_{z~p*} [ log ρ_h(z) ]  =  ∫ p* log(ρ * φ_h),
```

which is maximized not at `ρ = p*` but wherever `ρ * φ_h = p*`, i.e. by
submitting the deconvolution of your belief by the kernel. Gaussian closed
form: belief `N(0, τ²)`, bandwidth `h` ⇒ the optimal cloud is

```
N(0, τ² − h²)        (exists iff τ² > h²),
```

*shave exactly `h²` off your variance.* When `τ² ≤ h²` the deconvolution is not
a distribution and the optimum degenerates toward point masses, precisely the
regime of Theis, van den Oord & Bethge's (2016) exploit, where a cloud of
k-means centroids out-scores the true distribution. The improperness *itself*
is theirs (they name it: "an improper scoring function"); the fair-scores
literature established the same genre for the ensemble CRPS (Fricker, Ferro &
Stephenson 2013 derive the optimal cheat in closed form; Ferro 2014 quantifies
the variance-shaving). What we add is the characterization: the optimal
misreport is the deconvolution, with the Gaussian `τ² − h²` formula, and the
incentive gain is fourth-order, `≈ h⁴/4` for `h ≪ τ`: small, but a real slope,
and any optimizing submitter (human or fitted) will walk down it.

Scope note: this is exact for log-scored KDEs and for the `b = 1` (full-Kelly)
wealth dynamics of the pool, where `log W' = log ρ_h(z) + const`. The small-`b`
pot-split has a separate, already-flagged incentive caveat (linearity in the
report; see the nearest-the-pin paper's finite-`b` discussion) which is
orthogonal to the smoothing issue treated here.

## 2. The repair: if you smooth the forecasts, smooth the outcome too

Define the mollified log score

```
S_h(ρ, z)  =  (φ_h * log ρ_h)(z)  =  E_ε [ log ρ_h(z + hε) ],   ε ~ N(0, I),
```

operationally: *jitter the pin* by the same kernel the mechanism applies to the
clouds (or settle on the analytic convolution, which derandomizes it). Then

```
E_{z~p*} S_h(ρ, z) = ∫ p*_h log ρ_h,
```

maximized iff `ρ_h = p*_h` iff `ρ = p*`, because Gaussian convolution is
injective (its characteristic function never vanishes). The score is
strictly proper for the *pre-smoothing* density.

Attribution here is precise, because the pieces are older than they look:

- Properness of the convolved score against a noised outcome is Bröcker &
  Smith (2007, §5) and, worked out for exactly this white-noise model, Ferro
  (2017, Prop. 3). Both treat the noise as *measurement error to be endured*.
- Strictness is the part they left open. Bröcker & Smith, verbatim: "If S
  is strictly proper though, S̄ is not necessarily strictly proper, because if
  q̄(z) = p̄(z), this does not necessarily mean equality of p(x) and q(x)."
  For kernels with nonvanishing characteristic function it *does* mean that;
  the closure is one line.
- The discrete-channel analog exists in the label-noise literature: composing
  the prediction with an invertible noise-transition matrix restores the
  clean minimizer (Patrini et al. 2017, Thm. 2; van Rooyen & Williamson 2018's
  "reconstructible" Markov operators; Liu, Wang & Chen's surrogate scoring
  rules for finite outcomes). §2 is the continuous, elicitation-side version.
- The general lemma this instantiates (the kernel-channel lemma)
  extends the deterministic injective-transform propositions of Allen,
  Ginsbourger & Ziegel (2023, Prop. 4 published / Prop. 3 in arXiv v1) and Pic
  et al. (2025, Prop. 1) to Markov kernels: *scoring forecast and outcome
  through the same stochastic channel preserves properness always, and strict
  properness iff the channel is injective on measures.* No published statement
  of this was found in the scoring-rule literature; it is a short step from
  Bröcker–Smith plus injectivity, and we claim it only as that.

The flip in framing matters: the verification literature treats outcome noise
as a nuisance (Ferro explicitly doubts the "efficacy" of perturbing
observations). Read as *mechanism design*, deliberately injecting jitter
matched to the mechanism's own smoothing is what makes the sample-cloud game
strictly proper. One line of code: **jitter the pin by the KDE bandwidth.**

Two qualifications (paper v0.2, after referee feedback): the guarantee is
population-level (the cloud idealised by its law; finite `m` open), and the
bandwidth must be exogenous, frozen before submissions are observed. A
bandwidth computed from the participant's own cloud puts the channel under the
participant's control and is outside the theorem.

Disclosure: the microprediction platform (Cotton 2022, and the platform paper)
added a small amount of noise to submissions and ground truth, a merely
intuitive implementation detail at the time, without incentive analysis. This
section is, retroactively, the theory of that detail.

## 3. The ladder: run the proper rung at every scale and the edge decomposes

Let `p_t = p * N(0, t)` denote the heat flow (`∂_t p = ½Δp`). Run the §2 rung
at a ladder of scales `t ∈ [0, T]`; for sample clouds this is free: the rung
at scale `t` is *the same cloud* scored with bandwidth `√(h² + t)` against a
correspondingly jittered pin. A truthful participant's expected edge at scale
`t` is `KL(p*_t ‖ ρ_t)`, and de Bruijn's identity in integral form gives the
exact decomposition

```
KL(p* ‖ ρ)  =  KL(p*_T ‖ ρ_T)  +  ½ ∫₀ᵀ D_F(p*_t ‖ ρ_t) dt,
```

where `D_F(p‖q) = ∫ p ‖∇log p − ∇log q‖²` is the Fisher divergence. (The
differential form is Stam 1959 / Barron 1986; the relative version is Lyu
2009; the same identity prices the likelihood of diffusion models in Song et
al. 2021. The identity is not ours; the mechanism reading is.)

Read as a payment schedule:

- Rung differences pay the Fisher divergence: they are
  [Hyvärinen-scored](../mechanisms/local_scoring.py) *shape* elicitation:
  scale-invariant, partition-function-free (the infinitesimal log score, per
  the de Bruijn edge on the [map](../docs/map.html)).
- The coarse top rung pays for between-mode mass: `KL(p*_T ‖ ρ_T)` at a
  scale `T` large enough to connect the modes retains exactly the mixing-
  proportion information that score matching is blind to (Wenliang & Kanagawa
  2020; Zhang et al. 2022; Koehler, Heckett & Risteski 2023).
- **The telescoped total is the full log-score edge.** Nothing is double-
  counted and nothing is lost: the ladder is a *partition of the log score's
  Bregman edge across smoothing scales*.

## 4. The mechanism: the heat-ladder pool

Each of `n` participants stakes `sᵢ` and submits one cloud. Fix scales
`t₀ = 0 < t₁ < … < t_K = T` and pot weights `w_k ≥ 0`. At settlement, rung `k`
splits `w_k · Σsᵢ` by any budget-balanced rule driven by the rung score
`S_{√(h²+t_k)}(ρᵢ, z)` (additive WSWM form, or the multiplicative pot-split at
full Kelly). Every rung is self-funding, so the tower is; every rung is
strictly proper for the cloud by §2. One precision from the paper (v0.2):
paying rung *levels* with weights `w_k` puts *cumulative* weights on the
Fisher bands; to pay a band directly, use rung *differences*
`S_k − S_{k+1}`, which are themselves strictly proper (regret = the band's
Fisher integral, zero iff truthful). Either way the weights are an explicit,
interpretable dial over *what is being paid for*: fine rungs buy
local shape (gradients), coarse rungs buy global mass placement. The two
failure modes of the repo's existing pools annihilate pairwise: the
[local-score pool](local-score-wagering-pool.md)'s mode-mass blindness is
carried by the coarse rungs; the density pool's deconvolution gaming (§1) is
killed rung-by-rung by the jitter.

Practicalities: rung scores are positively correlated across
scales (the same cloud, resmoothing), so the ladder's *discriminative* value
concentrates in a few well-separated scales — `K` of order 3–5, geometrically
spaced, is the sensible default, mirroring the noise ladders of annealed score
matching (Song & Ermon 2019). And for clouds of `m` samples there is a
finite-`m` fairness correction at each rung in the spirit of Ferro (2014) that
we have not derived; see §7.

## 5. What this repairs in this repository

- [`nearest_the_pin.pot_split`](../mechanisms/nearest_the_pin.py) scores KDEs
  at the raw pin: §1 applies. The repair is a bandwidth-matched jitter (or the
  analytic mollified score) — implemented as `mollified_log_score` alongside
  the existing functions, with the incentive warning in the module docstring.
- The [local-score pool note](local-score-wagering-pool.md) §5 proposed
  annealing as the fix for mode-mass blindness; the heat ladder is that fix
  *as a mechanism*, with the exact accounting of §3.

## 6. Prior art and novelty (calibrated)

| Claim | Verdict |
|---|---|
| KDE + log score at the raw outcome is improper | Known — Theis, van den Oord & Bethge (2016); genre: Fricker–Ferro–Stephenson (2013), Ferro (2014), Bröcker (2012) for ensemble CRPS |
| Optimal misreport = deconvolution; Gaussian `τ²−h²`; `h⁴/4` gain | Likely novel, narrow gap — not found; elementary once stated (shrink-to-compensate arithmetic appears in Bröcker & Smith 2008's dressing, as post-processing not strategy) |
| Mollified score properness (per rung) | Known — Bröcker & Smith (2007, §5); Ferro (2017, Prop. 3) |
| Strictness via injectivity; jitter as deliberate design | Likely novel, narrow gap — closes a gap Bröcker & Smith explicitly flagged; discrete analog in Patrini et al. (2017), van Rooyen & Williamson (2018) |
| Kernel-channel lemma (Markov-kernel extension of AGZ Prop. 4 / Pic et al. Prop. 1) | Likely novel, narrow gap — unstated in the scoring-rule literature after diligent search |
| Integral de Bruijn identity | Known — Stam (1959), Barron (1986), Lyu (2009), Song et al. (2021) |
| The heat-ladder pool (assembly: budget-balanced pot across smoothing scales; rungs pay Fisher, top pays mass) | Likely novel, narrow gap — survived adversarial search. Nearest neighbours, each missing a leg: Ferro 2017 (one rung, no ladder, no mechanism); Lang–Leutbecher–Maciel 2025 (Gaussian scale-ladder of proper scores as a training loss, no properness of the composite, no payments); Dudík et al. 2021 (multi-resolution *market* over one outcome via partition refinement, subsidized not budget-balanced, no divergence decomposition); Lambert et al. 2008/2015 + the microprediction pool (self-funding cloud wagering at a single scale) |

Residual risk: Wenliang & Kanagawa is workshop-only (Zhang et al. 2022 and
Koehler et al. 2023 are the archival support); Claim 1b is elementary enough
that an unindexed remark could exist; the microprediction platform is public
prior disclosure of single-scale jitter (by the author), sans theorem.

## 7. Open questions

1. **Fair rungs.** The finite-`m` correction making each rung's expected score
   optimized by *sampling from* the belief (the log-score/KDE analog of
   Ferro's fair CRPS), and whether it interacts with the jitter.
2. **Optimal weights `w_k`.** For a wealth-concentration objective (pools as
   selection dynamics), is there a closed-form optimal scale measure `w(t)` —
   and does it look like the likelihood weighting of Song et al. (2021)?
3. **Anisotropic rungs.** Replacing isotropic heat flow with a
   reference-covariance flow connects to
   [anisotropic-sliced-scores.md](anisotropic-sliced-scores.md); the same
   fixed-reference discipline should preserve strict properness rung-wise.
4. **The `b`-interpolation.** Between full Kelly (`b=1`, log score, truthful at
   every rung) and `b → 0` (linear pot-split, mode-seeking), characterize the
   rung-wise optimal misreport as a function of `b` — the mechanism-design
   analog of risk-sensitive score matching.

## References

*The flaw and the fair-scores genre.*
- Theis, L., van den Oord, A. & Bethge, M. (2016). "A Note on the Evaluation of
  Generative Models." *ICLR*. arXiv:1511.01844.
- Fricker, T. E., Ferro, C. A. T. & Stephenson, D. B. (2013). "Three
  Recommendations for Evaluating Climate Predictions." *Meteorological
  Applications* 20(2), 246–255. doi:10.1002/met.1409.
- Ferro, C. A. T. (2014). "Fair Scores for Ensemble Forecasts." *QJRMS*
  140(683), 1917–1923. doi:10.1002/qj.2270.
- Bröcker, J. (2012). "Evaluating Raw Ensembles with the Continuous Ranked
  Probability Score." *QJRMS* 138(667), 1611–1617. doi:10.1002/qj.1891.
- Bröcker, J. & Smith, L. A. (2008). "From Ensemble Forecasts to Predictive
  Distribution Functions." *Tellus A* 60(4), 663–678.
  doi:10.1111/j.1600-0870.2008.00333.x.
- Kimpara, D., Frongillo, R. & Waggoner, B. (2023). "Proper Losses for Discrete
  Generative Models." *ICML*, PMLR 202. arXiv:2211.03761.
- Krüger, F., Lerch, S., Thorarinsdottir, T. & Gneiting, T. (2021). "Predictive
  Inference Based on Markov Chain Monte Carlo Output." *International
  Statistical Review* 89(2), 274–301. doi:10.1111/insr.12405.
- Siegert, S., Ferro, C. A. T., Stephenson, D. B. & Leutbecher, M. (2019). "The
  Ensemble-Adjusted Ignorance Score." *QJRMS* 145(S1), 129–139.
  doi:10.1002/qj.3447.

*The repair: convolved scores, observation error, noisy channels.*
- Bröcker, J. & Smith, L. A. (2007). "Scoring Probabilistic Forecasts: The
  Importance of Being Proper." *Weather and Forecasting* 22(2), 382–388.
  doi:10.1175/WAF966.1.
- Ferro, C. A. T. (2017). "Measuring Forecast Performance in the Presence of
  Observation Error." *QJRMS* 143(708), 2665–2676. doi:10.1002/qj.3115.
- Saetra, Ø., Hersbach, H., Bidlot, J.-R. & Richardson, D. S. (2004). "Effects
  of Observation Errors on the Statistics for Ensemble Spread and Reliability."
  *Monthly Weather Review* 132(6), 1487–1501.
- Candille, G. & Talagrand, O. (2008). "Impact of Observational Error on the
  Validation of Ensemble Prediction Systems." *QJRMS* 134(633), 959–971.
  doi:10.1002/qj.268.
- Bessac, J. & Naveau, P. (2021). "Forecast Score Distributions with Imperfect
  Observations." *ASCMO* 7, 53–71. doi:10.5194/ascmo-7-53-2021.
- Patrini, G., Rozza, A., Menon, A. K., Nock, R. & Qu, L. (2017). "Making Deep
  Neural Networks Robust to Label Noise: A Loss Correction Approach." *CVPR*.
  arXiv:1609.03683.
- van Rooyen, B. & Williamson, R. C. (2018). "A Theory of Learning with
  Corrupted Labels." *JMLR* 18(228), 1–50.
- Liu, Y., Wang, J. & Chen, Y. (2022). "Surrogate Scoring Rules." *ACM TEAC*
  10(3) (conf. EC 2020).
- Allen, S., Ginsbourger, D. & Ziegel, J. (2023). "Evaluating Forecasts for
  High-Impact Events Using Transformed Kernel Scores." *SIAM/ASA JUQ* 11(3),
  906–940. arXiv:2202.12732 (Prop. 4 published; Prop. 3 in arXiv v1).
- Pic, R., Dombry, C., Naveau, P. & Taillardat, M. (2025). "Proper Scoring
  Rules for Multivariate Probabilistic Forecasts based on Aggregation and
  Transformation." *ASCMO* 11, 23–58. doi:10.5194/ascmo-11-23-2025.

*The identity and the ladder.*
- Stam, A. J. (1959). "Some Inequalities Satisfied by the Quantities of
  Information of Fisher and Shannon." *Information and Control* 2(2), 101–112.
- Barron, A. R. (1986). "Entropy and the Central Limit Theorem." *Annals of
  Probability* 14(1), 336–342. doi:10.1214/aop/1176992632.
- Lyu, S. (2009). "Interpretation and Generalization of Score Matching." *UAI*.
  arXiv:1205.2629.
- Song, Y., Durkan, C., Murray, I. & Ermon, S. (2021). "Maximum Likelihood
  Training of Score-Based Diffusion Models." *NeurIPS*. arXiv:2101.09258.
- Song, Y. & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of
  the Data Distribution." *NeurIPS*. arXiv:1907.05600.
- Vincent, P. (2011). "A Connection Between Score Matching and Denoising
  Autoencoders." *Neural Computation* 23(7), 1661–1674.
- Wenliang, L. K. & Kanagawa, H. (2020). "Blindness of Score-Based Methods to
  Isolated Components and Mixing Proportions." arXiv:2008.10087 (workshop).
- Zhang, M., Key, O., Hayes, P., Barber, D., Paige, B. & Briol, F.-X. (2022).
  "Towards Healing the Blindness of Score Matching." arXiv:2209.07396.
- Koehler, F., Heckett, A. & Risteski, A. (2023). "Statistical Efficiency of
  Score Matching: The View from Isoperimetry." *ICLR*. arXiv:2210.00726.
- Parry, M., Dawid, A. P. & Lauritzen, S. (2012). "Proper Local Scoring Rules."
  *Ann. Statist.* 40(1), 561–592.

*Nearest mechanism neighbours.*
- Lang, S., Leutbecher, M. & Maciel, P. (2025). "A Multi-Scale Loss Formulation
  for Learning a Probabilistic Model with Proper Score Optimisation."
  arXiv:2506.10868.
- Dudík, M., Wang, X., Pennock, D. M. & Rothschild, D. M. (2021). "Log-time
  Prediction Markets for Interval Securities." *AAMAS*. arXiv:2102.07308.
- Lambert, N. S. et al. (2008/2015). Self-financed wagering mechanisms; the
  axiomatic characterization. *EC'08*; *JET* 156, 389–416.
- Wei, J., Fu, Z., Liu, Y., Li, X., Yang, Z. & Wang, Z. (2021). "Sample
  Elicitation." *AISTATS*, PMLR 130, 2692–2700. arXiv:1910.03155.
- Cotton, P. (2022). *Microprediction: Building an Open AI Network.* MIT Press.
  (Public prior disclosure of single-scale jitter in a live cloud-wagering
  pool, without incentive analysis.)
