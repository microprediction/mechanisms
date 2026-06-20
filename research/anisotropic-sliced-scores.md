# Anisotropic (Schur-damped) sliced scoring rules

*Status: a worked development of an open conjecture from the nearest-the-pin
paper (§5, §7.3), with an honest novelty assessment and one load-bearing
technical caveat. Likely novel as a combination; the central properness claim is
false-until-proven and the note is built around that boundary. See §4 and §6.*

## 1. The conjecture

The nearest-the-pin paper scores high-dimensional forecasts through random
one-dimensional projections and flags, as conjecture, a *covariance-shaped*
refinement (§5):

> a *Schur-damped* projection score, in which the directions `u` are not isotropic
> but shaped by a damped estimate of the forecast covariance (project more often
> along the well-estimated directions), interpolating between [isotropic and
> adapted] with a single reliability dial `γ`.

and asks (§7.3) whether anisotropic slicing dominates isotropic slicing in the
`p > n` regime and inherits a closed-form `γ⋆`. This note develops the
construction, identifies what is actually being traded off, and pins down the
exact condition under which the object remains a *strictly proper* scoring rule —
which, it turns out, is the whole game.

## 2. Background: the sliced energy score

The multivariate energy score is an average of one-dimensional CRPS over
projection directions. For `u` uniform on the sphere `S^{d−1}`,

```
ES(P, y) = c_d⁻¹ · E_u[ CRPS(P_u, ⟨u, y⟩) ],
```

where `P_u` is the law of the projected sample `⟨u, X⟩`, `X ∼ P`. This rests on
the identity `E_u|⟨u, x⟩| = c_d ‖x‖`, so the slice average reconstructs the
`‖·‖` in the energy score exactly. Because `ES` is strictly proper (Gneiting &
Raftery 2007) and the averaging measure is **fixed**, the sliced quantity is a
strictly proper score requiring only 1-D evaluations.

The load-bearing words are *uniform* and *fixed*.

## 3. The generalization

Replace the uniform slicing measure with an **anisotropic** one, `u ∼ μ_Σ`, that
concentrates on directions selected by a shrinkage estimate of a covariance `Σ`,
with a single dial `γ ∈ [0,1]` interpolating

```
γ = 0:  μ_Σ = uniform on the sphere      (recover the plain sliced score)
γ = 1:  μ_Σ = fully Σ-adapted slicing.
```

Two notes before anything else. First, the anisotropy must enter the *direction
law*, not a Gaussian sampler: slicing with `u ∼ N(0, I)` equals uniform-sphere
slicing (Nadjahi et al. 2021), so only a genuinely anisotropic `N(0, Σ)`-type law
(or a von Mises–Fisher / pushforward concentration) differs from §2. Second — and
this is the crux — **whose covariance is `Σ`?**

## 4. The properness boundary (the whole game)

There are two candidate sources for `Σ`, and they are not interchangeable.

**(a) `Σ` from the forecast `P` being scored — breaks strict properness.**
A kernel/energy score is strictly proper iff its kernel is characteristic and
*fixed* (Steinwart & Ziegel 2021). If the slicing law `μ_{Σ(P)}` reads the
covariance off the very forecast under test, the effective kernel becomes
forecast-dependent, and the divergence `E_{Y∼G}[S(F,Y) − S(G,Y)]` is no longer
guaranteed `≥ 0` with equality only at `F = G` — both the integrand and the
measure move with `F`. This is the same failure mode that makes naïve
outcome-weighted CRPS improper (Gneiting & Ranjan 2011; Allen et al. 2023). There
is no theorem rescuing it; the "project more along the forecast's own
well-estimated directions" reading of the conjecture is, taken literally,
**not strictly proper.** Treat that claim as false until a direct
divergence-nonnegativity proof is exhibited.

**(b) `Σ` from a fixed reference (climatology) — stays strictly proper.**
Let `Σ_ref` be a covariance estimated *once*, from a reference distribution
(observation climatology, a held-out sample, the pooled field), and held fixed
across every forecast being scored and compared. Then `μ_{Σ_ref}` is a fixed
characteristic-kernel measure, §2 applies verbatim, and the score is strictly
proper for all `γ`. The reliability dial becomes a *tuning knob on a still-proper
score*, exactly the position the proper localized scores occupy (twCRPS/owCRPS,
Allen et al. 2023; de Punder et al. 2025): properness is preserved precisely
because the adaptation depends on a fixed object, not on the candidate forecast.

So the honest form of the conjecture is: **anisotropic slicing against a fixed
reference covariance, with a reliability dial `γ`.** That is the version worth
building, and it keeps strict properness.

## 5. What `γ` actually trades off

The paper's phrase "project more often along the well-estimated directions"
conflates two objectives that pull in *opposite* directions, and seeing them
apart is most of the content:

- **Estimator variance.** With finitely many slices, the Monte-Carlo error of the
  score is smallest when directions concentrate where the projected distributions
  are well-estimated — in `p > n`, the top eigen-directions of `Σ_ref`. Adapting
  here (large `γ`) reduces score variance.
- **Discriminative power.** Two forecasts differ most along the directions of
  their *mean/shape discrepancy*, which high-dimensional two-sample tests reach
  through `Σ⁻¹` — the *low*-variance directions (Lopes, Jacob & Wainwright 2011),
  the **opposite** of the top eigen-directions. Adapting toward power pushes the
  other way.

`γ` is the dial between *low-variance estimation of the score* and *high power to
separate forecasts*. That reframing matters because it says there is no single
"correct" anisotropy — the optimum depends on what the score is *for* (ranking
forecasters vs. concentrating wealth on skill), and the two regimes want `γ`
shaped by `Σ_ref` and by `Σ_ref⁻¹` respectively.

## 6. The closed-form dial (imported, honestly)

The convex-combination dial with a closed-form optimum is mature shrinkage
mathematics: James–Stein (1961), Ledoit–Wolf linear shrinkage with closed-form
intensity (2004), optimal ridge in the `p ∼ n` regime (Dobriban & Wager 2018).
The author's own Schur-damping work derives
`γ⋆ = (n−2)ρ² / [ (n−2)ρ² + (1−ρ²) ]` as a simultaneous James–Stein/Ledoit–Wolf
intensity in the `p > n` regime (Cotton, *Two Sides of Schur Damping*, 2026;
*Schur Complementary Allocation*, 2024) — but for **covariance/portfolio
construction, not scoring rules.**

The honest claim is therefore narrow: that `γ⋆` is the right intensity for
building `Σ_ref` *as a covariance estimator*; whether it is also optimal for the
*scoring objective* (power as a test, or wealth-concentration speed inside a
[nearest-the-pin](../mechanisms/nearest_the_pin.py) pool) is a separate
optimization and is **open**. Importing `γ⋆` gives a principled default, not a
proof of score-optimality. A referee will (correctly) see the dial as imported
and the contribution as its *application* inside a proper sliced score.

## 7. Prior art and novelty

Calibrated verdict: **likely novel as a combination, narrow gap.** No prior work
combines (i) covariance-parametrized anisotropic slicing, (ii) of an
*energy-score / proper-scoring-rule* (not Wasserstein), (iii) with a scalar
reliability dial, (iv) proven strictly proper. The nearest neighbours each have
at most two of the four:

- Bonet, Drumetz & Courty, *Sliced-Wasserstein on Cartan–Hadamard Manifolds*
  (JMLR 2025): Mahalanobis "whiten-then-slice" — closest on the anisotropy axis,
  but Wasserstein, learned metric, no scoring rule, no dial.
- Nguyen, Ho, Pham & Bui, *Distributional Sliced-Wasserstein* (ICLR 2021):
  non-uniform learned slicing law with a concentration dial — closest on the dial
  axis, but discriminative/learned, not covariance-parametrized, not a score.
- Paty & Cuturi, *Subspace Robust Wasserstein* (ICML 2019): second-moment-driven
  direction selection — closest on the covariance flavour, but max-optimized, no
  dial, not a score.
- Gneiting & Raftery (2007); Steinwart & Ziegel (2021); Allen et al. (2023): the
  fixed-measure / fixed-kernel properness results this construction must respect.

A sharper gap surfaced in the search and is worth stating plainly: **there appears
to be no prior work that builds a *proper scoring rule* by averaging a univariate
proper score (CRPS) over projection directions at all** — sliced constructions to
date slice an *objective* (Sliced Score Matching, Song et al. 2019) or a
*divergence* (Sliced Kernelized Stein Discrepancy, Gong et al. 2021), not a
forecast-evaluation score. So even the *isotropic* sliced-CRPS-as-proper-score is
under-documented as such; the anisotropic dial is a further step on top.

The defensible claim: *the first covariance-adapted sliced proper scoring rule
with a reliability dial, kept strictly proper by adapting to a fixed reference
covariance rather than the forecast under test.* No claim is made for the
forecast-dependent variant.

## 8. Open questions

1. **Score-optimal `γ`.** Is `γ⋆` from the shrinkage literature optimal for a
   scoring objective (power, or wealth-concentration rate), or only for covariance
   estimation? Derive the score-side optimum directly.
2. **Can the forecast-dependent variant be rescued?** Is there a renormalization
   (à la owCRPS's `w̄_F`) that restores strict properness while letting the
   anisotropy read each forecast's own `Σ`? Or is the improper object still useful
   as a *discrimination diagnostic* (not for incentives)?
3. **Two dials.** §5 suggests the estimation-optimal and power-optimal anisotropies
   are `Σ_ref`- and `Σ_ref⁻¹`-shaped. Is the right object a *two-parameter* slicing
   law, and does a single `γ` suffice in practice?

## References

*Energy score / CRPS / kernel scores.*
- Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction,
  and Estimation." *JASA* 102(477), 359–378.
- Székely, G. J. & Rizzo, M. L. (2013). "Energy Statistics: A Class of Statistics
  Based on Distances." *J. Statist. Plann. Inference* 143(8), 1249–1272.
- Rizzo, M. L. & Székely, G. J. (2016). "Energy Distance." *WIREs Comput. Statist.*
  8(1), 27–38.
- Gneiting, T., Stanberry, L. I., Grimit, E. P., Held, L. & Johnson, N. A. (2008).
  "Assessing Probabilistic Forecasts of Multivariate Quantities." *TEST* 17(2),
  211–264.
- Hersbach, H. (2000). "Decomposition of the CRPS for Ensemble Prediction
  Systems." *Weather and Forecasting* 15(5), 559–570.
- Steinwart, I. & Ziegel, J. F. (2021). "Strictly Proper Kernel Scores and
  Characteristic Kernels on Compact Spaces." *Appl. Comput. Harmon. Anal.* 51,
  510–542. arXiv:1712.05279.
- Sejdinovic, D., Sriperumbudur, B., Gretton, A. & Fukumizu, K. (2013).
  "Equivalence of Distance-Based and RKHS-Based Statistics in Hypothesis Testing."
  *Ann. Statist.* 41(5), 2263–2291. arXiv:1207.6076.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B. & Smola, A. (2012).
  "A Kernel Two-Sample Test." *JMLR* 13, 723–773.

*Sliced Wasserstein and direction-adaptation variants.*
- Rabin, J., Peyré, G., Delon, J. & Bernot, M. (2012). "Wasserstein Barycenter and
  Its Application to Texture Mixing." *SSVM*, LNCS 6667, 435–446.
- Bonneel, N., Rabin, J., Peyré, G. & Pfister, H. (2015). "Sliced and Radon
  Wasserstein Barycenters of Measures." *J. Math. Imaging Vis.* 51(1), 22–45.
- Deshpande, I. et al. (2019). "Max-Sliced Wasserstein Distance and Its Use for
  GANs." *CVPR*, 10648–10656. arXiv:1904.05877.
- Kolouri, S., Nadjahi, K., Şimşekli, U., Badeau, R. & Rohde, G. K. (2019).
  "Generalized Sliced Wasserstein Distances." *NeurIPS*. arXiv:1902.00434.
- Nguyen, K., Ho, N., Pham, T. & Bui, H. (2021). "Distributional Sliced-Wasserstein
  and Applications to Generative Modeling." *ICLR*. arXiv:2002.07367.
- Chen, X., Yang, Y. & Li, Y. (2022). "Augmented Sliced Wasserstein Distances."
  *ICLR*. arXiv:2006.08812.
- Nguyen, K. & Ho, N. (2023). "Energy-Based Sliced Wasserstein Distance."
  *NeurIPS*. arXiv:2304.13586.
- Nguyen, K., Ren, T. & Ho, N. (2023). "Markovian Sliced Wasserstein Distances."
  *NeurIPS*. arXiv:2301.03749.
- Nguyen, K., Zhang, S., Le, T. & Ho, N. (2024). "Sliced Wasserstein with
  Random-Path Projecting Directions." *ICML*. arXiv:2401.15889.
- Paty, F.-P. & Cuturi, M. (2019). "Subspace Robust Wasserstein Distances." *ICML*,
  5072–5081. arXiv:1901.08949.
- Lin, T., Fan, C., Ho, N., Cuturi, M. & Jordan, M. I. (2020). "Projection Robust
  Wasserstein Distance and Riemannian Optimization." *NeurIPS*. arXiv:2006.07458.
- Bonet, C., Drumetz, L. & Courty, N. (2025). "Sliced-Wasserstein Distances and
  Flows on Cartan–Hadamard Manifolds" (defines Mahalanobis-SW). *JMLR* 26(32),
  1–76. arXiv:2403.06560.
- Nadjahi, K. et al. (2020). "Statistical and Topological Properties of Sliced
  Probability Divergences." *NeurIPS*. arXiv:2003.05783.
- Song, Y., Garg, S., Shi, J. & Ermon, S. (2019). "Sliced Score Matching." *UAI*
  (objective slicing — nearest neighbour, not a proper score).
- Gong, W., Li, Y. & Hernández-Lobato, J. M. (2021). "Sliced Kernelized Stein
  Discrepancy." *ICLR*. arXiv:2006.16531 (divergence slicing — nearest neighbour).

*Weighted / localized proper scoring rules (the properness boundary).*
- Gneiting, T. & Ranjan, R. (2011). "Comparing Density Forecasts Using Threshold-
  and Quantile-Weighted Scoring Rules." *JBES* 29(3), 411–422.
- Diks, C., Panchenko, V. & van Dijk, D. (2011). "Likelihood-Based Scoring Rules
  for Comparing Density Forecasts in Tails." *J. Econometrics* 163(2), 215–230.
- Holzmann, H. & Klar, B. (2017). "Focusing on Regions of Interest in Forecast
  Evaluation." *Ann. Appl. Stat.* 11(4), 2404–2431.
- Allen, S., Ginsbourger, D. & Ziegel, J. (2023). "Evaluating Forecasts for
  High-Impact Events Using Transformed Kernel Scores." *SIAM/ASA JUQ* 11(3),
  906–940. arXiv:2202.12732.
- de Punder, R., Diks, C., Laeven, R. & van Dijk, D. (2025). "Localizing Strictly
  Proper Scoring Rules." *JASA* (online-first). doi:10.1080/01621459.2025.2576189.
- Pic, R., Dombry, C., Naveau, P. & Taillardat, M. (2025). "Proper Scoring Rules
  for Multivariate Probabilistic Forecasts based on Aggregation and
  Transformation." *ASCMO* 11, 23–58. arXiv:2407.00650.

*Covariance shrinkage & closed-form intensity.*
- Stein, C. (1956). "Inadmissibility of the Usual Estimator for the Mean of a
  Multivariate Normal Distribution." *Proc. 3rd Berkeley Symp.* 1, 197–206.
- James, W. & Stein, C. (1961). "Estimation with Quadratic Loss." *Proc. 4th
  Berkeley Symp.* 1, 361–379.
- Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional
  Covariance Matrices." *J. Multivariate Anal.* 88(2), 365–411.
- Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix."
  *J. Portfolio Management* 30(4), 110–119.
- Ledoit, O. & Wolf, M. (2012). "Nonlinear Shrinkage Estimation of
  Large-Dimensional Covariance Matrices." *Ann. Statist.* 40(2), 1024–1060.
- Ledoit, O. & Wolf, M. (2020). "Analytical Nonlinear Shrinkage of
  Large-Dimensional Covariance Matrices." *Ann. Statist.* 48(5), 3043–3065.
- Dobriban, E. & Wager, S. (2018). "High-Dimensional Asymptotics of Prediction:
  Ridge Regression and Classification." *Ann. Statist.* 46(1), 247–279.
- Schäfer, J. & Strimmer, K. (2005). "A Shrinkage Approach to Large-Scale
  Covariance Matrix Estimation." *Stat. Appl. Genet. Mol. Biol.* 4(1), Art. 32.

*High-dimensional two-sample tests via random projection.*
- Bai, Z. & Saranadasa, H. (1996). "Effect of High Dimension: By an Example of a
  Two Sample Problem." *Statistica Sinica* 6(2), 311–329.
- Srivastava, M. S. & Du, M. (2008). "A Test for the Mean Vector with Fewer
  Observations than the Dimension." *J. Multivariate Anal.* 99(3), 386–402.
- Chen, S. X. & Qin, Y.-L. (2010). "A Two-Sample Test for High-Dimensional Data."
  *Ann. Statist.* 38(2), 808–835. arXiv:1002.4547.
- Lopes, M. E., Jacob, L. J. & Wainwright, M. J. (2011). "A More Powerful Two-Sample
  Test in High Dimensions Using Random Projection." *NeurIPS* 24. arXiv:1108.2401.

*The reliability-dial machinery (author's own).*
- Cotton, P. (2024). "Schur Complementary Allocation: A Unification of Hierarchical
  Risk Parity and Minimum Variance Portfolios." arXiv:2411.05807.
- Cotton, P. (2026). "Two Sides of Schur Damping: High-Dimensional
  Pseudo-Likelihoods and Portfolio Allocation." arXiv:2606.14798.
