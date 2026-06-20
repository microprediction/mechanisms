# Composing mechanisms: conservation laws and residual markets

*Status: three open questions from
[composition-and-the-algebra-of-mechanisms.md](composition-and-the-algebra-of-mechanisms.md)
(§5), worked up with honest novelty assessments. One is **retired** (it is a known
theorem); two are **likely novel, narrow gap** and are stated here with the
specific non-trivial consequence each must deliver to be more than "composing
known results." Prior art is deliberately exhaustive — that is the point.*

The composition note treats a strictly proper scoring rule (generator `G`, Bregman
divergence `D_G`) as something that can be turned into a market maker
(sequentialise: Hanson's LMSR; Abernethy–Chen–Wortman Vaughan cost functions) or a
pool (parimutuel / wagering), and treats mechanisms as composable into pipelines.
Three of its flagged open questions are addressed below.

---

## Q1. Properness under transformation — **retired (known theorem)**

The composition note asks whether an arbitrary *conjugation* (reparametrize the
outcome or the report by a map `T`, score in the new coordinates) preserves strict
properness, noting that the probability integral transform does (Dawid) while a
learned non-invertible map may not. **This is a published dichotomy, not an open
question**, and the honest move is to cite it and retire the novelty claim.

- **Allen, Ginsbourger & Ziegel (2023)**, "Evaluating Forecasts for High-Impact
  Events Using Transformed Kernel Scores," *SIAM/ASA JUQ*, arXiv:2202.12732,
  **Prop. 4**: a transformed kernel score is strictly proper **iff the transform
  is injective**. Exactly "invertible preserves, non-invertible can fail."
- **Pic, Dombry, Naveau & Taillardat (2025)**, "Proper Scoring Rules for
  Multivariate Probabilistic Forecasts based on Aggregation and Transformation,"
  *ASCMO* 11, arXiv:2407.00650, **Prop. 1**: properness preserved under any `T`;
  strict properness preserved iff `T` injective; dimension-reducing maps lose it.
  This is a full transformation-and-aggregation (i.e. composition) calculus.
- **Parry, Dawid & Lauritzen (2012)**, "Proper Local Scoring Rules," *Ann.
  Statist.* 40(1): the log score is the unique `0`-local rule invariant under smooth
  invertible reparametrization — the canonical "PIT preserves it" instance.
- **Dawid (1984)**, "Statistical Theory: The Prequential Approach," *JRSS A*
  147(2): the prequential principle and the PIT origin.
- **Diebold, Gunther & Tay (1998)**, "Evaluating Density Forecasts," *Int. Econ.
  Rev.* 39(4): PIT-based forecast evaluation in practice.
- **Holzmann & Klar (2017)**, "Focusing on Regions of Interest with Weighted
  Proper Scoring Rules"; **Gneiting & Ranjan (2011)**, *JBES* 29(3): the weighted
  scoring rules whose properness turns on the weight being a *fixed function of the
  outcome*, the same injectivity-flavoured condition from the other side.

**Residual novelty (narrow).** The published results are *qualitative* (injective
vs. not). A *quantitative* theory — bounding the propriety gap of an
*approximately* non-invertible learned map `T` by its distance from injectivity,
or characterizing the bias when a non-invertible normalizing flow is composed
before a proper score — is not done, and would be a defensible, narrow contribution
*if framed explicitly against* Allen et al. and Pic et al. Claim nothing more.

---

## Q2. A conservation law for composed self-funding mechanisms — **likely novel, narrow gap**

The note conjectures that "a well-formed pipeline should conserve wealth
stage-to-stage (zero-sum), the analogue of a skater faithfully passing its
posterior state forward." The subtlety — and the consequence that lifts this above
bookkeeping — is that **wealth conservation is automatic and is the wrong
invariant.**

**Wealth conservation is free.** Each stage, if self-funding, already has
`Σ_i ΔW_i = 0` (Lambert et al. 2008/2015 for wagering pools; Hanson 2007's
"sequentially shared" scoring rule telescopes payments with bounded patron loss;
Abernethy–Chen–Wortman Vaughan 2013 give path-independence / no-arbitrage for cost
functions). A pipeline of self-funding stages conserves wealth because it is a sum
of zeros. Stating that as the law would be vacuous.

**The non-trivial invariant is conservation of *edge*, and it requires interface
sufficiency.** A trader's value in a proper-scoring market is the Bregman
divergence `D_G(q, π)` — the edge an honest belief `q` holds over the price `π`
(Savage 1971; Hanson 2003). Compose two stages through an interface map `T` that
reduces the carried state (stage 2 sees only `T(·)` of stage 1's output). Then:

> **Claim (conservation of edge).** The composite is truthful and loses no elicited
> information iff `T` is a *sufficient* reduction — equivalently (by Q1) injective
> on the relevant statistic. If `T` is non-sufficient, wealth still balances at
> every stage, but the Bregman edge does **not** telescope across the seam: edge
> (and hence elicited information) leaks, and the composite is no longer strictly
> proper even though each stage is.

This is exactly the consequence the prior art does not state: budget balance and
truthfulness *come apart* under composition, and the gap between them is interface
sufficiency. It ties Q2 to Q1 — the admissible interfaces are the injective/
sufficient ones — and it is invisible at the single-stage level, where there is no
seam to leak across.

Closest prior art (none states this):

- **Lambert, Langford, Wortman Vaughan, Chen, Reeves, Shoham & Pennock (2015)**,
  "An Axiomatic Characterization of Wagering Mechanisms," *JET* 156: per-mechanism
  budget balance and anonymity; single stage.
- **Hanson (2007)**, "Logarithmic Market Scoring Rules for Modular Combinatorial
  Information Aggregation," *J. Prediction Markets* 1(1): sequentially-shared MSR,
  telescoping payments — *one* market over time / a combinatorial outcome space,
  not a chain of heterogeneous mechanisms.
- **Abernethy, Chen & Wortman Vaughan (2013)**, "Efficient Market Making via Convex
  Optimization," *ACM TEAC* 1(2): no-arbitrage, path-independence; single maker.
- **Barrieu & El Karoui (2005)**; **Jouini, Schachermayer & Touzi (2008)**: optimal
  risk sharing as infimal convolution — the *parallel* (one-shot) composition whose
  minimiser is the Pareto allocation; the conservation here is its *sequential*
  analogue.
- **Frongillo, Della Penna & Reid (2012)**, "Interpreting Prediction Markets: A
  Stochastic Approach," *NeurIPS*: wealth dynamics of Kelly traders as the
  conserved threaded state.

Verdict: **likely novel, narrow gap.** The framing (pipeline of heterogeneous
self-funding mechanisms, wealth as threaded state, edge-vs-wealth distinction)
appears unoccupied; the gap is small because each ingredient is standard, so the
*edge-conservation* consequence is what carries the contribution.

---

## Q3. Residual / boosting markets — **likely novel, narrow gap**

The note names a "residual chaining" operator — "one stage models what the previous
stage got wrong; a correction/boosting market on the residual stream" — with no
mechanism realizing it. Two mature equivalences flank the gap:

- **Markets are online learners / gradient descent.** Abernethy, Chen & Wortman
  Vaughan (2013) and Chen & Wortman Vaughan (2010, "A New Understanding of
  Prediction Markets via No-Regret Learning," *EC*, arXiv:1003.0034): a
  cost-function maker *is* Follow-the-Regularized-Leader. Frongillo, Della Penna &
  Reid (2012): Kelly traders make the market run stochastic mirror descent.
  Nueve & Waggoner (2025), "Smooth Quadratic Prediction Markets," arXiv:2505.02959:
  markets implementing general gradient descent, and the under-explored
  *budget-limited / self-funded* regime.
- **Boosting is functional gradient descent on residuals.** Mason, Baxter,
  Bartlett & Frean (1999), "Boosting Algorithms as Gradient Descent," *NeurIPS*;
  Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine,"
  *Ann. Statist.* 29(5); Freund & Schapire (1997), AdaBoost, *JCSS* 55(1).

Between them sits an unoccupied object: **one market per boosting stage, whose
tradeable is the previous stage's residual, with wealth flowing between stages as
credit routing.** The consequence that meets the bar:

> **Claim (boosting = wealth routing).** In a chain of self-funding markets where
> stage `t` prices the residual `r_{t−1}` of stage `t−1`, a log-wealth (Kelly)
> trader's optimal stake at stage `t` equals the gradient-boosting weight on the
> `t`-th weak learner, and the wealth transferred into stage `t` equals Friedman's
> functional-gradient line-search coefficient `β_t`. The market's bounded-loss /
> regret guarantee then becomes a generalization bound on the boosted ensemble.

That is a falsifiable identity, not an analogy; proving it is the contribution, and
the *self-funding, budget-limited* regime (Nueve–Waggoner) is what distinguishes it
from the unbounded-liquidity cost-function maker.

Closest prior art (none instantiates the three-part object):

- **Storkey (2011)**, "Machine Learning Markets," *AISTATS*, PMLR 15; **Storkey,
  Millin & Geras (2012)**, "Isoelastic Agents and Wealth Updates in Machine
  Learning Markets," *ICML*, arXiv:1206.6443: wealth-weighted equilibrium prices
  recover the weighted means "used in boosting and random forests" — but *parallel*
  ensembling at one market's equilibrium, not sequential residual correction.
- **Hu & Storkey (2014)**, "Multi-period Trading Prediction Markets with Connections
  to Machine Learning," *ICML*, arXiv:1403.0648: sequential, but one shared
  objective, not staged residuals.
- **Barbu & Lay (2012)**, "An Introduction to Artificial Prediction Markets for
  Classification," *JMLR* 13: a market of classifiers betting on labels — ensemble
  pricing, again not residual-staged.
- **Chen & Wortman Vaughan (2010)**; **Abernethy, Frongillo & Kutty (2014)**, "On
  Risk Measures, Market Making, and Exponential Families," *SIGecom Exchanges*
  13(2): the exponential-family / risk-measure backbone a residual market inherits.
- **Mason et al. (1999)**; **Friedman (2001)**: the boosting side of the identity.

Verdict: **likely novel, narrow gap.** Both pillars are mature, so the value is the
explicit `β_t`-equals-wealth-transfer identity and the budget-limited regret-to-
generalization translation.

---

## Open questions

1. Prove (or refute) the **edge-conservation** claim (Q2): does the Bregman edge
   telescope across an injective/sufficient interface and leak across a
   non-sufficient one? Exhibit a natural-looking but ill-formed composition where
   wealth balances yet truthfulness fails.
2. Prove (or refute) the **`β_t` = wealth-transfer** identity (Q3) for a Kelly
   trader, and translate the market's regret bound into an ensemble generalization
   bound.
3. The **quantitative** propriety gap for approximately-non-invertible learned `T`
   (Q1 residual), framed against Allen et al. (2023) and Pic et al. (2025).

## References

*Prediction markets as online learning / convex optimization.*
- Hanson, R. (2003). "Combinatorial Information Market Design." *Information Systems
  Frontiers* 5(1), 107–119.
- Hanson, R. (2007). "Logarithmic Market Scoring Rules for Modular Combinatorial
  Information Aggregation." *J. Prediction Markets* 1(1), 3–15.
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2011). "An Optimization-Based
  Framework for Automated Market-Making." *EC*, 297–306.
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). "Efficient Market Making
  via Convex Optimization, and a Connection to Online Learning." *ACM TEAC* 1(2).
- Chen, Y. & Wortman Vaughan, J. (2010). "A New Understanding of Prediction Markets
  via No-Regret Learning." *EC*, 189–198. arXiv:1003.0034.
- Frongillo, R. M., Della Penna, N. & Reid, M. D. (2012). "Interpreting Prediction
  Markets: A Stochastic Approach." *NeurIPS* 25.
- Frongillo, R. M. & Reid, M. D. (2015). "Convergence Analysis of Prediction
  Markets via Randomized Subspace Descent." *NeurIPS* 28.
- Nueve, E. & Waggoner, B. (2025). "Smooth Quadratic Prediction Markets."
  *NeurIPS*. arXiv:2505.02959.

*Machine-learning markets.*
- Storkey, A. (2011). "Machine Learning Markets." *AISTATS*, PMLR 15, 716–724.
  arXiv:1106.4509.
- Storkey, A., Millin, J. & Geras, K. (2012). "Isoelastic Agents and Wealth Updates
  in Machine Learning Markets." *ICML*. arXiv:1206.6443.
- Hu, J. (2012). "Combinatorial Modelling and Learning with Prediction Markets."
  arXiv:1201.3851.
- Hu, J. & Storkey, A. (2014). "Multi-period Trading Prediction Markets with
  Connections to Machine Learning." *ICML*, PMLR 32(2), 1773–1781. arXiv:1403.0648.
- Barbu, A. & Lay, N. (2012). "An Introduction to Artificial Prediction Markets for
  Classification." *JMLR* 13, 2177–2204. arXiv:1102.1465.
- Lay, N. & Barbu, A. (2012). "The Artificial Regression Market." arXiv:1204.4154.
- Othman, A. & Sandholm, T. (2011). "Liquidity-Sensitive Automated Market Makers
  via Homogeneous Risk Measures." *WINE*, LNCS 7090, 314–325.
- Othman, A., Pennock, D. M., Reeves, D. M. & Sandholm, T. (2013). "A Practical
  Liquidity-Sensitive Automated Market Maker." *ACM TEAC* 1(3).

*Boosting as functional gradient descent.*
- Freund, Y. & Schapire, R. E. (1997). "A Decision-Theoretic Generalization of
  On-Line Learning and an Application to Boosting." *J. Comput. Syst. Sci.* 55(1).
- Mason, L., Baxter, J., Bartlett, P. L. & Frean, M. (1999). "Boosting Algorithms
  as Gradient Descent." *NeurIPS* 12, 512–518.
- Mason, L., Baxter, J., Bartlett, P. L. & Frean, M. (2000). "Functional Gradient
  Techniques for Combining Hypotheses." In *Advances in Large Margin Classifiers*,
  221–246. MIT Press.
- Friedman, J., Hastie, T. & Tibshirani, R. (2000). "Additive Logistic Regression:
  A Statistical View of Boosting." *Ann. Statist.* 28(2), 337–407.
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting
  Machine." *Ann. Statist.* 29(5), 1189–1232.
- Schapire, R. E. & Freund, Y. (2012). *Boosting: Foundations and Algorithms.*
  MIT Press.

*Wagering / self-financing mechanisms.*
- Lambert, N. S. et al. (2008). "Self-Financed Wagering Mechanisms for
  Forecasting." *EC*, 170–179.
- Lambert, N. S. et al. (2015). "An Axiomatic Characterization of Wagering
  Mechanisms." *J. Econ. Theory* 156, 389–416.
- Witkowski, J., Freeman, R., Wortman Vaughan, J., Pennock, D. M. & Krause, A.
  (2018). "Incentive-Compatible Forecasting Competitions." *AAAI* 32(1), 1282–1289.

*Risk sharing / infimal convolution / aggregating makers (parallel composition).*
- Borch, K. (1962). "Equilibrium in a Reinsurance Market." *Econometrica* 30(3),
  424–444.
- Wilson, R. (1968). "The Theory of Syndicates." *Econometrica* 36(1), 119–132.
- Barrieu, P. & El Karoui, N. (2005). "Inf-Convolution of Risk Measures and Optimal
  Risk Transfer." *Finance and Stochastics* 9(2), 269–298.
- Jouini, E., Schachermayer, W. & Touzi, N. (2008). "Optimal Risk Sharing for Law
  Invariant Monetary Utility Functions." *Math. Finance* 18(2), 269–292.
- Abernethy, J. D., Frongillo, R. M. & Kutty, S. (2014). "On Risk Measures, Market
  Making, and Exponential Families." *ACM SIGecom Exchanges* 13(2), 21–25.
- Bhaskara, A., Frongillo, R. & Papireddygari, M. (2023). "A General Theory of
  Liquidity Provisioning for Prediction Markets." arXiv:2311.08725.
- Frongillo, R., Papireddygari, M. & Waggoner, B. (2024). "An Axiomatic
  Characterization of CFMMs and Equivalence to Prediction Markets." *ITCS*, LIPIcs
  287, Art. 51. arXiv:2302.00196.

*Scoring-rule invariance / transformation / sufficiency.*
- Savage, L. J. (1971). "Elicitation of Personal Probabilities and Expectations."
  *JASA* 66(336), 783–801.
- Dawid, A. P. (1984). "Statistical Theory: The Prequential Approach." *JRSS A*
  147(2), 278–292.
- Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998). "Evaluating Density
  Forecasts." *Int. Econ. Rev.* 39(4), 863–883.
- Gneiting, T. & Ranjan, R. (2011). "Comparing Density Forecasts Using Threshold-
  and Quantile-Weighted Scoring Rules." *JBES* 29(3), 411–422.
- Diks, C., Panchenko, V. & van Dijk, D. (2011). "Likelihood-Based Scoring Rules
  for Comparing Density Forecasts in Tails." *J. Econometrics* 163(2), 215–230.
- Parry, M., Dawid, A. P. & Lauritzen, S. (2012). "Proper Local Scoring Rules."
  *Ann. Statist.* 40(1), 561–592.
- Ehm, W., Gneiting, T., Jordan, A. & Krüger, F. (2016). "Of Quantiles and
  Expectiles: Consistent Scoring Functions, Choquet Representations and Forecast
  Rankings." *JRSS B* 78(3), 505–562.
- Allen, S., Ginsbourger, D. & Ziegel, J. (2023). "Evaluating Forecasts for
  High-Impact Events Using Transformed Kernel Scores." *SIAM/ASA JUQ* 11(3),
  906–940. arXiv:2202.12732.
- Pic, R., Dombry, C., Naveau, P. & Taillardat, M. (2025). "Proper Scoring Rules
  for Multivariate Probabilistic Forecasts based on Aggregation and
  Transformation." *ASCMO* 11, 23–58. arXiv:2407.00650.
