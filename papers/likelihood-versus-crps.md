# Likelihood versus CRPS: A Compositional Perspective

### Composition, Chained Elicitation, and the Score under Which Density Refinements Add Up

Peter Cotton · *Working draft v0.1* · 2026

---

## Abstract

The choice between the logarithmic score and the continuous ranked probability
score (CRPS) is usually argued on four grounds: locality, propriety, robustness,
and interpretability. A fifth applies when the elicited object is a reusable
predictive *density* and forecasts are *chained*, each stage refining the
residual left by the last. The log-likelihood of a composed forecast is the base
score plus the log-density of each residual refinement, so credit is additive and
each stage's increment is itself the proper score of what that stage added; CRPS
has no comparable per-stage identity. A population-level example shows the two
scores preferring measurably different forecasts, and CRPS remains the right
target when the decision is coarse or the tail unmodelable. Conformal prediction
is the limiting case, a method that produces no density and is scored on a metric
that does not require one.

---

## 1. The debate and the axis it usually turns on

A scoring rule is a settlement rule. It fixes what a forecaster is paid to
produce, and forecast evaluation is coherent only when the score is matched to
the object being forecast [@gneiting2011making; @gneiting2007strictly]. For a
full predictive distribution the two standard proper scores are the logarithmic
score [@good1952rational], which reads the density at the realized outcome, and
CRPS [@matheson1976scoring], which integrates squared distribution-function error
across the line. Both are proper, so in expectation each is optimized by the
truth; the argument between them is about which imperfect forecasts they reward.

That argument is usually conducted on four axes. *Locality*: the log score
depends only on the density assigned to what happened, and among scores with that
property it is essentially unique up to affine transformation
[@bernardo1979expected], though broader derivative-local scores exist once the
restriction is relaxed [@parry2012proper]. *Propriety*: both qualify, so this
settles nothing between them. *Robustness*: the log score is unbounded below and
CRPS is not, which is read for and against depending on whether an extreme outcome
signals a fixable model or an unmodelable tail [@machete2013contrasting;
@bolin2023local]. *Interpretability*: CRPS is measured in the units of the outcome and
the log score in nats. Reviews weigh these and reasonably conclude that the right
score depends on the use [@gneiting2007strictly; @dawid2014theory].

A fifth consideration is absent from that list. It concerns not a
single forecast but a *chain* of them, and it is decisive precisely when
distributional prediction is organized as a supply chain
[@cotton2019roar; @cotton2022microprediction].

## 2. Composition: the score under which refinements add up

Take a base forecast with density $p_1$ and distribution function $F_1$. A second
stage forecasts the residualized variable
$$z = \Phi^{-1}\!\big(F_1(y)\big),$$
the base forecast's own probability integral transform pushed to the Gaussian
scale [@rosenblatt1952remarks; @diebold1998evaluating]. If the second stage assigns density
$g$ to $z$, the change of variables $dz/dy = p_1(y)/\varphi(z)$ gives the composed
density $p(y) = p_1(y)\,g(z)/\varphi(z)$, so
$$\log p(y) \;=\; \log p_1(y) \;+\; \big(\log g(z) - \log\varphi(z)\big).$$
The log score of the composed forecast is the base score plus the incremental
score of the refinement. A stage that reports the crowd's own residual law,
$g=\varphi$, adds nothing and is paid nothing; a stage that finds structure the
base missed is paid exactly its contribution. Credit is additive, and skill is
attributable stage by stage.

This is the accounting a prediction supply chain
needs, and it is developed for chained pools and stacked lotteries in
[@cotton2026multistage; @cotton2020lottery] under the algebra of
[@cotton2026algebra]. The dependence case factors the same way: by Sklar's
theorem [@sklar1959] a joint log-density splits into marginal terms plus a copula
term, one stage each.

Telescoping does not separate the two scores: any additive bookkeeping of any
score's differences telescopes. Nor does path independence: a market scoring
rule built from *any* strictly proper score has a path-independent cost function,
namely the convex conjugate of its expected-score function
[@abernethy2013efficient], so the property is shared by the whole family rather
than particular to the logarithm. In Hanson's market form a trader who moves the report
from $p$ to $p'$ is paid $S(p',y)-S(p,y)$ and a chain of them telescopes to
$S(p_{\text{final}},y)-S(p_0,y)$ [@hanson2007logarithmic], for the same reason,
whatever $S$ is.

The property that separates them is narrower. Each increment $\log
g(z)-\log\varphi(z)$ is itself the logarithmic score of the stage's residual
refinement, strictly proper for the law of $z$, so the stage is paid exactly, and
only, for the density it added. Telescoping says the parts sum to the whole;
this says each part is a proper score of the right object. That is a property of
the residual construction and not of the market wrapper, and it follows
from locality: the increment sees the forecast only through the density at the
realized point, and the logarithm is essentially the unique local strictly proper
score [@bernardo1979expected]. Whether the property *characterizes* the log score
among all proper scores under composition is open.

Three further constructions are the same identity. A residual chain is gradient
boosting under log loss: each stage multiplies the running density by a likelihood
ratio fitted to what the chain so far gets wrong, the greedy stagewise step of
boosting [@mason1999boosting; @friedman2001greedy] applied to density estimation
with the log-likelihood as the loss [@rosset2002boosting], with participants in
place of weak learners and staked wealth in place of the learning rate. Sequential
prediction scored by cumulative log loss is the prequential view
[@dawid1984prequential], under which a probability model and a sequential code are
the same object, the connection minimum description length makes exact
[@gruenwald2019minimum]. And an autoregressive or normalizing-flow model is
already such a chain, its log-likelihood a sum of a base log-density and per-stage
change-of-variables terms [@papamakarios2021normalizing].

CRPS has no matching per-stage identity. Its values along a chain telescope, as
any score's differences do, and its market form has a path-independent cost like
any other proper score's; what it lacks is the identification of the increment
with the CRPS of the stage's residual refinement. The composition acts on
densities by multiplication and on distribution functions not at all simply, so
the difference of two CRPS values is a quantity about the composed forecasts and
not a score of what the stage contributed. This is distinct from the
reliability–resolution decomposition of a
single CRPS value [@hersbach2000decomposition], which splits one score into
calibration components rather than a chain into per-stage credit. CRPS is neither improper nor uninformative, but it does not give a chain
per-stage accounting.

## 3. Two geometries of error

The two scores measure different things. Their regret against the truth $p$
under a forecast $q$ is
$$\text{log-score regret} = D_{\mathrm{KL}}(p \,\|\, q), \qquad
  \text{CRPS regret} = \int_{-\infty}^{\infty}\!\big(F_q(t) - F_p(t)\big)^2\,dt.$$
Both are Bregman divergences of the respective proper score
[@gneiting2007strictly]. The log score penalizes relative density error where the
truth puts its mass; CRPS penalizes integrated distribution-function error across
thresholds. The first is the divergence that governs a likelihood ratio, a Bayesian update,
and a growth-optimal bet; the value a conditioning forecaster extracts over a
crowd that ignores a covariate is exactly the mutual information $I(R;X)$
[@cotton2026conformalbetting]. The second is gentler in the tails: a distant
outcome costs CRPS on the order of its distance, but costs the log score
quadratically, or without bound where the density vanishes.

There is a market reading of the first. In a complete, arbitrage-free market
normalized Arrow–Debreu state prices define a risk-neutral state-price density; in
an incomplete market there may be many, and none need equal the physical law
[@cochrane2005asset]. Either way prices behave like a density-ratio settlement,
not an arbitrary point forecast, and the growth-optimal investor stakes her
physical density to maximize expected log wealth [@kelly1956newinterpretation].
A logarithmic market scoring rule is the mechanism that pays exactly this
[@hanson2007logarithmic]. CRPS has no such reading.

The difference in the tails is measurable in a one-parameter example. Let the
truth be Student-$t$ with three degrees of freedom, heavy-tailed, and fit a single
Gaussian $N(0,\sigma^2)$ to it under each score. The likelihood optimum is
variance-matching, $\sigma=\sqrt{3}\approx1.73$, and covers the tail. The CRPS
optimum is $\sigma\approx1.25$, a $28\%$ narrower forecast that fits the center
and underweights the tail.

![Fitting a Gaussian to Student-$t_3$ truth. Mean CRPS is minimized by a narrow forecast ($\sigma\approx1.25$) that squeezes the tail in; mean negative log-likelihood is minimized by the wider, tail-covering forecast ($\sigma=\sqrt3\approx1.73$). The two proper scores prefer measurably different approximations. Population values by numerical integration.](figures/crps-tail-blindness)

The narrower forecast wins CRPS and loses log-likelihood:

| Gaussian fit to $t_3$ | mean CRPS | mean NLL |
|---|---|---|
| $\sigma=1.73$, likelihood-optimal | 0.848 | 1.968 |
| $\sigma=1.25$, CRPS-optimal | 0.829 | 2.102 |

Two qualifications. This does not beat a correct model: CRPS is proper, so
at the population level the true $t_3$ scores best on both (CRPS $0.827$, NLL
$1.773$), and the trade-off arises only among imperfect forecasts. And it runs the
other way when the tail is genuinely unmodelable, where the narrower forecast is
the sensible one and CRPS is right to prefer it; that same tail-insensitivity is
what the robustness literature counts in CRPS's favor [@bolin2023local]. The two
proper scores rank the same imperfect forecasts differently, and only the log
score's ranking adds up along a chain.

Conformal prediction is this taken to its limit. A standard split-conformal
predictor is not a density forecast at all; it targets finite-sample marginal
coverage [@lei2018distribution]. Coerce its calibration residuals into an
unsmoothed empirical predictive law and it assigns zero density outside the
observed range, hence $-\infty$ log-likelihood on any outcome beyond it, while its
CRPS stays finite and competitive. A method that need not produce a density, yet
scores well on CRPS, is the metric's tail-insensitivity turned into a feature of
the approach, and the conditional information $I(R;X)$ it forfeits is exactly what
a log-scored competitor collects against it [@cotton2026conformalbetting].

## 4. Where CRPS is the right target

None of this makes CRPS a bad score. It answers a different question, and it is
well matched to broad distribution-function accuracy, threshold and quantile
behavior, robustness under tails one cannot model, and interpretability in the
outcome's own units. When the decision is known and coarse, a threshold, a
quantile, an interval, an inventory limit, a smooth payoff integral, the
decision-relevant functional should be scored directly, with a threshold- or
quantile-weighted rule aimed at the region that matters
[@gneiting2011comparing; @gneiting2011making]. CRPS and its relatives are the
right tools there, and as a secondary diagnostic CRPS is a useful check that a
log-score winner has not won by pathological sharpness [@gneiting2007probabilistic].
The claim here is narrower: when the elicited object
is a reusable density and the future use is unknown, and above all when forecasts
are to be chained, the log score is the default because it is the score under
which refinements compose.

## 5. Related work

Each ingredient is old. The residual identity is the probability integral
transform [@rosenblatt1952remarks] and Sklar's theorem [@sklar1959];
probability-integral-transform diagnostics are standard for density forecasts
[@diebold1998evaluating; @gneiting2007probabilistic], and the same change-of-variables
log-additivity is what autoregressive and normalizing-flow models exploit
[@papamakarios2021normalizing].

That the log score is additive under sequential
prediction is the prequential principle [@dawid1984prequential] and the
compression view of minimum description length [@gruenwald2019minimum]. That any
proper score's market form telescopes and has a path-independent cost is Hanson's
market scoring rule [@hanson2007logarithmic] and its convex-duality account
[@abernethy2013efficient]; those are the facts that do *not* separate the log
score from CRPS.

Log-score optimization is likewise the natural objective when predictive
densities are stacked [@yao2018using]. Residual refinement under log loss is
gradient boosting [@mason1999boosting; @friedman2001greedy], applied to density
estimation in [@rosset2002boosting].

The conventional log-versus-CRPS axes,
locality, propriety, sensitivity, robustness, interpretability, are set out in
[@gneiting2007strictly; @machete2013contrasting; @bolin2023local], with locality's
role for the log score going back to [@bernardo1979expected]. The convex-duality
background is consolidated for these mechanisms in [@cotton2026algebra].

These facts belong to separate literatures: market scoring, prequential
analysis, boosting, flows. The log-versus-CRPS *evaluation* debate has drawn on
almost none of them and is conducted as if the object were a single forecast
graded once. Once the object is a *reusable density* and the process is a *chain
of elicitations*, only the log score's stage increment is itself the proper score
of what the stage added, and the reviews above do not weigh this. That is the
natural reading of a distributional prediction supply chain
[@cotton2019roar; @cotton2022microprediction; @cotton2026multistage], and it
recasts conformal prediction, often defended on CRPS and coverage, as the case
in which those metrics do not penalize the missing density.

One question appears not to have been asked. Is an author's choice of evaluation metric correlated with the comparative
advantage of the method being proposed? If methods tend to be reported on the
score they happen to win, the field's stated preferences between log-likelihood
and CRPS would partly reflect selection rather than merit. We are not aware of a
study measuring this correlation.

## 6. Summary

The right score matches the target. When the target is a threshold, a quantile, an
interval, or an action, score it directly, and CRPS or a weighted relative may be
the better tool. When the target is a reusable density under unknown later use,
and when densities are to be composed, held-out log-likelihood is the default: it
is local, its regret is a divergence one can read as information, it is the
settlement rule of a density-ratio market, and it is the score under which
refinements add up. CRPS remains a useful secondary diagnostic and a legitimate
primary target for coarser questions, but it is not the settlement rule for a
chain.
