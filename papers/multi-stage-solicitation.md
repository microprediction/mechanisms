# Multi-Stage Solicitation

### Chained elicitation markets, and lessons from the microprediction platform

Peter Cotton · *Working draft v0.1* · 2026

---

## Abstract

Deployed forecasting contests almost always run one level of competition
against one internal model. The microprediction platform was built the other
way: it chained elicitation mechanisms, one market's probabilistic output
becoming the message or the settlement of the next. Community percentiles from
univariate z-streams were themselves the object of further games; bivariate
and trivariate dependence streams priced copulas; a stacked-lottery design
composed monotone calibration maps contributed by competing algorithms. This
note reports that design and the theory it rests on. The single-stage
dictionary — proper scores, market makers, and pools as one convex object — is
a companion paper [@cotton2026algebra]; here the object is the chain. The
message type deployed contests collect is a finite cloud of samples, and the
sample-native forms of the composition operators are given; multivariate
settlement factors into marginal stages and a rank-settled copula stage; and a
residual market behind a conformal predictor collects the predictor's
discarded conditional information as bankroll growth. The platform is retired;
its nearest live successor, monteprediction, runs one stage of the chain
repeatedly.

---

## 1. Introduction

Deployed forecasting contests are thinner than the theory allows: with few
exceptions they run one level of competition against one internal model.
Numerai pools staked submissions into a single stake-weighted meta-model and
pays each forecast its marginal contribution to it [@craib2017numeraire];
CrunchDAO blends each contest into one ensemble; the IARPA prediction polls
aggregated forecasts by track record, comparing favourably with head-to-head
markets [@atanasov2017distilling], with reputation rather than wealth as the
threaded state. In every case, one pool and one internal aggregate.

The microprediction platform was built the other way. A market's
*probabilistic* output — a distribution, a percentile, or a rank emitted by
one stage — became the message or the settlement transform of the next, so
calibration and dependence were themselves the subject of further games.
Univariate streams spawned z-streams of community percentiles; bivariate and
trivariate dependence streams priced copulas; and a stacked-lottery design
composed monotone calibration maps contributed by competing algorithms,
presented at MIT CSAIL in 2020 [@cotton2020lottery, slides 29-31;
@cotton2022microprediction]. The platform is retired. Its nearest live
successor, monteprediction, is a weekly self-funding pool over
million-scenario joint submissions in eleven dimensions with wealth threaded
across rounds since January 2024 [@cotton2024monteprediction]: one stage of
the chain, repeated, rather than a chain.

This note is a report on that design. The mechanisms compose because each is a
stateful transducer over one message type, a distributional belief, with
wealth as the threaded state; the single-stage dictionary that generates them
— proper scores, market makers, and pools as one convex object — is a
companion paper [@cotton2026algebra], and §2 recalls only what the chain
needs. The rest is the multi-stage account the platform demanded: the finite
sample cloud as the message type deployed contests actually collect (§3), the
factoring of a multivariate settlement into marginal stages and a rank-settled
copula stage (§4), and the residual chain behind a conformal predictor whose
degenerate case leaves an extractable rent (§5). Play is stagewise throughout:
no participant deviates across stages; the cross-stage game is largely open
(§6).

## 2. Stages and their composition

A *stage* is a transducer (Mealy machine) over one message type. Let
$\mathrm{Dist}$ be the set of distributional beliefs; a stage carries a wealth
state $w$, consumes participant reports in $\mathrm{Dist}^m$ together with,
where it settles, a realized outcome $x$, and emits an aggregate belief and a
transfer vector,

$$M:\ (\mathrm{Dist}^m,\ w,\ x)\ \longmapsto\ (\mathrm{Dist},\ w',\ \pi).$$

A scoring rule is the transfer component of a stage; a market maker is a stage
whose state is the inventory vector and whose emitted belief is the price; an
opinion pool is a stage with no outcome argument and zero transfers; a
parimutuel is a stage whose transfers are a pot split. Composition wires the
belief output of one stage to the belief inputs of the next while state and
transfers thread through.

The single-stage operators — Sequentialise (a proper score run against a
wealth state is a cost-function market maker), Pool, Ensemble, Merge, and
Conjugate — and the convex dictionary behind them are the companion algebra
[@cotton2026algebra]. Two operators do the chaining, and their guarantees are
what let a pipeline be trusted.

**Residual.** Let a stage emit the aggregate $F_1$ for an outcome $Y$, and
let a second market elicit a distribution for the residual $U=F_1(Y)$,
settling at the realised $u=F_1(y)$. If $F_1$ is the true conditional law
then $U$ is uniform (the probability integral transform [@cotton2026algebra,
Prop. 4]) and the second market has nothing to price; whatever structure
remains in the residual is the second stage's edge. The corrected forecast
composes the two reports.

**Proposition 1 (the correction is a multiplicative reweighting).** *Let
$F_1$ be strictly increasing onto $(0,1)$ with density $p_1>0$ and let the
residual market's consensus be a distribution $H$ on $[0,1]$ with $H(0)=0$,
$H(1)=1$ and density $g$. The composed forecast $F=H\circ F_1$ has density*

$$p(y) \;=\; p_1(y)\, g\!\big(F_1(y)\big),$$

*so $\log p(y)=\log p_1(y)+\log g(u)$ with $u=F_1(y)$: the chain's log score
is the sum of stage log scores, and the residual stage is paid by a proper
score on $u$ alone.*

**Proof.** Chain rule: $F'(y)=g(F_1(y))\,p_1(y)$; take logarithms. The
residual score $\log g(u)$ is the logarithmic score [@cotton2026algebra, Thm
1] applied to the report $g$ and outcome $u$, hence strictly proper for the
law of $U$. $\blacksquare$

Multiplying the density by a ratio fitted to what the current model gets
wrong is the functional-gradient step of boosting under log loss
[@mason1999boosting; @friedman2001greedy], so a chain of residual markets is
stagewise boosting with wealth as the learning rate. What Proposition 1 does
not settle is the game across stages: who funds the residual pot, and whether
a forecaster free to enter both stages prefers to withhold information from
the first and sell it to the second (§6).

**Spec.** Serialise a pipeline to data and search over it; the mechanism
analogue is a market over pipelines. Also open.

**Stagewise play.** Call a profile of reports a *stagewise equilibrium* if
no participant gains by a deviation confined to a single stage, all other
stages' reports held fixed *and the inputs and settlement transforms of
every other stage clamped at their pre-deviation values*. This is the
pipeline version of the myopic-trader assumption standard in the
market-scoring-rule literature [@hanson2007logarithmic;
@chen2010newunderstanding]. The clamp is not cosmetic: a stage's output is
wired into later settlement transforms (the residual point $u=F_1(y)$, or a
rank vector in a copula stage), so an upstream deviation moves downstream
payoffs even when every downstream report is held fixed.

**Proposition 2 (single-stage guarantees compose under clamped stagewise
play).** *Suppose each stage of a pipeline, taken in isolation with its
inputs and settlement transform fixed, makes the truthful report a best
response: strict propriety for externally funded scoring stages
[@cotton2026algebra, Thm 1], the sequential scoring [@cotton2026algebra, Thm
2] for market stages, the price-taking pot-split analysis of the point-cloud
paper [@cotton2026pointcloud] for parimutuel stages, and the residual score
of Proposition 1 for correction stages. Then truthful reporting at every
stage is a stagewise equilibrium of the pipeline. If moreover no participant
reports to a stage upstream of one in which they hold a position (disjoint
stage membership suffices), the clamp is vacuous for every feasible
deviation, and truthful reporting survives unrestricted single-stage
deviations.*

**Proof.** With the other stages' inputs and transforms clamped, a deviation
confined to stage $k$ changes the deviator's payoff only through stage $k$'s
transfer map, and the stage-$k$ hypothesis makes the truthful report a best
response. For the second claim: a stage-$k$ deviation moves stage $k$'s
transfer and, through stage $k$'s output, transfers strictly downstream of
$k$; a deviator with no downstream position collects none of the latter, so
the propagation is payoff-irrelevant to them. $\blacksquare$

Proposition 2 is not a Nash equilibrium of the composed game. When the same
participant reports upstream and holds exposure downstream, the deviation
propagates through the settlement transform, and a downstream stake is a
derivative written on an upstream settlement, with the attendant incentives
to distort the underlying [@kumar1992futures; @jarrow1994derivative;
@hanson2009manipulator; @ostrovsky2012information]. Disjoint membership,
zero downstream exposure for upstream reporters, or exogenous freezing of
the settlement transforms restore the proposition; the dynamic game without
them is outside this note's scope.

The rest of this note reports the two message types the platform chained on —
sample clouds (§3) and copula ranks (§4) — and the residual chain that a
conformal predictor degenerates (§5).

## 3. Samples as messages

The message type has so far been a distribution given exactly. Deployed
contests collect something rougher: a finite cloud of samples, smoothed by
the contest into a density before settlement. The companion point-cloud paper
gives the sample-based elicitation result the algebra needs
[@cotton2026pointcloud]:

**Proposition 3 (sample-based elicitation; @cotton2026pointcloud, Thms
1-2).** *Score the bandwidth-$h$ kernel density estimate of a submitted
cloud by the logarithmic score. Settled at the raw outcome, the optimal
cloud is drawn from a deconvolution of the belief when one exists (for
Gaussian beliefs and kernels with belief variance exceeding $h^2$, the
belief with $h^2$ removed from the variance). Settled at the
outcome jittered by the same kernel, truthful sampling is optimal, and
strictly so whenever the kernel's characteristic function is nonvanishing on
a dense set: the smoothing channel is injective on laws.*

The proof is in the companion paper. With jittered settlement the cloud
*is* a valid message, and the operators of the algebra [@cotton2026algebra]
act on clouds directly.

- *Conjugate* acts pointwise: the pushforward of a cloud under $\psi$ is
  $\psi$ applied to each sample, with no Jacobian computed anywhere. This is
  the practical reason to prefer sample messages: transformation of
  densities requires calculus, transformation of clouds requires
  arithmetic.
- *Pool* (linear) is a wealth-weighted union: sample from participant $i$'s
  cloud with probability proportional to $w_i$. The linear pool is the
  sample-native aggregate; the logarithmic pool has no comparably clean
  sample form, since multiplying densities requires importance weights.
- *Residual* acts by ranking: pass each sample through the upstream CDF.
  Composition of monotone maps, the stacked-lottery operation, is again
  pointwise on samples.
- *Sequentialise* is the nearest-the-pin pool itself: the market's state is
  the field's aggregate cloud.

One caveat governs the order of operations. Smoothing and conjugation
commute only for affine maps: for scalar $\psi(x)=ax+b$ and a Gaussian
kernel, the KDE of the mapped cloud at bandwidth $|a|h$ equals the
pushforward of the bandwidth-$h$ KDE (in higher dimension the bandwidth
matrix must transform with the map), and for nonlinear $\psi$ no bandwidth
makes this an identity. The smoothing seam and its jitter must therefore be applied in the
settlement coordinates, after all transforms, which in pipeline terms says
the KDE stage belongs immediately before settlement and nowhere else.

## 4. Margins and copulas

A multivariate settlement can be factored. By Sklar's theorem [@sklar1959]
a joint law with continuous margins $F_1,\dots,F_d$ and copula $C$ has
density

$$p(x)\;=\;\prod_{i=1}^d f_i(x_i)\;\cdot\;
c\big(F_1(x_1),\dots,F_d(x_d)\big),$$

with $c$ the copula density on $[0,1]^d$.

**Proposition 4 (the log score factors).** *For a joint report assembled
from margin reports $f_i$ and a rank-stage report $c$, a density on
$[0,1]^d$,*

$$\log p(x)\;=\;\sum_{i=1}^d\log f_i(x_i)\;+\;\log c(u),
\qquad u_i=F_i(x_i),$$

*so a pipeline of $d$ margin stages and one rank stage settled on the rank
vector $u$ pays every stage a logarithmic score of its own object, and the
chain's log score is the sum. Given the margin reports, the rank stage's
score is strictly proper for the law of the rank vector. That law has
uniform margins, and is the copula of the joint, iff the margin reports are
correct; keeping the rank stage's report class as arbitrary densities on
$[0,1]^d$ keeps the stage closed under wrong upstream reports.*

**Proof.** Take logarithms in Sklar's density factorization; the summands
are exactly the stage scores. Strict propriety of the rank stage is the
logarithmic score's strict propriety [@cotton2026algebra, Thm 1] applied to
densities on $[0,1]^d$, the transform $x\mapsto u$ being fixed once the
margin reports are given. The final clause is Sklar's theorem applied to the
true joint law. $\blacksquare$

This is the two-stage estimation logic of inference functions for margins
[@joe1997multivariate], and the factorization underlies out-of-sample
copula comparison [@diks2010copula]; here it is read as a market design.
The rank stage is the multivariate Residual operator of §2: under correct
margins the rank vector has uniform
margins and its joint law is the dependence structure alone, so a
rank-settled market elicits dependence separately from level. When the
margins are correct the settled object is invariant to monotone
transformations of the coordinates; the settlement transform itself moves
with the reported margin maps. The microprediction platform ran exactly this
factoring: alongside univariate z-streams it operated bivariate and
trivariate streams in which community-implied percentiles were embedded by a
space-filling curve into one settled scalar [@cotton2022microprediction], a
copula market in production. Sample messages compose with the factoring:
rank each coordinate of a submitted cloud through the margin CDFs and the
result is a cloud on $[0,1]^d$ for the copula stage, with the §3 seam
discipline applied at that stage's settlement.

## 5. A residual chain: betting against a conformal predictor

The purest deployed chain is a residual market run behind a point predictor.
A point predictor leaves a residual $R$; re-leveling its marginal law is
conformalization, and a market on the residual rank prices whatever
conditional structure the re-leveling ignores. A single-shape conformal
predictor, one that applies the same residual law at every input, prices the
residual pool flat: its marginal coverage is exact and its conditional
information is discarded. An entrant who conditions on the input $X$ collects
that discarded information as bankroll growth, at the rate $I(R;X)$, the
mutual information between residual and input, while the band's marginal
coverage stays exact. Marginal coverage is the break-even statement; the
conditional information is the rent.

The full account — the pool payoff, the Kelly-Breiman growth identity, the
Gaussian rate $-\tfrac12\log(1-\rho^2)$, and the anytime-valid measurement
that turns the rent into a test — is a standalone note
[@cotton2026conformalbetting]. The mechanism ran in production: the
microprediction nearest-the-pin pool [@cotton2022microprediction] paid each
entry its sample density at the realised value relative to the field's, and
the MidOne contest [@crunchdao_midone] priced an explicit residual density
[@cotton_density] the same way. A conformal predictor entering such a pool is
the participant that prices flat in $X$; its better-informed competitor's
bankroll is the rent.

## 6. Open problems

1. *Conservation of edge.* Wealth conservation is automatic, a sum of
   stage-level zeros, and is not the invariant. The invariant that can fail
   is the edge a truthful participant holds over the price, $D_G(q,\pi)$.
   For the logarithmic score the edge is $\mathrm{KL}(q\Vert\pi)$ and the
   data-processing inequality settles it: an interface
   channel cannot increase it, and preserves it exactly when the channel is
   sufficient for the pair [@csiszar1967information]. No such inequality
   holds for a general Bregman edge. For the quadratic generator, merging
   the first two of three outcomes sends
   $\pi=(\tfrac13,\tfrac13,\tfrac13)$ and
   $q=(\tfrac{13}{30},\tfrac{13}{30},\tfrac2{15})$, with
   $D_G(q,\pi)=\tfrac3{50}$, to a pair with
   $D_G(qK,\pi K)=\tfrac2{25}$: coarsening increased the Brier edge. Open:
   characterize the generators and channels for which
   $D_G(qK,\pi K)\le D_G(q,\pi)$, and quantify the contraction or
   amplification of edge through a given interface.
2. *Residual markets.* The single-step correction is a multiplicative
   reweighting of the upstream density (Proposition 1); a
   chain of residual markets is stagewise boosting with wealth as the
   learning rate. Open: the microstructure (who funds each residual pot, and
   when it settles relative to the stage before), whether a forecaster free
   to enter several stages prefers withholding information upstream to sell
   it downstream, and whether the iterated correction converges to the true
   conditional law as boosting does. Split-conformal prediction is the
   one-participant degenerate case, with its rent $I(R;X)$ as the value left
   on the table [@cotton2026conformalbetting]. The closest live analogue is
   in point prices: a virtual bid in a two-settlement electricity market is
   written on the day-ahead stage's pricing error, and its convergence
   bidders are the residual stage's informed entrants [@jha2023financial].
   The distributional version is the gap.
3. *Copula markets.* A rank-settled stage elicits dependence separately
   from margins (Proposition 4), and under correct margins its settled
   object is invariant to monotone transformations of the coordinates.
   The racetrack's win and exotic pools price margins and joint orders in
   parallel books on a finite outcome space, with consistency left to
   arbitrage and the Harville map as the bridge [@harville1973assigning;
   @hausch1981efficiency]; the rank-settled stage differs by construction,
   chaining through the reported margins so the dependence market cannot
   disagree with them. Open: the equilibrium when the same participants
   trade margin and copula stages, and the choice of embedding for
   $d\ge2$, the space-filling curves as deployed against the random
   one-dimensional projections of the point-cloud paper
   [@cotton2026pointcloud].

## References

::: {#refs}
:::
