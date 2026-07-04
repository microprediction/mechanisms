# Multi-Stage Solicitation

### Chained forecast elicitation: representations, and the designs the theory supports

Peter Cotton · *Working draft v0.2* · 2026

---

## Abstract

Elicitation mechanisms chain when one mechanism's output is the next one's
message or settled outcome. What can be chained, and how to score it, turns on
what a mechanism emits: a *point* (a price or a forecast) or a *distribution*
(a density, a percentile, or a rank). This note sorts the chains by
representation, marks which the theory supports, and then reviews which have
been built. Point outputs chain as ordinary derivatives. Distributions chain
more richly, provided an exogenous outcome anchors the chain: a residual stage
is then scored on the part its predecessor left uniform, and a downstream
contribution may be judged in its own market or, for the log score
equivalently, converted into a top-level forecast and judged there. Without an
anchor the chain is indeterminate. Only then do we turn to practice: the
microprediction platform's percentile and copula streams, monteprediction, and
the residual contests at CrunchDAO.

---

## 1. Introduction

Deployed forecasting contests almost always run one level of competition
against one internal model: Numerai pays each staked submission its marginal
contribution to a single meta-model [@craib2017numeraire], CrunchDAO blends a
contest into one ensemble, the IARPA prediction polls aggregated by track
record [@atanasov2017distilling]. The alternative is to *chain*: let one
mechanism's output be the message, or the settled outcome, of the next, so
that calibration and dependence become the subject of their own games.

Whether two mechanisms chain turns on what the upstream one emits, and that is
the organizing question. A stage emits either a point or a distribution, and
sorting the four ways one can feed the next leaves only one combination worth
much (§2). The rest follows from there. Section 3 gives the operator that
chains and the two ways to score a stage inside a chain. Section 4 draws the
line between the designs the theory backs and the one it does not. Section 5 is
what has been built. The single-stage dictionary the chains draw on, proper
scores and market makers and pools as one convex object, is a companion paper
[@cotton2026algebra]; here the object is the chain.

## 2. Two representations, and the chains between them

A stage of a chain emits one of two things.

A **point output** is a scalar summary: a price, a point forecast, a single
reported percentile. A **probabilistic output** is a full object over the
outcome space: a density or distribution function, the probability-integral
transform (rank) of an outcome, or a finite cloud of samples standing in for a
density. The distinction is not cosmetic: it decides what a downstream stage
can be paid to get right.

Write the upstream stage's output as the message or the settlement transform of
the downstream stage. Four chains are possible, and only some carry content.

- **Point → point.** A second market settles on the first's price: every
  derivative, the electricity virtual bids and transmission rights written on a
  day-ahead price, a market on another market's reported number. This is
  ordinary derivative structure, a translation of a point output; it is
  ubiquitous and is not composition of *elicitation* — no distribution is
  elicited about the upstream object.
- **Probabilistic → probabilistic.** The upstream distribution, percentile, or
  rank is the downstream stage's message or settled object. This is the case
  with content, and the rest of the note is about it. It needs one thing to be
  well posed: an exogenous outcome somewhere in the chain to anchor the scores
  (§4).
- **Point → probabilistic.** A point output carries no distribution, so a
  downstream distributional stage has nothing to elicit *about it* unless a
  distribution is supplied. The natural way to supply one is to model the point
  forecast's error: a stage that emits a point leaves a residual, and a second
  stage elicits the residual's law. Point-to-probabilistic chaining *is* the
  residual construction of §3.
- **Probabilistic → point.** Collapsing a distribution to a summary — its mean,
  a chosen quantile — and chaining a point market on the summary throws away
  the distributional content the upstream stage worked to produce. It is
  well defined and lossy, the reverse of the residual construction, and of
  little interest here.

So the representation answers the question of what chains to what: point
outputs chain among themselves as derivatives, distributions chain among
themselves as elicitation, and the only bridge between the two worlds is the
residual — reintroduce a distribution by scoring what a point forecast got
wrong.

## 3. Composing a chain, and the two ways to score it

A *stage* is a transducer (Mealy machine) over one message type. Let
$\mathrm{Dist}$ be the set of distributional beliefs; a stage carries a wealth
state $w$, consumes participant reports in $\mathrm{Dist}^m$ together with,
where it settles, a realized outcome $x$, and emits an aggregate belief and a
transfer vector,

$$M:\ (\mathrm{Dist}^m,\ w,\ x)\ \longmapsto\ (\mathrm{Dist},\ w',\ \pi).$$

A scoring rule is the transfer component of a stage; a market maker is a stage
whose state is the inventory vector and whose emitted belief is the price; an
opinion pool is a stage with no outcome argument and zero transfers; a
parimutuel is a stage whose transfers are a pot split. The operators that build
and combine stages (Sequentialise, Pool, Ensemble, Merge, Conjugate) and the
convex dictionary behind them are the companion algebra [@cotton2026algebra].
One operator does the chaining.

**Residual.** Let a stage emit the aggregate $F_1$ for an outcome $Y$, and
let a second market elicit a distribution for the residual $U=F_1(Y)$,
settling at the realised $u=F_1(y)$. If $F_1$ is the true conditional law
then $U$ is uniform (the probability integral transform [@cotton2026algebra,
Prop. 4]) and the second market has nothing to price; whatever structure
remains in the residual is the second stage's edge.

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

**Local scoring versus top-level conversion.** A contribution to the second
market can be judged two ways. *Locally*, the residual stage is scored on its
own object $u$, by $\log g(u)$. *At the top level*, the contribution is
converted into a forecast of $Y$ — the composed density $p_1(y)\,g(F_1(y))$ —
and scored against the ultimate outcome by $\log p(y)$. Proposition 1 says these
coincide: $\log p(y)=\log p_1(y)+\log g(u)$, so up to the constant $\log p_1(y)$
that no downstream report can move, the two scores rank downstream
contributions identically. This is the choice raised in the stacked-lottery
design [@cotton2020lottery, slides 29-31]: run the secondary market on its own
residual, or fold every secondary contribution up to the top level and score it
there. For the log score they are the same mechanism; the equivalence is
special to scores that are additive under composition, and fails for a general
proper score.

Multiplying the density by a ratio fitted to what the current model gets wrong
is the functional-gradient step of boosting under log loss [@mason1999boosting;
@friedman2001greedy], so a chain of residual markets is stagewise boosting with
wealth as the learning rate.

**Spec.** Serialise a pipeline to data and search over it; the mechanism
analogue is a market over pipelines. Open.

**Stagewise play.** Call a profile of reports a *stagewise equilibrium* if no
participant gains by a deviation confined to a single stage, all other stages'
reports held fixed *and the inputs and settlement transforms of every other
stage clamped at their pre-deviation values*. This is the pipeline version of
the myopic-trader assumption standard in the market-scoring-rule literature
[@hanson2007logarithmic; @chen2010newunderstanding]. The clamp is not cosmetic:
a stage's output is wired into later settlement transforms (the residual point
$u=F_1(y)$, or a rank vector in a copula stage), so an upstream deviation moves
downstream payoffs even when every downstream report is held fixed.

**Proposition 2 (single-stage guarantees compose under clamped stagewise
play).** *Suppose each stage of a pipeline, taken in isolation with its inputs
and settlement transform fixed, makes the truthful report a best response:
strict propriety for externally funded scoring stages [@cotton2026algebra, Thm
1], the sequential scoring [@cotton2026algebra, Thm 2] for market stages, the
price-taking pot-split analysis of the point-cloud paper [@cotton2026pointcloud]
for parimutuel stages, and the residual score of Proposition 1 for correction
stages. Then truthful reporting at every stage is a stagewise equilibrium of
the pipeline. If moreover no participant reports to a stage upstream of one in
which they hold a position (disjoint stage membership suffices), the clamp is
vacuous for every feasible deviation, and truthful reporting survives
unrestricted single-stage deviations.*

**Proof.** With the other stages' inputs and transforms clamped, a deviation
confined to stage $k$ changes the deviator's payoff only through stage $k$'s
transfer map, and the stage-$k$ hypothesis makes the truthful report a best
response. For the second claim: a stage-$k$ deviation moves stage $k$'s
transfer and, through stage $k$'s output, transfers strictly downstream of $k$;
a deviator with no downstream position collects none of the latter, so the
propagation is payoff-irrelevant to them. $\blacksquare$

Proposition 2 is not a Nash equilibrium of the composed game. When the same
participant reports upstream and holds exposure downstream, the deviation
propagates through the settlement transform, and a downstream stake is a
derivative written on an upstream settlement, with the attendant incentives to
distort the underlying [@kumar1992futures; @jarrow1994derivative;
@hanson2009manipulator; @ostrovsky2012information]. Disjoint membership, zero
downstream exposure for upstream reporters, or exogenous freezing of the
settlement transforms restore the proposition; the dynamic game without them is
outside this note's scope.

## 4. Which designs the theory supports

The propriety of the chain rests on one requirement: an exogenous outcome
anchors it. Every score above is paid against a realised $y$ or its rank $u$;
strip the outcome out and nothing pins the reports down. This is the line
between the designs the theory supports and the one it does not.

**Anchored probabilistic chains (supported).** A pipeline whose final stage
settles on a real outcome, with proper local scores at each stage, is
well founded: Proposition 2 makes truthful reporting a stagewise equilibrium,
and Proposition 1's additivity makes local scoring and top-level conversion the
same mechanism. Residual chains (boosting), and the copula factoring below, are
of this kind.

**Unanchored chains (not supported).** A chain in which one market settles on
another market's displayed probability, with no exogenous outcome anywhere,
has no scoring anchor: the equilibrium is indeterminate and the mechanism runs
as a beauty contest. Manifold's resolves-to-market markets are the live
example; they are play-money precisely because there is nothing to be proper
about.

**Samples as the message (supported, and what contests collect).** The message
type has so far been a distribution given exactly. Deployed contests collect
something rougher: a finite cloud of samples, smoothed into a density before
settlement. The companion point-cloud paper gives the elicitation result the
chain needs [@cotton2026pointcloud]:

**Proposition 3 (sample-based elicitation; @cotton2026pointcloud, Thms 1-2).**
*Score the bandwidth-$h$ kernel density estimate of a submitted cloud by the
logarithmic score. Settled at the raw outcome, the optimal cloud is drawn from
a deconvolution of the belief when one exists (for Gaussian beliefs and kernels
with belief variance exceeding $h^2$, the belief with $h^2$ removed from the
variance). Settled at the outcome jittered by the same kernel, truthful
sampling is optimal, and strictly so whenever the kernel's characteristic
function is nonvanishing on a dense set: the smoothing channel is injective on
laws.*

With jittered settlement the cloud is a valid message, and the operators act on
clouds pointwise: Conjugate pushes each sample through the map, a linear Pool is
a wealth-weighted union of clouds, Residual ranks each sample through the
upstream CDF. One caveat governs the order of operations — smoothing and a
nonlinear map do not commute, so the kernel and its jitter belong in the
settlement coordinates, immediately before settlement and nowhere else.

**Dependence by rank (supported).** A multivariate settlement factors. By
Sklar's theorem [@sklar1959] a joint law with continuous margins $F_1,\dots,F_d$
and copula $C$ has density $p(x)=\prod_i f_i(x_i)\cdot c(F_1(x_1),\dots,F_d(x_d))$.

**Proposition 4 (the log score factors).** *For a joint report assembled from
margin reports $f_i$ and a rank-stage report $c$, a density on $[0,1]^d$,*

$$\log p(x)\;=\;\sum_{i=1}^d\log f_i(x_i)\;+\;\log c(u),
\qquad u_i=F_i(x_i),$$

*so a pipeline of $d$ margin stages and one rank stage settled on the rank
vector $u$ pays every stage a logarithmic score of its own object, and the
chain's log score is the sum. Given the margin reports, the rank stage's score
is strictly proper for the law of the rank vector, which has uniform margins
and is the copula of the joint iff the margin reports are correct.*

**Proof.** Take logarithms in Sklar's factorization; the summands are the stage
scores. Strict propriety of the rank stage is the logarithmic score
[@cotton2026algebra, Thm 1] on densities over $[0,1]^d$, the transform
$x\mapsto u$ fixed once the margins are given. $\blacksquare$

This is the same additivity as Proposition 1, one dimension per margin: the
rank stage is the multivariate Residual operator, eliciting dependence
separately from level, and again local scoring of the rank stage equals its
top-level conversion. The construction is the two-stage estimation logic of
inference functions for margins [@joe1997multivariate] read as a market design.

**A worked anchored chain: betting against a conformal predictor.** A point
predictor leaves a residual $R$; conformalization re-levels its marginal law
but prices the residual flat in the input. An entrant who conditions on the
input $X$ collects what the flat predictor discards as bankroll growth, at the
rate $I(R;X)$, the mutual information between residual and input, while the
band's marginal coverage stays exact; marginal coverage is the break-even
statement, the conditional information is the rent. The full account, with the
Gaussian rate $-\tfrac12\log(1-\rho^2)$ and the anytime-valid measurement, is a
standalone note [@cotton2026conformalbetting]. It is a point-to-probabilistic
residual chain with the outcome as anchor, so the theory above applies whole.

## 5. What has been built

The supported designs are rare in practice; one platform ran several at once.

**The microprediction platform.** The base game was a pool on a single live
number. Anyone could open a stream for a quantity they cared about;
contributors submitted a fixed-size bundle of Monte Carlo scenarios, 225 of
them, and when the number arrived the pot was split by how near the scenarios
fell. Paying by log-wealth made a contributor's best move to submit scenarios
that matched her honest distribution [@cotton2022microprediction]. That is the
nearest-the-pin rule, and it only ever settled one scalar.

To make calibration and dependence into games of their own, the platform
turned each of them into another scalar. Calibration first. Once the community
has forecast a quantity, ask where the outcome actually landed in the forecast
distribution: its percentile, or the z-score you get by pushing that percentile
through a normal. Forecast well and those z-scores look standard normal;
forecast badly and they drift or spread out. A z-score is just another number,
so it opened its own stream, `z1~`, and predicting `z1~` meant predicting the
first game's miscalibration. In the language of §3 this is the
probability-integral transform run as a residual stage.

Dependence used the same idea in reverse. The joint behaviour of two streams
is two-dimensional, but a pool settles one number, so the two community
percentiles were folded into a single number by the Morton z-curve: interleave
the binary digits of the two coordinates, the way a geohash packs latitude and
longitude into one string. A pool on the folded number was a market on the
copula of the pair, `z2~`; three streams folded the same way gave `z3~`. A 2020
copula contest ran exactly this on the five-minute comovements of five
cryptocurrencies, contributors submitting 225 samples packed through the
z-curve [@cotton2020copula]. This is the rank factoring of Proposition 4, in
production. A related stacked-lottery design let competing algorithms contribute
monotone calibration maps that composed into one forecast [@cotton2020lottery,
slides 29-31]. The platform is retired.

**Its successors.** monteprediction is the base game with one stage repeated
rather than chained. Each weekly submission is about a million joint scenarios
of eleven sector-ETF returns, and the pot is split in proportion to the density
a participant places on the realised vector [@cotton2024monteprediction;
@cotton2024eleven]; wealth threads across rounds, and the contest has run since
January 2024. The MidOne contests at CrunchDAO priced residual densities
directly [@crunchdao_midone; @cotton_density], a residual stage without the
sample smoothing.

**For contrast, the point-output world.** Chaining on point outputs is
everywhere: derivatives, the electricity virtual bids and transmission rights
on day-ahead prices [@jha2023financial], a market on another market's reported
number. But it is derivative structure, not elicitation composition. And the
racetrack's win versus exotic pools, index versus single-name option books, and
tranche versus single-name CDS run margins and dependence in *parallel* books
that settle independently, consistency left to arbitrage
[@harville1973assigning; @hausch1981efficiency]; parallel is not chained, and
the books can disagree.

## 6. Open problems

1. *Conservation of edge.* Wealth conservation is automatic, a sum of
   stage-level zeros, and is not the invariant. The invariant that can fail
   is the edge a truthful participant holds over the price, $D_G(q,\pi)$.
   For the logarithmic score the edge is $\mathrm{KL}(q\Vert\pi)$ and the
   data-processing inequality settles it: an interface channel cannot increase
   it, and preserves it exactly when the channel is sufficient for the pair
   [@csiszar1967information]. No such inequality holds for a general Bregman
   edge. For the quadratic generator, merging the first two of three outcomes
   sends $\pi=(\tfrac13,\tfrac13,\tfrac13)$ and
   $q=(\tfrac{13}{30},\tfrac{13}{30},\tfrac2{15})$, with $D_G(q,\pi)=\tfrac3{50}$,
   to a pair with $D_G(qK,\pi K)=\tfrac2{25}$: coarsening increased the Brier
   edge. Open: characterize the generators and channels for which
   $D_G(qK,\pi K)\le D_G(q,\pi)$.
2. *Residual markets.* The single-step correction is a multiplicative
   reweighting of the upstream density (Proposition 1); a chain of residual
   markets is stagewise boosting with wealth as the learning rate. Open: the
   microstructure (who funds each residual pot, and when it settles relative to
   the stage before), whether a forecaster free to enter several stages prefers
   withholding information upstream to sell it downstream, and whether the
   iterated correction converges to the true conditional law as boosting does.
   Split-conformal prediction is the one-participant degenerate case, with its
   rent $I(R;X)$ as the value left on the table
   [@cotton2026conformalbetting].
3. *Copula markets.* A rank-settled stage elicits dependence separately from
   margins (Proposition 4). Open: the equilibrium when the same participants
   trade margin and copula stages, and the choice of embedding for $d\ge2$, the
   space-filling curves as deployed against the random one-dimensional
   projections of the point-cloud paper [@cotton2026pointcloud].

## References

::: {#refs}
:::
