# Multi-Stage Solicitation of Probability Distributions

### Experiments and Theory

Peter Cotton · *Working draft v0.3* · 2026

---

## Abstract

The microprediction platform built forecasting supply chains, chaining pools so
that one's output became the message or the settled outcome of the next. This
note describes those games and asks, of each, whether truthful reporting was
really the best move. A chain is proper where an exogenous outcome anchors every
stage; a downstream stage can then be scored on its own residual or, for the log
score, at the top level. By that test the games mostly hold up, one of them only
by a lucky accident.

---

## 1. The games that were built

Adam Smith's pin factory [@smith1776wealth] made pins cheaply by splitting the
work into specialized steps. Prediction can be organized the same way. A
forecast that has to be produced over and over is cheaper if it is broken into
standard sub-forecasts, each contributed by whoever is best at it, the supply
chain the microprediction project set out to build [@cotton2022microprediction].
Markets and contests supply one half of that, the competition that makes any
single sub-forecast cheap, but not the other half: a market pays for the best
forecast of its own number and stops, with no reason to pass its output to a
next stage. Chaining supplies the rest. The output of one mechanism becomes the
message, or the settled outcome, of the next.

A handful of forecasting contests have chained one elicitation mechanism onto
another this way. One platform ran several such chains at once.

**The microprediction platform.** The base game was a pool on a single live
number. Anyone could open a stream for a quantity they cared about;
contributors submitted a fixed-size bundle of Monte Carlo scenarios, 225 of
them, and when the number arrived the pot was split in proportion to how near
the scenarios fell. A contributor who bets to maximize her long-run wealth then
does best to place scenarios according to her honest distribution: the
log-optimal, all-in Kelly bet [@cotton2022microprediction]. This is a property
of the log-optimal player, not of the pot split alone; a contributor
optimizing something else stakes differently. That is the nearest-the-pin rule,
and it only ever settled one scalar.

To make calibration and dependence into games of their own, the platform
turned each of them into another scalar. Calibration first. Once the community
has forecast a quantity, ask where the outcome actually landed in the forecast
distribution: its percentile, or the z-score you get by pushing that percentile
through a normal. Forecast well and those z-scores look standard normal;
forecast badly and they drift or spread out. A z-score is just another number,
so it opened its own stream, `z1~`, and predicting `z1~` meant predicting the
first game's miscalibration. In the language of §2 this is the
probability-integral transform run as a residual stage.

Dependence used the same idea in reverse. The joint behaviour of two streams
is two-dimensional, but a pool settles one number, so the two community
percentiles were folded into a single number by the Morton z-curve: interleave
the binary digits of the two coordinates, the way a geohash packs latitude and
longitude into one string. A pool on the folded number was a market on the
copula of the pair, `z2~`; three streams folded the same way gave `z3~`. A 2020
copula contest ran exactly this on the five-minute comovements of five
cryptocurrencies, contributors submitting 225 samples packed through the
z-curve [@cotton2020copula]. A related stacked-lottery design let competing
algorithms contribute monotone calibration maps that composed into one forecast
[@cotton2020lottery, slides 29-31]. The platform is retired.

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
tranche versus single-name CDS run margins and dependence in parallel books
that settle independently, consistency left to arbitrage [@harville1973assigning;
@hausch1981efficiency]; parallel is not chained, and the books can disagree.

## 2. What can be proven

Each of these games composes stages that share one form. A *stage* is a
transducer over one message type: it carries a wealth state, consumes reports
that are distributional beliefs, and where it settles takes a realized outcome
and returns transfers. A scoring rule, a market maker, a pool, and a parimutuel
are all stages, and the operators that build and combine them, together with
the convex dictionary behind them, are a companion paper [@cotton2026algebra].
Two facts govern whether a chain of them is valid.

**A chain is proper only where an outcome anchors it.** Every score is paid
against a realized value or its rank; strip the outcome out and nothing pins
the reports down. A chain whose final stage settles on a real outcome, with a
proper score at each stage, is well founded, and truthful reporting is a
stagewise equilibrium.

**Proposition 1 (single-stage guarantees compose).** *If each stage of a
pipeline, taken with its inputs and settlement transform fixed, makes the
truthful report a best response, then truthful reporting at every stage is a
stagewise equilibrium: no participant gains by a deviation confined to one
stage. If moreover no participant reports upstream of a stage in which they hold
a position, truthfulness survives unrestricted single-stage deviations.*

**Proof.** With the other stages clamped, a deviation confined to stage $k$
moves the deviator's payoff only through stage $k$'s transfer, where the
single-stage hypothesis makes truth a best response. A stage-$k$ deviation also
moves transfers strictly downstream of $k$; a deviator with no downstream
position collects none of them. $\blacksquare$

This is not a Nash equilibrium of the full game: a participant who reports
upstream and holds a downstream stake holds a derivative on an upstream
settlement, with the usual incentive to distort the underlying
[@kumar1992futures; @jarrow1994derivative; @hanson2009manipulator;
@ostrovsky2012information]. The platform's chains lived inside this caveat.

**The residual stage, and two ways to score it.** The operator that chains is
the residual: a stage emits an aggregate $F_1$ for an outcome $Y$, and a second
market elicits the law of the residual $U=F_1(Y)$, settling at $u=F_1(y)$. If
$F_1$ were the truth then $U$ is uniform (the probability integral transform),
and whatever structure remains is the second stage's edge.

**Proposition 2 (the residual correction is a reweighting).** *For $F_1$
strictly increasing with density $p_1>0$ and a residual report with density
$g$ on $[0,1]$, the composed forecast $F=H\circ F_1$ has density $p(y)=p_1(y)\,
g(F_1(y))$, so*
$$\log p(y)\;=\;\log p_1(y)+\log g(u),\qquad u=F_1(y).$$

**Proof.** Chain rule on $F=H\circ F_1$; take logarithms. The term $\log g(u)$
is the logarithmic score [@cotton2026algebra, Thm 1] for the report $g$ against
$u$, strictly proper for the law of $U$. $\blacksquare$

The additive form answers a design question raised by the stacked lottery
[@cotton2020lottery]. A contribution to the second market can be judged
*locally*, by $\log g(u)$ on its own residual, or converted to a *top-level*
forecast of $Y$ and judged by $\log p(y)$. Proposition 2 makes these the same
mechanism, since they differ by $\log p_1(y)$, which no downstream report can
move. The equivalence is special to the log score and to scores additive under
composition; for a general proper score, local and top-level scoring rank
downstream contributions differently.

**Dependence factors the same way.** By Sklar's theorem [@sklar1959] a joint
density is $\prod_i f_i(x_i)\cdot c(F_1(x_1),\dots,F_d(x_d))$, so

$$\log p(x)\;=\;\sum_{i=1}^d \log f_i(x_i)+\log c(u),\qquad u_i=F_i(x_i).$$

A pipeline of $d$ margin stages and one rank stage settled on the rank vector
pays each stage a log score of its own object, and the rank stage's report is
strictly proper for the copula given correct margins — the multivariate
residual, one dimension per margin.

**What the pool actually elicits.** The base game does not receive a density;
it receives samples and smooths them. That smoothing is where propriety is won
or lost.

**Proposition 3 (sample elicitation; @cotton2026pointcloud, Thms 1-2).** *Score
the bandwidth-$h$ kernel density estimate of a submitted cloud by the log score.
Settled at the raw outcome, the optimal cloud is drawn from a deconvolution of
the belief, not the belief. Settled at the outcome jittered by the same kernel,
truthful sampling is optimal, strictly so when the kernel's characteristic
function is nonvanishing on a dense set.*

The correction is a jitter of the settlement, and it is the hinge on which the
built games turn.

## 3. Were the games valid?

Take the games of §1 to the tests of §2.

**The base pool: proper by lucky accident.** The nearest-the-pin pool paid each
contributor the density their smoothed samples placed at the realized value.
By Proposition 3, scored at the raw outcome that rewards a deconvolution of the
belief, not the belief: a contributor whose honest law is Gaussian would be paid
most by submitting samples with variance $h^2$ *below* the truth. The platform
escaped this, but by luck. Discrete outcomes caused computational trouble (ties
when the outcome fell on a submitted sample, degenerate estimates), so the
implementation jittered the settlement to smooth them over. A jitter of the
outcome is the repair Proposition 3 prescribes, and it removed the deconvolution
incentive — a fortunate accident, even though the amount added for numerical
comfort was almost certainly not the amount matched to the smoothing scale that
strict propriety asks. The hack that kept the arithmetic well behaved was the
hinge that kept the game roughly honest. monteprediction settles on continuous
returns, so the same accident is not available to it; whether its plug-in
density split needs a fair finite-sample correction is a live question.

**The `z1~` calibration stream: a residual pool that rewards sharpness.**
Predicting the realized z-score is a proper elicitation of its law, anchored by
the exogenous outcome (with the same jitter caveat as the base pool). It is a
residual stage on the base game's PIT, and it does reward sharpness. If the
community's forecast was miscalibrated, its z-scores drift or spread, and
whoever predicts that is paid. And if the base forecast ignored a covariate, an
entrant who conditions on it forecasts the z-score's conditional law, beats the
flat uniform prediction, and collects the discarded conditional information as
growth, at the rate $I(R;X)$ [@cotton2026conformalbetting]. What `z1~` does not
deliver is a certificate from marginal statistics: a uniform PIT does not by
itself witness a sharp forecast, so the premium for sharpness is claimed by an
entrant who actually holds the better conditional information and stakes on it,
not read off the calibration of the first game.

**The `z2~`/`z3~` copula streams: proper elicitation, distorted metric.** Given
correct margins, folding two percentiles onto one axis and pricing the folded
law elicits the copula, and for the log score the folding is a fixed bijection
that changes nothing. The trouble is the *choice* of fold. The Morton z-curve
is not nearness-preserving: two joint outcomes that are close on the plane can
be far apart on the curve. A nearest-the-pin pool on the folded scalar therefore
prices a metric that the curve, not the problem, chose, and a contributor is
rewarded for placing mass near the truth *on the curve*. The elicitation is
valid; the settlement metric is an artifact. The principled high-dimensional
route scores random one-dimensional projections rather than one space-filling
projection [@cotton2026pointcloud], averaging a defensible metric instead of
fixing an arbitrary one.

**The stacked lottery: a calibration diagnostic, composed.** Composing the
monotone calibration maps of competing algorithms is the residual/PIT operator
applied repeatedly, and it inherits the same standing as `z1~`: proper as
elicitation of each stage's rank, informative as a calibration diagnostic, but
not by itself a strictly proper score for the conditional forecast, since the
PIT is blind to conditional sharpness.

**The verdict.** The chains were the right idea and mostly the right mechanism.
Where they settled on an exogenous outcome and scored a rank, they were proper
(the residual and copula elicitations). The base pool would have leaked an edge
of order the bandwidth, but an accidental jitter introduced for discrete
outcomes stood in for the settlement correction and kept it roughly honest.
Where they folded a joint onto a space-filling curve, the elicitation held but
the implied metric did not. A chain is as valid as its weakest anchor and its
settlement transform.

## 4. Open problems

1. *Conservation of edge.* Wealth conservation across a chain is automatic and
   is not the invariant; the edge a truthful participant holds over the price,
   $D_G(q,\pi)$, can grow or shrink through an interface stage. For the log
   score the edge is $\mathrm{KL}(q\Vert\pi)$ and the data-processing inequality
   caps it [@csiszar1967information]; no such inequality holds for a general
   Bregman edge — coarsening can *raise* the Brier edge. Open: characterize the
   generators and channels for which an interface cannot increase the edge.
2. *Residual markets.* A chain of residual markets is stagewise boosting with
   wealth as the learning rate (Proposition 2). Open: who funds each residual
   pot, whether a forecaster free to enter several stages withholds upstream to
   sell downstream, and whether the iteration converges to the conditional law.
   Split-conformal prediction is the one-participant degenerate case, its rent
   the discarded information $I(R;X)$ [@cotton2026conformalbetting].
3. *Copula settlement.* The `z2~`/`z3~` streams show that a copula elicits
   properly but its settlement metric is the fold's, not the problem's. Open:
   the right embedding for $d\ge2$, the space-filling curves as deployed against
   the random projections of the point-cloud paper [@cotton2026pointcloud], and
   the equilibrium when the same participants trade margin and copula stages.

## References

::: {#refs}
:::
