# An Algebra of Prediction-Rewarding Mechanisms

### Scoring rules, market makers, and pools as composable transducers

Peter Cotton · Microprediction · peter.cotton@microprediction.com · *July 2026*

---

## Abstract

Scoring rules, market makers, parimutuels, and opinion pools compose when
each stage is a stateful transducer over one message type, and the message
deployed contests actually collect is a finite cloud of samples. This note
provides sample-based elicitation inside that algebra: the companion paper's
jittered settlement makes cloud submission strictly proper; here the
composition operators act on clouds pointwise, and joint laws factor into
margin stages plus a copula stage settled on the rank vector. The single-stage theory underneath is
classical convex duality: proper scores from convex entropies, cost-function
market makers by Fenchel conjugacy, the two opinion pools as the two
Kullback-Leibler barycenters, merged market makers as the infimal
convolution of risk sharing. The benefit of composing is illustrated by the
simplest two-stage example: a residual market collects the conformal
predictor's information gap $I(R;X)$ as bankroll growth while its marginal
coverage stays exact.

**Keywords:** proper scoring rules; cost-function market makers; parimutuel
mechanisms; opinion pools; convex duality; mechanism composition; conformal
prediction markets

---

## 1. Introduction

Mechanisms for eliciting and aggregating forecasts are usually studied one at
a time. A proper scoring rule is analysed as a one-shot contract; a market
maker as a sequential trading venue; an opinion pool as an estimator; a
calibration test as a diagnostic. Deployed systems are thinner than the
theory allows: with few exceptions they run one level of competition against
one internal model. Numerai pools staked submissions into a single
stake-weighted meta-model and pays each forecast its marginal contribution
to it [@craib2017numeraire]; CrunchDAO blends each contest into one
ensemble; the IARPA prediction polls aggregated forecasts by track record,
comparing favourably with head-to-head markets [@atanasov2017distilling],
with reputation rather than wealth as the threaded state. In every case,
one pool and one internal aggregate.

Adjacent markets contain pieces of the structure, none of them the piece
this note needs. Derivatives chain point outputs: every futures contract
settles on another market's price, and in electricity the virtual bids and
transmission rights that settle on day-ahead prices pay forecasters for
correcting the day-ahead consensus [@jha2023financial]. These are
translations of a price, fixed transforms of a point output, not
composition of elicitation mechanisms. The racetrack tote prices margins
and joint finishing orders in parallel books, neither settling on the
other's output, consistency left to arbitrage [@harville1973assigning;
@hausch1981efficiency]. Reuse of a market's *probabilistic* output, a
distribution, percentile, or rank emitted by one stage becoming the message
or the settlement transform of the next, is scarce, and is the chain this
note studies. Its one deployed instance ran on the microprediction platform
[@cotton2022microprediction]: streams spawned z-streams of community
percentiles, and bivariate and trivariate dependence streams priced copulas,
so calibration and dependence were themselves the subject of further games;
the stacked-lottery design behind it was presented at MIT CSAIL in 2020
[@cotton2020lottery, slides 29-31], and the platform is retired. The nearest
live thing is monteprediction, a weekly self-funding pool over
million-scenario joint submissions in eleven dimensions with wealth threaded
across rounds since January 2024 [@cotton2024monteprediction]: the
Sequentialise operator of §4 in production, one stage repeated, not a
chain.

What a stage elicits moves under transformation of its message or its
outcome. Scoring a
kernel-smoothed submission at the raw outcome elicits the deconvolution of
the belief rather than the belief; jittering the settlement by the smoothing
kernel repairs it [@cotton2026pointcloud]. This is a property of a single
stage, not of the chain, and it is the third question below. The repair
makes a finite sample cloud a legitimate message type, and §6 runs the
algebra on the objects deployed contests actually collect.

Conformal prediction supplies a second motivation, from the opposite
direction. Split-conformal is itself a composition, a point predictor chained
into a rank-based calibration stage, but a degenerate one: the pool step is
skipped by fiat, all credit assigned to a single model in advance, and the
calibration stage prices the residual flat in the input. The wasted log
score of a single-shape conformal predictor, one that applies the same
residual law at every input, is the mutual information
$I(R;X)$ between residual and input, and
an entrant to a parimutuel residual pool who conditions on the input collects
it as bankroll growth at exactly that rate (Theorem 9).
Running the residual stage as an actual pool, rather than assuming its winner,
is what the operators below are for; Proposition 4 locates the gap in the
theory of the probability integral transform, and §5 prices it.

This note organizes the catalogue around three questions that are often
blurred together:

1. *Message closure.* Can every stage consume and emit one common object, so
   that stages plug together at all?
2. *Convex generation.* Is each individual mechanism generated by a convex
   potential, so that one dictionary covers scoring rules, market makers, and
   their duals?
3. *Propriety under transformation.* When a stage transforms forecasts or
   outcomes, does strict propriety survive?

The first is an engineering convention (§2, §8). The second and third admit
theorems, which are stated with complete proofs in §3 and §4; that
mathematics is classical, collected in one place and one convention. The
contributions are the parts built on it: sample-based elicitation, running
the algebra on clouds rather than exact densities (§6); the factoring of
multivariate elicitation into margin stages and a rank-settled copula stage
(§7); and the priced conformal example (§5), where the residual market's
bankroll growth is the conformal predictor's information gap. Play is
stagewise throughout: no participant deviates across stages (§4).
Cross-stage strategy is outside the note's scope.

## 2. Preliminaries

**The finite setting.** Section 3 works with a finite outcome set
$\{1,\dots,n\}$ and reports $p$ in the probability simplex $\Delta$; from
the aggregation operators of §4 onward the paper moves to the continuous
setting, densities and distribution functions on $\mathbb R$ or $[0,1]^d$.
The finite statements transfer with the usual measure-theoretic care; the
one place where the transfer is not routine is the probability integral
transform, treated separately in Proposition 4.

**Scores.** A scoring rule assigns $S(p,i)\in\mathbb R\cup\{-\infty\}$ to a
report $p$ and outcome $i$; the expected score of reporting $p$ under belief
$q$ is $S(p;q)=\sum_i q_i S(p,i)$, assumed well defined with $S(q;q)$ finite
(this is the regularity used throughout). The rule is *proper* if
$S(q;q)\ge S(p;q)$ for all $p,q$ and *strictly proper* if equality forces
$p=q$. Scores are written in reward form: higher is better.

**Convex tools.** For convex $G$ with subgradient selection
$G'(p)\in\partial G(p)$, the Bregman divergence is
$D_G(q,p)=G(q)-G(p)-\langle G'(p),q-p\rangle$. The Legendre-Fenchel conjugate
of a function $R$ on $\Delta$ is
$R^*(\mathbf q)=\sup_{p\in\Delta}\langle p,\mathbf q\rangle-R(p)$. The infimal
convolution of $f$ and $g$ is
$(f\,\square\,g)(x)=\inf_y f(y)+g(x-y)$. "Closed proper convex" is used in the
standard sense [@rockafellar1970convex]. Differentiable statements are on the
relative interior of $\Delta$, with extended-real values allowed on the
boundary; subgradients on $\Delta$ act on its tangent space and are defined
up to addition of multiples of $\mathbf 1$.

## 3. One potential, three mechanisms

**Theorem 1 (characterisation of proper scoring rules;
@savage1971elicitation, @mccarthy1956measures).** *A regular scoring rule $S$
is proper iff there is a convex $G:\Delta\to\mathbb R$ with*

$$S(p,i) \;=\; G(p) + \langle\, G'(p),\; e_i - p\,\rangle,
\qquad G'(p)\in\partial G(p),$$

*where $e_i$ is the $i$-th unit vector. It is strictly proper iff $G$ is
strictly convex relative to $\Delta$, and then $G(p)=S(p;p)$ is the expected
score of a truthful forecaster.*

**Proof.** ($\Leftarrow$) Since $\sum_i q_i(e_i-p)=q-p$, the expected score is
affine in the belief: $S(p;q)=G(p)+\langle G'(p),q-p\rangle$. Hence

$$S(q;q)-S(p;q) \;=\; G(q)-G(p)-\langle G'(p),\,q-p\rangle \;=\; D_G(q,p),$$

which is non-negative by the supporting-hyperplane inequality, and zero only
at $q=p$ when $G$ is strictly convex relative to $\Delta$; so $S$ is
(strictly) proper.

($\Rightarrow$) Define $G(q):=S(q;q)$. Properness says
$G(q)=\sup_p S(p;q)$, and each $q\mapsto S(p;q)=\langle S(p,\cdot),q\rangle$
is affine, so $G$ is a pointwise supremum of affine functions, hence convex.
For any fixed $p$,

$$S(p;q) \;=\; G(p) + \langle S(p,\cdot),\, q-p\rangle
\quad\text{with}\quad S(p;q)\le G(q)\ \ \forall q,$$

so the vector $S(p,\cdot)$ is a subgradient of $G$ at $p$ (acting on the
tangent space of $\Delta$, where subgradients are defined up to a constant
shift along $\mathbf 1$, which the representation absorbs). Substituting
$G'(p)=S(p,\cdot)$ into the display returns $S(p,i)$ identically. If $S$ is
strictly proper the supremum has a unique maximiser at every $q$, which for
a supremum of affine functions is equivalent to strict convexity relative to
$\Delta$. $\blacksquare$

The content of Theorem 1 is the dictionary *proper scoring rule
$\leftrightarrow$ convex function $\leftrightarrow$ Bregman divergence*
[@gneiting2007strictly; @banerjee2005optimality]. The classics are three
choices of $G$, with the divergences computed directly from the definition:

| Score | Generator $G(p)=S(p;p)$ | Bregman divergence $D_G(q,p)$ |
|--------------------|--------------------------|-------------------------------|
| logarithmic | $\sum_i p_i\log p_i$ | $\mathrm{KL}(q\Vert p)$ |
| Brier (quadratic) | $\lVert p\rVert_2^2$ | $\lVert q-p\rVert_2^2$ |
| spherical | $\lVert p\rVert_2$ | $\lVert q\rVert_2\,(1-\cos\theta_{p,q})$ |

For the spherical row: $\nabla G(p)=p/\lVert p\rVert$, so
$D_G(q,p)=\lVert q\rVert-\langle p,q\rangle/\lVert p\rVert
=\lVert q\rVert(1-\cos\theta_{p,q})$, an angular term scaled by
$\lVert q\rVert$. For the logarithmic row the representation gives
$S(p,i)=\log p_i$ on the relative interior, with $-\infty$ on the boundary.

**Theorem 2 (scoring rule to market maker; @hanson2007logarithmic,
@abernethy2013efficient).** *Let $R$ be closed, proper, and strictly convex on
$\Delta$, extended by $+\infty$ off $\Delta$ so that $C=R^*$ below is the
Fenchel conjugate on $\mathbb R^n$ (in the sequel, $R=G$, the generator of
Theorem 1 read as a regulariser), and define*

$$C(\mathbf q)=\sup_{p\in\Delta}\big(\langle p,\mathbf q\rangle - R(p)\big).$$

*Then:*

*(i) $C$ is convex and finite, the maximiser $p^\ast(\mathbf q)$ is unique,
and $\nabla C(\mathbf q)=p^\ast(\mathbf q)\in\Delta$: prices are a probability
vector. (Without strict convexity, prices live in $\partial C(\mathbf q)$.)*

*(ii) $C$ is translation-equivariant, $C(\mathbf q+\alpha\mathbf 1)
=C(\mathbf q)+\alpha$; costs telescope over any trade path, so round trips
cost zero and buying the full bundle costs its payout: the market is
arbitrage-free.*

*(iii) With initial state $\mathbf q_0=\mathbf 0$, the maker's loss when the
market settles on outcome $i$ after net sales $\mathbf q$ is*

$$\mathrm{loss}_i(\mathbf q) = q_i - C(\mathbf q) + C(\mathbf 0)
\;\le\; R(e_i) - \inf_{p\in\Delta} R(p)
\;\le\; \sup_{p\in\Delta} R(p) - \inf_{p\in\Delta} R(p).$$

*(iv) Taking $R(p)=b\sum_i p_i\log p_i$ gives
$C(\mathbf q)=b\log\sum_i e^{q_i/b}$, Hanson's LMSR, with worst-case loss
$b\log n$.*

**Proof.** (i) $C$ is a supremum of affine functions of $\mathbf q$, hence
convex; the supremum of a continuous function over the compact $\Delta$ is
attained, and strict concavity of
$p\mapsto\langle p,\mathbf q\rangle-R(p)$ makes the maximiser unique. Danskin's
theorem gives $\nabla C(\mathbf q)=p^\ast(\mathbf q)$.
(ii) $\langle p,\mathbf q+\alpha\mathbf 1\rangle
=\langle p,\mathbf q\rangle+\alpha$ for $p\in\Delta$, so the supremum shifts
by $\alpha$. A trader moving the state $\mathbf q\to\mathbf q'$ pays
$C(\mathbf q')-C(\mathbf q)$ by definition, so costs over any path telescope;
a round trip costs zero, and by translation equivariance the bundle
$\alpha\mathbf 1$ costs exactly $\alpha$, its sure payout.
(iii) The maker collects $C(\mathbf q)-C(\mathbf 0)$ and pays $q_i$, so
$\mathrm{loss}_i=q_i-C(\mathbf q)+C(\mathbf 0)$. By Fenchel-Moreau,
$\sup_{\mathbf q}\big(\langle e_i,\mathbf q\rangle - C(\mathbf q)\big)
=R^{**}(e_i)=R(e_i)$, and $C(\mathbf 0)=\sup_p -R(p)=-\inf_p R(p)$: by
Fenchel-Moreau the first bound is the exact supremum of the loss over
$\mathbf q$. The final inequality holds because $e_i\in\Delta$.
(iv) Lagrange: maximising $\langle p,\mathbf q\rangle-b\sum p_i\log p_i$
subject to $\sum p_i=1$ gives $q_i-b(\log p_i+1)=\lambda$, so
$p_i\propto e^{q_i/b}$, and substituting back yields
$C(\mathbf q)=b\log\sum_i e^{q_i/b}$. Then $R(e_i)=0$ and
$\inf R=-b\log n$ at the uniform distribution, so the loss bound is
$b\log n$. $\blacksquare$

**Proposition 3 (cost-function markets and CFMMs).** *Under the monotonicity,
concavity, and reserve-domain hypotheses of @frongillo2024axiomatic,
cost-function prediction markets and constant-function market makers can be
converted into one another. In the unconstrained-reserve sign convention a
cost function $C$ gives a concave CFMM potential by
$\varphi(\mathbf r)=-C(-\mathbf r)$; bounded-reserve versions require a
perspective (level-set) construction. The equivalence is a convex level-set
duality; it is not the bare Fenchel conjugacy $C\mapsto C^*$.*

We do not reprove the general equivalence here; @frongillo2024axiomatic give
it in full, and @angeris2023primer and @angeris2021uniswap develop the CFMM
side.

**Example (constant product).** Take two outcome tokens with reserves
$r_1,r_2$ and invariant $r_1r_2=k$. The pool's portfolio value at prices
$(p,1-p)$ is

$$V(p)=\inf\{p\,r_1+(1-p)\,r_2:\ r_1r_2= k\}
      =2\sqrt{k\,p(1-p)},$$

by the AM-GM inequality, with the infimum attained at
$r_1=\sqrt{k(1-p)/p}$, $r_2=\sqrt{kp/(1-p)}$. $V$ is concave; reading $R(p)=-V(p)$ as the
regulariser of Theorem 2 gives a maker whose worst-case loss is the range of
$R$ on $[0,1]$, namely $\sqrt k$ (between $p=\tfrac12$ and the endpoints):
the geometric mean of the initial reserves. The corresponding generator is
not entropic, which is the convex-analytic content of the observation that
constant-product markets and the LMSR price bounded-payout claims
differently.

**Proposition 4 (the probability integral transform;
@rosenblatt1952remarks, @dawid1984prequential).** *If $X$ has continuous CDF
$F$ then $U=F(X)\sim\mathrm{Uniform}(0,1)$, and $z=\Phi^{-1}(U)\sim N(0,1)$.
In the prequential setting, if $F_t$ is the true conditional law of $X_t$
given the past, then $U_t=F_t(X_t)$ is conditionally uniform and the PIT
stream is iid uniform.*

**Proof.** For continuous $F$, with the generalized inverse
$F^{-}(u)=\inf\{x:F(x)\ge u\}$, the event $\{F(X)\le u\}$ differs from
$\{X\le F^{-}(u)\}$ by a set of probability zero, and
$\Pr(X\le F^{-}(u))=F(F^{-}(u))=u$ by continuity. The prequential statement
applies this conditionally at each $t$. $\blacksquare$

Two scope remarks. First, the converse fails in the strong
sense: marginally uniform PITs do not certify an informative forecast.
Reporting the unconditional law $F$ of an iid sequence gives
$F(X_t)\sim\mathrm{Uniform}(0,1)$ even if valuable covariates were ignored. A
PIT critic therefore witnesses miscalibration relative to its test class; it
complements, and does not replace, a proper score that rewards sharp
conditional distributions. Conformal prediction lives on exactly this
distinction: a split-conformal predictor achieves marginal coverage by
construction, the uniform-PIT guarantee, while free to ignore the
conditional information in the input; §5 prices the gap (Theorem 9).
Second, for discrete forecasts the randomized PIT
preserves the exact uniform null, while the mid-PIT is a convenient
deterministic diagnostic with a different null distribution.

## 4. Operators on mechanisms

**The common signature.** A *transducer* (Mealy machine) is a tuple
$(S,A,B,\delta,\lambda)$ with state space $S$, input alphabet $A$, output
alphabet $B$, transition map $\delta:S\times A\to S$, and output map
$\lambda:S\times A\to B$; run on an input stream it produces
$s_{t+1}=\delta(s_t,a_t)$ and $b_t=\lambda(s_t,a_t)$, a causal map of input
streams to output streams. The mechanisms of §3 share one instantiation. Let
$\mathrm{Dist}$ denote the set of distributional beliefs and take
$S$ the wealth states, $A=\mathrm{Dist}^m\times\mathcal X$ ($m$ participant
reports and, where the stage settles, a realized outcome), and
$B=\mathrm{Dist}\times\mathbb R^m$ (an aggregate belief and transfers). A
*stage* is such a transducer, written

$$M:\ (\mathrm{Dist}^m,\ w,\ x)\ \longmapsto\ (\mathrm{Dist},\ w',\ \pi).$$

A scoring rule is the transfer component $\lambda$ of a stage, not itself a
map $\mathrm{Dist}\to\mathrm{Dist}$; a market maker is a stage whose state
is the inventory vector and whose emitted belief is the price; an opinion
pool is a stage with no outcome argument and zero transfers. Composition
wires the belief output of one stage to the belief inputs of the next while
state and transfers thread through.

The operators below act on stages.

**Sequentialise.** Theorem 2: a proper score run sequentially against a
wealth state is a cost-function market maker.

**Pool.** A proper score gives a batch elicitation mechanism when reports are
scored independently and funded externally: properness is inherited report by
report. Parimutuel and budget-balanced versions are a different game, because
the pot split couples payoffs through the denominator; the price-taking
analysis [@cotton2026pointcloud, §2] gives truthful all-in, a symmetric
equilibrium at fractional stakes, and degeneracy as the stake fraction
vanishes, and beyond that the equilibrium theory is open.

**Ensemble (Proposition 5: the two pools are the two KL barycenters).** *Let
$q_1,\dots,q_m$ be densities with respect to a common dominating measure,
$w_i\ge0$, $\sum w_i=1$, and for the second display assume
$0<\int\prod_i q_i^{w_i}<\infty$. Then the linear pool
minimizes the forward divergences and the logarithmic pool the reverse:*

$$\textstyle\arg\min_p \sum_i w_i\,\mathrm{KL}(q_i\Vert p)=\sum_i w_i q_i,
\qquad
\arg\min_p \sum_i w_i\,\mathrm{KL}(p\Vert q_i)\ \propto\ \prod_i q_i^{w_i}.$$

**Proof.** For the first, $\sum_i w_i\,\mathrm{KL}(q_i\Vert p)
=\mathrm{const}-\int(\sum_i w_iq_i)\log p$, and $\int \bar q\log p$ is
maximized over densities at $p=\bar q$ (Gibbs). For the second,
$\sum_i w_i\,\mathrm{KL}(p\Vert q_i)=\int p\log p-\int p\,\overline{\log q}$
with $\overline{\log q}=\sum_i w_i\log q_i$; the Lagrange condition is
$\log p=\overline{\log q}+\mathrm{const}$. $\blacksquare$

Training weights by cumulative log score
and then mixing linearly, $\sum_m w_m F_m$, is Bayesian model averaging, a
linear pool with score-trained weights; the logarithmic pool multiplies
densities and renormalizes. They are different aggregates with different
sharpness [@genest1986combining].

**Merge (Proposition 6: merging makers is infimal convolution;
@rockafellar1970convex, @bhaskara2023general).** *For closed
proper convex $f,g$: $(f\,\square\,g)^*=f^*+g^*$. Consequently, merging two
cost-function makers with regularisers $R_1,R_2$ (cost functions
$C_i=R_i^*$) yields the maker with regulariser $R_1+R_2$, and merging
$\mathrm{LMSR}_{b_1}$ with $\mathrm{LMSR}_{b_2}$ yields
$\mathrm{LMSR}_{b_1+b_2}$: liquidity adds.*

**Proof.** $(f\,\square\,g)^*(p)
=\sup_x\langle p,x\rangle-\inf_y\{f(y)+g(x-y)\}
=\sup_{y,z}\langle p,y\rangle-f(y)+\langle p,z\rangle-g(z)
=f^*(p)+g^*(p)$. For the makers, $C_1\,\square\,C_2=(R_1+R_2)^*$ by
biconjugacy, and $b_1G+b_2G=(b_1+b_2)G$ for the entropic generator.
$\blacksquare$

This is also the risk-sharing literature's composition law: the aggregate of
several agents' convex risk measures is their infimal convolution, with the
Pareto allocation as minimiser and the common subgradient as the clearing
price [@barrieu2005inf; @jouini2008optimal]; the market-liquidity reading is
developed by @bhaskara2023general.

**Conjugate.** Run a stage in a transformed coordinate and map back. This is
where propriety needs care, and the theory splits in two:

- *Fixed transformations.* Properness is preserved under any fixed
  transformation of forecast and outcome, and strict propriety iff the
  transformation is injective [@allen2023transformed, Prop. 4;
  @pic2025proper, Prop. 1]. The Markov-kernel (stochastic channel) extension,
  strict propriety iff the channel is injective on laws, is Theorem 2 of the
  companion point-cloud paper, whose Theorem 1 also exhibits the canonical
  failure: a KDE smoothing seam whose raw-outcome score elicits a
  deconvolution.
- *Forecast-dependent transformations.* The PIT critic transforms outcomes by
  the reported $F$ itself, so the fixed-transformation theorems do not apply
  to it; Proposition 4 and its scope remarks are the correct warrant, and the
  critic is a calibration diagnostic rather than a strictly proper score. The
  stacked-lottery design [@cotton2020lottery, slides 29-31] is this operator
  in practice: percentiles from one game feed the next, and calibration is
  produced by composing monotone maps contributed by competing algorithms.

**Residual.** Let a stage emit the aggregate $F_1$ for an outcome $Y$, and
let a second market elicit a distribution for the residual $U=F_1(Y)$,
settling at the realised $u=F_1(y)$. If $F_1$ is the true conditional law
then $U$ is uniform (Proposition 4) and the second market has nothing to
price; whatever structure remains in the residual is the second stage's
edge. The corrected forecast composes the two reports.

**Proposition 7 (the correction is a multiplicative reweighting).** *Let
$F_1$ be strictly increasing onto $(0,1)$ with density $p_1>0$ and let the
residual market's consensus be a distribution $H$ on $[0,1]$ with $H(0)=0$,
$H(1)=1$ and density $g$. The
composed forecast $F=H\circ F_1$ has density*

$$p(y) \;=\; p_1(y)\, g\!\big(F_1(y)\big),$$

*so $\log p(y)=\log p_1(y)+\log g(u)$ with $u=F_1(y)$: the chain's log score
is the sum of stage log scores, and the residual stage is paid by a proper
score on $u$ alone.*

**Proof.** Chain rule: $F'(y)=g(F_1(y))\,p_1(y)$; take logarithms. The
residual score $\log g(u)$ is the logarithmic score of Theorem 1 applied to
the report $g$ and outcome $u$, hence strictly proper for the law of $U$.
$\blacksquare$

Multiplying the density by a ratio fitted to what the current model gets
wrong is the functional-gradient step of boosting under log loss
[@mason1999boosting; @friedman2001greedy], so a chain of residual markets is
stagewise boosting with wealth as the learning rate. What Proposition 7 does
not settle is the game across stages: who funds the residual pot, and whether
a forecaster free to enter both stages prefers to withhold information from
the first and sell it to the second (§9).

**Spec.** Serialise a pipeline to data and search over it; the mechanism
analogue is a market over pipelines. Also open.

**Stagewise play.** Call a profile of reports a *stagewise equilibrium* if
no participant gains by a deviation confined to a single stage, all other
stages' reports held fixed *and the inputs and settlement transforms of
every other stage clamped at their pre-deviation values*. This is the
pipeline version of the myopic-trader assumption standard in the
market-scoring-rule literature [@hanson2007logarithmic;
@chen2010newunderstanding]. The clamp is not cosmetic: a stage's output is
wired into later settlement transforms (the residual point $u=F_1(y)$, the
rank vector of §7), so an upstream deviation moves downstream payoffs even
when every downstream report is held fixed.

**Proposition 8 (single-stage guarantees compose under clamped stagewise
play).** *Suppose each stage of a pipeline, taken in isolation with its
inputs and settlement transform fixed, makes the truthful report a best
response: strict propriety for externally funded scoring stages (Theorem 1),
the sequential scoring of Theorem 2 for market stages, the price-taking
pot-split analysis of the companion paper for parimutuel stages, and the
residual score of Proposition 7 for correction stages. Then truthful
reporting at every stage is a stagewise equilibrium of the pipeline.
If moreover no participant reports to a stage upstream of one in which they
hold a position (disjoint stage membership suffices), the clamp is vacuous
for every feasible deviation, and truthful reporting survives unrestricted
single-stage deviations.*

**Proof.** With the other stages' inputs and transforms clamped, a deviation
confined to stage $k$ changes the deviator's payoff only through stage $k$'s
transfer map, and the stage-$k$ hypothesis makes the truthful report a best
response. For the second claim: a stage-$k$ deviation moves stage $k$'s
transfer and, through stage $k$'s output, transfers strictly downstream of
$k$; a deviator with no downstream position collects none of the latter, so
the propagation is payoff-irrelevant to them. $\blacksquare$

Proposition 8 is not a Nash equilibrium of the composed game. When the same
participant reports upstream and holds exposure downstream, the deviation
propagates through the settlement transform, and a downstream stake is a
derivative written on an upstream settlement, with the attendant incentives
to distort the underlying [@kumar1992futures; @jarrow1994derivative;
@hanson2009manipulator; @ostrovsky2012information]. Disjoint membership,
zero downstream exposure for upstream reporters, or exogenous freezing of
the settlement transforms restore the proposition; the dynamic game without
them is outside the note's scope.

## 5. Betting against a conformal predictor

The introduction claimed that split-conformal prediction is a degenerate
composition and that running the residual stage as a market collects what
the degeneracy wastes. This section proves it. The analysis is
population-level throughout: laws are continuous, the ranking uses the true
marginal CDF, and the finite-sample, exchangeability-based conformal
construction enters only through the measurement caveat at the end of the
section. A point predictor leaves a
residual $R$; conformalization re-levels its marginal law. Write $U=F_R(R)$
for the PIT of the residual, $F_R$ the marginal CDF of $R$, so $U$ is
marginally uniform whatever the model (Proposition 4), and let
$g(u\mid x)$ be the conditional density of $U$ given $X=x$. Since
$R\mapsto F_R(R)$ is almost surely invertible, $I(U;X)=I(R;X)$, and because
the marginal of $U$ is uniform,

$$I(R;X)\;=\;\mathbb E_X\,\mathrm{KL}\big(P_{R\mid X}\,\Vert\,P_R\big)
\;=\;\mathbb E_X\!\int_0^1 g(u\mid X)\log g(u\mid X)\,\mathrm du.$$

The residual stage is the parimutuel of the companion paper run on the rank
scale. Two facts about that pool are needed.

**Lemma 1 (pool payoff).** *Let the crowd's aggregate stake have density $q$
on the outcome space, normalised to one, and let a price-taking entrant
stake $\epsilon\,b(u)\,\mathrm du$ alongside it. As $\epsilon\to0$ the
entrant's gross payoff per unit of their own wealth, after outcome $u$, is
$W(u)=b(u)/q(u)$.*

**Proof.** Bin the outcome axis at width $\delta$. The crowd stakes
$q(u)\delta$ on the bin at $u$, the entrant $\epsilon\,b(u)\delta$.
If the outcome lands there the pool is split in proportion to stake, so the
per-unit payoff is $1/(q(u)\delta)$ up to $O(\epsilon)$, and the entrant
collects $\epsilon\,b(u)\delta\cdot 1/(q(u)\delta)=\epsilon\,b(u)/q(u)$: per
unit of entrant wealth, $b(u)/q(u)$. The bin width cancels and the limit is
the Radon-Nikodym derivative $\mathrm db/\mathrm dq$. $\blacksquare$

The cancellation is what makes a lottery on a point outcome well posed: the
pool divides two vanishing quantities and the ratio survives, with no
reference measure and no posted odds.

**Lemma 2 (log-optimal growth).** *If the outcome is drawn from $p$, the
expected log-growth of an entrant staking $b$ against a crowd staking $q$
is*

$$\mathbb E_{U\sim p}\Big[\log\frac{b(U)}{q(U)}\Big]
=\mathrm{KL}(p\Vert q)-\mathrm{KL}(p\Vert b)\;\le\;\mathrm{KL}(p\Vert q),$$

*with equality iff $b=p$: the log-optimal stake is the truth.*

**Proof.** Add and subtract $\log p$ inside the expectation; Gibbs'
inequality kills the second term exactly at $b=p$. $\blacksquare$

The lemmas are the parimutuel form of the Kelly-Breiman log-optimality
principle [@kelly1956newinterpretation; @cover2006; @barroncover1988]; when
the crowd prices
the marginal, the growth of belief $b$ is
$I(R;X)-\mathbb E_X\,\mathrm{KL}(g(\cdot\mid X)\Vert b(\cdot\mid X))$, a
decomposition written by @kemp2022bayesian for population growth in
stochastic environments.

**Theorem 9 (parimutuel rent of a conformal predictor).** *Put the pool on
the rank scale. The single-shape conformal predictor is the crowd that
prices it flat, $q\equiv 1$ on $[0,1]$. Against it:*

*(i) an entrant who knows only the marginal stakes $b\equiv1$ and grows at
rate zero: marginal coverage stated as wealth;*

*(ii) an entrant who observes $X$ and stakes $b=g(\cdot\mid X)$ grows per
round at rate $\mathbb E_X\,\mathrm{KL}(g(\cdot\mid X)\Vert 1)=I(R;X)$;*

*(iii) with a track take $\tau\in[0,1)$ the informed rate is
$I(R;X)+\log(1-\tau)$, positive iff $I(R;X)>-\log(1-\tau)$.*

**Proof.** Lemma 1 with $q\equiv1$ makes the payoff $b(U)$; Lemma 2 makes
the growth $\mathrm{KL}(p\Vert 1)-\mathrm{KL}(p\Vert b)$ for true
conditional law $p=g(\cdot\mid x)$. For (i), $b=1$ and the marginal of $U$
is uniform, so the rate is zero. For (ii), conditioning on $X=x$ the optimal
stake is $g(\cdot\mid x)$ with conditional rate
$\mathrm{KL}(g(\cdot\mid x)\Vert 1)=\int g\log g$; averaging over $X$ gives
$I(R;X)$ by the display above. For (iii) every payoff is multiplied by
$1-\tau$, adding $\log(1-\tau)$ to the rate. $\blacksquare$

The certificate and the leak are two readings of one pool. Marginal
coverage is part (i), a break-even statement; part (ii) is what the
certificate cannot see, and re-leveling cannot close it because re-leveling
acts on the marginal, where the leak is invisible. Conformal re-leveling
takes no rake, so any conditional information at all is pure profit.

**Example (Gaussian residual).** Let $(R,X)$ be jointly Gaussian with
correlation $\rho$, marginally $R\sim N(0,\sigma_R^2)$,
$X\sim N(0,\sigma_X^2)$: the canonical case, left by a linear predictor
whose residual retains correlation with the input.
The informed entrant's report is
$R\mid X=x\sim N(\rho(\sigma_R/\sigma_X)x,\,(1-\rho^2)\sigma_R^2)$, and with
$m(x)=\rho(\sigma_R/\sigma_X)x$, $s^2=(1-\rho^2)\sigma_R^2$,

$$\mathbb E_X\,\mathrm{KL}\big(N(m,s^2)\,\Vert\,N(0,\sigma_R^2)\big)
=\mathbb E_X\Big[\log\frac{\sigma_R}{s}+\frac{s^2+m^2}{2\sigma_R^2}
-\frac12\Big]=-\tfrac12\log(1-\rho^2)=I(R;X).$$

At $\rho=\tfrac12$ the rent is $0.144$ nats per observation and the informed
bankroll doubles every five observations; at $\rho=0.3$, every fifteen.
Throughout, the conformal band's marginal coverage remains exact. The
composed forecast of Proposition 7 built from the entrant's rank report
$g(u\mid x)=p_{R\mid X}(F_R^{-}(u))/p_R(F_R^{-}(u))$ recovers the
conditional law exactly: $p_R(r)\,g(F_R(r)\mid x)=p_{R\mid X}(r)$.

**The mechanism has run in production.** The microprediction platform's
nearest-the-pin pool [@cotton2022microprediction] paid each entry its sample
density at the realised value relative to the field's, the payoff of Lemma 1
with densities estimated from submitted samples; the MidOne contest
[@crunchdao_midone] priced an explicit density [@cotton_density] the same
way, and its priced object was a residual stream. A conformal predictor
entering such a pool is the participant that prices flat in $X$, and Theorem
9 is the bankroll of its better-informed competitor.

**Measuring the rent.** The rent is defined through a log score, so it is
estimated rather than read off. Two routes:

**Proposition 10 (certified lower bound).** *Let $\mathrm{HSIC}(U,X)$ be
computed with a product kernel satisfying $\sup_z k(z,z)\le K$. Then
$I(R;X)\ \ge\ \mathrm{HSIC}(U,X)/(2K)$.*

**Proof.** Write $\mathrm{TV}=\sup_A|P(A)-Q(A)|$ for the total variation
between the joint law and the product of marginals.
$\mathrm{MMD}=\mathrm{HSIC}^{1/2}$ is the integral probability
metric over the unit ball of the tensor RKHS, whose witnesses satisfy
$\lVert f\rVert_\infty\le\sqrt K$, so $\mathrm{MMD}\le 2\sqrt K\,\mathrm{TV}$.
Pinsker's inequality in the same convention [@cover2006] gives
$I(U;X)=\mathrm{KL}\ge 2\,\mathrm{TV}^2$, and $I(U;X)=I(R;X)$. Combining,
$I(R;X)\ge 2(\mathrm{MMD}/2\sqrt K)^2=\mathrm{HSIC}/(2K)$. $\blacksquare$

Distance covariance [@szekely2007measuring] is a fixed multiple of an HSIC
[@sejdinovic2013equivalence], so a measured dependence between rank and
input certifies
a minimum extractable rent. No matching upper bound exists with a fixed
kernel: MMD metrizes weak convergence [@simongabriel2023metrizing] and KL
does not.

**Proposition 11 (anytime-valid test and consistent estimate).** *At round
$t$ choose a betting density $b_t(\cdot\mid x)\ge0$ with
$\int_0^1 b_t(u\mid x)\,\mathrm du=1$ from the history, and set
$W_t=\prod_{s\le t}b_s(U_s\mid X_s)$. (i) Under the null of conditional rank
uniformity, $U_t\mid X_t,\mathcal F_{t-1}\sim\mathrm{Uniform}(0,1)$, the
process $(W_t)$ is a
non-negative martingale with mean one, so
$\Pr(\sup_t W_t\ge1/\alpha)\le\alpha$ [@ville1939]: rejecting when
$W_t\ge1/\alpha$ is an anytime-valid level-$\alpha$ test of conditional rank
uniformity, that is, of conditional dependence of the residual rank on the
input. (ii) If the rounds are iid and $\log b_t\to\log g$ uniformly
with a bounded envelope, then $t^{-1}\log W_t\to I(R;X)$ almost surely.*

**Proof.** (i) Under the null, each factor has conditional mean
$\int_0^1 b_t=1$; Ville's
inequality applies to the resulting martingale. (ii) Write
$\log b_s=\log g+(\log b_s-\log g)$: the second terms vanish in Cesàro mean
by uniform convergence, and the average of the first tends almost surely to
$\mathbb E[\log g(U\mid X)]=I(R;X)$ by the law of large numbers and the
display opening this section. $\blacksquare$

The wealth process is simultaneously the test and the estimator; it is the
log-optimal e-variable of safe testing [@gruenwald2024safe;
@vovk2021evalues] for the null $U\perp X$, with conformal test martingales
as ancestor [@vovk2003testing]. Sequential tests of forecast calibration
[@arnold2023sequentially] and conditional independence by betting
[@shaer2023modelx] use the same mechanics; the identification of the growth
rate with the conformal predictor's information gap, and with the realised
bankroll in a deployed pool, is the reading here. One measurement caveat:
$U=F_R(R)$ uses the true marginal, and with empirical ranks the null is
exact in the full-conformal, leave-one-out sense. And one scope remark: by the
distribution-free no-go results [@lei2014distribution; @barber2021limits;
@vovk2012conditional] no procedure certifies conditional coverage from data
alone, so a quiet e-process means no rent found at this power, never
conditional validity.

## 6. Samples as messages

The message type has so far been a distribution given exactly. Deployed
contests collect something rougher: a finite cloud of samples, smoothed by
the contest into a density before settlement. The companion paper gives the
sample-based elicitation result the algebra needs
[@cotton2026pointcloud]:

**Proposition 12 (sample-based elicitation; @cotton2026pointcloud, Thms
1-2).** *Score the bandwidth-$h$ kernel density estimate of a submitted
cloud by the logarithmic score. Settled at the raw outcome, the optimal
cloud is drawn from a deconvolution of the belief when one exists (for
Gaussian beliefs and kernels with belief variance exceeding $h^2$, the
belief with $h^2$ removed from the variance). Settled at the
outcome jittered by the same kernel, truthful sampling is optimal, and
strictly so whenever the kernel's characteristic function is nonvanishing on
a dense set: the smoothing channel is injective on laws.*

The proof is in the companion paper. With jittered settlement the cloud
*is* a valid message, and the operators of §4 act on clouds directly.

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

## 7. Margins and copulas

A multivariate settlement can be factored. By Sklar's theorem [@sklar1959]
a joint law with continuous margins $F_1,\dots,F_d$ and copula $C$ has
density

$$p(x)\;=\;\prod_{i=1}^d f_i(x_i)\;\cdot\;
c\big(F_1(x_1),\dots,F_d(x_d)\big),$$

with $c$ the copula density on $[0,1]^d$.

**Proposition 13 (the log score factors).** *For a joint report assembled
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
are exactly the stage scores. Strict propriety of the rank stage is Theorem
1 applied to densities on $[0,1]^d$, the transform $x\mapsto u$ being fixed
once the margin reports are given. The final clause is Sklar's theorem
applied to the true joint law. $\blacksquare$

This is the two-stage estimation logic of inference functions for margins
[@joe1997multivariate], and the factorization underlies out-of-sample
copula comparison [@diks2010copula]; here it is read as a market design.
The rank stage is the multivariate Residual operator: under correct
margins the rank vector has uniform margins and its joint law is the
dependence structure alone, so a rank-settled market elicits dependence
separately from level. When the margins are correct the settled object is
invariant to monotone transformations of the coordinates; the settlement
transform itself moves with the reported margin maps. The microprediction platform ran exactly this factoring: alongside
univariate z-streams it operated bivariate and trivariate streams in which
community-implied percentiles were embedded by a space-filling curve into
one settled scalar [@cotton2022microprediction], a copula market in
production. Sample messages compose with the factoring: rank each
coordinate of a submitted cloud through the margin CDFs and the result is a
cloud on $[0,1]^d$ for the copula stage, with the §6 seam discipline
applied at that stage's settlement.

## 8. Implementation notes

The mechanisms of the catalogue are implemented against a single interface
patterned on the skaters time-series contract [@cotton_skaters], a successor
to timemachines [@cotton_timemachines]: every stage consumes and emits the
same distributional type and threads its own state, so the signature of §4
is the forecasting contract with wealth in place of model state. The closure
of the message type is what makes a small operator set sufficient; the
transforms, ensembles, and residual constructions of §4 all consume and emit
the one type. Scores are implemented in loss form; the text uses reward
form.

## 9. Open problems

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
2. *Residual markets.* Proposition 7 gives the single-step correction; a
   chain of residual markets is stagewise boosting with wealth as the
   learning rate. Open: the microstructure (who funds each residual pot, and
   when it settles relative to the stage before), whether a forecaster free
   to enter several stages prefers withholding information upstream to sell
   it downstream, and whether the iterated correction converges to the true
   conditional law as boosting does. Split-conformal prediction is the
   one-participant degenerate case, with its rent $I(R;X)$ as the value left
   on the table (Theorem 9). The closest live analogue is in point prices:
   a virtual bid in a two-settlement electricity market is written on the
   day-ahead stage's pricing error, and its convergence bidders are the
   residual stage's informed entrants [@jha2023financial]. The
   distributional version is the gap.
3. *Copula markets.* A rank-settled stage elicits dependence separately
   from margins (Proposition 13), and under correct margins its settled
   object is invariant to monotone transformations of the coordinates.
   The racetrack's win and exotic pools price margins and joint orders in
   parallel books on a finite outcome space, with consistency left to
   arbitrage and the Harville map as the bridge [@harville1973assigning;
   @hausch1981efficiency]; the rank-settled stage differs by construction,
   chaining through the reported margins so the dependence market cannot
   disagree with them. Open: the equilibrium when the same participants
   trade margin and copula stages, and the choice of embedding for
   $d\ge2$, the space-filling curves as deployed against the random
   one-dimensional projections of the companion paper.

## References

::: {#refs}
:::
