# Predictors as Markets

### One maker, makers in parallel, markets in series

Peter Cotton · *Working draft v0.5* · August 29, 2026

---

## Abstract

There are two algebraic reasons markets appear inside learning: convex
duality turns optimization into trading, and graphical factorization turns
inference into networks of local markets. We develop both. For one maker,
non-convexity does not obstruct coherence: no-arbitrage is a chord
condition, rational flow trades the biconjugate, and a proportional fee of
at least the chord excursion restores no-arbitrage, the fee being exactly a
bid-ask spread by conjugation. For makers in parallel, combining
fee-bearing makers is an infimal convolution solved by one monotone
clearing-price root-find with sparse fills, and the aggregate supply curve
is a consolidated limit order book. For markets in series, one market per
chain-rule factor makes the market an exact min-plus inference engine:
parallel merge implements factor multiplication and cheapest routing
implements variable elimination, both for arbitrary convex potentials.
A finite-state chain of markets runs Viterbi exactly; sum-product
semantics are recovered exactly on log-quadratic families, where partial
minimization equals marginalization up to a constant, and the
market-implemented Kalman filter and exact tree marginals live on this
intersection. A cost-based predictor admits a coherent market
implementation if and only if its arbitrage depth is dominated by the
permitted friction; friction extends the construction beyond convexity,
and arbitrage enforces coherence across markets, every convex feasibility
violation being a separating portfolio. The resulting object is the
incentive closure of the predictor: the same operator, with every
information source paying to move the state.

---

## 1. The correspondences, and the question

A cost-function market maker posts a potential $C$ over inventory and
charges $C(q+s) - C(q)$ for a fill $s$; prices are gradients and, when $C$
is convex, the standard theory applies [@hanson2003combinatorial;
@abernethy2013efficient]. A run of published identities makes many learning
procedures markets in this literal sense. Trading against such a maker is
follow-the-regularized-leader with the cost as conjugate regularizer
[@chen2010newunderstanding; @abernethy2013efficient], with trader wealth as
the learning rate [@frongillo2012interpreting]. A market of Kelly bettors
performs Bayesian model averaging with wealths as posterior weights
[@beygelzimer2012kelly], and equilibrium prices of utility-maximizing agents
implement mixtures and products of experts [@pennock1997aggregate;
@storkey2011machine; @storkey2012isoelastic; @barbu2012artificial].

Two further identities are used throughout. Ordinary least squares is a
market of data points: each observation a quadratic maker quoting its
value with capital equal to its precision, the estimate their merge, since
merging makers is the infimal convolution of their costs and liquidity
adds [@bhaskara2023general; @barrieu2005inf]. And a proximal step is a
trade against a fee-bearing maker: the proximal operator of
$f\lvert\cdot\rvert$ is the soft-threshold, which is the optimal response
to a proportional fee (Lemma 4 below), and the prox of any convex $g$ is
the response to a maker charging $g$. The two canonical penalties are then
the two market primitives: ridge is a zero-quoting participant with
capital $\lambda$, lasso is a fee of $\lambda$, and the theorem that
regularization is robustness to data perturbation [@elghaoui1997robust;
@xu2009robustness] acquires a market reading in which the adversary's
budget is priced rather than assumed.

We ask the converse: which predictors are markets, and what does the
market form demand? The answer has two independent parts, and the paper is
organized around them. Convex duality turns optimization into trading:
sections 2–4 treat one maker and sections 5–7 makers in parallel.
Graphical factorization turns inference into networks of local markets:
sections 8–11 treat markets in series, with the precise semiring statement
in section 8. Section 12 states the characterization. Throughout, a scalar
security settles at
$\varphi(\omega) = \omega \in [-1,1]$; vector statements substitute the
convex hull of payoff vectors. Reference implementations and numerical
theorem tests accompany the paper in the `mechanisms` repository.

## 2. Coherence is a chord condition

**Proposition 1 (no-arbitrage without convexity).** *For a path-independent
maker with cost $C$ (no convexity assumed), there is no outcome-independent
profitable fill from any state if and only if every chord slope of $C$ lies
in the convex hull of payoffs: for all $q, s$,*

$$C(q+s) - C(q) \;\ge\; \min_{\omega}\, \varphi(\omega)\, s
\qquad\text{(scalar: } C \text{ is } 1\text{-Lipschitz).}$$

**Proof.** A fill $s$ from state $q$ has sure profit
$\min_\omega \varphi(\omega) s - [C(q+s) - C(q)]$; positivity for some pair
$(q,s)$ is exactly the failure of the displayed inequality, and the
inequality for all pairs says every chord slope is supported by the hull.
$\blacksquare$

Round trips are refunds for any $C$, convex or not, because the charge
telescopes over closed paths. When the chord condition fails the exploit is
the accumulation of sure-profit net positions at states where quotes exit
the hull.

Convexity enters the standard axiomatics through *information
incorporation*, the marginal-cost monotonicity condition of
@abernethy2013efficient, not through no-arbitrage. The habit of assuming
convexity has bundled two different guarantees: monotone price response to
flow, and absence of sure-loss opportunities. The axiom sets impose the
first, convexity follows, and no-arbitrage is then derived with convexity
in hand, so the possibility of a coherent maker with non-monotone quotes
never arises. Proposition 1 separates the guarantees: dropping information
incorporation while keeping the chord condition leaves a coherent
non-convex maker, exhibited numerically in the companion repository.
Arbitrage theory without convexity is developed in a different formalism
by @lepinette2017nonconvex.

## 3. The market trades the biconjugate

Let $\hat C$ denote the lower convex envelope of a chord-coherent cost $C$
(its biconjugate on the line [@rockafellar1970convex]) and
$g = C - \hat C \ge 0$ the gap.

**Proposition 2 (contact and pass-through).** *A myopic risk-neutral trader
with believed mean $\mu$ in the hull, facing the maker at state $q$:
(i) has optimal fills landing on the contact set $\{C = \hat C\}$, and
(ii) earns maximal expected profit*

$$\Pi_C(q,\mu) \;=\; \Pi_{\hat C}(q,\mu) + g(q),$$

*the envelope profit plus the gap at the starting state.*

**Proof.** $\sup_s \mu s - [C(q+s) - C(q)] = \sup_x [\mu x - C(x)] - \mu q
+ C(q)$. Since $C \ge \hat C$ with equality on the contact set, and a
maximizer of an affine function minus $C$ is a point where an affine
minorant touches $C$, hence touches $\hat C$, the supremum equals
$\sup_x [\mu x - \hat C(x)]$ and is attained on the contact set. Adding and
subtracting $\hat C(q)$ gives (ii). $\blacksquare$

The maker behaves observationally like its convex envelope. Concave
stretches are unquotable intermediate states, holes in the book that
rational flow jumps across. Whoever lands inside one (noise) overpays the
gap at the landing state, and by (ii) the next rational trader recoups it,
so the maker is a conduit keeping envelope differences over any
rational-to-rational span, and off-contact states are transient. The
trade-set analogue on the CFMM side is the canonical concave trading
function: an arbitrary invariant is behaviorally equivalent to a concave
one [@angeris2024geometry], with the limits of concavification mapped by
@frongillo2024axiomatic.

Read as beliefs, a hole is a multimodal quote: the maker takes prices on
either side of the gap and refuses every price inside, jumping
discontinuously as flow accumulates. This is the book-side shape of the
bimodal implied densities documented around binary events
[@melick1997crude; @clark2017brexit].

## 4. Frictions price the remaining failure

The failure mode surviving §2 is chord slopes exiting the hull.

**Proposition 3 (fees buy bounded incoherence).** *If chord slopes exit the
hull by at most $\varepsilon$ per unit, i.e. $C(q+s) - C(q) \ge
\min_\omega \varphi(\omega)s - \varepsilon\lvert s\rvert$, then a
proportional fee $f \ge \varepsilon$ restores no-arbitrage.*

**Proof.** Sure profit with the fee is
$\min_\omega \varphi(\omega)s - \Delta C - f\lvert s\rvert \le
(\varepsilon - f)\lvert s\rvert \le 0$. $\blacksquare$

This is the mechanism-level counterpart of a theorem of mathematical
finance: arbitrarily small proportional transaction costs restore
no-arbitrage for price processes that are arbitrageable frictionlessly,
with a consistent price system inside the spread as certificate
[@guasoni2006transaction; @guasoni2010ftap]. In prediction markets the
precedent is fees sized to expected arbitrage profit restoring bounded loss
under privacy noise [@cummings2016privacy; @frongillo2018private]. The
minimum viable spread of a cost is its arbitrage depth, and a venue
quoting a wide spread to cover a badly non-convex cost is coherent but
uninformative in proportion: the market prices the model's incoherence as
uncertainty.

## 5. A linear fee is a bid-ask spread

Why a fee at all? A cost-function maker's charge telescopes, so round trips
are free: the maker earns nothing on uninformed flow and cannot recover
adverse-selection losses from volume. That no state-dependent cost can
charge a round trip, and that a path-dependent volume charge repairs it, is
due to @othman2012profitcharging; the linear charge $f\lvert s\rvert$ is
their volume levy in its simplest convex form, chosen because it conjugates
in closed form. Write $g_q(s) = C(q+s) - C(q)$ for a convex $C$ and
$m = C'(q)$ for the marginal price.

**Lemma 4 (fee–spread duality).** *Let
$T_f(x) = \operatorname{sign}(x)\max(\lvert x\rvert - f,\, 0)$ denote the
soft-threshold. Then the fee-bearing cost $\tilde C_q = g_q +
f\lvert\cdot\rvert$ has conjugate*

$$\tilde C_q^*(p) \;=\; g_q^*\!\big( m + T_{f}(p - m) \big),$$

*zero if and only if $\lvert p - m\rvert \le f$.*

**Proof.** The conjugate of a sum of closed proper convex functions with
overlapping relative interiors of domains is the infimal convolution of the
conjugates [@rockafellar1970convex, Thm. 16.4], and the conjugate of
$f\lvert\cdot\rvert$ is the indicator of $[-f, f]$. Hence
$\tilde C_q^*(p) = \min_{\lvert u\rvert \le f} g_q^*(p - u)$, the minimum of
the convex function $g_q^*$ over $[p - f,\, p + f]$. Since $g_q^* \ge 0$
with equality exactly at $m$, the minimum is attained at the projection of
$m$ onto the interval, which is $m + T_{f}(p - m)$. $\blacksquare$

The maker quotes $\mathrm{ask} = m + f$ and $\mathrm{bid} = m - f$ and
trades nothing in between: a proportional fee is a bid-ask spread, the
same duality by which a proportional transaction cost confines the pricing
functional to the bid-ask band [@jouini1995transaction]. Because
conjugation reads only biconjugates, the lemma survives non-convexity with
the envelope in place of the cost:

**Lemma 5 (the envelope form).** *For any chord-coherent $C$ with envelope
$\hat C$, the fee-bearing maker's no-trade band at state $q$ is the
interval of beliefs within $f$ of the envelope's marginal price, non-empty
exactly on the contact set; at off-contact states every belief in the hull
yields profit at least $g(q)$.*

**Proof.** By Proposition 2, profit at belief $\mu$ is the envelope profit
plus $g(q)$; with the fee the envelope profit is the soft-thresholded
conjugate of Lemma 4, zero on the band around $\hat C'(q)$. On the contact
set $g(q) = 0$ and the band survives; off it the additive $g(q) > 0$
leaves no zero. $\blacksquare$

## 6. Makers in parallel: routing, the order book, Moreau

Let makers $i = 1..n$ hold inventories $q_i$ with convex costs $C_i$,
liquidities of their choosing, and fees $f_i$ of their choosing.

**Lemma 6 (combination).** *Let
$\tilde C = \tilde C_1 \,\square\, \cdots \,\square\, \tilde C_n$ and define
each maker's supply*

$$s_i(p) \;=\;
\begin{cases}
(C_i')^{-1}(p - f_i) - q_i, & p \ge \mathrm{ask}_i,\\[2pt]
0, & \mathrm{bid}_i < p < \mathrm{ask}_i,\\[2pt]
(C_i')^{-1}(p + f_i) - q_i, & p \le \mathrm{bid}_i,
\end{cases}$$

*each non-decreasing in $p$. Fix a demand $\Delta$ for which a clearing
price $p^*$ with $\sum_i s_i(p^*) = \Delta$ exists. Then:
(i) $\tilde C^* = \sum_i \tilde C_i^*$, a sum of soft-thresholded profit
functions; (ii) the split $s_i = s_i(p^*)$ attains $\tilde C(\Delta)$, and
any optimal split satisfies
$C_i'(q_i + s_i) + f_i \operatorname{sign}(s_i) = p^*$ for $s_i \ne 0$ and
$\lvert p^* - m_i \rvert \le f_i$ for $s_i = 0$; (iii) the split is sparse:
every maker whose quote band strictly contains $p^*$ trades exactly zero.*

**Proof.** (i) is the conjugate-sum identity applied to the convolution
[@rockafellar1970convex]. For (ii), the split is feasible by choice of
$p^*$, and $p^* \in \partial \tilde C_i(s_i(p^*))$ for every $i$: when
$s_i(p^*) \ne 0$ the subgradient is
$C_i'(q_i+s_i) + f_i\operatorname{sign}(s_i) = p^*$, and when
$s_i(p^*) = 0$ it is the interval $[m_i - f_i,\, m_i + f_i] \ni p^*$. A
common multiplier certifying every coordinate is exactly the optimality
condition for $\min\{\sum_i \tilde C_i(s_i) : \sum_i s_i = \Delta\}$.
(iii) restates the zero branch. $\blacksquare$

The computation is a scalar monotone root-find whatever $n$ is, and the fee
costs nothing beyond a horizontal shift of each supply curve. The
$\lvert s\rvert$ terms are an $\ell_1$ penalty, so sparsity arrives for the
same reason it does in the lasso, and for the same reason proportional
transaction costs produce no-trade regions and sparse portfolios
[@olivaresnadal2018robust]. A small trade routes entirely to the tightest
quote; a growing trade pushes that maker's fee-adjusted marginal price
through the next band and spills over, consuming makers in fee order.

**Corollary 7 (zero fees).** *With $f_i \equiv 0$ the convolution reduces
to the fee-free merge: conjugate regularisers add, and for a perspective
family $C_b(q) = b\,C_1(q/b)$ liquidity adds,
$C_{b_1} \square C_{b_2} = C_{b_1+b_2}$* [@bhaskara2023general].

**Corollary 8 (the order book).** *The aggregate supply
$S(p) = \sum_i s_i(p)$ is non-decreasing, identically zero on
$\big(\max_i \mathrm{bid}_i,\ \min_i \mathrm{ask}_i\big)$ when that
interval is non-empty, flat wherever every maker's band covers $p$, and
smooth and strictly increasing wherever some maker is in the money.* Read
as a market: best bid and ask are the tightest quotes, depth at each price
is the sum of the active makers' closed-form supplies, and large orders
walk the levels. The aggregate of linear-fee makers is a consolidated
limit order book, and in producer-theory terms Lemma 6 is Marshall's
horizontal summation of firm supply curves [@marshall1890principles;
@mascolell1995microeconomic] with the reversibility of share production
patched by the fee. The economics of the book assembled from competing
liquidity suppliers is @glosten1994limit, with convergence of strategic
schedules in @biais2000competing.

**Proposition 9 (a deep co-quoter is Moreau smoothing).** *Merging a
maker with cost $C$ with a quadratic co-quoter of cost $s^2/(2\lambda)$
yields the venue with cost the Moreau envelope
$e_\lambda C(x) = \min_y C(y) + (x-y)^2/(2\lambda)$. Then:
(i) coherence is preserved: if $C$ is chord-coherent, so is $e_\lambda C$
for every $\lambda$; (ii) the convexity defect of $e_\lambda C$, the
largest slope drop at a branch crossing, equals the separation of the
competing minimizers divided by $\lambda$, so if all competing minimizers
lie in an interval of diameter $D$ (as when the set where $C$ exceeds its
convex envelope is bounded and $C$ is convex outside it), the defect is at
most $D/\lambda$; (iii) the symmetric double well attains the bound at
every depth.*

**Proof.** The merge is infimal convolution, and inf-convolution with the
quadratic is the Moreau envelope [@rockafellar1970convex]. For (i),
wherever the infimum is attained at $y^*(x)$ with $C$ differentiable
there, the first-order condition gives $e_\lambda C'(x) = (x - y^*)/
\lambda = C'(y^*)$: the envelope's derivative is a selection of $C$'s
derivatives, chords average derivatives, and $C$'s derivatives lie in the
payoff hull by hypothesis. For (ii), at a crossing $x_0$ with competing
minimizers $y_1^* < y_2^*$ the branch slopes are $(x_0 - y_i^*)/\lambda$,
so the drop is exactly $(y_2^* - y_1^*)/\lambda$. $\blacksquare$

Part (i) corrects a possible misreading of the construction: the merge
never creates arbitrage, at any depth, so what the deep co-quoter buys is
expressiveness. The holes in the book close at rate $1/\lambda$, and the
residual kink of the symmetric double well is priced by a fee of the same
order. Here $\lambda$ is the co-quoter's liquidity: its price impact is
$1/\lambda$, so large $\lambda$ means a deep book, not a strong pull.

The two repairs are the two market primitives again: friction ($\ell_1$,
the fee) and participant depth ($\ell_2$, capital), lasso and ridge. The
identification maps the penalty terms, not the statistical procedures; it
is a dictionary of primitives, not an equivalence of estimators.

## 7. Self-set fees, stabilizers, adaptivity

Nothing requires the fees to be administered. Each maker may quote its own
$f_i$: a quote inside the aggregate spread earns nothing, a quote too tight
is picked off by informed flow, and the undercutting happens inside the
same minimisation that clears the trade, since the $\inf$ in the
convolution is a minimum over quotes. The surviving fee has the
interpretation of the competitive adverse-selection charge of classical
microstructure [@glosten1985bidask; @biais2000competing], reached through
routing rather than a dealer game; this is an argument about discipline,
and the strategic equilibrium of the quote game is not solved here.

Optimization already pays these frictions: a proximal step charges
$\lVert\Delta\theta\rVert^2/(2\eta)$ per move, a trust region is an
infinite fee outside a band, weight decay is a zero-quoting participant.
In market language the stabilizers that make non-convex training behave
are the frictions that make a non-convex venue non-exploitable.
Path-dependent optimizers correspond to makers whose quotes depend on flow
history, the adaptive-liquidity territory where path independence is
deliberately traded away: no maker combines path independence, translation
invariance and liquidity sensitivity [@othman2013practical], the adaptive
class is axiomatized by @li2013adaptive with the homogeneous-risk-measure
characterization in @othman2011liquidity, no trade-history maker achieves
every desideratum at once [@abernethy2014vpm], and liquidity selection
itself can be run as online learning [@nueve2026adaptiveliquidity;
@nueve2025smooth].

## 8. Two operators, one semiring

Sections 5–7 composed makers on one quantity. A model is a factorization,

$$P(\text{everything}) \;=\; \prod_{\text{nodes}} P(x_i \mid
\text{parents}(x_i)),$$

and the remaining operator runs one market per factor, each pricing its
conditional given what is upstream. The count of two is structural.
Message-passing inference is an algorithm over a commutative semiring
[@aji2000gdl]: one operation combines evidence about a variable (the
product), one moves evidence between variables (the sum), and sum-product,
max-product and min-sum are the one algorithm over different semirings.

The correspondence with the market algebra is exact once the objects are
fixed, and the fixing matters: the two compositions live on the two sides
of the Legendre transform. Assign each factor its potential, the negative
log-density $\varphi_i = -\log P(x_i \mid \text{parents})$. For parallel
composition, identify each maker's factor potential with its *conjugate*,
$\varphi_i = C_i^*$: infimal convolution of costs in inventory space is
addition of potentials in price space. For serial composition, the stage
potentials are the *primal* leg costs of a route. The dictionary is

$$\begin{array}{ccc}
\text{factor algebra} & \longleftrightarrow & \text{market algebra}\\[2pt]
\text{density } p_i & \longleftrightarrow & \text{potential } \varphi_i = -\log p_i\\
\text{multiply factors} & \longleftrightarrow & \text{parallel merge: } \varphi_i = C_i^* \text{ add as costs inf-convolve}\\
\text{eliminate a variable} & \longleftrightarrow & \text{route through its market: } \inf \text{ over the shared leg}
\end{array}$$

with the two sides related by the Legendre transform, which is the Laplace
transform of the min-plus semiring [@litvinov2007idempotent]. The parallel
law is a commuting square,

$$\begin{array}{ccc}
C_1,\, C_2 & \xrightarrow{\ \square\ } & C_1 \,\square\, C_2\\[2pt]
\downarrow{\scriptstyle *} & & \downarrow{\scriptstyle *}\\[2pt]
C_1^*,\, C_2^* & \xrightarrow{\ +\ } & C_1^* + C_2^*
\end{array}$$

while serial composition stays primal, obeying the composition law
$(\varphi_2 \circ \varphi_1)(x,z) = \inf_y [\varphi_1(x,y) +
\varphi_2(y,z)]$, the min-plus kernel product. The Legendre transform is
not a computational trick here but the change of representation between
the two composition laws.

**Proposition 10 (the market computes min-plus inference).** *(i) Parallel:
merging makers multiplies factors, since $(C_1 \square C_2)^* = C_1^* +
C_2^*$ and potentials add exactly when densities multiply. (ii) Serial:
the cheapest route to a terminal exposure through the chain of stage
markets prices the min-plus elimination
$\inf_{z_1,\dots,z_{n-1}} \sum_i \varphi_i(z_{i-1}, z_i)$, the tropical
product of the stage kernels. (iii) On log-quadratic families, partial
minimization of a potential equals its marginalization up to an additive
constant independent of the retained variables, so the market's min-plus
messages are the sum-product messages and the market computes exact
Bayesian inference.*

**Proof.** (i) is the conjugate-sum identity together with
$-\log(p_1 p_2) = \varphi_1 + \varphi_2$. (ii) is the definition of the
cheapest route: the trader chooses intermediate exposures to minimize the
sum of stage costs. For (iii), write a jointly quadratic potential
$q(x,y)$ with positive definite $y$-block $Q_{yy}$; completing the square,
$-\log \int e^{-q(x,y)}\,dy = \min_y q(x,y) + \tfrac12\log\det(Q_{yy}/2\pi)$,
the Schur-complement identity, and the constant does not depend on $x$.
$\blacksquare$

The proposition divides its labor as follows. The min-plus identities of
(i) and (ii) are algebraic and hold for arbitrary potentials; the
generalized distributive law needs no convexity [@aji2000gdl]. Convexity
is what the market adds: it is the condition
under which the potentials are implementable as coherent cost-function
makers (§2), so that the min-plus computation is carried out by
self-interested trading rather than by a solver.

**Corollary (a chain of markets runs Viterbi).** *Take finite state
spaces and stage potentials $\varphi_t(i,j) = -\log P(X_t = j \mid
X_{t-1} = i) - \log P(y_t \mid X_t = j)$. The cheapest route of
Proposition 10(ii) is $\min_{x_{1:T}} \sum_t \varphi_t(x_{t-1}, x_t)$:
the Viterbi decoding of the hidden Markov model, computed exactly, with
no Gaussian structure anywhere.* The three worked examples of this paper
now form a progression: least squares is parallel composition, Viterbi is
serial min-plus composition, and the Kalman filter of §10 is the Gaussian
intersection where the serial computation is also Bayesian.

**Principle (the Gaussian intersection).** *Log-quadratic families are
exactly where probabilistic inference and market optimization coincide: a
market whose traders optimize computes min-plus, an inference engine
integrates, and Proposition 10(iii) says the two agree on this family up
to constants that cancel in every price.* The Kalman and tree-marginal
propositions below are instances of the intersection, not evidence for a
universal serial thesis. Away from log-quadratics the market prices the
max-product (MAP) posterior rather than the sum-product marginal, and the
discrepancy is the Laplace-approximation gap of the potential. Risk
aversion is an exponential tilt and plausibly interpolates between the two
semirings; §12 poses this as the open question of which inference
algorithm a risk-averse market runs.

## 9. One market per factor

The locality that makes the factorization tradable is Hanson's modularity:
in a combinatorial LMSR a bet on $A \mid B$ moves that conditional and
provably nothing else, uniquely among market scoring rules
[@hanson2007logarithmic; @hanson2003combinatorial]. Hanson runs one joint
market over the product space; the substance of pricing it is
probabilistic inference, which is why exact pricing is #P-hard
[@chen2008complexity], why tournament markets price by Bayes-net inference
[@chen2008tournaments], why a deployed combinatorial market ran its price
and asset updates on the junction-tree algorithm [@sun2012junction], and
why approximate designs price over the marginal polytope
[@dudik2012tractable; @dudik2021logtime]. Securities structured by a
Bayes-net factorization appear in @pennock2000compact, with the trades
that preserve the structure characterized by @xia2011structure.

The serial architecture takes the factors as separate venues rather than
one joint book. Market $i$ opens on $P(x_i \mid \text{parents})$ when its
conditioning information freezes, conditions on upstream quotes while
upstream is live, and re-references when upstream settles; settlements
cascade through the DAG like dataflow, and the schedule of markets is an
unrolling of the graph. Observable nodes carry settled markets; a latent
node either settles through its observable footprint, a market on a hidden
state being a market on a functional of future observables scored through
the model, or remains indicated but unsettled, priced by whoever
warehouses the model risk.

## 10. The market step is the Kalman update

Take the linear-Gaussian chain $x_{t+1} = a x_t + \varepsilon_t$,
$y_t = x_t + \eta_t$, and alternate two steps: a *model step*, propagating
the current belief through the dynamics in mean-precision form, and a
*market step*, merging the propagated belief with an observation maker
quoting $y_t$ whose capital is the observation precision.

**Proposition 11 (the market step is the Kalman update).** *The
alternation is the Kalman filter [@kalman1960filtering]: after each
observation the market state equals the filtered mean and variance
exactly.*

**Proof.** The model step is the standard prediction of mean and variance.
Merging quadratic makers adds precisions and precision-weights means (the
fusion form of Corollary 7), which is the information-form measurement
update; the covariance-form update follows by the usual algebra.
$\blacksquare$

The prediction step is computed and the correction step is traded, with
capital setting the weight of the data. The nearest published object is
the Bayesian market maker of @brahma2012bayesianmm, whose trade update is
a scalar Gaussian measurement update with covariance inflation for jumps;
we find no filtering reading, and no alternation with a dynamic model, in
the literature.

## 11. Trees and loops

On a Gaussian chain with an observation at every node, the posterior at an
interior node is the merge of three sources: the forward message (the
filter of §10 run up to the node), the local observation, and the backward
message (downstream observations pulled back through the dynamics, each
pull-back a reparametrization and each combination a precision addition).

**Proposition 12 (trades as messages, exact on trees).** *All three
messages are market operations, and the merged posterior equals the
brute-force conditioning of the joint at every node: Gaussian belief
propagation [@pearl1988probabilistic] with trades as messages.*

The repository verifies the identity to ten digits. The contrast with the
deployed combinatorial engines is architectural: there the junction tree
is the pricing algorithm inside one joint market [@sun2012junction]; here
the messages pass between venues, and the graph's edges are market
boundaries.

On loopy graphs belief propagation double-counts and its fixed points
drift. In a market the same inconsistency is self-punishing.

**Proposition 13 (cycle inconsistency is a sure profit).** *Assemble
pairwise correlation quotes around a cycle
into a quote matrix $P$ with unit diagonal. The quotes admit a joint
distribution iff $P \succeq 0$. Otherwise, with $w$ the eigenvector of a
negative eigenvalue $\lambda$, the bundle with weights $ww^\top$ has price
$w^\top P w = \lambda < 0$ and payoff $(w^\top x)^2 \ge 0$: a sure profit
of at least $\lvert\lambda\rvert$ per unit.*

This is the second-moment coherence condition, prices of products must
form a PSD matrix [@daspremont2005market], read as cycle consistency.
Arbitrage-enforced coherence is an old idea. Coherence is no-arbitrage
[@nau1991arbitrage]; @pennock1996marketbayes built arbitrageur agents
enforcing the additivity identities of a Bayes-net economy, with
equilibrium prices equal to the network's probabilities and distributed
bidding as distributed inference; the combinatorial literature removes the
arbitrage algorithmically over the marginal polytope [@kroer2016arbitrage];
and @saguillo2025arbitrage measure Polymarket arbitrageurs extracting
\$39.6M enforcing logical coherence across markets.

The scope of Proposition 13 is one class of consistency failure: locally
quoted beliefs that cannot be embedded in any joint distribution, detected
here at second moments. Not every loopy-propagation error takes this form,
but the class it covers is, in a market, free money, and the flow that
harvests it pushes the quotes back toward the cone. Whether the corrected
fixed point is the true marginal, a Bethe-like surrogate, or something the
liquidity profile selects is open.

The PSD cone is not special. Coherent price sets are convex (they are
convex hulls of payoff vectors, or their conic images), and separation
turns every exterior point into a trade:

**Proposition 14 (arbitrage is separation).** *Let $K$ be the closed
convex hull of the attainable payoff vectors (or its image under a linear
security map), so that $\inf_{z \in K} \langle y, z\rangle \le \langle y,
\varphi(\omega)\rangle$ for every outcome $\omega$ and portfolio $y$. If a
quoted configuration $x \notin K$, any separating functional $y$ with
$\langle y, x\rangle < \inf_{z \in K} \langle y, z\rangle$ is a portfolio
whose price is strictly less than its realized payoff in every state: a
sure profit. Proposition 13 is the instance $K = \{P \succeq 0\}$ with the
separating functional $ww^\top$.*

The proof is the separating-hyperplane theorem read as a trade
[@nau1991arbitrage; @daspremont2005market]; the payoff-hull hypothesis is
what upgrades the separation certificate to an arbitrage, and for an
abstract consistency set the construction yields the certificate only. In
an ordinary numerical method an infeasibility is a residual to be driven
down by the algorithm; in a market it is a payoff, and whoever finds it
is paid to act as the separation oracle. Arbitrageurs are decentralized
separation oracles, and the friction of §4 sets the tolerance below which
infeasibility is allowed to persist.

## 12. The characterization, and open problems

The characterization is for the cost-based class. Call a predictor
*cost-based* if it is specified by a path-independent potential $C$ over a
security inventory, and call friction of size $f$ *permitted* if the
mechanism may charge up to $f$ per unit traded. A cost-based predictor
admits a coherent market implementation if and only if its arbitrage depth
is dominated by the permitted friction; convexity is the $f = 0$ special
case. Within the coherent class, non-convexity spends expressiveness, the
contact set in place of a full quoting range, and spends nothing else
against rational flow.

The construction throughout is one operation: *marketization*. Given an
operator $T$ from inputs to outputs, a marketization is a mechanism whose
clearing computes $T$; whose local contributions compose, in parallel and
in series, to compute composite operators; in which inconsistent
contributions create exploitable trades (Proposition 14); in which
friction bounds how much inconsistency can persist (Proposition 3); and in
which every participant pays to perturb the computation. One estimator
becomes one maker, combining evidence becomes parallel composition,
composing conditional operators becomes serial composition. A statistical
procedure specifies how information would be combined if supplied
honestly; its marketization implements the same operator against
self-interested sources. In this sense a market is the incentive closure
of a predictor: the computation, plus the dual certificates of its
constraints, plus payment to whoever holds one.

Nothing in the definition mentions prediction. The serial law is the
min-plus kernel composition of dynamic programming, control, and shortest
paths; the parallel law is the additive combination of local potentials;
and the closure of the whole construction is a boundary statement:

**Proposition 15 (the effective boundary maker).** *Let a network of
makers have boundary variables $b$, internal variables $h$, and local
costs $C_e$, each jointly convex, and define
$C_{\mathrm{eff}}(b) = \inf_h \sum_e C_e(b, h)$. Then
$C_{\mathrm{eff}}$ is convex; the elimination may be performed variable
by variable in any order; for quadratic costs each single-variable
elimination is a Schur complement; and on a chain the computation is the
dynamic program of Proposition 10(ii). A network of local makers is one
effective maker at its boundary.*

**Proof.** Partial minimization of a jointly convex function is convex,
and iterated infima may be taken in any order [@rockafellar1970convex];
the quadratic case is the completion of the square in the proof of
Proposition 10(iii). $\blacksquare$

This is the operation that reduces resistor networks to terminal
impedances, eliminates latent Gaussian variables, and takes Schur
complements; prediction is the application in which beliefs and prices
share units. We leave the general theory, including the categorical
formulation in which markets are min-plus kernels composed by shared
inventory, outside this paper's scope.

Open problems, in rough order of tractability:

*Which inference algorithm does a risk-averse market run?* Proposition 10
says a risk-neutral market computes min-plus exactly and sum-product only
on the Gaussian intersection. Risk aversion is an exponential tilt,
suggesting it interpolates between the two semirings; whether a
risk-averse trading population prices the marginal, the mode, or a
temperature in between would complete the correspondence.

*Market representations of general prediction maps.* The
characterization above is confined to cost-based predictors. Define what
it means for an arbitrary prediction map $T$, from data sets to forecasts,
to possess a market representation, and give conditions on $T$ equivalent
to existence; the cost-based case suggests path independence and a chord
bound are the shadow of the general condition.

*The equilibrium fee.* Section 7 argues discipline, not equilibrium. In a
sparse-signal flow model, does the Bertrand-equilibrium fee reproduce the
universal-threshold rate of the shrinkage literature? If so, the
regularization constant that statistics tunes by cross-validation equals
the adverse-selection cost of the data source.

*Loopy fixed points.* Under a concrete flow model, does
arbitrage-corrected quoting converge, and to what?

*Arbitrage depth as a capacity measure.* The minimum viable spread of a
trained landscape is computable; whether it tracks quantities learning
theory names (sharpness, mode connectivity) is open.

*Effective friction.* Proposition 15 eliminates frictionless networks.
With per-maker fees, what are the effective fees and liquidities of the
reduced boundary maker in terms of the network's, and which fee-bearing
classes are closed under elimination?

*Books around binary events.* The multimodal-quote reading of book holes
(§3) is testable against limit-order data around court rulings and FDA
decisions, where bimodal implied densities are documented
[@clark2017brexit].

## References

::: {#refs}
:::
