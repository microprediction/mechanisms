# Predictors as Markets

### One maker, makers in parallel, markets in series

Peter Cotton · *Working draft v0.2* · August 28, 2026

---

## Abstract

Many learning procedures are cost-function markets by identity rather than
analogy: mirror descent trades against a convex-cost maker, Bayesian model
averaging is a market of Kelly bettors, least squares is the
precision-weighted merge of one quadratic maker per observation, and a
proximal step is the optimal response to a maker charging the penalty. We
ask the converse: which predictors are markets. For one maker, we show
non-convexity does not obstruct coherence: no-arbitrage is a chord
condition, rational flow trades the biconjugate, and a proportional fee of
at least the chord excursion restores no-arbitrage, the fee being exactly a
bid-ask spread by conjugation. For makers in parallel, combining
fee-bearing makers is an infimal convolution solved by one monotone
clearing-price root-find with sparse fills, the aggregate supply curve is a
consolidated limit order book, and a deep quadratic co-quoter is Moreau
smoothing. For markets in series, running one market per chain-rule factor
makes the alternation of model step and market step the Kalman filter,
makes market messages on a tree exact posterior marginals, and turns the
inconsistency that degrades loopy belief propagation into an arbitrage that
traders remove. A predictor is a market when its arbitrage depth is finite
relative to affordable friction; the two compositions are the two
operations of the inference semiring.

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
market form demand? Sections 2–4 treat one maker, sections 5–7 makers in
parallel, sections 8–11 markets in series, and section 12 states the
characterization. Throughout, a scalar security settles at
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
@abernethy2013efficient, not through no-arbitrage. Dropping that axiom
while keeping the chord condition leaves a coherent non-convex maker,
exhibited numerically in the companion repository. Arbitrage theory
without convexity is developed in a different formalism by
@lepinette2017nonconvex.

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
chord-coherent (possibly non-convex) maker with a quadratic co-quoter of
liquidity $\lambda$ yields the venue with cost the Moreau envelope
$e_\lambda C(x) = \min_y C(y) + (x-y)^2/(2\lambda)$, whose arbitrage depth
decreases to zero as $\lambda \to \infty$; generically (tilted gaps) the
merged venue is exactly convex at finite depth, while a symmetric double
well retains a concave kink of magnitude $O(1/\lambda)$ at every depth,
priced by a fee of the same order.*

**Proof sketch.** The merge is infimal convolution, and inf-convolution
with the quadratic is the Moreau envelope [@rockafellar1970convex].
$e_\lambda C$ is a minimum of convex branches
$y \mapsto C(y) + (\cdot - y)^2/(2\lambda)$; non-convexity survives only at
branch crossings, where the slope jump is the branch-minimizer separation
over $\lambda$, hence $O(1/\lambda)$, vanishing or merging entirely when
the crossing is eliminated by tilt. $\blacksquare$

The two repairs are the two market primitives again: friction ($\ell_1$,
the fee) and participant depth ($\ell_2$, capital), lasso and ridge.

## 7. Self-set fees, stabilizers, adaptivity

Nothing requires the fees to be administered. Each maker may quote its own
$f_i$: a quote inside the aggregate spread earns nothing, a quote too tight
is picked off by informed flow, and the undercutting happens inside the
same minimisation that clears the trade, since the $\inf$ in the
convolution is a minimum over quotes. The surviving fee is the competitive
adverse-selection charge of classical microstructure [@glosten1985bidask;
@biais2000competing], reached through routing rather than a dealer game.

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
The market algebra is that pair: parallel merge is the product, densities
multiplying as precisions add, and serial propagation is the sum.

The parallel operation has two equivalent forms, addition of conjugates
and infimal convolution of costs, because inf-convolution is convolution
in the min-plus semiring and the Legendre transform is its Laplace
transform [@litvinov2007idempotent]: the convex machinery of this paper is
the min-plus form of the probability calculus. Every Gaussian exactness
below has one source: log-quadratics are the family on which the
sum-product and min-sum semirings coincide, so a market whose traders
optimize agrees with an inference engine that integrates. Off the family
they diverge by the Laplace-approximation gap, which is §12's open
question.

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

**Proposition 10 (the market step is the Kalman update).** *The
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

**Proposition 11 (trades as messages, exact on trees).** *All three
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

**Proposition 12 (cycle inconsistency is a sure profit).** *Assemble
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

Proposition 12 identifies the object being corrected: the inconsistency
that degrades loopy belief propagation is, in a market, free money, and
the flow that harvests it pushes the quotes back toward the cone. Whether
the corrected fixed point is the true marginal, a Bethe-like surrogate, or
something the liquidity profile selects is open.

## 12. The characterization, and open problems

A predictor is a market when its cost has finite arbitrage depth relative
to affordable friction; convexity is the $f = 0$ special case. Within the
coherent class, non-convexity spends expressiveness, the contact set in
place of a full quoting range, and spends nothing else against rational
flow.

The market form adds three things the estimation literature obtains
otherwise. Tuning constants become prices set by competition rather than
formulas. Composition becomes an algebra, parallel and serial as the two
semiring operations. And the procedure becomes robust to strategic data,
since every source pays to move the state: a market is the incentive
closure of a predictor.

Open problems, in rough order of tractability:

*Which inference algorithm is a market?* A market of optimizing traders
natively computes min-plus, so off the Gaussian family it prices the
max-product posterior rather than the sum-product marginal, the
discrepancy being the Laplace-approximation gap. Risk aversion is an
exponential tilt, suggesting it interpolates between the two semirings;
whether a risk-averse trading population prices the marginal, the mode, or
a temperature in between would say precisely which inference algorithm a
market runs.

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

*Books around binary events.* The multimodal-quote reading of book holes
(§3) is testable against limit-order data around court rulings and FDA
decisions, where bimodal implied densities are documented
[@clark2017brexit].

## References

::: {#refs}
:::
