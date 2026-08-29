# Predictors as Markets

### Coherence without convexity, frictions as regularizers, and the market as biconjugate

Peter Cotton · *Working draft v0.1* · August 28, 2026

---

## Abstract

Many learning procedures are cost-function markets by identity rather than
analogy: mirror descent trades against a convex-cost maker, Bayesian model
averaging is a market of Kelly bettors with wealths as posterior weights,
least squares is the precision-weighted merge of one quadratic maker per
observation, and a proximal step is the optimal response to a maker charging
the penalty. This note asks the converse, which predictors are markets, and
finds that the apparent obstruction, non-convexity, does not obstruct
coherence: charges telescope for any cost, and the exact no-arbitrage
condition is that chord slopes stay in the convex hull of payoffs. Rational
flow trades the biconjugate, landing on the contact set with profit equal to
the envelope profit plus the starting gap, so non-convexity costs
expressiveness, unquotable states that read as holes in the book and
multimodal quotes, rather than soundness. Frictions price the remaining
failure: a proportional fee of at least the chord excursion restores
no-arbitrage, the mechanism-level analogue of no-arbitrage under small
transaction costs, and the fee-spread lemma survives non-convexity with the
envelope in place of the cost; a deep quadratic co-quoter drives the merged
venue's incoherence to zero like one over its depth via the Moreau envelope.
A predictor is a market when its arbitrage depth is finite relative to
affordable friction; convexity is the zero-friction case.

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
Ordinary least squares is a market of data points: each observation a
quadratic maker quoting its value with capital equal to its precision, the
estimate their merge, since merging makers is the infimal convolution of
their costs and liquidity adds [@bhaskara2023general]. A proximal step is a
trade against a fee-bearing maker: the proximal operator of
$f\lvert\cdot\rvert$ is the soft-threshold, which is the optimal response to
a proportional fee, and the prox of any convex $g$ is the response to a
maker charging $g$. The two canonical penalties are then the two market
primitives: ridge is a zero-quoting participant with capital $\lambda$,
lasso is a fee of $\lambda$, and the theorem that regularization is
robustness to data perturbation [@elghaoui1997robust; @xu2009robustness]
acquires a market reading in which the adversary's budget is priced rather
than assumed.

The question is the converse. Which predictors are markets, and what
exactly does the market form demand? The suspect requirement is convexity,
without which a maker is said to be exploitable [@angeris2020oracles].
Sections 2–4 show the requirement is weaker and more informative than that.

Throughout, a scalar security settles at $\varphi(\omega) = \omega \in
[-1,1]$; vector statements substitute the convex hull of payoff vectors.
Reference implementations and numerical theorem tests accompany the note in
the `mechanisms` repository.

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

Two remarks. Cycling was never the exploit: the charge telescopes over any
closed path whatever $C$ is, so round trips are refunds; the exploit, when
the chord condition fails, is the accumulation of sure-profit net positions
at states where quotes exit the hull. And convexity enters the standard
axiomatics not through no-arbitrage but through *information incorporation*
(the marginal-cost monotonicity condition of @abernethy2013efficient);
dropping that axiom while keeping the chord condition leaves a coherent
non-convex maker, exhibited numerically in the companion repository.
Arbitrage theory without convexity is developed in a different formalism by
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
rational flow jumps across; whoever lands inside one (noise) overpays the
gap at the landing state, and by (ii) the next rational trader recoups it,
so the maker is a conduit keeping envelope differences over any
rational-to-rational span, and off-contact states are transient. The
trade-set analogue on the CFMM side is the canonical concave trading
function: an arbitrary invariant is behaviorally equivalent to a concave
one [@angeris2024geometry], with the limits of concavification mapped by
@frongillo2024axiomatic. Read as beliefs, a hole is a multimodal quote:
the maker takes prices on either side of the gap and refuses every price
inside, jumping discontinuously as flow accumulates, the book-side shape of
the bimodal implied densities documented around binary events
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
minimum viable spread of a cost is its arbitrage depth, and a venue quoting
a wide spread to cover a wild landscape is coherent but uninformative in
proportion: the market prices the model's incoherence as uncertainty.

**Lemma 4 (the fee lemma, absorbed).** *For any chord-coherent $C$ with
envelope $\hat C$, the fee-bearing maker's no-trade band at state $q$ is
the interval of beliefs within $f$ of the envelope's marginal price, and it
is non-empty exactly on the contact set; at off-contact states every belief
in the hull yields profit at least $g(q)$.*

**Proof.** By Proposition 2, profit at belief $\mu$ is the envelope profit
plus $g(q)$, and with the fee the envelope profit is the soft-thresholded
conjugate of the convex case, zero on the band around $\hat C'(q)$; on the
contact set $g(q) = 0$ and the band survives, off it the additive $g(q) >
0$ leaves no zero. $\blacksquare$

The convex case, contact set everywhere, is the fee–spread lemma of the
companion note *Combining Linear-Fee Market Makers*; conjugation reads only
biconjugates, so the lemma survives non-convexity verbatim with the
envelope in place of the cost.

**Proposition 5 (a deep co-quoter is Moreau smoothing).** *Merging the
$C$-maker with a quadratic co-quoter of liquidity $\lambda$ yields the
venue with cost the Moreau envelope $e_\lambda C(x) = \min_y C(y) +
(x-y)^2/(2\lambda)$, whose arbitrage depth decreases to zero as
$\lambda \to \infty$; generically (tilted gaps) the merged venue is exactly
convex at finite depth, while a symmetric double well retains a concave
kink of magnitude $O(1/\lambda)$ at every depth, priced by a fee of the
same order.*

**Proof sketch.** The merge is infimal convolution [@bhaskara2023general],
and inf-convolution with the quadratic is the Moreau envelope
[@rockafellar1970convex]. $e_\lambda C$ is a minimum of convex branches
$y \mapsto C(y) + (\cdot - y)^2/(2\lambda)$; non-convexity survives only at
branch crossings, where the slope jump is the branch-minimizer separation
over $\lambda$, hence $O(1/\lambda)$, vanishing or merging entirely when
the crossing is eliminated by tilt. $\blacksquare$

The two repairs are the two market primitives again: friction ($\ell_1$,
the fee) and participant depth ($\ell_2$, capital), lasso and ridge.
Optimization already pays them: a proximal step charges
$\lVert\Delta\theta\rVert^2/(2\eta)$ per move, a trust region is an
infinite fee outside a band, weight decay is a zero-quoting participant.
Path-dependent optimizers correspond to makers whose quotes depend on flow
history, the adaptive-liquidity territory where path independence is
deliberately traded away: no maker can combine path independence,
translation invariance and liquidity sensitivity [@othman2013practical],
the adaptive class is axiomatized by @li2013adaptive with the
homogeneous-risk-measure characterization in @othman2011liquidity, no
trade-history maker achieves every desideratum at once [@abernethy2014vpm],
and liquidity selection itself can be run as online learning
[@nueve2026adaptiveliquidity; @nueve2025smooth].

## 5. The characterization

A predictor is a market when its cost has finite arbitrage depth relative
to affordable friction; convexity is the $f = 0$ special case. Within the
coherent class, non-convexity spends expressiveness, the contact set in
place of a full quoting range, and spends nothing else against rational
flow. What the market form adds to a predictor is not new estimates but
three things the estimation literature obtains otherwise: tuning constants
become prices set by competition rather than formulas, composition becomes
an algebra (merge as infimal convolution, chains as products), and the
procedure becomes robust to strategic data, since every source pays to move
the state. A market is the incentive closure of a predictor; a predictor is
a market with the strategyproofing stripped out.

## 6. Open problems

*Arbitrage depth as a capacity measure.* The minimum viable spread of a
trained landscape is computable; whether it tracks quantities learning
theory already names (sharpness, mode connectivity) is open.

*Two-agent pumps.* Adversarial-training oscillation resembles a money pump
between path-dependent agents, and gradient penalties resemble
spread-widening; a precise statement is needed or the analogy retired.

*Noise as subsidy.* Minibatch noise lands states off-contact and the next
informed step recoups the gap; whether SGD noise plays the economic role of
noise traders awaits a model.

*Books around binary events.* The multimodal-quote reading of book holes
(§3) is testable against limit-order data around court rulings and FDA
decisions, where bimodal implied densities are already documented
[@clark2017brexit].

## References

::: {#refs}
:::
