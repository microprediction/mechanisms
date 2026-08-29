# Predictors as Markets

### One maker, makers in parallel, markets in series

Peter Cotton · *Working draft v0.11* · August 29, 2026

---

## Abstract

There are two algebraic reasons markets appear inside learning: convex
duality turns optimization into trading, and graphical factorization turns
inference into networks of local markets. We develop both. For one maker,
non-convexity does not obstruct coherence: no-arbitrage is a chord
condition, rational flow trades the biconjugate, and a proportional fee of
at least the arbitrage depth (the worst per-unit excursion of chord slopes
beyond the payoff hull) restores no-arbitrage, the fee being exactly a
bid-ask spread by conjugation. For makers in parallel, combining
fee-bearing makers is an infimal convolution solved by one monotone
clearing-price root-find with sparse fills, and the aggregate supply curve
is a consolidated limit order book. For markets in series we separate
three things usually run together: a min-plus routing algebra, in which
parallel merge multiplies factors and cheapest routing eliminates
variables for arbitrary potentials; its market implementation, which we
exhibit for quadratic makers on unrestricted real securities, where the
alternation of model and market steps is exactly the Kalman filter and
market messages give exact tree marginals; and probabilistic sum-product
inference, which the routing algebra reproduces on log-quadratic families,
where partial minimization equals marginalization up to a constant. A
finite-state chain runs Viterbi in the routing algebra; we do not claim a
coherent cost-function implementation of arbitrary finite-state kernels.
A cost-based predictor admits a coherent market implementation if and only
if its arbitrage depth is dominated by the permitted friction; friction
extends the construction beyond convexity, and separation turns
feasibility violations into portfolios that pay their discoverer. The
resulting object is the incentive closure of the predictor: the same
operator, with every information source paying to move the state.

---

## 1. Many predictors are already markets

A cost-function market maker posts a potential $C$ over its inventory $q$
of outstanding shares and charges $C(q+s) - C(q)$ for a fill of $s$
shares; prices are gradients of $C$, and when $C$ is convex the standard
theory applies [@hanson2003combinatorial; @abernethy2013efficient]. A run of published identities makes many learning
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
value with capital equal to its precision (inverse variance), the estimate
their merge, since merging makers is the infimal convolution of their
costs and liquidity, the inverse of price impact, adds
[@bhaskara2023general; @barrieu2005inf]. And a proximal step is a
trade against a fee-bearing maker: the proximal operator of
$f\lvert\cdot\rvert$ is the soft-threshold, which is the optimal response
to a proportional fee (Lemma 4 below), and the prox of any convex $g$ is
the response to a unit quadratic maker charging $\tfrac12 s^2 + g(s)$,
since $\operatorname{prox}_g(x) = \arg\max_s \{x s - \tfrac12 s^2 -
g(s)\}$; the quadratic term is the maker's own curvature, and without it
the trader's problem is a bare conjugate, unbounded for
$\lvert\mu\rvert > f$ in the $\ell_1$ case. Lemma 4 carries the base cost
$g_q$ for exactly this reason. The two canonical penalties are then
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
in section 8. Section 12 states the characterization.

Two payoff regimes appear and must not be conflated. In the *bounded*
regime a scalar security settles at $\varphi(\omega) = \omega \in [-1,1]$,
vector statements substituting the convex hull of payoff vectors; this is
where Proposition 1 bites, where bounded worst-case loss is meaningful,
and where sections 2–5 live. In the *unrestricted* regime the security
settles at a real-valued quantity with payoff hull $\mathbb{R}$, so the
chord condition is vacuous for finite positions and a quadratic maker
$C(q) = mq + q^2/(2\lambda)$, whose chord slopes are unbounded, is
admissible; this is the estimation regime of the Gaussian fusion, Kalman
and tree results (§§6, 10–11) and of Proposition 13. Coherence statements
transfer between the regimes only through the hull that defines them.
Reference implementations and numerical theorem tests accompany the paper
in the `mechanisms` repository.

## 2. Coherence is a chord condition

**Proposition 1 (no-arbitrage without convexity).** *For a
path-independent maker (one whose charge depends only on the inventory
endpoints) with cost $C$, no convexity assumed, there is no
outcome-independent profitable fill from any state if and only if every
chord slope $[C(q+s) - C(q)]/s$ lies in the convex hull of payoffs: for
all $q, s$,*

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
(i) has every attained optimal fill landing on the contact set
$\{C = \hat C\}$, an optimum existing whenever $x \mapsto C(x) - \mu x$
attains its infimum (for instance when it is coercive); and
(ii) earns maximal expected profit*

$$\Pi_C(q,\mu) \;=\; \Pi_{\hat C}(q,\mu) + g(q)$$

*as an identity of extended-real suprema, the envelope profit plus the gap
at the starting state.*

**Proof.** $\sup_s \mu s - [C(q+s) - C(q)] = \sup_x [\mu x - C(x)] - \mu q
+ C(q)$. Since $C \ge \hat C$ with equality on the contact set, and a
maximizer of an affine function minus $C$ is a point where an affine
minorant touches $C$, hence touches $\hat C$, the supremum equals
$\sup_x [\mu x - \hat C(x)]$, attained on the contact set when attained at
all. Adding and subtracting $\hat C(q)$ gives (ii). $\blacksquare$

Attainment is a genuine hypothesis, not a formality. The chord-coherent
cost $C(x) = \arctan x$ has lower convex envelope the constant
$-\pi/2$ and hence empty contact set; at $\mu = 0$ the supremum $\pi/2$ is
approached only as $x \to -\infty$. Where the relevant contact sits at
infinity, the language below about flow landing on contact points and
off-contact states being transient describes a limit that no fill
realizes.

The maker behaves observationally like its convex envelope. Concave
stretches are unquotable intermediate states, holes in the maker's book of
standing quotes that rational flow jumps across. Whoever lands inside one (noise) overpays the
gap at the landing state, and by (ii) the next rational trader recoups it,
so the maker is a conduit keeping envelope differences over any
rational-to-rational span, and off-contact states are transient. The
trade-set analogue for constant-function market makers (CFMMs, the
decentralized-exchange design) is the canonical concave trading function:
an arbitrary invariant is behaviorally equivalent to a concave one
[@angeris2024geometry], with the limits of concavification mapped by
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
finance: under regularity conditions such as sticky paths or full
support, arbitrarily small proportional transaction costs restore
no-arbitrage for a large class of processes that are arbitrageable
frictionlessly, with a consistent price system inside the spread as
certificate [@guasoni2006transaction; @guasoni2010ftap]. In prediction markets the
precedent is fees sized to expected arbitrage profit restoring bounded loss
under privacy noise [@cummings2016privacy; @frongillo2018private]. Call
the smallest $\varepsilon$ for which the hypothesis of Proposition 3
holds the cost's *arbitrage depth*: the worst per-unit excursion of its
chord slopes beyond the payoff hull, and so the minimum viable half-spread
(the quoted spread being $2f$, since bid and ask sit at $m \mp f$). A
venue quoting a wide spread to cover a deeply incoherent cost is
arbitrage-free but uninformative in proportion: the market prices the
model's incoherence as uncertainty.

## 5. A linear fee is a bid-ask spread

Why a fee at all? A cost-function maker's charge telescopes, so round trips
are free: the maker earns nothing on uninformed flow and cannot recover
from volume its adverse-selection losses, the losses to better-informed
traders. That no state-dependent cost can
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
functional to the bid-ask band [@jouini1995transaction]. Without
convexity the exact description needs no conjugation at all:

**Lemma 5 (the no-trade interval, exactly).** *For any cost $C$ (no
convexity assumed), state $q$, and fee $f \ge 0$, write
$d_q(s) = [C(q+s) - C(q)]/s$ for the chord slope. A belief $\mu$ admits no
profitable trade if and only if*

$$\sup_{s<0} d_q(s) - f \;\le\; \mu \;\le\; \inf_{s>0} d_q(s) + f,$$

*so the set of no-trade beliefs is exactly this interval intersected with
the payoff hull. For differentiable convex $C$ both bounds equal $C'(q)$
and the interval is Lemma 4's band $[m - f, m + f]$. Write
$\Delta_q = \sup_{s<0} d_q - \inf_{s>0} d_q$. The untruncated interval is
non-empty if and only if $2f \ge \Delta_q$; if in addition $C$ is
chord-coherent, both bounds lie in the hull, so their midpoint does and
the intersection with the hull is non-empty under the same condition. If
$C$ is chord-coherent and Proposition 2's supremum is attained, then off
the contact set $\Delta_q > 0$.*

**Proof.** No profitable trade means $\mu s \le C(q+s) - C(q) +
f\lvert s\rvert$ for all $s$. Dividing by $s > 0$ gives $\mu \le d_q(s) +
f$; dividing by $s < 0$ reverses the inequality to $\mu \ge d_q(s) - f$;
together these are the displayed interval, whose non-emptiness is
$\sup_{s<0} d_q - f \le \inf_{s>0} d_q + f$. Under chord coherence both
one-sided bounds are limits of chord slopes and so lie in the hull, which
is convex, so the midpoint of a non-empty interval lies in the hull. For
the last claim, off the contact set Proposition 2 gives every belief a
strictly positive frictionless profit, so the $f = 0$ interval is empty.
$\blacksquare$

The hull intersection is not decoration. Without chord coherence
$C(q) = 100q$ has $\Delta_q = 0$, so the untruncated interval $\{100\}$
is non-empty for every $f \ge 0$, yet no admissible belief in $[-1,1]$
declines to trade: the maker is arbitraged from every state.

So friction does not merely widen an existing spread: a spread with
$2f \ge \Delta_q$ fills the quote hole that non-convexity created, and a
state inside the hole becomes tenable. Non-convexity creates holes,
frictionless rational flow jumps them (§3), and a large enough spread
stabilizes points inside them. For $C(q) = a\sin q$ at $q = 0$, a state
sitting $a$ above its flat envelope, the belief $\mu = 0$ admits no
profitable trade exactly when $f \ge a$, though every belief profits there
frictionlessly.

## 6. Makers in parallel

Let makers $i = 1..n$ hold inventories $q_i$ with costs $C_i$ that are
differentiable and strictly convex, with $C_i'$ onto the price range of
interest so that $(C_i')^{-1}$ is defined there; fees $f_i$ of their
choosing; and write $m_i = C_i'(q_i)$ for maker $i$'s marginal price and
$\mathrm{ask}_i = m_i + f_i$, $\mathrm{bid}_i = m_i - f_i$ for its
quotes, as in Lemma 4. Without differentiability and strict convexity the
supply curves below are set-valued and the statements hold with
$(C_i')^{-1}$ read as a subgradient correspondence.

One normalization is needed before the merged object is a maker. The
infimal convolution of the incremental costs need not vanish at zero: for
two zero-fee quadratic makers $g_i(s) = m_i s + s^2/(2\lambda_i)$,

$$(g_1 \,\square\, g_2)(0) \;=\; -\frac{(m_1-m_2)^2}
{2(1/\lambda_1 + 1/\lambda_2)} \;<\; 0$$

whenever the makers' quotes differ, because zero net external demand
still admits a profitable internal cross-trade. We therefore assume the
component inventories have been cleared against each other and the
effective cost normalized to vanish at the resulting origin; otherwise
evaluating at $\Delta = 0$ repeatedly appears to pay the same internal
arbitrage again and again.

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
[@olivaresnadal2018robust]. A small trade routes to whichever maker
posts the best quote, and to that maker alone unless the best quote is
tied; a growing trade pushes that maker's fee-adjusted marginal price
through the next band and spills over, consuming makers in quote-price
order, which coincides with fee order only when their marginal prices
agree.

**Corollary 7 (zero fees).** *With $f_i \equiv 0$ the convolution reduces
to the fee-free merge: conjugate regularisers add, and for a perspective
family $C_b(q) = b\,C_1(q/b)$ liquidity adds,
$C_{b_1} \square C_{b_2} = C_{b_1+b_2}$ [@bhaskara2023general]. For
quadratic makers the merge is Gaussian fusion: the merged quote is the
precision-weighted mean of the makers' quotes, and the merged liquidity
is the sum of their liquidities (precisions).*

**Corollary 8 (the order book).** *The aggregate supply
$S(p) = \sum_i s_i(p)$ is non-decreasing, identically zero on
$\big(\max_i \mathrm{bid}_i,\ \min_i \mathrm{ask}_i\big)$ when that
interval is non-empty, flat wherever every maker's band covers $p$, and
smooth and strictly increasing wherever some maker is active.* Read
as a market: best bid and ask are the tightest quotes, $S(p)$ is the
cumulative quantity executable up to price $p$ with local depth $S'(p)$
where it exists, and large orders walk the levels. The aggregate of linear-fee makers is a consolidated
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
(i) coherence is preserved: if the chord slopes of $C$ lie in $[a, b]$,
so do those of $e_\lambda C$, for every $\lambda$; (ii) at any point where
two distinct minimizing branches coexist (a crossing), the downward jump
in marginal price is exactly the separation of the competing minimizers
divided by $\lambda$, at most $D/\lambda$ when all competing minimizers
lie in an interval of diameter $D$; (iii) for the symmetric double well
$C(y) = \alpha \min(\lvert y - c\rvert, \lvert y + c\rvert)$ with
$\alpha < 1$, the jump at $x = 0$ equals $2\alpha$ while
$\lambda\alpha < c$ and $2c/\lambda$ thereafter, so it is constant in the
depth until the co-quoter is deep enough to reach across the wells. The
bound in (ii) measures crossing defects only: the
envelope can be smoothly non-convex with a unique minimizer everywhere,
as for $C(y) = a \sin y$ with $\lambda a < 1$, where
$(e_\lambda C)''(x) = C''(y^*)/(1 + \lambda C''(y^*))$ is negative
wherever $C''$ is.*

**Proof.** The merge is infimal convolution, and inf-convolution with the
quadratic is the Moreau envelope [@rockafellar1970convex]. For (i), no
differentiability is needed: for $h > 0$, substituting $y = z + h$,

$$e_\lambda C(x+h) = \inf_z \Big\{ C(z+h) + \frac{(x-z)^2}{2\lambda}
\Big\} \;\le\; e_\lambda C(x) + b\,h,$$

since $C(z+h) - C(z) \le b h$ for every $z$; the reverse substitution
gives the lower bound $a h$, so every chord slope of $e_\lambda C$ lies
in $[a, b]$. For (ii), at a crossing $x_0$ with competing minimizers
$y_1^* < y_2^*$ the branch slopes are $(x_0 - y_i^*)/\lambda$, so the
drop is exactly $(y_2^* - y_1^*)/\lambda$. For the smooth case,
differentiate the first-order condition $y^* = x - \lambda C'(y^*)$ to
get $dy^*/dx = 1/(1 + \lambda C''(y^*))$ and hence the displayed
curvature. For (iii), the well's competing minimizers sit at
$\pm\lambda\alpha$ while $\lambda\alpha < c$, giving separation
$2\lambda\alpha$ and jump $2\alpha$, and at $\pm c$ once the co-quoter
reaches the wells, giving $2c/\lambda$. $\blacksquare$

Part (i) is the durable statement: adding depth reshapes a non-convex
maker without ever creating arbitrage, at any $\lambda$, so what the deep
co-quoter buys is expressiveness. Crossing kinks close at rate
$1/\lambda$ only where the competing minimizers stay within a
$\lambda$-independent diameter; the double well shows what happens
otherwise, its jump holding at $2\alpha$ until the depth exceeds
$c/\alpha$. Smooth concave stretches lie outside the bound's scope
entirely. Here $\lambda$ is the
co-quoter's liquidity: its price impact is $1/\lambda$, so large
$\lambda$ means a deep book, not a strong pull.

The two repairs are the two market primitives again: friction ($\ell_1$,
the fee) and participant depth ($\ell_2$, capital), lasso and ridge. The
identification maps the penalty terms, not the statistical procedures; it
is a dictionary of primitives, not an equivalence of estimators.

## 7. Self-set fees and adaptivity

Nothing requires the fees to be administered. Each maker may quote its own
$f_i$, and routing then disciplines the posted menu: a maker quoting
inside the prevailing spread becomes the best quote and takes the flow
first, a maker quoting behind the best price receives nothing until the
book in front of it is consumed, and a maker quoting too tight is picked
off by informed flow.

The discipline is a property of Lemma 6's routing, not of its
minimization. The infimal convolution minimizes over allocations $s_i$
with the fees held fixed; it does not minimize over the fees, and nothing
in it constitutes a game in which makers choose $f_i$. What the surviving
fee would be, and whether it is the competitive adverse-selection charge
of classical microstructure [@glosten1985bidask; @biais2000competing],
requires the strategic quote game, which is not solved here.

Optimization already pays these frictions: a proximal step charges
$\lVert\Delta\theta\rVert^2/(2\eta)$ per move, a trust region is an
infinite fee outside a band, weight decay is a zero-quoting participant.
In market language the stabilizers of non-convex training are the
market's two repairs: friction prices chord excursions (Proposition 3)
and depth fills holes (Proposition 9); non-convexity by itself is not
exploitability (§2).
Path-dependent optimizers correspond to makers whose quotes depend on flow
history, the adaptive-liquidity territory where path independence is
deliberately traded away: no maker combines path independence, translation
invariance (complete-bundle prices summing to the sure payoff) and
liquidity sensitivity [@othman2013practical], the adaptive
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
\varphi_2(y,z)]$, the min-plus product of kernels (two-argument
potentials). The Legendre transform is
not a computational trick here but the change of representation between
the two composition laws.

**Proposition 10 (the routing algebra).** *(i) Parallel: merging makers
multiplies factors, since $(C_1 \square C_2)^* = C_1^* + C_2^*$ and
potentials add exactly when densities multiply. (ii) Serial: the cheapest
route to a terminal position through a chain of stage kernels costs
$\inf_{z_1,\dots,z_{n-1}} \sum_i \varphi_i(z_{i-1}, z_i)$, the min-plus
product of the kernels. (iii) On log-quadratic (Gaussian) families,
partial minimization of a potential equals its marginalization up to an
additive constant independent of the retained variables, so the min-plus
messages are the sum-product messages.*

**Proof.** (i) is the conjugate-sum identity together with
$-\log(p_1 p_2) = \varphi_1 + \varphi_2$. (ii) is the definition of the
cheapest route: the trader chooses intermediate exposures to minimize the
sum of stage costs. For (iii), write a jointly quadratic potential
$q(x,y)$ with positive definite $y$-block $Q_{yy}$; completing the square,
$-\log \int e^{-q(x,y)}\,dy = \min_y q(x,y) + \tfrac12\log\det(Q_{yy}/2\pi)$,
the Schur-complement identity, and the constant does not depend on $x$.
$\blacksquare$

Three things must be kept apart here, and the rest of the paper keeps
them apart. The *routing algebra* is Proposition 10: min-plus identities
that hold for arbitrary potentials, (ii) being little more than the
definition of a cheapest route, and needing no convexity, as the
generalized distributive law does not [@aji2000gdl]. *Sum-product
inference* is what (iii) recovers on the log-quadratic family. A *market
implementation* is a third thing, and strictly more: it requires each
kernel to be posted as a path-independent maker with named securities and
settlements, chord slopes inside the payoff hull, and the property that
self-interested trades perform the minimization rather than a solver
performing it. Proposition 10 does not supply that, and this paper
exhibits it only for quadratic makers on unrestricted real securities,
in §§10–11.

The implementability condition itself is the one §2 identified: a kernel
is tradeable as a path-independent maker when its cost's chord slopes
respect the payoff hull. For closed convex $C$ this reads dually as the
potential $\varphi = C^*$ having effective domain inside the hull, so
that $C = \varphi^*$ has subgradients there; a non-convex coherent maker
is not its own biconjugate, $C^{**} = \hat C \ne C$, and instead shares
this conjugate representation with its envelope, which is exactly the
observational equivalence of §3. Convexity is the separate, further
property of monotone information incorporation, and it is what gives the
clean dual representation, not coherence.

**Corollary (the serial algebra runs Viterbi).** *Take finite state
spaces and stage potentials $\varphi_t(i,j) = -\log P(X_t = j \mid
X_{t-1} = i) - \log P(y_t \mid X_t = j)$. The cheapest route of
Proposition 10(ii) is $\min_{x_{1:T}} \sum_t \varphi_t(x_{t-1}, x_t)$:
the Viterbi decoding of the hidden Markov model, exactly, with no
Gaussian structure anywhere. The statement is about the serial algebra;
a coherent cost-function implementation of arbitrary finite-state
kernels is not claimed.* The three worked examples of this paper
now form a progression: least squares is parallel composition and is
implemented by quadratic makers, Viterbi is serial min-plus composition
in the routing algebra alone, and the Kalman filter of §10 is the
Gaussian intersection, where the serial computation is implemented by
makers and is also Bayesian.

**Principle (the Gaussian intersection).** *Log-quadratic families are a
family on which probabilistic inference and min-plus optimization
coincide, and one closed under both compositions: optimizing traders
compute min-plus, an inference engine integrates, and Proposition 10(iii)
says the two agree here up to constants that cancel in every price.* The
Kalman and tree-marginal propositions below live on this intersection and
are not evidence for a universal serial thesis. The coincidence is not
peculiar to quadratics: for $q(x,y) = x^2 + (y-x)^4$, minimization and
marginalization also differ by an $x$-independent constant, so
"log-quadratic" is sufficient and not necessary. Characterizing the
families on which the two semirings agree, presumably by closure under a
rich enough class of products, affine maps and eliminations, is open.

Away from the intersection, min-plus elimination returns the max-marginal
$x \mapsto \inf_y \varphi(x,y)$, the profile potential, rather than the
sum-product marginal; the discrepancy is the Laplace-approximation gap.
Note the max-marginal is a function, not the MAP point, which is its
argmin. Risk aversion is an exponential tilt and plausibly interpolates
between the two semirings; §12 poses this as the open question of which
inference algorithm a risk-averse market runs.

## 9. One market per factor

The locality that makes the factorization tradable is Hanson's modularity:
in a combinatorial logarithmic market scoring rule (LMSR) a bet on
$A \mid B$ moves that conditional and provably nothing else, uniquely
among market scoring rules
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
conditioning information freezes, that is, when the upstream quantities it
conditions on stop changing; it conditions on upstream quotes while
upstream is live, and re-references when upstream settles. Settlements
cascade through the directed acyclic graph like dataflow, and the schedule
of markets is an unrolling of the graph. Observable nodes carry settled markets; a latent
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
interior node $t$ is the merge of three sources: the *predictive* forward
message $p(x_t \mid y_{1:t-1})$, which is the filter of §10 run up to but
not including $y_t$; the local observation $p(y_t \mid x_t)$; and the
backward message $p(y_{t+1:T} \mid x_t)$, downstream observations pulled
back through the dynamics, each pull-back a reparametrization and each
combination a precision addition. Taking the filtered posterior through
node $t$ as the forward message would count $y_t$ twice.

**Proposition 12 (trades as messages, exact on trees).** *All three
messages are market operations, and their merge equals the conditioning
of the joint at every node: Gaussian belief propagation
[@pearl1988probabilistic] with trades as messages.*

**Proof.** In information form the joint of a Gaussian chain factorizes
into node and edge potentials. Induct on the chain: the predictive
message at $t$ is the model step applied to the merge at $t-1$, a
precision-scaled reparametrization; merging it with the local
observation's potential adds precisions (Corollary 7), which is the
information-form conditioning identity; the backward recursion is the
same argument run on the reversed chain, its pull-backs the Schur
complements of Proposition 10(iii). Since each merge is exactly the
information-form update and the three potentials multiply to the
conditional, the merge is the posterior. $\blacksquare$

The repository verifies the identity to ten digits. The contrast with the
deployed combinatorial engines is architectural: there the junction tree
is the pricing algorithm inside one joint market [@sun2012junction]; here
the messages pass between venues, and the graph's edges are market
boundaries.

On loopy graphs belief propagation double-counts and its fixed points
drift. In a market the same inconsistency is self-punishing.

**Proposition 13 (quote inconsistency is a sure profit).** *Let $x$ be a
vector of unrestricted real random variables and let separate venues
quote securities paying the products $x_i x_j$ for every pair, the
diagonal quoted at one, at fixed linear prices executable in the size
traded; assemble the quotes into a matrix $P$ with unit diagonal. The
quotes admit a joint distribution iff $P \succeq 0$ (a Gaussian with
covariance $P$ witnesses sufficiency). Otherwise let $w$ be a unit
eigenvector of a negative eigenvalue $\lambda_{\min}$ and buy $w_i^2$
units of security $ii$ and $2 w_i w_j$ units of each security $ij$,
$i < j$. The bundle costs $w^\top P w = \lambda_{\min} < 0$ and pays
$(w^\top x)^2 \ge 0$: a sure profit of $\lvert\lambda_{\min}\rvert$. On three coordinates the quoted
pairs are exactly the edges of a triangle, so the loopy reading is
literal.*

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

Two boundaries of the proposition mark the general picture. If only a
subgraph of pairs is quoted, the realizability condition is not positive
semidefiniteness of a filled-in matrix but PSD-completability of the
partial one [@grone1984completions], and the arbitrage bundle must be
supported on quoted pairs. If the coordinates are constrained, say binary
$x_i \in \{\pm 1\}$, the feasible quote set is the cut polytope, a strict
subset of the PSD body [@deza1997cuts], so quotes can be positive
semidefinite yet jointly unrealizable. In every case the coherent set is
the convex hull of attainable payoffs, and Proposition 14 below is the
universal form.

The scope of Proposition 13 is in any case one class of consistency
failure: locally
quoted beliefs that cannot be embedded in any joint distribution, detected
here at second moments. Not every loopy-propagation error takes this form,
but the class it covers is, in a market, free money, and the flow that
harvests it pushes the quotes back toward the cone. Whether the corrected
fixed point is the true marginal, a surrogate of the kind loopy
propagation converges to, or something the liquidity profile selects is
open.

The PSD cone is not special. Coherent price sets are convex (they are
convex hulls of payoff vectors, or their conic images), and separation
turns every exterior point into a trade:

**Proposition 14 (arbitrage is separation).** *Let $K$ be the closed
convex hull of the attainable payoff vectors (or its image under a linear
security map), so that $\inf_{z \in K} \langle y, z\rangle \le \langle y,
\varphi(\omega)\rangle$ for every outcome $\omega$ and portfolio $y$, and
let the venue quote fixed linear prices $x$, executable in the size
traded. If $x \notin K$, any separating functional $y$ with
$\langle y, x\rangle < \inf_{z \in K} \langle y, z\rangle$ is a portfolio
whose price is strictly less than its realized payoff in every state: a
sure profit. Proposition 13 is the instance $K = \{P \succeq 0\}$.*

*For a nonlinear cost-function maker the conclusion is local. There $x$
is the current gradient and a trade of size $\delta y$ costs
$\delta \langle y, x \rangle + o(\delta)$, since the marginal price moves
as the trade is filled, so a strict separating margin yields sure profit
for all sufficiently small $\delta$, at a rate rather than in the size of
$y$; market impact and fees then set how large a position the certificate
supports.*

The proof is the separating-hyperplane theorem read as a trade
[@nau1991arbitrage; @daspremont2005market]; the payoff-hull hypothesis is
what upgrades the separation certificate to an arbitrage, and for an
abstract consistency set the construction yields the certificate only. In
an ordinary numerical method an infeasibility is a residual to be driven
down by the algorithm; in a market it is a payoff, and whoever finds it
is paid to act as the separation oracle. Arbitrageurs are decentralized
separation oracles, and the friction of §4 sets the tolerance below which
infeasibility is allowed to persist.

## 12. Closing

The characterization is for the cost-based class. Call a predictor
*cost-based* if it is specified by a path-independent potential $C$ over a
security inventory, and call friction of size $f$ *permitted* if the
mechanism may charge up to $f$ per unit traded. A cost-based predictor
admits a coherent market implementation if and only if its arbitrage depth
is dominated by the permitted friction; chord coherence is the $f = 0$
special case, and convexity is the separate, further property of monotone
information incorporation. Three words are kept apart throughout:
*coherence* (chord slopes stay in the payoff hull), *convexity* (marginal
prices respond monotonically to flow), and *expressiveness* (the quoting
range, which non-convexity restricts to the contact set). Within the
coherent class, non-convexity spends expressiveness and spends nothing
else against rational flow.

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
constraints, plus payment to whoever holds one. Duality alone supplies
the middle column of

$$\begin{array}{lll}
\text{computation} & \text{certificate} & \text{payment}\\
\text{primal solve} & \text{dual multiplier} & \text{none}\\
\text{verified computation} & \text{proof of violation} & \text{none or fixed}\\
\text{incentive closure} & \text{separating portfolio} & \text{the certificate's value}
\end{array}$$

and the closure adds the third: a marketized computation is self-policing
in proportion to the value of its own errors, with friction the exchange
rate between tolerated error and the cost of participation.

The definition has neighbors in four literatures, and each holds one
column. Markets that compute are old: a combinatorial auction cleared
airport slots by integer programming [@rassenti1982slots], electricity
markets pay the dual variables of the dispatch program as nodal prices
[@schweppe1988spot], and market-oriented programming solved distributed
allocation by computing competitive equilibria of artificial economies
[@wellman1993market]. Algorithmic mechanism design asks when optimization
is compatible with incentives [@nisan2001algorithmic], and the
classification of what can be computed incentive-compatibly was posed as
a program by @feigenbaum2002distributed. The middle column carries a
lower bound: any communication protocol realizing an efficient allocation
must reveal supporting prices [@nisan2006communication], so certificates
are not optional. The third column exists in practice without the
duality: fraud-proof systems pay whoever exhibits an invalid state
transition [@teutsch2019truebit; @kalodner2018arbitrum], for discrete
transitions rather than convex programs. And the friction bound has an
incentive-free ancestor: the auction algorithm's
$\varepsilon$-complementary slackness permits suboptimality at most
$n\varepsilon$, vanishing for $\varepsilon < 1/n$ on integer data
[@bertsekas1992auction]; Proposition 3 is the same shape of bound with
the $\varepsilon$ charged to traders. The connective tissue, the closure
operation itself, serial composition of mechanisms, and the
identification of violation certificates with separating portfolios,
appears unoccupied.

Nothing in the definition mentions prediction. The serial law is the
min-plus kernel composition of dynamic programming, control, and shortest
paths; the parallel law is the additive combination of local potentials;
and the closure of the whole construction is a boundary statement:

**Proposition 15 (the effective boundary maker).** *Let a network of
makers have boundary variables $b$, internal variables $h$, and local
costs $C_e$, each closed, proper and jointly convex, with
$\sum_e C_e(b, \cdot)$ coercive in $h$ for each $b$ in the boundary
domain, and define $C_{\mathrm{eff}}(b) = \inf_h \sum_e C_e(b, h)$. Then
$C_{\mathrm{eff}}$ is convex, proper and finite there, the infimum is
attained, and the elimination may be performed variable by variable in
any order. For quadratic costs with positive definite internal block each
single-variable elimination is a Schur complement, and on a chain the
computation is the dynamic program of Proposition 10(ii). If in addition
the chord slopes of $C_{\mathrm{eff}}$ lie in the boundary payoff hull,
the network is one coherent effective maker at its boundary.*

**Proof.** Partial minimization of a jointly convex function is convex,
coercivity gives attainment and rules out the value $-\infty$, and
iterated infima may be taken in any order [@rockafellar1970convex]; the
quadratic case is the completion of the square in the proof of
Proposition 10(iii), which needs the internal block invertible.
$\blacksquare$

Both hypotheses earn their place. A single local term linear and
decreasing in an unconstrained internal variable drives
$C_{\mathrm{eff}} \equiv -\infty$; and convexity of $C_{\mathrm{eff}}$
says nothing about coherence against a bounded boundary hull, which is
why the last clause is separate.

This is the operation that reduces resistor networks to terminal
impedances, eliminates latent Gaussian variables, and takes Schur
complements: internal competition disappears into an external price law,
as internal nodes disappear into a Dirichlet-to-Neumann map. One caution
attaches to "any order": the effective boundary market is
elimination-order invariant, but the computational cost of producing it
is not, and the blowup of intermediate scopes is the treewidth phenomenon
of variable elimination. Prediction is the application in which beliefs
and prices share units. We leave the general theory, including the
categorical formulation in which markets are min-plus kernels composed by
shared inventory, outside this paper's scope.

One principle deserves separation from the open problems it generates,
because it qualifies everything above.

**Principle (accessible arbitrage).** *Market coherence is relative not
only to friction but to the cost of discovering violations. Write $K$ for
the exactly coherent quotes (Proposition 1) and $K_f$ for those admitting
no sure profit net of a fee $f$ (Proposition 3). Let $\mathcal{A}$ be a
class of admissible certificate-search procedures, and for a quoted
configuration $x$ let $V_f(A(x); x)$ be the *maximum realizable* profit
from the portfolio a procedure $A$ finds, computed against the venue's
actual books and net of trading friction, under the position or capital
constraint the arbitrageur faces, and let $C_A(x)$ be the search cost.
The accessible-coherence set is*

$$K_{\mathcal{A}, f} \;=\; \Big\{\, x :\ \sup_{A \in \mathcal{A}}
\mathbb{E}\big[\, V_f(A(x); x) - C_A(x) \,\big] \le 0 \,\Big\}.$$

*With nonnegative search costs, $K \subseteq K_f \subseteq
K_{\mathcal{A}, f}$: a violation persists whenever every available
separation procedure costs at least as much, net of friction, as the
certificate it finds is worth.*

Both features of $V_f$ are load-bearing. The supremum ranges over
procedures, not certificates: two procedures can find the same portfolio
at very different costs, so coherence is relative to a class of
arbitrageurs. And the profit must be the realizable one. A separating
functional may be scaled by any positive constant, so against fixed
linear prices its nominal value is unbounded and no finite search cost
could ever leave a violation standing; it is the nonlinear book, the
finite executable depth, and the position constraint that make
$K_f \subsetneq K_{\mathcal{A}, f}$ possible at all. Equivalently one may
normalize certificates to $\lVert y \rVert \le 1$ and price them against
available depth. Instantiating $\mathcal{A}$ gives
polynomial-time coherence, bounded-budget coherence, and
latency-constrained coherence, and a venue can be arbitrage-free to
ordinary participants while arbitrageable to a better-equipped class.
Three nested notions of coherence result: exact (no sure-profit chord,
Proposition 1), frictional (violations below the spread survive,
Proposition 3), and accessible (violations too expensive to discover
survive). Proposition 14 addresses the existence of arbitrage; whether a
procedure in $\mathcal{A}$ can profitably prove $x \notin K$ is its
accessibility, a different property. The self-policing of the
marketization table is qualified accordingly: effective self-correction
is the error's value net of discovery cost and friction.

Open problems, in rough order of tractability:

*Which inference algorithm does a risk-averse market run?* Proposition 10
says risk-neutral routing computes min-plus exactly and sum-product only
on the Gaussian intersection. Risk aversion is an exponential tilt,
suggesting it interpolates between the two semirings; whether a
risk-averse trading population prices the marginal, the mode, or a
temperature in between would complete the correspondence.

*A serial market mechanism in general.* Proposition 10(ii) is routing
algebra and §§10–11 implement it only for quadratic makers on
unrestricted real securities. Give, for a useful class of kernels beyond
the quadratic, the securities, settlements and inventory variables of
each stage market, verify chord coherence stage by stage, and show that
self-interested trading performs the elimination. The interface between
the two compositions is part of the problem: parallel merge acts on
conjugates and serial routing on primal leg costs, so a factor graph
alternating the two needs an explicit change of representation at each
junction.

*Market representations of general prediction maps.* The
characterization above is confined to cost-based predictors. Define what
it means for an arbitrary prediction map $T$, from data sets to forecasts,
to possess a market representation, and give conditions on $T$ equivalent
to existence. A first conjecture: $T$ is representable when it factors
through a network of cost kernels whose boundary reduction is $T$, with
the chord bound applying kernel by kernel; path independence and the
chord condition would then be the image of the general condition in the
cost-based case.

*Accessible coherence, characterized.* Compute $K_{\mathcal{A}, f}$ for
concrete classes: polynomial-time arbitrageurs against cones whose
separation is hard, budgeted arbitrageurs against fee-bearing venues,
latency-constrained arbitrageurs against cascading settlements. And
characterize the mechanisms in which the stream of certificate payments
funds the ongoing computation.

*The equilibrium fee.* Section 7 argues discipline, not equilibrium. In a
flow model where most coefficients are zero and informed trades carry
signal plus noise, does the fee that survives undercutting reproduce the
universal threshold $\sigma\,(2\log p)^{1/2}$ of the shrinkage
literature?
If so, the regularization constant that statistics tunes by
cross-validation equals the adverse-selection cost of the data source.

*Loopy fixed points.* Under a concrete flow model, does
arbitrage-corrected quoting converge, and to what?

*Arbitrage depth as a capacity measure.* The arbitrage depth of a trained
model's loss landscape is computable; whether it tracks quantities
learning theory names (sharpness, mode connectivity) is open.

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
