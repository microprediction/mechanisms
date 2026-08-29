# Markets in Series

### Filtering, belief propagation, and arbitrage as the loop correction

Peter Cotton · *Working draft v0.1* · August 28, 2026

---

## Abstract

Parallel composition of cost-function market makers is settled algebra:
merging is infimal convolution, liquidities add, and for quadratic makers
the merge is Gaussian precision fusion exactly. This note develops the
serial operator. Factorize a joint distribution by the chain rule and run
one market per conditional factor, each opening when its conditioning
information freezes and settling in cascade through the graph. For
linear-Gaussian structure the correspondence with inference is exact:
alternating a model step with a market step reproduces the Kalman filter,
because the market step is the Kalman update, and market messages on a tree
compute exact posterior marginals, belief propagation with trades as
messages. On loopy graphs the pathology of belief propagation materializes
as arbitrage: pairwise quotes around a cycle that admit no joint
distribution form a matrix with a negative eigenvalue, handing any trader a
bundle with negative price and nonnegative payoff, so arbitrageurs are the
loop correction. The two operators are the two operations of the inference
semiring, parallel the product and serial the sum, and exactness for
Gaussians has a single source: the family on which sum-product and min-sum
agree. The prior art is heavy and credited; the serial architecture, the
market-implemented filter, and the loopy-arbitrage certificate appear to be
the unoccupied residue.

---

## 1. Two operators, one semiring

Merging market makers that quote the same quantity is infimal convolution
of their costs: conjugates add, liquidity adds [@bhaskara2023general], and
the composition law is the risk-sharing algebra [@barrieu2005inf]. For
quadratic makers the merged clearing price is the precision-weighted mean
of quotes and the merged price impact the inverse of summed precisions:
parallel composition is Gaussian fusion. This note is about the other
operator. A model is a factorization,

$$P(\text{everything}) \;=\; \prod_{\text{nodes}} P(x_i \mid
\text{parents}(x_i)),$$

and serial composition runs one market per factor, each pricing its
conditional given what is upstream.

That there are exactly two operators is not an accident of taste.
Message-passing inference is an algorithm over a commutative semiring
[@aji2000gdl]: one operation combines evidence about a variable (the
product), one moves evidence between variables (the sum), and sum-product,
max-product and min-sum are the one algorithm over different semirings. The
market algebra is that pair: parallel merge is the product, densities
multiplying as precisions add, and serial propagation is the sum. The
parallel operation wears two costumes, addition of conjugates and infimal
convolution of costs, because inf-convolution is convolution in the
min-plus semiring and the Legendre transform is its Laplace transform
[@litvinov2007idempotent]: the convex machinery of this and the companion
notes is the tropical shadow of the probability calculus. Exactness in
everything Gaussian below has one source: log-quadratics are the family on
which the sum-product and min-sum semirings coincide, so a market whose
traders optimize agrees with an inference engine that integrates. Off the
family they diverge by the Laplace-approximation gap, which is §6's open
question.

## 2. One market per factor

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

## 3. The market step is the Kalman update

Take the linear-Gaussian chain $x_{t+1} = a x_t + \varepsilon_t$,
$y_t = x_t + \eta_t$, and alternate two steps: a *model step*, propagating
the current belief through the dynamics in mean-precision form, and a
*market step*, merging the propagated belief with an observation maker
quoting $y_t$ whose capital is the observation precision.

**Proposition 1.** *The alternation is the Kalman filter
[@kalman1960filtering]: after each observation the market state equals the
filtered mean and variance exactly.*

**Proof.** The model step is the standard prediction of mean and variance.
By the fusion identity, merging quadratic makers adds precisions and
precision-weights means, which is the information-form measurement update;
the covariance-form update follows by the usual algebra. $\blacksquare$

Prediction is computation; correction is a market; capital decides how
hard the data pulls. The nearest published object is the Bayesian market
maker of @brahma2012bayesianmm, whose trade update is a scalar Gaussian
measurement update with covariance inflation for jumps; the filtering
reading, and any alternation with a dynamic model, appear unstated in the
literature.

## 4. Trees: messages are trades

On a Gaussian chain with an observation at every node, the posterior at an
interior node is the merge of three sources: the forward message (the
filter of §3 run up to the node), the local observation, and the backward
message (downstream observations pulled back through the dynamics, each
pull-back a reparametrization and each combination a precision addition).

**Proposition 2.** *All three messages are market operations, and the
merged posterior equals the brute-force conditioning of the joint at every
node: Gaussian belief propagation [@pearl1988probabilistic] with trades as
messages, exact on trees.*

The companion repository verifies the identity to ten digits. The contrast
with the deployed combinatorial engines is architectural: there the
junction tree is the pricing algorithm inside one joint market
[@sun2012junction]; here the messages pass between venues, and the
graph's edges are market boundaries.

## 5. Loops: cycle inconsistency is arbitrage

On loopy graphs belief propagation double-counts and its fixed points
drift. The market version of the pathology is sharper and self-punishing.

**Proposition 3.** *Assemble pairwise correlation quotes around a cycle
into a quote matrix $P$ with unit diagonal. The quotes admit a joint
distribution iff $P \succeq 0$. Otherwise, with $w$ the eigenvector of a
negative eigenvalue $\lambda$, the bundle with weights $ww^\top$ has price
$w^\top P w = \lambda < 0$ and payoff $(w^\top x)^2 \ge 0$: a sure profit
of at least $\lvert\lambda\rvert$ per unit.*

This is the second-moment coherence condition, prices of products must
form a PSD matrix [@daspremont2005market], read as cycle consistency. The
lineage of "arbitrageurs enforce coherence" is long and must be owned:
coherence is no-arbitrage [@nau1991arbitrage]; @pennock1996marketbayes
built arbitrageur agents enforcing the additivity identities of a
Bayes-net economy, with equilibrium prices equal to the network's
probabilities and distributed bidding as distributed inference; the
combinatorial literature removes the arbitrage algorithmically over the
marginal polytope [@kroer2016arbitrage]; and Polymarket's arbitrageurs
have been measured doing the enforcement for eight figures
[@saguillo2025arbitrage]. What the loop reading adds is the object being
corrected: the inconsistency that degrades loopy belief propagation is,
in a market, not a numerical nuisance but free money, and the flow that
harvests it pushes the quotes back toward the cone. Whether the corrected
fixed point is the true marginal, a Bethe-like surrogate, or something the
liquidity profile selects is open.

## 6. Discussion

Three structures elsewhere in this project are serial compositions:
boosting is sequential residual markets, each stage trading what the
composite so far got wrong; a microprediction supply chain is a sequence
of markets each adding value to an intermediate product; and Fermi-style
decomposition of a question into a chain of small markets is the
factorization run by hand. The equilibria-as-graphical-models
correspondence is older than all of them [@pennock1996marketbayes;
@storkey2011machine]; what the serial operator contributes is the
mechanism version: separate venues, an opening rule, cascading
settlement, and exactness theorems where the Gaussian family permits them.

The open question underneath is the semiring gap. A market of optimizing
traders natively computes min-plus, so off the Gaussian family it prices
the max-product posterior rather than the sum-product marginal, and the
discrepancy is the Laplace-approximation gap. Risk aversion is an
exponential tilt, which suggests it interpolates between the two
semirings; whether a risk-averse trading population prices the marginal,
the mode, or a temperature in between is the right next theorem, and it
would say precisely which inference algorithm a market is.

## References

::: {#refs}
:::
