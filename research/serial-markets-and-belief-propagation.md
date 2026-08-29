# Serial markets and belief propagation

*Status: a worked generalization with a calibrated novelty assessment.
Reference implementation:
[`serial_markets.py`](../mechanisms/serial_markets.py), theorem-tested in
[`test_serial_markets.py`](../tests/test_serial_markets.py).*

Parallel composition of makers is infimal convolution: multiple participants
on one quantity, liquidities adding
([liquidity-is-precision.md](liquidity-is-precision.md); Proposition 6 of
the [composition paper](../papers/composition-and-the-algebra-of-mechanisms.md)).
This note works out the other operator. Serial composition is the chain
rule: factorize the joint as $P(\text{everything}) = \prod P(\text{node}
\mid \text{parents})$ and run one market per factor, each pricing its
conditional given what is upstream. A model is a factorization; the markets
are its factors. For linear-Gaussian structure the correspondence is exact:
alternating a model step with a market step is the Kalman filter, market
messages on a tree compute exact marginals, and on loopy graphs the
inconsistency that breaks belief propagation materializes as arbitrage,
with arbitrageurs as the loop correction.

## 1. One market per factor

The design is as old as the combinatorial market: Hanson's modularity is
precisely the statement that in a combinatorial LMSR a trade on a
conditional quantity moves that conditional and nothing else, so the joint
decomposes into independently tradable factors. The serial operator takes
the factors as separate venues: market $i$ opens on
$P(x_i \mid \text{parents}(x_i))$, conditions on upstream quotes while
upstream is live, and re-references when upstream settles. Settlements
cascade through the DAG like dataflow, and a node's market opens when its
conditioning information freezes; the schedule of markets is an unrolling
of the graph.

Why exactly two operators. Message-passing inference is an algorithm over a
commutative semiring (the generalized distributive law of Aji & McEliece
2000): one operation combines evidence about the same variable (the
product), one moves evidence between variables (the sum), and the algorithm
is indifferent to which semiring supplies them. The market algebra is that
pair: parallel merge is the product (densities multiply, conjugates add,
precisions add, costs inf-convolve), and serial propagation is the sum. The
parallel operation wears two costumes because inf-convolution is
convolution in the min-plus semiring and the Legendre transform is its
Laplace transform (Maslov's idempotent analysis): the convex machinery of
these notes is the tropical shadow of the probability calculus. Exactness
throughout the Gaussian sections has one source: log-quadratics are the
family on which the sum-product and min-sum semirings agree, so the market
(whose traders optimize) and inference (which integrates) coincide; off the
family they diverge by exactly the Laplace-approximation gap, an open
question below.

Latent nodes get the treatment the fusion note gave covariance: observables
carry settled markets; a hidden state either settles through its observable
footprint (a market on a latent is a market on a functional of future
observables, scored through the model) or remains indicated but unsettled,
priced by whoever will warehouse the model risk.

## 2. Filtering: the market step is the Kalman update

Take the linear-Gaussian chain $x_{t+1} = a x_t + \varepsilon$,
$y_t = x_t + \eta$. The serial machinery alternates:

- *model step* (computed): propagate the belief through the dynamics in
  mean/precision form;
- *market step*: fuse the propagated belief with an observation maker
  quoting $y_t$ with capital equal to the observation precision.

By the fusion identity (merging quadratic makers adds precisions and
precision-weights means), the market step is the Kalman update, exactly,
and the alternation reproduces the classical filter to machine precision
([`test_serial_markets.py`](../tests/test_serial_markets.py)). Prediction
is computation; correction is a market; wealth decides how hard the data
pulls.

## 3. Trees: messages are trades

On a Gaussian chain with an observation at every node, the posterior of an
interior node is the fusion of three sources: the forward message (filter
up to the node), the local observation, and the backward message (each
downstream observation pulled back through the dynamics). All three are
market operations — propagation is a reparametrization, fusion is
precision addition — and the result equals the brute-force conditioning of
the full joint at every node (tested to $10^{-10}$). This is Gaussian
belief propagation with trades as messages, and on trees it is exact for
the same reason belief propagation is.

## 4. Loops: cycle inconsistency is arbitrage

On loopy graphs belief propagation double-counts and its fixed points
drift. The market version of the pathology is sharper. Assemble the
pairwise quotes around a cycle into a quote matrix with unit diagonal; if
the quotes admit no joint distribution the matrix has a negative
eigenvalue, and the bundle along the offending eigenvector has negative
price and nonnegative payoff $(w^\top x)^2$: a sure profit, the
correlation-triangle bound of §1 of
[covariance-market.md](covariance-market.md) read as a cycle-consistency
condition. Cycle inconsistency is not a numerical nuisance but free money,
and the flow that harvests it pushes the quotes back toward the PSD cone:
arbitrageurs are the loop correction
([`test_serial_markets.py`](../tests/test_serial_markets.py) exhibits the
certificate and its absence for consistent quotes). The lineage of that
sentence is old: coherence is no-arbitrage (de Finetti; Nau & McCardle
1991), and Pennock & Wellman (1996) built arbitrageur agents that enforce
the additivity identities of a Bayes-net economy in equilibrium; the
combinatorial-market literature instead removes the arbitrage
algorithmically (Kroer et al. 2016), and Polymarket arbitrageurs have been
measured doing the enforcement for real money (Saguillo et al. 2025). The
loop-specific form — non-PSD cycle quotes as the certificate, loopy belief
propagation as the pathology being corrected — appears unclaimed. Whether
the corrected fixed point is the true marginal or a Bethe-like surrogate
is open (§8).

## 5. Reframings

Three structures already in this repository are serial compositions.
Boosting is sequential residual markets, each stage trading what the
composite so far got wrong
([composing-mechanisms-conservation-and-boosting.md](composing-mechanisms-conservation-and-boosting.md)).
The microprediction supply chain is the economic name for the operator: a
sequence of markets each adding value to an intermediate product. And
Fermi-style decomposition of a large question into a chain of small
markets is the factorization run by hand.

## 6. Implementation

[`serial_markets.py`](../mechanisms/serial_markets.py) provides
`propagate` (the model step), `market_update` (the market step),
`market_kalman_filter` (their alternation), `chain_posterior` (tree
inference by market operations), and `cycle_arbitrage` (the loop
certificate). The tests verify Kalman equivalence to machine precision,
chain marginals against joint conditioning at every node, and the
arbitrage certificate's presence for inconsistent quotes and absence for
consistent ones.

## 7. Prior art and novelty (calibrated)

Compiled from a targeted web search. Claim by claim:

*One market per chain-rule factor (§1).* Partially known, and the novelty
is narrower than the framing suggests. Hanson's modularity (2007) is a
locality theorem *inside one joint market*: an LMSR bet on $A\mid B$ moves
that conditional and provably nothing else, uniquely among market scoring
rules; he never decomposes into separate venues. Pennock & Wellman (2000)
structure a *security set* by a Bayes-net factorization and ask when it is
operationally complete; Xia & Pennock (2011) characterize the
structure-preserving trades (decomposable graphs). Not found: separate
cost-function markets chained serially, each pricing one factor
conditional on upstream venues, with the open-when-frozen lifecycle.

*Pricing is inference (§1-§3).* Known in substance, unclaimed as a
statement. LMSR pricing is #P-hard because prices are marginals (Chen,
Fortnow, Lambert, Pennock & Wortman 2008); tournament markets price by
Bayes-net inference (Chen, Goel & Pennock 2008); the junction-tree
algorithm ran the price and asset updates of a deployed combinatorial
market (Sun, Hanson, Laskey & Twardy 2012, the DAGGRE/SciCast engine);
constraint generation prices approximately over the marginal polytope
(Dudík, Lahaie & Pennock 2012). No source states "price updating in a
structured market is belief propagation" as an equivalence.

*Markets implementing graphical-model inference (§3).* Known, with one
major citation: Pennock & Wellman (1996) map a Bayes net to a
general-equilibrium economy, one agent per conditional-probability entry,
arbitrageur producers enforcing the coherence identities, equilibrium
prices equal to the network's probabilities, and distributed bidding as
distributed inference. Storkey (2011) has market equilibria factorizing as
products of local potentials. Not found: trades literally serving as
messages along a factorization of separate sequential markets.

*Filtering by markets (§2).* Not found; the most open claim of the five.
The nearest object is the Bayesian market maker of Brahma, Chakraborty,
Das, Lavoie & Magdon-Ismail (2012), whose trade update is a scalar
Gaussian measurement update with covariance inflation for jumps; the
Kalman connection is unstated in print, and no work alternates model steps
with market steps or does market-based data assimilation.

*Cycle inconsistency is arbitrage (§4).* The coherence-is-no-arbitrage
root is classical (de Finetti; Nau & McCardle 1991); arbitrageur agents
enforcing coherence are in Pennock & Wellman (1996); algorithmic arbitrage
removal over the marginal polytope is Kroer, Dudík, Lahaie & Balakrishnan
(2016); measured human arbitrageurs enforcing logical coherence on
Polymarket are in Saguillo et al. (2025). The correlation-triangle bounds
exist in mathematical finance as no-arbitrage constraints on implied
correlations. Not found: the loopy-BP reading, the PSD cycle certificate
as holonomy, or arbitrageurs framed as the loop correction.

Overall: the semiring observation is Aji & McEliece (2000) plus Maslov's
dequantization, cited as such; the substance of pricing-as-inference and
coherence-as-no-arbitrage is thoroughly occupied and credited above. The
unoccupied residue is the serial architecture itself, the exact
market-implemented Kalman filter and tree marginals, and the loopy
arbitrage framing, each carried by a theorem test here.

## 8. Open questions

- *Loopy fixed points.* Under a concrete flow model, does
  arbitrage-corrected quoting converge, and to what — the true marginals,
  a Bethe-like surrogate, or something the choice of liquidities selects?
- *Scheduling.* The open-when-frozen lifecycle rule turns a DAG into a
  calendar; formalize the strip as a scheduled unrolling and ask what
  happens when conditioning information only partially freezes.
- *Latent settlement.* Settling a hidden-state market through its
  observable footprint scores a functional of future observables through
  the model; when is that strictly proper for the latent's conditional
  law?
- *Beyond Gaussians.* The exactness here leans on precision arithmetic;
  the exponential-family maker of
  [covariance-market.md](covariance-market.md) suggests conjugate-family
  chains as the next tractable class.
- *Max-product markets.* A market of optimizing traders natively computes
  in the min-plus semiring, so off the Gaussian family it prices the
  max-product (MAP) posterior rather than the sum-product marginal, and
  the discrepancy is the Laplace-approximation gap. Which object does a
  risk-averse trading population actually price, and does risk aversion
  (an exponential tilt) interpolate between the two semirings?

## References

*Combinatorial markets, modularity, and pricing as inference.*
- Hanson, R. (2003). "Combinatorial Information Market Design."
  *Information Systems Frontiers* 5(1); (2007). "Logarithmic Market
  Scoring Rules for Modular Combinatorial Information Aggregation."
  *J. Prediction Markets* 1(1).
- Chen, Y., Fortnow, L., Lambert, N., Pennock, D. M. & Wortman, J. (2008).
  "Complexity of Combinatorial Market Makers." *EC*.
- Chen, Y., Goel, S. & Pennock, D. M. (2008). "Pricing Combinatorial
  Markets for Tournaments." *STOC*.
- Xia, L. & Pennock, D. M. (2011). "Price Updating in Combinatorial
  Prediction Markets with Bayesian Networks." *UAI*.
- Sun, W., Hanson, R., Laskey, K. B. & Twardy, C. (2012). "Probability and
  Asset Updating using Bayesian Networks for Combinatorial Prediction
  Markets." *UAI*.
- Dudík, M., Lahaie, S. & Pennock, D. M. (2012). "A Tractable
  Combinatorial Market Maker Using Constraint Generation." *EC*.
- Dudík, M., Wang, X., Pennock, D. M. & Rothschild, D. (2021). "Log-time
  Prediction Markets for Interval Securities." *AAMAS*.

*Markets as inference engines; coherence as no-arbitrage.*
- Pennock, D. M. & Wellman, M. P. (1996). "Toward a Market Model for
  Bayesian Inference." *UAI*.
- Pennock, D. M. & Wellman, M. P. (2000). "Compact Securities Markets for
  Pareto Optimal Reallocation of Risk." *UAI*.
- Storkey, A. J. (2011). "Machine Learning Markets." *AISTATS*.
- Nau, R. F. & McCardle, K. F. (1991). "Arbitrage, Rationality, and
  Equilibrium." *Theory and Decision* 31(2-3), 199–240.
- Kroer, C., Dudík, M., Lahaie, S. & Balakrishnan, S. (2016).
  "Arbitrage-Free Combinatorial Market Making via Integer Programming."
  *EC*.
- Saguillo, O., Ghafouri, V., Kiffer, L. & Suarez-Tangil, G. (2025).
  "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets."
  *AFT*.
- Brahma, A., Chakraborty, M., Das, S., Lavoie, A. & Magdon-Ismail, M.
  (2012). "A Bayesian Market Maker." *EC*.

*Inference, filtering, and the semiring.*
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction
  Problems." *J. Basic Engineering* 82(1), 35–45.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems.*
  Morgan Kaufmann.
- Aji, S. M. & McEliece, R. J. (2000). "The Generalized Distributive Law."
  *IEEE Trans. Information Theory* 46(2), 325–343.
- Litvinov, G. L. (2005). "The Maslov Dequantization, Idempotent and
  Tropical Mathematics: A Brief Introduction." arXiv:math/0507014.

*Companions in this repository.*
- [liquidity-is-precision.md](liquidity-is-precision.md) — the fusion
  identity the market step instantiates.
- [covariance-market.md](covariance-market.md) — the PSD cone whose
  boundary the cycle certificate detects.
- [composing-mechanisms-conservation-and-boosting.md](composing-mechanisms-conservation-and-boosting.md)
  — sequential chains, conservation, and residual (boosting) markets.
- The [composition paper](../papers/composition-and-the-algebra-of-mechanisms.md)
  and the [multi-stage paper](../papers/multi-stage-solicitation.md) — the
  operator algebra and when chains are proper.
