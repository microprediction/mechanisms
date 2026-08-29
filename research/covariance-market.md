# A covariance market

*Status: a worked design with a calibrated novelty assessment. Reference
implementation: [`covariance_market.py`](../mechanisms/covariance_market.py),
theorem-tested in
[`test_covariance_market.py`](../tests/test_covariance_market.py). Companion
to [liquidity-is-precision.md](liquidity-is-precision.md), which flagged the
absence of an incentive-designed market for a covariance matrix, and to the
fee algebra of
[proportional-fees-and-the-order-book.md](proportional-fees-and-the-order-book.md).*

The prior-art search behind the fusion note found no prediction-market design
whose elicited object is a covariance matrix with positive-semidefinite
consistency. This note observes that the object mostly assembles itself from
machinery already in this repository: PSD consistency is the no-arbitrage
condition, not a constraint to engineer; the canonical maker is the
exponential-family market scoring rule instantiated for the Gaussian family,
whose inventory is a precision matrix and whose Bregman divergence is Stein's
loss; restricted coverage quotes Dempster's covariance selection natively; and
the fee lemma goes spectral, landing on singular-value thresholding.
monteprediction is the batch (pool) form of the same elicitation, and the
comparison is instructive in both directions.

## 1. Coherence is the cone

Let securities pay the products $X_j X_k$ of a random vector's coordinates,
so a price matrix $P$ quotes a second-moment estimate. The payoff matrix
$XX^\top$ is rank-one positive semidefinite, and any PSD matrix is a convex
combination of rank-ones, so the convex hull of payoffs is exactly the PSD
cone. Coherence of prices — membership in the convex hull of payoffs, the
no-arbitrage condition of the cost-function framework — is therefore exactly
$P \succeq 0$. Concretely, if $w^\top P w < 0$ for some $w$, the bundle with
weights $ww^\top$ costs a negative amount and pays $(w^\top X)^2 \ge 0$:
free money. PSD consistency is the precise analogue of "prices lie in the
simplex" for LMSR, and any coherent design gets it for nothing. The same
fact in static-arbitrage language is d'Aspremont (2005): absence of
buy-and-hold arbitrage is equivalent to positive semidefiniteness of the
price matrices of securities and their products. There it is a test applied
to given prices; here it is enforced by construction (§2).

## 2. The Gaussian market maker

LMSR is the log-partition function of the categorical family run as a cost
function, and the general recipe — cost = log-partition, prices = mean
parameters, trader profit = Bregman divergence = KL — is the exponential
family market of Abernethy, Kutty, Lahaie & Sami (2014), whose Example 2.5
is the univariate Gaussian and whose one-sentence multivariate remark stops
at the scoring rule (the Dawid-Sebastiani score). This section develops the
step they did not take: run the construction on the family whose sufficient
statistic is $xx^\top$, the zero-mean Gaussian:

$$C(Q) = b\,A(\Theta_0 + Q/b), \qquad
A(\Theta) = -\tfrac12 \log\det(-2\Theta),$$

with symmetric-matrix inventory $Q$. The quoted price is the mean parameter

$$\nabla C(Q) \;=\; \big(-2(\Theta_0 + Q/b)\big)^{-1} \;=\; \Sigma,$$

positive definite by construction: the maker cannot be traded out of the
coherent cone (the cost is $+\infty$ there). Three readings follow.

*Inventory is precision.* The natural parameter is $-\tfrac12\Sigma^{-1}$, so
the share vector a trader accumulates is, up to scale, a precision-matrix
increment. The identification argued statistically in
[liquidity-is-precision.md](liquidity-is-precision.md) is, for this maker,
the coordinate system: traders deposit precision, the market quotes
covariance.

*The payment rule is Stein's loss.* The Bregman divergence of $A$ is the KL
divergence between zero-mean Gaussians. A myopic trader with believed second
moment $M$ maximizes expected profit by moving the quote to $M$ exactly, and
collects

$$b \cdot \mathrm{KL}\!\big(\mathcal N(0, M)\,\Vert\, \mathcal N(0, \Sigma)\big)
\;=\; \tfrac b2\big[\operatorname{tr}(\Sigma^{-1}M) - d - \log\det(\Sigma^{-1}M)\big],$$

the classical Stein (entropy) loss of the standing quote. The
decision-theoretic covariance literature's loss function is this market's
payment rule
([`test_covariance_market.py`](../tests/test_covariance_market.py)).

*Liquidity adds.* The family $C_b = b\,A(\Theta_0 + \cdot/b)$ is a
perspective family, so merging makers adds their $b$, with the
depth-proportional split optimal (tested), and the fusion algebra of the
companion note applies with matrix-valued everything.

Means are co-elicited by adding the linear securities $X_j$ and running the
full Gaussian family (variance has elicitation complexity two, jointly with
the mean); the note stays with second moments for notation.

## 3. Coverage is a graphical model

A maker willing to quote only a subgraph $E$ of pairs is the exponential
family restricted to the sufficient statistics on $E$: a Gaussian graphical
model. Starting from a diagonal quote and trading only on $E$, the maker's
precision matrix is supported on $E$ plus the diagonal, so its quoted
covariance is the maximum-entropy completion of the on-graph moments with
unquoted partial correlations exactly zero — Dempster's covariance selection,
not as a default rule bolted on but as what the restricted maker's prices
mean (tested). What
[liquidity-is-precision.md](liquidity-is-precision.md) §5 framed as a
completion convention is this maker's idle state.

## 4. Fees are spectral spreads

The fee-spread lemma of the
[fee lemma](../papers/predictors-as-markets.md) goes
matrix-valued by changing the norm. A proportional fee on the *nuclear norm*
of the fill, $f\lVert S\rVert_*$, conjugates to the indicator of the
spectral-norm ball, so the maker's dead zone is
$\{\lVert P - \Sigma\rVert_2 \le f\}$: a bid-ask spread in operator norm. No
trade is profitable while the mispricing's top eigenvalue is inside the band,
and beyond it the profitable deviation is rank-one in the top eigendirection
(tested); the trading response soft-thresholds eigenvalues, the
singular-value-thresholding operator of matrix completion. Low-rank fills
arrive for the same reason lasso fills were sparse. An *entrywise* $\ell_1$
fee instead gives per-edge spreads, and self-set per-edge fees are the market
analogue of the graphical-lasso penalty, quoted competitively rather than
tuned — the conjecture left open in the fusion note, §8.

## 5. monteprediction is the primal form

monteprediction elicits the joint law of eleven sector returns by scenario
submission, so it already elicits the covariance, in sample space. The
comparison with the maker above is the repo's recurring parimutuel-versus-MSR
duality, and each form solves the other's hard problem.

*Coherence.* An empirical covariance of submitted scenarios is PSD by
construction; indeed a cloud cannot violate any moment inequality at any
order. Sample submission is the primal answer to the coherence problem the
moment cone solves dually.

*Bounded loss.* The securities design inherits unbounded payoffs ($X_jX_k$
is unbounded, the Gaussian score unbounded below), the one genuinely hard
problem of §6. The pool form never had it: payments redistribute wagers.
The price paid is the absence of continuous prices, an order book, and a
hedging instrument.

*Aggregation.* The merged maker fuses by adding precisions (the log pool,
sharpening with participation); a pool of clouds aggregates as a
wealth-weighted mixture (the linear pool), whose covariance carries the
between-participant disagreement term. The mixture's humility is arguably
correct when sources share data, Winkler's dependent-sources caveat playing
out mechanically; the two mechanisms sit on the two sides of the
covariance-intersection question.

*Selection.* The pot-split dynamic is the Kelly machinery, so a participant
submitting clouds from a diagonal model is the Ledoit-Wolf target
participant of the fusion note, with wealth share as intensity — in mixture
form, at a venue that exists.

*The jitter tilt is covariance shrinkage.* The multi-stage paper's verdict
on monteprediction (an unjittered kernel score elicits clouds slightly
tighter than the belief) reads, at the level of this note's estimand, as an
unintended shrinkage incentive: the settlement leans the elicited covariance
inward by roughly the kernel scale, in Ledoit-Wolf's direction but for no
statistical reason and by an arbitrary amount. The matched-jitter repair
removes exactly this tilt.

## 6. What is genuinely hard

- *Bounded loss.* Winsorized settlement (variance swaps cap payoffs in
  practice for the same reason), compact support, or a bounded-cost wrapping
  all work, and all degrade the clean conjugate structure. This is the open
  design problem of the securities form.
- *Settlement noise.* A single realized $xx^\top$ is rank-one; propriety
  holds in expectation, but fourth moments govern the variance of the
  settlement and slow the capital dynamics that do the tuning. Streams are
  the natural setting.
- *Scale.* $d(d+1)/2$ securities; projection securities $(w^\top X)^2$ trade
  variance along directions and connect to the sliced machinery of the
  point-cloud paper — and to dispersion trading, which is how practice
  synthesizes exactly these exposures (variance swaps via the log-contract;
  correlation swaps; implied-correlation indices), priced but not
  incentive-designed.

## 7. The dictionary

The chain of notes closes a dictionary between regularized estimation and
market design. Calibrated entry by entry:

| statistics | market design | status |
|---|---|---|
| regularizer | conjugate of the cost function | theorem (the FTRL duality) |
| penalty weight $\lambda$ | proportional fee $f$ | theorem at the program level: the fee enters the clearing problem exactly as an $\ell_1$ weight, with the same soft-threshold and active set |
| posterior weight | wealth | theorem for discrete Kelly bettors; drift, not convergence, in the fusion note's setting |
| variable enters the lasso path | maker/edge starts receiving flow | same active-set condition; the *path* reading (fee decline traversing it) needs a competition dynamics model |
| shrinkage intensity | capital share | exact in the quadratic fusion; tuning by selection is drift, not yet an equilibrium |
| graphical-lasso penalty | per-edge fee | conjectural: exact at the stationarity conditions, open as an equilibrium statement |
| ridge ($\ell_2$) penalty | a zero-quoting participant with capital $\lambda$ | exact (the fusion identity); the two canonical penalties are the two market primitives, participant and friction |
| robust-optimization budget | the fee, priced rather than assumed | regularization = robustness is a theorem (El Ghaoui & Lebret 1997; Xu, Caramanis & Mannor 2009); endogenizing the budget is the open equilibrium |
| oracle tuning ($\lambda$ by CV) | fee by Bertrand competition | economics classical (zero-profit spread = adverse-selection cost), unproven inside this algebra |

The pattern of the right column: every quantity statistics tunes by formula
or cross-validation reappears as a price, set by a self-interested party
bearing the consequences. Where the entries are theorems they are collected
in this repository's notes and tests; where they are dynamics claims they
are named as open.

## 8. Prior art and novelty (calibrated)

Compiled from a targeted web search on each ingredient.

*The exponential-family construction (§2).* Known. Cost = log-partition,
prices = mean parameters, informed-trader profit = Bregman divergence = KL is
Abernethy, Kutty, Lahaie & Sami (2014), restated in Abernethy, Frongillo &
Kutty (2015) with the risk-measure reading; the general convex-cost frame is
Abernethy, Chen & Wortman Vaughan (2013), the FTRL link Chen & Wortman
Vaughan (2010), continuous-outcome makers Gao, Chen & Pennock (2010) and
Chen, Ruberry & Wortman Vaughan (2013). The univariate Gaussian appears as
their example; the multivariate case appears as a one-sentence remark at the
scoring-rule level, where the score is Dawid & Sebastiani (1999). No novelty
in the template.

*The covariance market itself (§2-§3).* Not found. No published market
whose quoted price is a PSD matrix, whose cost is the log-det, or whose
inventory is a precision matrix; no log-det trading function in the
CFMM/DeFi literature. Nearest neighbours to cite and distinguish: the EC
2014 multivariate remark; ParlayMarket (Rana et al. 2026), a pairwise
exponential-family (Ising) AMM over binary events, the discrete cousin; and
Squeeth/power perpetuals (White, Robinson et al. 2021), a traded $S^2$
instrument with no moment-structured maker. The graphical (restricted
coverage = Dempster completion) reading was likewise not found. Verdict:
the instantiation and its matrix semantics are the unoccupied slot; the
mathematics is the published template plus classical convex analysis.

*PSD as no-arbitrage (§1).* Partially known. It is an immediate corollary
of Abernethy-Chen-Wortman Vaughan's prices-in-the-convex-hull theorem, and
d'Aspremont (2005) proves the equivalent statement in static-arbitrage
language (a test on given prices of securities and their products);
correlation-triangle feasibility is options-desk folklore (Walter & Lopez
2000). The packaging as mechanism design — coherence enforced by the cost
function rather than tested — was not found stated.

*Nuclear fees and SVT (§4).* Not found in any market or trading context.
Singular-value thresholding is Cai, Candès & Shen (2010) in matrix
completion; fee-induced no-trade intervals are established for scalar
CFMMs. The matrix fee with a spectral dead zone appears unoccupied.

*Practice (§6).* All standard and cited below: the log contract (Neuberger
1994), variance-swap replication (Carr & Madan 1998; Demeterfi, Derman,
Kamal & Zou 1999), correlation swaps and dispersion (Bossu 2005; Jacquier &
Slaoui 2010), implied-correlation indices (Cboe), the correlation risk
premium (Driessen, Maenhout & Vilkov 2009).

Overall: the template is Abernethy et al.'s, the coherence fact is a
corollary of theirs and d'Aspremont's, and the loss function is Dawid &
Sebastiani's; the market instantiation, the precision-inventory and
Dempster-coverage semantics, the Stein-loss payment identity, and the
spectral fee are the connective tissue this note adds, each with a theorem
test in [`test_covariance_market.py`](../tests/test_covariance_market.py).

## 9. Open questions

- *Bounded loss with clean conjugates.* Is there a cost family that is
  simultaneously coherent on the moment cone, bounded-loss under winsorized
  settlement, and closed under merge? The log-cosh trick does not obviously
  matricize.
- *The fee-decline path.* Model entry and Bertrand competition on per-edge
  fees and characterize the trajectory of the quoted graph; does it traverse
  the graphical-lasso path, and is the stationary graph the one whose edges
  earn their adverse-selection cost?
- *Hybrid: clouds as collateral.* A maker whose quotes are backed by a
  scenario cloud (every generative model a collateralized market maker)
  would give the covariance market continuous prices while keeping
  sample-space coherence. What is the correct cloud-to-cost-function map,
  and is it the empirical-measure exponential tilt?
- *Fourth moments.* The maker's local liquidity in mean-parameter space is
  the inverse Fisher information, a fourth-moment object; heavy tails thin
  the market exactly where estimates are worst. Quantify.

## References

*Exponential-family and continuous-outcome market makers.*
- Hanson, R. (2003). "Combinatorial Information Market Design." *Information
  Systems Frontiers* 5(1).
- Chen, Y. & Wortman Vaughan, J. (2010). "A New Understanding of Prediction
  Markets via No-Regret Learning." *EC*. arXiv:1003.0034.
- Gao, X., Chen, Y. & Pennock, D. M. (2010). "An Axiomatic Characterization
  of Continuous-Outcome Market Makers." *WINE*.
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). "Efficient Market
  Making via Convex Optimization, and a Connection to Online Learning."
  *ACM TEAC* 1(2).
- Chen, Y., Ruberry, M. & Wortman Vaughan, J. (2013). "Cost Function Market
  Makers for Measurable Spaces." *EC*.
- Abernethy, J., Kutty, S., Lahaie, S. & Sami, R. (2014). "Information
  Aggregation in Exponential Family Markets." *EC*. arXiv:1402.5458.
- Abernethy, J. D., Frongillo, R. M. & Kutty, S. (2015). "On Risk Measures,
  Market Making, and Exponential Families." *ACM SIGecom Exchanges* 13(2).
- Abernethy, J. & Frongillo, R. (2012). "A Characterization of Scoring Rules
  for Linear Properties." *COLT*.
- Rana, A., Nadkarni, S., Moshrefi, A. & Viswanath, P. (2026).
  "ParlayMarket: Automated Market Making for Parlay-style Joint Contracts."
  arXiv:2603.22596.

*Scoring and elicitation of moments.*
- Dawid, A. P. & Sebastiani, P. (1999). "Coherent Dispersion Criteria for
  Optimal Experimental Design." *Annals of Statistics* 27(1), 65–81.
- Lambert, N. S., Pennock, D. M. & Shoham, Y. (2008). "Eliciting Properties
  of Probability Distributions." *EC*.
- Frongillo, R. & Kash, I. A. (2021). "Elicitation Complexity of Statistical
  Properties." *Biometrika* 108(4).

*Coherence, moments, and arbitrage.*
- d'Aspremont, A. (2005). "A Market Test for the Positivity of Arrow-Debreu
  Prices." arXiv:cs/0510027.
- Bertsimas, D. & Popescu, I. (2002). "On the Relation Between Option and
  Stock Prices: A Convex Optimization Approach." *Operations Research*
  50(2), 358–374.
- Walter, C. & Lopez, J. A. (2000). "Is Implied Correlation Worth
  Calculating?" *Journal of Derivatives* 7(3); FRBSF WP 99-04.

*Covariance selection and matrix regularization.*
- Dempster, A. P. (1972). "Covariance Selection." *Biometrics* 28(1).
- Grone, R., Johnson, C. R., Sá, E. M. & Wolkowicz, H. (1984). "Positive
  Definite Completions of Partial Hermitian Matrices." *Linear Algebra
  Appl.* 58.
- Friedman, J., Hastie, T. & Tibshirani, R. (2008). "Sparse Inverse
  Covariance Estimation with the Graphical Lasso." *Biostatistics* 9(3).
- Cai, J.-F., Candès, E. J. & Shen, Z. (2010). "A Singular Value
  Thresholding Algorithm for Matrix Completion." *SIAM J. Optimization*
  20(4), 1956–1982.
- El Ghaoui, L. & Lebret, H. (1997). "Robust Solutions to Least-Squares
  Problems with Uncertain Data." *SIAM J. Matrix Anal. Appl.* 18(4).
- Xu, H., Caramanis, C. & Mannor, S. (2009). "Robustness and Regularization
  of Support Vector Machines." *JMLR* 10.

*Variance and correlation in practice.*
- Neuberger, A. (1994). "The Log Contract." *J. Portfolio Management* 20(2).
- Carr, P. & Madan, D. (1998). "Towards a Theory of Volatility Trading." In
  Jarrow, R. (ed.), *Volatility*, Risk Books, 417–427.
- Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999). "A Guide to
  Volatility and Variance Swaps." *J. Derivatives* 6(4), 9–32.
- Bossu, S. (2005). "Arbitrage Pricing of Equity Correlation Swaps."
  JPMorgan working paper.
- Jacquier, A. & Slaoui, S. (2010). "Variance Dispersion and Correlation
  Swaps." arXiv:1004.0125.
- Driessen, J., Maenhout, P. J. & Vilkov, G. (2009). "The Price of
  Correlation Risk: Evidence from Equity Options." *J. Finance* 64(3).
- White, D., Robinson, D., Adams, H. et al. (2021). "Power Perpetuals."
  Paradigm research.

*Companions in this repository.*
- [liquidity-is-precision.md](liquidity-is-precision.md) — the fusion
  algebra and the shrinkage-by-selection reading.
- [proportional-fees-and-the-order-book.md](proportional-fees-and-the-order-book.md)
  and the [fee lemma](../papers/predictors-as-markets.md)
  — the scalar fee lemma the spectral spread generalizes.
- The [point-cloud paper](../papers/scoring-point-cloud-distributional-submissions.md)
  and [multi-stage paper](../papers/multi-stage-solicitation.md) — the
  monteprediction machinery and the jitter repair.
