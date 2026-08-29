# Predictors as markets

*Status: a worked characterization with a calibrated novelty assessment; two
claims made in conversation are corrected here (§3, §5). Reference
implementation: [`nonconvex_maker.py`](../mechanisms/nonconvex_maker.py),
theorem-tested in
[`test_nonconvex_maker.py`](../tests/test_nonconvex_maker.py).*

The dictionary assembled across
[proportional-fees-and-the-order-book.md](proportional-fees-and-the-order-book.md),
[liquidity-is-precision.md](liquidity-is-precision.md) and
[covariance-market.md](covariance-market.md) raises the converse question: is
every predictor a market? The answer is a characterization. A large class of
learning procedures are markets literally, by published correspondences and
by the identities in this repository's notes. The apparent obstruction,
non-convexity, turns out not to obstruct coherence at all: no-arbitrage is a
chord condition, rational flow trades the biconjugate, and what non-convexity
costs is expressiveness (unquotable inventory states), not
soundness. Frictions then price the remaining failure mode, bounded slope
excursions, and a deep convex co-quoter fills the holes.

## 1. The literal correspondences

Procedures that are markets by identity rather than analogy:

- *Mirror descent and FTRL* trade against a cost-function maker: cost =
  regularizer's conjugate (Chen & Wortman Vaughan 2010; Abernethy, Chen &
  Wortman Vaughan 2013), with wealth as the learning rate (Frongillo, Della
  Penna & Reid 2012).
- *Bayesian model averaging* is a market of Kelly bettors with wealths as
  posterior weights (Beygelzimer, Langford & Pennock 2012); wealth updates
  are posterior updates in machine-learning markets (Storkey 2011; Storkey,
  Millin & Geras 2012; Barbu & Lay 2012).
- *Least squares* is a market of data points: each observation a quadratic
  maker quoting its value with capital equal to its precision, the estimate
  the merge ([liquidity-is-precision.md](liquidity-is-precision.md)).
- *Proximal steps* are trades against fee-bearing makers: the prox of
  $f\lvert\cdot\rvert$ is the soft-threshold is the optimal response to a
  proportional fee (the fee lemma of the
  [working paper](../papers/predictors-as-markets.md), §5), and
  the prox of a general convex $g$ is the response to a maker charging $g$.
  ISTA for the lasso is repeated trading against a quadratic maker with a
  fee.
- *The two canonical penalties are the two market primitives*: ridge is a
  zero-quoting participant with capital $\lambda$; lasso is a fee of
  $\lambda$. Elastic net is both at once.

## 2. Coherence is a chord condition, not convexity

A path-independent maker charges $C(q+s) - C(q)$, which telescopes over any
closed path whatever $C$ is: cycles are refunds, convex or not. A
sure-profit trade requires $\min_\omega \langle \varphi(\omega), s\rangle >
C(q+s) - C(q)$, so the exact no-arbitrage condition is that every chord
slope of $C$ lies in the convex hull of the payoff vectors; for the scalar
maker settling in $[-1,1]$, that $C$ is 1-Lipschitz. Convexity is
sufficient (with gradient range in the hull) but not necessary: the test
suite exhibits a non-convex 1-Lipschitz maker admitting no arbitrage at all.
Non-convexity was never the arbitrage; incoherent quotes were.

## 3. The market trades the biconjugate

What non-convexity does change is which states rational flow visits. Write
$\hat C$ for the lower convex envelope of $C$ (so $\hat C = C^{**}$ on the
line) and $g = C - \hat C \ge 0$ for the gap. For a myopic risk-neutral
trader with believed mean $\mu$ in the hull:

- every *attained* optimal fill lands on the *contact set*
  $\{C = \hat C\}$, and
- the maximal profit is the envelope profit plus the gap at the starting
  state: $\Pi_C(q) = \Pi_{\hat C}(q) + g(q)$, as extended reals,

both verified to machine precision in
[`test_nonconvex_maker.py`](../tests/test_nonconvex_maker.py). Attainment
is a real hypothesis: $C = \arctan$ is chord-coherent with convex envelope
the constant $-\pi/2$, so its contact set is empty and the optimum runs
off to $-\infty$ (tested). Rational
flow reads $C^{**}$: the maker behaves observationally like its convex
envelope, with the concave stretches as unquotable intermediate states —
an excluded inventory range, which the book shows as a block of size
$q_2 - q_1$ at the single supporting price rather than as a missing price
range. Whoever lands off-contact (noise flow) overpays the gap
at the landing state; the profit identity hands exactly that amount to the
next rational trader; the maker is a conduit that keeps envelope
differences over any rational-to-rational span. Non-convexity costs
expressiveness, not coherence, and it costs the maker nothing against
rational flow.

Read as beliefs, a hole in the book is a multimodal quote: the maker will
take the price on either side of the gap but refuses every price inside it,
and accumulating flow moves the quote across the gap discontinuously. This
is the shape of books around binary events, stated here as an
interpretation rather than an empirical claim (§9).

## 4. Frictions price the remaining failure

The failure mode that survives §2 is chord slopes exiting the hull. If the
excursion is bounded by $\varepsilon$, a proportional fee $f \ge
\varepsilon$ restores no-arbitrage: the fee widens the coherent price set to
an $f$-band, so bounded incoherence is priced rather than fatal (tested:
a maker with slope excursions of $0.19$ is arbitrageable bare and coherent
under $f = 0.20$). Only unbounded excursions, which no finite spread
covers, disqualify a cost from being a market. The minimum viable spread of
a predictor is its arbitrage depth, and a venue quoting a wide spread to
cover a wild landscape is coherent but uninformative in proportion: the
market prices the model's incoherence as uncertainty.

The second repair is a participant rather than a friction. Merging with a
quadratic co-quoter of liquidity $\lambda$ is infimal convolution with
$\lvert\cdot\rvert^2/(2\lambda)$, the Moreau envelope. A deep co-quoter
convexifies the merged venue on the tested costs; a shallow one does not.
Depth never fills an excluded range at finite depth, however: for the
double well the midpoint gap decays like $1/\lambda$ but stays strictly
positive (tested). The required depth is set by the
*global* geometry of the gaps (how wide a concave stretch the quadratic
must bridge), not by the local weak-convexity constant, which governs prox
uniqueness instead — the numerical demonstration in the tests has the venue
still non-convex at the local scale $1/\rho$ and convex at several times
it. The merge also never creates arbitrage: the envelope's derivative is a
selection of $C$'s derivatives (first-order condition), so a chord-coherent
maker stays chord-coherent at every depth, and what the deep co-quoter buys
is expressiveness, not coherence (tested). The crossing-jump bound measures
crossing defects only: $C = a\sin y$ with $\lambda a < 1$ has a unique
minimizer everywhere yet envelope curvature $C''/(1+\lambda C'') < 0$
wherever $C'' < 0$, smooth non-convexity with no crossing (tested). Exact convexification at finite depth is generic but not universal: for
a symmetric double well the merged venue keeps a concave kink at every
finite depth (the minimum of two crossing parabolas), with defect decaying
like $1/\lambda$, so the residue is priced by a fee of the same order. The two repairs are exactly the dictionary's two primitives, friction
($\ell_1$) and participant depth ($\ell_2$), the same pair as lasso and
ridge.

The fee lemma (§5 of the
[working paper](../papers/predictors-as-markets.md)) extends without
convexity in exact form: the no-trade beliefs at state $q$ are the
fee-widened interval of one-sided chord bounds,
$[\sup_{s<0} d_q - f,\ \inf_{s>0} d_q + f]$, intersected with the payoff
hull, with $d_q$ the chord slope. For convex $C$ both bounds equal
$C'(q)$, the band $m \pm f$. Off the contact set the frictionless
interval is empty (chord gap $\Delta_q > 0$) and the fee fills the hole
exactly when $2f \ge \Delta_q$: at small fees every belief still profits
and the state is transient, while a wide enough spread stabilizes a state
inside a quote hole (both tested; the sine cost at its crest is the worked
instance). The intersection with the hull is not optional: $C(q) = 100q$
has $\Delta_q = 0$ yet no admissible belief declines to trade (tested).

Optimization already pays these frictions. A proximal step charges
$\lVert\Delta\theta\rVert^2/(2\eta)$ per move; a trust region is an
infinite fee outside a band; weight decay is a zero-quoting participant.
In market language the stabilizers that make non-convex training behave are
the frictions that make a non-convex venue non-exploitable. Path-dependent
optimizers (momentum) correspond to makers whose quotes depend on flow
history, the adaptive-liquidity territory where path independence, and with
it the free-round-trip guarantee, is deliberately traded away: the trilemma
of Othman, Pennock, Reeves & Sandholm (2013) prices adaptivity at a vig,
Li & Wortman Vaughan (2013) axiomatize the adaptive class, and Abernethy,
Frongillo, Li & Wortman Vaughan (2014) prove no trade-history maker gets
every desideratum at once.

## 5. The characterization

A predictor is a market when its cost has finite arbitrage depth relative
to affordable friction; convexity is the $f = 0$ special case. What
non-convexity spends is expressiveness (contact set in place of full
quoting range) and what unbounded incoherence spends is everything. Two
corrections to the conversational versions of this claim are recorded
here: cycles were never the exploit (telescoping refunds them, §2), and the
Moreau convexification threshold is global, not the weak-convexity constant
(§4).

## 6. Implementation

[`nonconvex_maker.py`](../mechanisms/nonconvex_maker.py) provides
`NonconvexMaker` (arbitrary scalar cost, proportional fee),
`lower_convex_envelope`, `max_sure_profit` (the arbitrage-depth
diagnostic), and `moreau_envelope` (the merged venue with a quadratic
co-quoter). The tests verify: coherence without convexity; rational landing
on the contact set and the profit identity $\Pi_C = \Pi_{\hat C} + g(q_0)$;
the off-contact pass-through; fee-priced slope excursions with round trips
costing exactly $2fs$; and deep-versus-shallow co-quoter convexification.

## 7. Prior art and novelty (calibrated)

Compiled from a targeted web search. Claim by claim:

*Coherence is a chord condition (§2).* Not found as stated. Abernethy,
Chen & Wortman Vaughan derive convexity from their *information
incorporation* axiom, not from no-arbitrage; given convexity, their Theorem
2 puts gradients in the payoff hull. No paper was found stating the
no-arbitrage condition for arbitrary non-convex $C$ as a chord condition,
and none studying non-convex cost-function prediction-market makers.
Lépinette & Tran (2017) develop arbitrage theory for non-convex market
models in the classical-finance formalism, evidence the question is
recognized elsewhere.

*The market trades the biconjugate (§3).* Partially known, on the CFMM
side. The Geometry of CFMMs (Angeris et al. 2023) proves every trade set
has a canonical concave, nondecreasing, homogeneous trading function —
behavioral replacement of an arbitrary invariant by a concave one — and
Frongillo, Papireddygari & Waggoner (2024) show quasiconcave potentials
need not concavify. The cost-function statement, rational flow lands on the
contact set with the profit identity and gap pass-through, was not found;
nor the unquotable-gap-states reading. Concavity in the CFMM literature is
imposed to avoid exploitability (Angeris & Chitra 2020), not derived.

*Frictions price incoherence (§4).* Known in the classical analogue,
partially known in prediction markets. Guasoni (2006) and Guasoni, Rásonyi
& Schachermayer (2010) prove that arbitrarily small proportional
transaction costs restore no-arbitrage for processes arbitrageable
frictionlessly, with consistent price systems inside the spread as the
certificate — the theorem-strength version of "chord slopes may exit the
hull by at most $f$". Frongillo & Waggoner (2018) size a per-trade fee to
expected arbitrage profit to restore bounded loss in private prediction
markets (noise-induced, not curvature-induced, incoherence; the negative
results are Cummings, Pennock & Wortman Vaughan 2016). The cost-function
statement with the excursion bound was not found.

*Moreau by merge (§4).* Partially known. Aggregation of parallel makers as
infimal convolution is Bhaskara et al. (2023), all inputs convex; CFMM
composition is in the Geometry paper. Reading the Moreau envelope as a deep
quadratic co-quoter convexifying a non-convex venue was not found.

*Adaptivity (§4).* Known: the trilemma (Othman, Pennock, Reeves & Sandholm
2013), the homogeneous-risk-measure characterization (Othman & Sandholm
2011), the axiomatic adaptive class (Li & Wortman Vaughan 2013), the
volume-parameterized impossibility (Abernethy, Frongillo, Li & Wortman
Vaughan 2014), and adaptive liquidity as online learning (Nueve, Nguyen,
Frongillo & Waggoner 2026). The optimizer-stabilizers-as-frictions mapping
was not found in either direction.

*Book holes and multimodality (§3).* Both halves documented, unconnected.
Bimodal option-implied densities around binary events are established
(Melick & Thomas 1997 for the Gulf crisis; Clark & Amen 2017 for Brexit);
order-book gap statistics and pre-announcement depth withdrawal are
documented empirically; discontinuous equilibrium prices arise from folded
demand (Gennotte & Leland 1990; Çetin & Sheynzon 2014). A mechanical link
from multimodal beliefs to book gaps was not found; §8 keeps it as the
testable open question.

Overall: the convex-analysis ingredients are classical, the transaction-cost
analogue is theorem-grade in mathematical finance, and the CFMM literature
has the concave-canonicalization half. The chord condition, the contact-set
and pass-through identities, the generalized fee lemma, and the
Moreau-by-merge reading appear unoccupied, each with a numerical theorem
test here.

## 8. Open questions

- *Arbitrage depth as a capacity measure.* The minimum viable spread of a
  trained model's landscape is computable in principle; does it correlate
  with anything the learning theory already names (sharpness, mode
  connectivity)?
- *Two-agent pumps.* Adversarial training oscillation reads as a money pump
  between two path-dependent agents, and gradient penalties as
  spread-widening; state and prove a version, or retire it.
- *Noise as subsidy.* Minibatch noise lands states off-contact and the
  next informed step recoups the gap; whether SGD noise plays the economic
  role of noise traders (subsidizing discovery) deserves a precise model
  before it is asserted.
- *Books around binary events.* The multimodal-quote reading of book holes
  is testable against limit-order data around court rulings and FDA
  decisions.

## References

*Cost-function makers and their axioms.*
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). "Efficient Market
  Making via Convex Optimization, and a Connection to Online Learning."
  *ACM TEAC* 1(2).
- Chen, Y. & Wortman Vaughan, J. (2010). "A New Understanding of Prediction
  Markets via No-Regret Learning." *EC*.
- Othman, A., Pennock, D. M., Reeves, D. M. & Sandholm, T. (2013). "A
  Practical Liquidity-Sensitive Automated Market Maker." *ACM TEAC* 1(3).
- Othman, A. & Sandholm, T. (2011). "Liquidity-Sensitive Automated Market
  Makers via Homogeneous Risk Measures." *WINE*.
- Li, X. & Wortman Vaughan, J. (2013). "An Axiomatic Characterization of
  Adaptive-Liquidity Market Makers." *EC*.
- Abernethy, J., Frongillo, R., Li, X. & Wortman Vaughan, J. (2014). "A
  General Volume-Parameterized Market Making Framework." *EC*.
- Nueve, E. & Waggoner, B. (2025). "Smooth Quadratic Prediction Markets."
  *NeurIPS*. arXiv:2505.02959.
- Nueve, E., Nguyen, M., Frongillo, R. & Waggoner, B. (2026). "Adaptive
  Liquidity in Prediction Markets via Online Learning." arXiv:2605.09599.

*Markets as learners; learners as markets.*
- Frongillo, R., Della Penna, N. & Reid, M. D. (2012). "Interpreting
  Prediction Markets: A Stochastic Approach." *NeurIPS*.
- Beygelzimer, A., Langford, J. & Pennock, D. M. (2012). "Learning
  Performance of Prediction Markets with Kelly Bettors." *AAMAS*.
- Storkey, A. J. (2011). "Machine Learning Markets." *AISTATS*; Storkey,
  A. J., Millin, J. & Geras, K. (2012). "Isoelastic Agents and Wealth
  Updates in Machine Learning Markets." *ICML*.
- Barbu, A. & Lay, N. (2012). "An Introduction to Artificial Prediction
  Markets for Classification." *JMLR* 13.

*Non-convexity, CFMMs, and canonicalization.*
- Angeris, G., Chitra, T., Diamandis, T., Evans, A. & Kulkarni, K. (2023).
  "The Geometry of Constant Function Market Makers." arXiv:2308.08066.
- Frongillo, R., Papireddygari, M. & Waggoner, B. (2024). "An Axiomatic
  Characterization of CFMMs and Equivalence to Prediction Markets." *ITCS*.
- Angeris, G. & Chitra, T. (2020). "Improved Price Oracles: Constant
  Function Market Makers." *ACM AFT*.
- Lépinette, E. & Tran, T. Q. (2017). "Arbitrage Theory for Non Convex
  Financial Market Models." *Stoch. Proc. Appl.* 127(10).
- Rockafellar, R. T. (1970). *Convex Analysis.* Princeton University Press.

*Frictions and no-arbitrage.*
- Guasoni, P. (2006). "No Arbitrage under Transaction Costs, with
  Fractional Brownian Motion and Beyond." *Math. Finance* 16(3), 569–582.
- Guasoni, P., Rásonyi, M. & Schachermayer, W. (2010). "The Fundamental
  Theorem of Asset Pricing for Continuous Processes under Small Transaction
  Costs." *Annals of Finance* 6(2), 157–191.
- Frongillo, R. & Waggoner, B. (2018). "Bounded-Loss Private Prediction
  Markets." *NeurIPS*.
- Cummings, R., Pennock, D. M. & Wortman Vaughan, J. (2016). "The
  Possibilities and Limitations of Private Prediction Markets." *EC*.
- Bhaskara, A., Frongillo, R., Lindgren, E. & Papireddygari, M. (2023). "A
  General Theory of Liquidity Provisioning for Prediction Markets."
  arXiv:2311.08725.

*Regularization as robustness; sparsity from costs.*
- El Ghaoui, L. & Lebret, H. (1997). "Robust Solutions to Least-Squares
  Problems with Uncertain Data." *SIAM J. Matrix Anal. Appl.* 18(4).
- Xu, H., Caramanis, C. & Mannor, S. (2009). "Robustness and Regularization
  of Support Vector Machines." *JMLR* 10.
- Olivares-Nadal, A. V. & DeMiguel, V. (2018). "A Robust Perspective on
  Transaction Costs in Portfolio Optimization." *Operations Research* 66(3).

*Multimodal beliefs and discontinuous prices.*
- Melick, W. R. & Thomas, C. P. (1997). "Recovering an Asset's Implied PDF
  from Option Prices: An Application to Crude Oil during the Gulf Crisis."
  *JFQA* 32(1), 91–115.
- Clark, I. J. & Amen, S. (2017). "Implied Distributions from GBPUSD
  Risk-Reversals and Implication for Brexit Scenarios." *Risks* 5(3), 35.
- Gennotte, G. & Leland, H. (1990). "Market Liquidity, Hedging, and
  Crashes." *American Economic Review* 80(5), 999–1021.
- Çetin, U. & Sheynzon, I. (2014). "A Simple Model for Market Booms and
  Crashes." *Math. Financ. Econ.* 8(3).
