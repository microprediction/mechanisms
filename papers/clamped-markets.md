# Clamped Markets

### How prediction mechanisms compose, and what familiar methods are when the competition is switched off

Peter Cotton · *Working draft v0.1* · August 29, 2026

---

## Abstract

This paper is partly synthetic. Many of the correspondences below are
known individually; the purpose is to place them in one compositional
calculus and to ask what a familiar statistical method is when read
through it. The organizing observation is that many familiar statistical methods are
market ecosystems with degrees of freedom clamped. Mechanisms compose in
two ways, and both are established. In parallel, competing makers merge by
infimal convolution, so conjugates add and liquidity adds. In series,
a factorization assigns one venue per conditional factor, the arrangement
Hanson's modularity makes tradable and that deployed combinatorial markets
price by junction-tree inference. Reading the two laws through the
generalized distributive law puts them in one semiring frame, in which a
market of optimizing traders computes min-plus and agrees with Bayesian
inference on the log-quadratic family. Against that background, clamping is
the operation that recovers ordinary statistics: freeze entry and a market
of Kelly bettors becomes Bayesian model averaging; freeze capital and a
market of data points becomes least squares; freeze the propagation step
and a chain of venues becomes the Kalman filter; freeze the residual venue
to a single unconditional entry and a two-stage solicitation becomes
split-conformal prediction. Each clamp forfeits a specific rent, and in
the conformal case the rent is exactly the conditional information
$I(R;X)$ that the unconditional entry declines to price. The paper
sets out the composition laws with their sources, catalogues the clamps,
and states what unclamping would require. Where the calculus stops is
itself informative: it locates the mechanism that is missing.

---

## 1. Two composition laws, and a dial

A prediction mechanism takes beliefs and pays for them. Two ways of
putting mechanisms together account for most of what is known.

*Parallel.* Several makers quote the same quantity. Their aggregate is
the infimal convolution of their cost functions, so conjugates add and
liquidity adds; this is the aggregation law of @bhaskara2023general, the
risk-sharing composition of @barrieu2005inf and @jouini2008optimal, and
in the cost-function framework of @abernethy2013efficient it is the
statement that merging two makers yields a deeper one. For quadratic
makers it is Gaussian fusion: the merged quote is the precision-weighted
mean and precisions add.

*Serial.* A model is a factorization, $P(\text{everything}) = \prod_i
P(x_i \mid \text{parents})$, and the serial arrangement runs one venue per
factor, each pricing its conditional given what is upstream. The locality
that makes this tradable is Hanson's modularity: in a combinatorial
logarithmic market scoring rule a bet on $A \mid B$ moves that conditional
and provably nothing else, uniquely among market scoring rules
[@hanson2007logarithmic; @hanson2003combinatorial].

Neither law is new here, and §§2–3 set them out with their sources. What
this paper adds is a dial. Between a full ecosystem, in which every stage is a live venue with
free entry, and a single fitted model, there is a continuum of
*clamped* mechanisms: the same composition, with some degree of freedom
frozen. Ordinary statistical procedures sit at the clamped end, and the
question worth asking of each is not whether it is a market but which
clamp it applies and what that clamp costs. Section 5 catalogues them,
and conformal prediction turns out to be the sharpest case, because there
the forfeited rent has a closed form.

Three kinds of statement appear below and are worth distinguishing as
they arrive. *Known identities* are used and cited, not claimed: that
trading against a cost-function maker is follow-the-regularized-leader,
that a market of Kelly bettors performs Bayesian model averaging, that
merging makers is infimal convolution, that Gaussian fusion is
information-form addition, that min-plus elimination is dynamic
programming. *Results* are the few small formal statements, mostly in the
companion paper, that the calculus turns up on its way through. And
*principles* are proposed organizing ideas that are not yet theorems,
chiefly clamping itself and the accounting of what each clamp forfeits.
The individual identities are mostly known; the algebra connecting them,
and what it says about methods that are not usually thought of as
markets, is the subject here.

## 2. Parallel composition: what is settled

Cost-function market makers price by a convex potential, with prices its
gradient, no-arbitrage from convexity and bounded loss from a bounded
conjugate range [@abernethy2013efficient; @hanson2003combinatorial]. The
duality with proper scoring rules is standard: the maker is the conjugate
of the scoring rule's entropy, and trading against it is
follow-the-regularized-leader [@chen2010newunderstanding].

Merging is infimal convolution. @bhaskara2023general prove that liquidity
providers submitting arbitrary cost functions operate, in parallel,
exactly as the maker whose cost is $\bigwedge_i C_i$, with the dual
generating functions adding and the split across providers computed
behind the scenes; @angeris2024geometry read Minkowski sums of trade sets
the same way. In risk-measure language this is the classical result that
the aggregate of convex risk measures is their infimal convolution, with
the Pareto allocation as minimizer and the common subgradient as clearing
price [@barrieu2005inf; @jouini2008optimal].

Fees complicate the picture and are treated in the companion paper: a
proportional fee is exactly a bid-ask spread, participation becomes
sparse, and the aggregate supply curve is a consolidated limit order
book. Nothing here depends on that development.

## 3. Serial composition: what is settled

Pricing a combinatorial market *is* probabilistic inference, which is why
exact pricing of an LMSR over a combinatorial space is #P-hard
[@chen2008complexity]; why tournament markets price by Bayes-net
inference [@chen2008tournaments]; why a deployed combinatorial market ran
its price and asset updates on the junction-tree algorithm
[@sun2012junction]; and why tractable designs price approximately over
the marginal polytope [@dudik2012tractable; @dudik2021logtime]. Which
securities preserve a Bayes-net structure under trade is characterized by
@xia2011structure, and securities structured by a factorization appear in
@pennock2000compact.

That trading and message passing are the same activity is likewise
established. @pennock1996marketbayes map a Bayes net to an economy with
one agent per conditional-probability entry and arbitrageur agents
enforcing the additivity identities, showing equilibrium prices equal the
network's probabilities and distributed bidding is distributed inference.
@storkey2011machine shows that agents caring about subsets of variables
give equilibria factorizing as products of local potentials, and derives
messages from optimized positions.

The arrangement called serial here differs in architecture rather
than in that observation: separate venues with settlement boundaries and
an opening rule, each factor a market that opens when its conditioning
information freezes and settles in cascade, rather than one joint book
priced by an inference algorithm. Whether self-interested trading can be
made to perform the elimination itself, rather than a solver performing
it inside one venue, is open, and is the subject of the companion paper's
final section.

## 4. One semiring

The two laws are the two operations of a commutative semiring, which is
the frame the generalized distributive law provides [@aji2000gdl]: one
operation combines evidence about a variable, one moves evidence between
variables, and sum-product, max-product and min-sum are one algorithm
over different semirings. Assigning each factor its potential
$\varphi_i = -\log p_i$, parallel merge adds potentials (equivalently
inf-convolves costs, the two related by the Legendre transform, which is
the min-plus Laplace transform [@litvinov2007idempotent]), and serial
composition is the min-plus kernel product $\inf_y[\varphi_1(x,y) +
\varphi_2(y,z)]$.

Two consequences are worth recording because they are often run together.
First, a market of optimizing traders computes min-plus natively: a
finite-state chain of stage potentials prices exactly the Viterbi
decoding of the corresponding hidden Markov model, with no Gaussian
structure anywhere. Second, min-plus agrees with sum-product on
log-quadratic families, where partial minimization equals marginalization
up to a constant independent of the retained variables, by the
Schur-complement identity. On that family the alternation of a
propagation step with a fusion step reproduces the Kalman filter
[@kalman1960filtering] exactly, the fusion being a parallel merge, and
messages on a Gaussian chain reproduce belief propagation
[@pearl1988probabilistic]. Off it, the two diverge by the
Laplace-approximation gap, and what a market prices is the max-marginal
rather than the marginal.

## 5. Clamping

Now the dial. In each case below a mechanism of §§2–4 is recovered from a
familiar procedure by *unfreezing* something, and the procedure is
recovered by clamping it again.

*Freeze entry: Bayesian model averaging.* A market of Kelly bettors
prices the wealth-weighted average of participants' beliefs, and wealth
updates are Bayes updates, so the market performs Bayesian model
averaging with wealths as posterior weights [@beygelzimer2012kelly;
@kelly1956newinterpretation; @breiman1961optimal]. Model averaging over a
fixed model list is that market with entry closed. What the clamp forfeits
is whatever a model outside the list would have earned; the market-selection
literature is the study of who survives when entry is open
[@blume2006smart].

*Freeze capital: least squares.* Precision-weighted estimation of a common
mean is a market of data points, each observation a quadratic maker
quoting its value with capital equal to its precision, the estimate their
merge. Fixing those capitals at the nominal precisions is a clamp: the
market version lets a source that has been wrong lose capital, which is
the difference between weighted least squares and its robust or adaptive
variants.

*Freeze the fee: frictionless aggregation.* A maker with no fee earns
nothing on uninformed flow and cannot recover adverse-selection losses
from volume. Setting fees to zero is the clamp that makes an aggregation
rule out of a venue; letting participants quote their own fee is what
turns the spread into a price. The companion paper develops this.

*Freeze propagation: the Kalman filter.* In the filtering recursion the
correction step is a parallel merge, executed by trades against an
observation maker, while the prediction step is computed by whoever owns
the model. That asymmetry is a clamp on the edge factor: nobody is paid
to move the state forward. Unclamping it means posting the edge as a
venue with securities on the pair $(x_t, x_{t+1})$, which is the open
problem the companion paper states.

*Freeze the residual venue: conformal prediction.* This is the sharpest
case and the one that motivates the vocabulary.

## 6. Conformal prediction as a clamped two-stage market

Split-conformal prediction forms a residual law from calibration data and
applies it at every input. Read as a mechanism it is a two-stage
solicitation: a base predictor, then a residual stage. The second stage
is a market with the field restricted to a single *unconditional* entry
[@cotton2026multistage].

The distinction that matters is not which contest a forecaster enters but
whether their submission moves with the covariates. An unconditional
forecaster submits one distribution and stands by it; a conditional
forecaster submits one that depends on the input she sees. In a
log-wealth pool settling on an outcome, the best conditional entrant
out-earns the best unconditional one by exactly the conditional
information the covariates carry, $I(R;X)$, the mutual information
between the residual and the input [@cotton2026conformalbetting]. That
number is the rent the clamp forfeits, and in the Gaussian case it is
$-\tfrac12\log(1-\rho^2)$.

Two consequences follow. Marginal coverage, the guarantee conformal
prediction offers [@lei2018distribution], is the break-even statement of
the clamped venue: the unconditional entry is priced correctly on
average, which is what makes it safe and also what makes it leave money
on the table. And the gap does not close when the conformists are right:
if the base forecast is calibrated so the residuals are marginally
standard, a conditional participant still profits at the same rate,
because marginal correctness is not conditional correctness
[@barber2021limits]. Adaptive conformal variants that let the law depend
on the input are conditional entries, which is to say they have partially
unclamped the venue and are forecasting.

Run the residual stage as an actual pool and the rent is collected rather
than forfeited. That mechanism ran in production as the microprediction
platform's `z1~` residual streams and its MidOne contest
[@cotton2026multistage].

## 7. Point-cloud games, and the unclamped end

At the other end of the dial sits a mechanism with nothing frozen. In a
nearest-the-pin parimutuel over a continuum, participants submit clouds
of samples, the pot is split in proportion to the density each placed at
the realized outcome, and entry is open; this was the reward engine
behind monteprediction [@cotton2024monteprediction]. Every degree of
freedom the clamped methods freeze is live here: who enters, how much
each stakes, what each conditions on, and how sharp a submission is.

Point-cloud submission also shows what the unclamped end costs. Scoring a
kernel density at the raw outcome elicits the *deconvolution* of the
participant's belief rather than the belief, an impropriety with a closed
form, repaired by jittering the pin with the same kernel; and in high
dimensions the pool runs on random projections. Those are the subject of
a separate paper, and the point here is only that an open mechanism has
design problems a clamped one never encounters, because a clamped
mechanism has already decided the answers.

The platform stacked such games: a pool on a live quantity, a stream
predicting that pool's own calibration, dependence streams pricing
copulas, and a lottery of calibration maps
[@cotton2026multistage]. Read through this calculus, that architecture is
the serial law of §3 with every stage left unclamped, and the familiar
methods of §5 are what remain when the stages are frozen one at a time.

## 8. What clamping costs, and what unclamping would require

The catalogue suggests a common shape. Each clamp replaces a price with
an assumption:

| clamp | assumption replacing a price | forfeited |
|---|---|---|
| entry closed | the model list is adequate | what an outside model would earn |
| capital fixed | nominal precisions are right | the correction from realized performance |
| fee zero | flow is uninformed | adverse-selection revenue |
| propagation computed | the dynamics are known | the price of the transition |
| residual unconditional | covariates carry nothing | $I(R;X)$ |

Only the last entry currently has a closed form, and finding the others
is the natural program. Two further problems are worth naming. Unclamping
a stage requires a settlement rule for that stage's securities, and a
latent quantity with no settlement functional is not a market however
convenient it is to speak of one. And the serial law, though it is what
the platform implemented, has no general theorem saying self-interested
trading performs the elimination; the companion paper states what such a
mechanism would need.

The claim is modest and, I think, useful: the individual identities are
mostly known, and the algebra connecting them is the subject here. The
clamped limits of these mechanisms are the methods statistics already
uses. Naming the clamp is a
quick way to see what a method has given up, and in one case it says
exactly how much.

## References

::: {#refs}
:::
