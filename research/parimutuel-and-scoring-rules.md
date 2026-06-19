# Parimutuel Markets and Proper Scoring Rules

This document surveys the academic literature underpinning two closely related
families of mechanisms for eliciting and aggregating probabilistic beliefs:
**parimutuel markets** (pool betting and its modern generalizations) and
**proper scoring rules** (incentive-compatible loss functions for probability
forecasts). The two strands converge in the theory of *market scoring rules*,
which power many modern prediction markets (see
[`market-scoring-rules-and-amms.md`](market-scoring-rules-and-amms.md)).

---

## Part I: Parimutuel Markets

### 1. The Basic Parimutuel Mechanism

A **parimutuel** (French *pari mutuel*, "mutual betting") market is a pool-betting
system. Bettors stake money on a set of mutually exclusive and exhaustive
outcomes. All stakes are combined into a single pool. After the outcome is
realized, the operator deducts a fixed proportion (the *takeout* or *track take*)
and distributes the remaining pool proportionally among those who backed the
winning outcome.

**Origin.** The mechanism was invented by the French entrepreneur **Pierre Oller**
in the 1860s as an alternative to bookmaking, which Oller regarded as
exploitative. It was mechanized in the 20th century by the *totalizator* (the
"tote board"), an automatic odds-computing machine, and remains the dominant
settlement system for horse and greyhound racing worldwide.

**Payout formula.** Let there be $n$ outcomes. Let $W_i$ be the total amount
wagered on outcome $i$, and let $W = \sum_{i=1}^{n} W_i$ be the total pool. Let
$\tau \in [0,1)$ be the takeout rate. If outcome $j$ wins, the net pool
$(1-\tau)W$ is distributed pro rata to the winning bets. A bettor who staked $w$
on the winning outcome $j$ receives

$$\text{Payout} = w \cdot \frac{(1-\tau)\,W}{W_j}.$$

The implied **parimutuel odds** (payout per unit staked) on outcome $i$ are
$(1-\tau)\,W / W_i$, and the implied probability of outcome $i$ is $p_i = W_i / W$.
A defining feature is that the *odds are not fixed at bet time*: they are
determined endogenously by the final distribution of the pool, so the operator
bears no risk on the outcome, the bettors collectively fund all payouts. This
self-funding, risk-free-to-the-house property distinguishes parimutuel betting
from fixed-odds bookmaking.

> Reference implementation: [`mechanisms/parimutuel.py`](../mechanisms/parimutuel.py) (`ParimutuelPool`).

### 2. The Favorite–Longshot Bias

The most robust empirical regularity in parimutuel betting markets is the
**favorite–longshot bias (FLB)**: bettors systematically overbet longshots
(low-probability outcomes) and underbet favorites. As a result, the expected
return on favorites is higher (less negative) than on longshots.

- **Thaler & Ziemba (1988), "Anomalies: Parimutuel Betting Markets."** A widely
  cited synthesis in the JEP "Anomalies" series. Documents the FLB and frames
  these markets as a clean testbed for studying market efficiency and behavioral
  biases.
- **Snowberg & Wolfers (2010), "Explaining the Favorite–Long Shot Bias."** Using a
  large dataset, constructs a test distinguishing *risk-loving preferences* from
  *misperception of probabilities*, finding the evidence favors misperceptions
  (probability-weighting).
- Earlier originating work: **Ali (1977), "Probability and Utility Estimates for
  Racetrack Bettors."**

### 3. Parimutuel Markets as Information Aggregation

Because the final pool fractions $W_i/W$ encode the market's *collective
subjective probability*, parimutuel markets function as information aggregation
mechanisms. Empirically, racetrack favorites are remarkably well-calibrated,
which is part of why these markets are treated as a laboratory for the *wisdom of
crowds* and the efficient-markets hypothesis, the conceptual bridge from
gambling markets to **prediction markets**.

### 4. Combinatorial and Call-Auction Parimutuel Markets

A line of work in financial engineering generalized the mechanism to a
**call-auction parimutuel** that can price *combinatorial* and *contingent* claims
while preserving the self-funding property.

- **Lange & Economides (2005), "A Parimutuel Market Microstructure for Contingent
  Claims."** Introduces a call-auction parimutuel mechanism for trading contingent
  claims; the mechanism commercialized by Longitude Inc. and used in bank
  "economic derivatives" auctions.
- **Baron & Lange (2007), *Parimutuel Applications in Finance: New Markets for New
  Risks*.** A book-length treatment of applying parimutuel principles to financial
  markets.

### 5. Dynamic Parimutuel Markets

- **Pennock (2004), "A Dynamic Pari-Mutuel Market for Hedging, Wagering, and
  Information Aggregation" (ACM EC'04).** Introduces the **Dynamic Parimutuel
  Market (DPM)**, combining the infinite-liquidity / no-operator-risk advantages
  of a pool with the continuous price discovery of a double auction. Shares are
  priced by a cost function so prices change continuously; the mechanism behind
  the *Yahoo! Buzz* market and a direct precursor to Hanson's market scoring
  rules.

> Reference implementation: [`mechanisms/parimutuel.py`](../mechanisms/parimutuel.py) (`DynamicParimutuelMarket`).

---

## Part II: Proper Scoring Rules

### 6. Definition

A **scoring rule** $S(\mathbf{p}, i)$ assigns a reward (or penalty) to a
probabilistic forecast $\mathbf{p}$ after outcome $i$ is observed. Writing the
expected score under true beliefs $\mathbf{q}$ of reporting $\mathbf{p}$ as
$S(\mathbf{p}, \mathbf{q}) = \sum_i q_i\,S(\mathbf{p}, i)$, the rule is **proper**
if $S(\mathbf{q},\mathbf{q}) \ge S(\mathbf{p},\mathbf{q})$ for all
$\mathbf{p},\mathbf{q}$, and **strictly proper** if equality holds only when
$\mathbf{p}=\mathbf{q}$. Propriety makes truthful reporting incentive-compatible;
strict propriety makes it the *unique* optimum.

### 7. The Three Classic Rules

For a forecast $\mathbf{p}$ over $n$ outcomes with realized outcome $j$:

**Logarithmic score** (Good, 1952), the unique smooth strictly proper *local*
rule: $S_{\log}(\mathbf{p}, j) = \ln p_j$. Underlies maximum likelihood, the LMSR,
and cross-entropy/log-loss.

**Brier / quadratic score** (Brier, 1950), strictly proper and bounded:
$S_{\text{Brier}} = 2p_j - \sum_i p_i^2 - 1$, often written as the loss
$\sum_i (p_i - \mathbf{1}\{i=j\})^2$.

**Spherical score**: $S_{\text{sph}}(\mathbf{p}, j) = p_j / \lVert\mathbf{p}\rVert_2$.

> Reference implementation: [`mechanisms/scoring_rules.py`](../mechanisms/scoring_rules.py).

### 8. Canonical References and Characterization

- **Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and
  Estimation"** (JASA). The standard modern reference; establishes the
  correspondence between strictly proper rules and **convex functions / Bregman
  divergences**, and popularized the CRPS and energy score.
- **Savage (1971), "Elicitation of Personal Probabilities and Expectations."** The
  **Savage representation**: the expected-score function $G(\mathbf{q}) =
  S(\mathbf{q},\mathbf{q})$ is convex and the rule is recovered from its
  subgradients.
- **Schervish (1989), "A General Method for Comparing Probability Assessors."**
  Characterizes proper scoring rules (binary) as mixtures of elementary
  threshold rules.
- Originating works: **Brier (1950)**, **Good (1952)**, **McCarthy (1956)**,
  **Matheson & Winkler (1976)** (CRPS).

### 9. From Scoring Rules to Market Scoring Rules

A **market scoring rule (MSR)** maintains a public probability report; any trader
may move it from $\mathbf{p}_{\text{old}}$ to $\mathbf{p}_{\text{new}}$ and on
resolution receives $S(\mathbf{p}_{\text{new}}, j) - S(\mathbf{p}_{\text{old}}, j)$.
Because the rule is proper, each trader is incentivized to report truthfully, and
the market maker's worst-case loss is bounded. Hanson's **LMSR** is the MSR built
from the logarithmic score, see
[`market-scoring-rules-and-amms.md`](market-scoring-rules-and-amms.md).

### 10. CRPS: Scoring Distributional Forecasts

For a real-valued forecast given by a predictive CDF $F$ and realized value $y$,
the **Continuous Ranked Probability Score** is

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \big(F(x) - \mathbf{1}\{x \ge y\}\big)^2\,dx
 = \mathbb{E}_F|X - y| - \tfrac{1}{2}\mathbb{E}_F|X - X'|,$$

a strictly proper rule reported in the units of the observation, reducing to
absolute error for a point forecast. Its multivariate generalization is the
**energy score** (see
[`perps-cda-monteprediction.md`](perps-cda-monteprediction.md)).

> Reference implementation: [`mechanisms/scoring_rules.py`](../mechanisms/scoring_rules.py) (`crps_ensemble`, `energy_score`).

---

## BibTeX

See the consolidated [`bibliography.bib`](bibliography.bib).

```bibtex
@article{thaler1988anomalies,
  author  = {Thaler, Richard H. and Ziemba, William T.},
  title   = {Anomalies: Parimutuel Betting Markets---Racetracks and Lotteries},
  journal = {Journal of Economic Perspectives}, year = {1988},
  volume  = {2}, number = {2}, pages = {161--174}}

@article{snowberg2010explaining,
  author  = {Snowberg, Erik and Wolfers, Justin},
  title   = {Explaining the Favorite--Long Shot Bias: Is It Risk-Love or Misperceptions?},
  journal = {Journal of Political Economy}, year = {2010},
  volume  = {118}, number = {4}, pages = {723--746}}

@inproceedings{pennock2004dynamic,
  author    = {Pennock, David M.},
  title     = {A Dynamic Pari-Mutuel Market for Hedging, Wagering, and Information Aggregation},
  booktitle = {Proceedings of the 5th ACM Conference on Electronic Commerce (EC '04)},
  year = {2004}, pages = {170--179}, publisher = {ACM}}

@article{gneiting2007strictly,
  author  = {Gneiting, Tilmann and Raftery, Adrian E.},
  title   = {Strictly Proper Scoring Rules, Prediction, and Estimation},
  journal = {Journal of the American Statistical Association}, year = {2007},
  volume  = {102}, number = {477}, pages = {359--378}}
```

> **Sourcing note.** Web verification was unavailable when this file was drafted;
> bibliographic details are drawn from established knowledge of these canonical
> works. Page numbers and venue details are worth a spot-check before relying on
> them (especially the working-paper-vs-journal dating of Lange & Economides and
> the Hanson 2003 venue).
