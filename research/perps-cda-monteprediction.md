# Perpetual Futures, Continuous Double Auctions, and Distributional Forecasting

Three mechanisms that round out the survey: the **perpetual futures** funding
tether, the **continuous double auction** that underlies modern exchanges, and
the **sample-based distributional forecasting** of monteprediction.com.

---

## Section 1 — Perpetual Futures (Perps)

### 1.1 Origin: Shiller's perpetual futures vs. crypto perps

The conceptual ancestor is **Shiller (1993), "Measuring Asset Values for Cash
Settlement in Derivative Markets: Hedonic Repeated Measures Indices and Perpetual
Futures"** (J. Finance). Shiller wanted liquid, cash-settled derivatives on
illiquid real-economy risks (home prices, national income). Two contributions:

- **Hedonic repeated-measures indices** — a manipulation-resistant index from
  sparse, heterogeneous transactions, so an underlier without a continuous traded
  price can still be measured for settlement.
- **Perpetual futures / perpetual claims** — a derivative that never expires, kept
  tethered to the index by a stream of periodic cash settlements tied to a
  dividend/rent-like income measure, so hedgers of long-lived assets need not roll
  expiring contracts.

**How crypto perps differ.** Crypto perpetual swaps inherit the skeleton (no
expiry + periodic cash flow that anchors to an index) but replace the anchoring
mechanism. Shiller anchored via an exogenous income flow because his underliers
had no continuous price; crypto underliers *do*, so the periodic cash flow is a
**market-driven funding rate keyed to the perp's premium/discount to spot**,
exchanged peer-to-peer between longs and shorts.

### 1.2 Crypto perpetual swaps: BitMEX (2016)

The modern crypto perpetual swap launched on **BitMEX** in 2016 with **XBTUSD**,
widely credited as the first crypto perpetual and the design that established the
funding-rate convention now standard industry-wide. XBTUSD is an *inverse*
contract (quoted in USD, margined and settled in BTC).

### 1.3 The funding rate mechanism

Funding makes it costly to sit on whichever side pushes the perp away from the
index, incentivizing arbitrageurs to push it back.

- **Positive** funding → **longs pay shorts** (perp at a premium).
- **Negative** funding → **shorts pay longs** (perp at a discount).
- `Funding Payment = Position Notional × Funding Rate`, exchanged directly between
  traders. Classic interval: every **8 hours**.

**BitMEX funding formula:**

$$F = P + \operatorname{clamp}\big(I - P,\ -0.05\%,\ +0.05\%\big)$$

where $P$ is the time-weighted **premium index** and $I$ is the **interest-rate**
term. When the premium is small, $F$ sits near $I$; when large, the premium term
drives funding to push the perp back to the index.

> Reference implementation: [`mechanisms/perp.py`](../mechanisms/perp.py).

### 1.4 Mark price, index price, margin, liquidation

- **Index Price** — external reference, typically a composite of spot prices.
- **Last Price** — most recent perp trade; manipulable in thin books.
- **Mark Price** — used for unrealized PnL and liquidation; computed from the
  index plus a decaying funding basis rather than the perp's own last trade. This
  prevents *unfair liquidations* from momentary wicks.
- **Initial Margin** (~$1/\text{leverage}$) to open; **Maintenance Margin** to
  keep open; **liquidation** triggers when mark-price equity falls below
  maintenance margin.

### 1.5 Funding-rate arbitrage / cash-and-carry

When funding is persistently positive, go **long spot, short perp** in equal
notional: roughly delta-neutral, while the short perp collects funding each
interval. This arbitrage is the force that keeps the perp tethered to spot.

References: Shiller (1993); BitMEX documentation ("Perpetual Contracts Guide,"
"Funding," "Fair Price Marking"); dYdX protocol docs; Perpetual Protocol's
**vAMM** (a constant-product curve used purely for price discovery, with funding
anchoring it to the index).

---

## Section 2 — Continuous Double Auction (CDA) & Order Books

### 2.1 The CDA mechanism

A **continuous double auction** underlies virtually all modern electronic
exchanges. Many buyers and sellers submit orders at any time; trades execute
immediately whenever a buy and sell are compatible. The **limit order book**
aggregates resting orders: **bids** (best = highest), **asks** (best = lowest),
the **bid–ask spread**, and **depth** at each price level.

- A **limit order** rests if it cannot execute, *providing* liquidity.
- A **market order** executes against the best prices, *consuming* liquidity.
- **Price-time priority**: better prices first, then earliest-submitted (FIFO).
- A trade prints at the **resting (passive) order's price**, so price improvement
  accrues to the aggressor.

> Reference implementation: [`mechanisms/cda.py`](../mechanisms/cda.py).

### 2.2 Smith (1962): experimental markets

**Smith (1962), "An Experimental Study of Competitive Market Behavior"** (JPE).
The seminal experimental-economics study: double-auction markets with human
subjects converge rapidly to competitive equilibrium even with few traders and no
participant knowing the aggregate curves. Central to Smith's 2002 Nobel.

### 2.3 Gjerstad & Dickhaut (1998): belief-based agents

**Gjerstad & Dickhaut (1998), "Price Formation in Double Auctions"** (GEB).
Introduces the **GD agent**, which estimates the probability any candidate
bid/ask is accepted from recent history and submits the price maximizing expected
surplus — a standard benchmark adaptive strategy.

### 2.4 Gode & Sunder (1993): zero-intelligence traders

**Gode & Sunder (1993), "Allocative Efficiency of Markets with Zero-Intelligence
Traders"** (JPE). **Budget-constrained ZI traders** that submit *random*
bids/asks achieve near-100% allocative efficiency in a CDA — "the market is a
partial substitute for individual rationality." (ZI prices are far more volatile
than human traders', a point later addressed by ZIP agents.)

### 2.5 CDA vs. call auctions vs. market makers

| Dimension | CDA | Call / Batch Auction | Dealer / Market Maker |
|---|---|---|---|
| Timing | Continuous | Periodic clearing | Continuous quotes |
| Price formation | Sequence of bilateral trades | Single uniform clearing price | Dealer-set bid/ask |
| Liquidity source | Resting limit orders | Pooled batch | Dealer inventory |
| Strengths | Immediacy; price discovery | Consolidates liquidity; resists gaming | Guaranteed immediacy |
| Weaknesses | Fleeting quotes; latency races | No continuous immediacy | Wider spreads |

**Budish, Cramton & Shim (2015), "The High-Frequency Trading Arms Race"** (QJE)
proposed **frequent batch auctions** as a market-design remedy to the CDA latency
arms race.

---

## Section 3 — Monteprediction & Distributional Forecasting

### 3.1 monteprediction.com

**monteprediction.com** is a forecasting contest associated with **Peter Cotton**
(microprediction). Its distinguishing feature: participants forecast by submitting
a **set of Monte Carlo sample points** — an empirical, sample-based representation
of a multivariate distribution — rather than a point estimate or parametric
distribution. The "cloud" of samples is scored against the realized outcome by a
proper scoring rule in the **energy score / Monte Carlo CRPS** family, rewarding
forecasts well-centered on and appropriately dispersed around the truth. An
open-source `monteprediction` Python package supports assembling and submitting
samples.

The contest targets **multivariate financial outcomes** (e.g. joint weekly returns
of the eleven SPDR sector ETFs), so the dependence-aware nature of the energy
score is essential: a forecaster is rewarded for capturing the *joint*
distribution, not just each margin.

> Operational details (sample count, target variables, cadence, prizes) change
> over time — verify on the live site.

### 3.2 The energy score and sample-based scoring

The **energy score** is the multivariate generalization of the CRPS. For a
predictive distribution $P$ (with independent draws $X, X'$) and realized $y$:

$$\mathrm{ES}(P, y) = \mathbb{E}\lVert X - y\rVert - \tfrac12\,\mathbb{E}\lVert X - X'\rVert$$

(generally with exponent $\beta \in (0,2)$; $\beta=1$ above). It is **strictly
proper**, negatively oriented, reduces to the CRPS in one dimension, and is
related to the **energy distance** of Székely & Rizzo. Crucially it is easy to
estimate from samples: the first term is the average distance from samples to $y$,
the second the average pairwise distance among samples — a Monte-Carlo estimator
needing only the sample cloud, never a density. This is what makes "submit a bag
of samples" a well-founded forecasting interface. (Gneiting & Raftery 2007.)

> Reference implementation: [`mechanisms/scoring_rules.py`](../mechanisms/scoring_rules.py) (`energy_score`, `crps_ensemble`).

### 3.3 Microprediction: Building an Open AI Network

**Cotton (2022), *Microprediction: Building an Open AI Network*** (MIT Press).
Lays out crowdsourced *distributional* prediction: continuously running open
prediction streams where algorithms repeatedly submit distributional forecasts
evaluated by proper scoring rules, composed into larger "open AI network"
services. monteprediction.com is a concrete, contest-shaped instantiation for
multivariate financial outcomes.

### 3.4 Sample-based forecasts, parimutuels, and scoring-rule mechanisms

There is a deep equivalence between **proper scoring rules** and **market/betting
mechanisms** for eliciting beliefs:

- A **proper scoring rule** pays a forecaster so that truthful reporting maximizes
  expected payoff — the single-agent analog of a market price.
- **Parimutuel betting** divides a pool among winners pro rata, so implied odds
  reflect the aggregate distribution of bets; settlement is batch, like a call
  auction over outcomes.
- **Market scoring rules** (Hanson's LMSR) literally turn a proper scoring rule
  into a market maker. This is the bridge unifying this repository: the same
  proper-scoring-rule machinery that scores monteprediction.com submissions is,
  wrapped as a market maker, the engine of a prediction market; a CDA aggregates
  beliefs through *trades*; a perp funding rate aggregates beliefs about the
  spot–perp basis through *carry payments*.

---

## BibTeX

See the consolidated [`bibliography.bib`](bibliography.bib).

```bibtex
@article{shiller1993measuring,
  author  = {Shiller, Robert J.},
  title   = {Measuring Asset Values for Cash Settlement in Derivative Markets:
             Hedonic Repeated Measures Indices and Perpetual Futures},
  journal = {The Journal of Finance}, year = {1993},
  volume  = {48}, number = {3}, pages = {911--931}}

@article{smith1962experimental,
  author  = {Smith, Vernon L.},
  title   = {An Experimental Study of Competitive Market Behavior},
  journal = {Journal of Political Economy}, year = {1962},
  volume  = {70}, number = {2}, pages = {111--137}}

@article{gode1993allocative,
  author  = {Gode, Dhananjay K. and Sunder, Shyam},
  title   = {Allocative Efficiency of Markets with Zero-Intelligence Traders},
  journal = {Journal of Political Economy}, year = {1993},
  volume  = {101}, number = {1}, pages = {119--137}}

@book{cotton2022microprediction,
  author = {Cotton, Peter},
  title  = {Microprediction: Building an Open AI Network},
  publisher = {MIT Press}, year = {2022}, address = {Cambridge, MA}}

@misc{monteprediction,
  author = {Cotton, Peter},
  title  = {Monteprediction: A Monte Carlo Distributional Forecasting Contest},
  howpublished = {\url{https://monteprediction.com}}}
```

> **Sourcing note.** Web verification was unavailable when this file was drafted.
> Live monteprediction.com operational details, exchange-doc URLs/parameters, the
> Shiller page range, the book ISBN, and the Székely–Rizzo energy-distance
> citation are worth confirming.
