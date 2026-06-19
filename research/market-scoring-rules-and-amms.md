# Market Scoring Rules (LMSR) and Automated Market Makers (AMMs)

This document surveys two closely related families of automated market makers:
the *market scoring rules* tradition from the prediction-markets literature
(Hanson's LMSR and its cost-function generalizations), and the *constant function
market makers* (CFMMs) that emerged from decentralized finance (DeFi). The two
literatures developed largely independently but are unified by the cost-function
/ convex-conjugate framework.

---

## Part I, Market Scoring Rules and Cost-Function Market Makers

### 1. Hanson's Logarithmic Market Scoring Rule (LMSR)

**Hanson (2003), "Combinatorial Information Market Design"** introduces the
*market scoring rule* concept: a proper scoring rule, normally used to elicit one
agent's report, becomes a market maker when each trader who changes the market's
probability estimate must pay off the previous trader's score and is paid off by
the next. Especially suited to *combinatorial* markets.

**Hanson (2007), "Logarithmic Market Scoring Rules for Modular Combinatorial
Information Aggregation"** is the canonical exposition of LMSR, establishing its
modularity and bounded-loss property.

#### Cost-function formulation

Over a vector of outstanding share quantities $q = (q_1,\dots,q_n)$:

$$C(q) = b \cdot \log\!\left(\sum_{i=1}^{n}\exp\!\left(\frac{q_i}{b}\right)\right).$$

A trader moving the quantity vector from $q$ to $q'$ pays $C(q') - C(q)$. The
**instantaneous price** is the softmax gradient

$$p_i(q) = \frac{\partial C}{\partial q_i}
        = \frac{\exp(q_i/b)}{\sum_j \exp(q_j/b)},$$

automatically non-negative and summing to one, the market's probability
estimate.

#### The liquidity parameter $b$

$b > 0$ controls depth vs. worst-case subsidy: large $b$ → deep, low-impact
market with larger potential subsidy; small $b$ → thin, responsive market with
smaller subsidy.

#### Bounded-loss property

The market maker's worst-case loss over all trading paths and final states is

$$b \cdot \log n,$$

which is precisely the budget a subsidizing institution pays for the information
the market aggregates.

#### Relationship to proper scoring rules

The logarithmic scoring rule $S(\mathbf{p}, i) = b\log p_i$ is the prototypical
strictly proper rule; LMSR is its sequential "market" form. The cost function
$C(q)$ is the convex conjugate whose induced prices are the log rule's optimal
report, the duality between Bregman-divergence proper scoring rules and convex
cost functions later made explicit by Abernethy, Chen & Wortman Vaughan.

> Reference implementations: [`mechanisms/lmsr.py`](../mechanisms/lmsr.py) and
> [`mechanisms/cmm.py`](../mechanisms/cmm.py).

### 2. Cost-function-based market makers in general

**Abernethy, Chen & Wortman Vaughan (2013), "Efficient Market Making via Convex
Optimization, and a Connection to Online Learning"** (ACM TEAC) is the unifying
framework. A cost-function market maker is defined by a *convex potential* $C$:
prices are $\nabla C$, *no-arbitrage* ⇔ $C$ convex, *bounded loss* ⇔ bounded
gradient range. The paper draws an explicit equivalence with **online convex
optimization / Follow-the-Regularized-Leader**, with the cost function as the
regularizer and the liquidity parameter as the learning rate. LMSR is the special
case where $C$ is the scaled log-partition function and the regularizer is
negative entropy.

### 3. Othman & Sandholm: liquidity-sensitive market makers

Plain LMSR fixes liquidity $b$ and runs at a guaranteed expected loss.

- **Othman, Pennock, Reeves & Sandholm (2010/2013), "A Practical
  Liquidity-Sensitive Automated Market Maker."** Replaces $b$ with
  $b(q) = \alpha\sum_i q_i$, so depth grows with volume and the maker earns a
  positive spread (can run at a profit). The trade-off is that prices no longer
  normalize exactly to one (a small over-round).
- **Othman & Sandholm (2011), "Liquidity-Sensitive Automated Market Makers via
  Homogeneous Risk Measures."** Grounds liquidity sensitivity in the theory of
  homogeneous (coherent) risk measures.
- **Othman & Sandholm (2011), "Automated Market-Making in the Large: The Gates
  Hillman Prediction Market"**, a large real-world deployment.

---

## Part II, Constant Function Market Makers (DeFi AMMs)

### 4. CFMMs and constant product

A **constant function market maker (CFMM)** holds reserves
$R = (R_1,\dots,R_n)$ and accepts any trade leaving a *trading function*
$\varphi(R)$ unchanged (up to fees). The most influential instance is the
**constant product market maker (CPMM)** (Uniswap), with two-asset invariant

$$x \cdot y = k.$$

The marginal price is $p = y/x$, and a fee $\gamma$ (e.g. 0.3%) is taken on the
input. Liquidity providers (LPs) deposit both assets and earn fees.

- **Angeris, Kao, Chiang, Noyes & Chitra (2019/2021), "An Analysis of Uniswap
  Markets."** The foundational academic analysis: models Uniswap as a
  convex-optimization / arbitrage system, proves conditions under which the AMM
  price tracks an external reference (via rational arbitrageurs), and
  characterizes LP risks.
- **Angeris & Chitra (2020), "Improved Price Oracles: Constant Function Market
  Makers"** (ACM AFT). Generalizes to arbitrary CFMMs defined by a concave,
  increasing trading function and analyzes them as decentralized price oracles.

> Reference implementation: [`mechanisms/amm.py`](../mechanisms/amm.py).

### 5. Constant mean (Balancer) and StableSwap (Curve)

- **Martinelli & Mushegian (2019), Balancer whitepaper.** Generalizes to a
  **constant weighted-geometric-mean** invariant $\prod_i R_i^{w_i} = k$,
  $\sum_i w_i = 1$, a self-rebalancing index fund; constant product is the
  two-asset equal-weight case.
- **Egorov (2019), StableSwap (Curve) whitepaper.** A hybrid interpolating between
  the constant-sum function (zero slippage, for pegged assets) and constant
  product (no-depletion guarantee) via an amplification coefficient $A$.

### 6. Impermanent (divergence) loss

Because arbitrageurs trade the pool back to the external price, an LP ends up
holding more of the depreciating asset. For a constant-product pool, if the price
ratio changes by a factor $r$, the loss relative to holding is

$$\text{IL}(r) = \frac{2\sqrt{r}}{1+r} - 1 \le 0,$$

zero at $r=1$ and strictly negative otherwise. Fees offset this; an LP profits
only when accumulated fees exceed impermanent loss.

- **Milionis, Moallemi, Roughgarden & Zhang (2022), "Automated Market Making and
  Loss-Versus-Rebalancing."** Introduces **LVR**, a cleaner metric than
  impermanent loss that isolates the cost LPs pay to arbitrageurs, benchmarking
  against a continuously rebalancing strategy.

> Reference implementation: [`mechanisms/amm.py`](../mechanisms/amm.py) (`impermanent_loss`).

### 7. Concentrated liquidity (Uniswap v3)

**Adams, Zinsmeister, Salem, Keefer & Robinson (2021), "Uniswap v3 Core."**
Introduces **concentrated liquidity**: LPs allocate capital to chosen price ranges
$[p_a, p_b]$, behaving like a constant-product market on shifted "virtual
reserves" within the range, far greater capital efficiency, at the cost of going
inactive once price exits the range.

### 8. The bridge between prediction-market AMMs and DeFi AMMs

Both families are special cases of the **cost-function / trading-function**
abstraction. A prediction-market maker is a convex potential $C(q)$ with prices
$\nabla C(q)$; a DeFi CFMM is a concave trading function $\varphi(R)$ with spot
prices from the ratio of its partials. Up to sign conventions and a change of
variables, these are the *same object viewed through convex duality*, convexity
⇔ no-arbitrage, bounded conjugate domain ⇔ bounded loss.

- **Angeris, Chitra, Evans & Boyd (2021/2023), "Constant Function Market Makers:
  Multi-Asset Trades via Convex Optimization."** Casts CFMM trading/routing as
  convex programs, paralleling Abernethy–Chen–Wortman Vaughan.

In short: Hanson's LMSR, the convex-optimization framework, Othman–Sandholm
liquidity sensitivity, and Uniswap/Balancer/Curve are all instances of
*convex-potential market making*. The prediction-market literature emphasized
bounded loss and information aggregation (a *subsidized* elicitation device); the
DeFi literature emphasized fees, impermanent loss / LVR, and capital efficiency (a
*fee-earning* liquidity venue), but the underlying mathematics is shared.

---

## BibTeX

See the consolidated [`bibliography.bib`](bibliography.bib).

```bibtex
@article{hanson2007lmsr,
  author  = {Hanson, Robin},
  title   = {Logarithmic Market Scoring Rules for Modular Combinatorial Information Aggregation},
  journal = {The Journal of Prediction Markets}, year = {2007},
  volume  = {1}, number = {1}, pages = {3--15}}

@article{abernethy2013efficient,
  author  = {Abernethy, Jacob and Chen, Yiling and Wortman Vaughan, Jennifer},
  title   = {Efficient Market Making via Convex Optimization, and a Connection to Online Learning},
  journal = {ACM Transactions on Economics and Computation}, year = {2013},
  volume  = {1}, number = {2}, articleno = {12}}

@article{angeris2021uniswap,
  author  = {Angeris, Guillermo and Kao, Hsien-Tang and Chiang, Rei and Noyes, Charlie and Chitra, Tarun},
  title   = {An Analysis of Uniswap Markets},
  journal = {Cryptoeconomic Systems}, year = {2021}, volume = {1}, number = {1},
  note    = {arXiv:1911.03380}}

@misc{milionis2022lvr,
  author = {Milionis, Jason and Moallemi, Ciamac C. and Roughgarden, Tim and Zhang, Anthony Lee},
  title  = {Automated Market Making and Loss-Versus-Rebalancing},
  howpublished = {arXiv:2208.06046}, year = {2022}}
```

> **Sourcing note.** Web verification was unavailable when this file was drafted;
> DOIs, page numbers, and arXiv-to-published mappings (especially LVR and the
> multi-asset CFMM paper) are worth confirming against the primary sources.
