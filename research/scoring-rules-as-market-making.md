# Proper scoring rules *are* market making

There is a small, cute observation that ties together half of this repository:
**a market maker is a forecaster being scored, and the scoring rule it is being
scored by is read straight off Savage's characterisation of proper scoring
rules.** Once you see it, Hanson's LMSR, cost-function market makers, constant-
function AMMs, and the no-regret view of prediction markets stop being four
topics and become one convex function looked at from four sides.

This note states the connection carefully. It is the longer version of the
keystone paragraph in
[composition and the algebra of mechanisms](composition-and-the-algebra-of-mechanisms.md);
references are citekeys in [`bibliography.bib`](bibliography.bib).

Cross-links to code: [`scoring_rules`](../mechanisms/scoring_rules.py),
[`lmsr`](../mechanisms/lmsr.py), [`cmm`](../mechanisms/cmm.py),
[`amm`](../mechanisms/amm.py), [`aggregation`](../mechanisms/aggregation.py).

---

## 1. Savage's characterisation

Fix a finite outcome space $\{1,\dots,n\}$. A forecaster reports a probability
vector $p\in\Delta$ and is paid $S(p,i)$ when outcome $i$ occurs. Writing the
expected score of report $p$ under belief $q$ as
$S(p;q)=\sum_i q_i\,S(p,i)$, the rule is **proper** if honesty is optimal,
$S(q;q)\ge S(p;q)$ for all $p$, and **strictly proper** if $p=q$ is the unique
maximiser.

Savage (1971) [savage1971elicitation], anticipated by McCarthy (1956)
[mccarthy1956measures] and sharpened by Gneiting & Raftery (2007)
[gneiting2007strictly], characterised *every* such rule. Define the
**expected score of an honest forecaster**

$$G(q)\;=\;S(q;q)\;=\;\sup_{p}\,S(p;q).$$

As a supremum of functions linear in $q$, $G$ is convex — it is the "generalised
entropy" or *Bayes risk* of the prediction problem. The characterisation is that
a (regular) rule is proper **iff** it is the supporting hyperplane of some convex
$G$:

$$S(p,i)\;=\;G(p)\;+\;\big\langle\,G'(p),\;e_i-p\,\big\rangle,\qquad G'(p)\in\partial G(p).$$

Every convex $G$ generates a proper rule this way, and the choice of subgradient
$G'$ is the only freedom. The honesty gap is then forced to be a **Bregman
divergence**, non-negative exactly because $G$ is convex:

$$S(q;q)-S(p;q)\;=\;G(q)-G(p)-\big\langle\,G'(p),\,q-p\,\big\rangle\;=\;D_G(q,p)\;\ge\;0.$$

The log score $S(p,i)=\log p_i$ has $G(q)=\sum_i q_i\log q_i$ (negative Shannon
entropy) and $D_G=\mathrm{KL}(q\Vert p)$; the Brier/quadratic score has
$G(q)=\lVert q\rVert^2-1$ and $D_G$ the squared Euclidean distance. Calibration
and refinement (Bregman geometry of forecasts) are the same picture
[dawid2007geometry], and which *functionals* of $q$ a single subgradient can
elicit is exactly the theory of property elicitation and elicitability
[lambert2008eliciting; gneiting2011making; abernethy2012characterization;
fissler2016higher].

## 2. The market-making reading

Now run the construction backwards. Put an automated market maker in the room
holding an inventory $\theta\in\mathbb{R}^n$ of Arrow–Debreu securities (security
$i$ pays \$1 if outcome $i$ occurs). Give it a convex **cost function** $C$, so a
trader who buys the bundle $\delta$ pays $C(\theta+\delta)-C(\theta)$, and define
the **instantaneous prices** as the gradient

$$\pi(\theta)\;=\;\nabla C(\theta)\in\Delta.$$

This is exactly the cost-function market maker of Chen & Pennock and
Abernethy–Chen–Vaughan [chen2007utility; abernethy2013efficient;
abernethy2011optimization] — see [`cmm`](../mechanisms/cmm.py).

The claim is that **quoting the price vector $\pi$ is identical to a forecaster
reporting the belief $\pi$ and being scored by the proper rule of §1.** Concretely:

- The market maker's cost $C$ is the **convex conjugate** of a generalised entropy
  $R$ on the simplex, $C(\theta)=\sup_{p\in\Delta}\{\langle p,\theta\rangle-R(p)\}$,
  and that $R$ is the Savage generator $G$ of a proper scoring rule.
- A trader who believes the truth is $q$ moves the market from its current price
  $p=\pi(\theta)$ to $q$ by buying the bundle that makes $\nabla C$ equal $q$.
  At settlement under outcome $i$ the trader's net payoff is precisely the
  **change in score**, $S(q,i)-S(p,i)$.
- Therefore the trader's *expected* profit under her own belief is

$$\mathbb{E}_{q}\big[S(q,i)-S(p,i)\big]\;=\;S(q;q)-S(p;q)\;=\;D_G(q,p).$$

So the trader is paid the Bregman divergence between her belief and the price she
found — the elicitation regret of §1, now denominated in money. Hanson's market
scoring rule [hanson2007logarithmic] is literally this: *the next trader pays off
the previous trader's score.* The market maker is a forecaster of last resort,
and each trade is one move in a relay of proper-scoring reports.

## 3. One convex function, two coordinate systems

The content of §2 is a single Legendre/Fenchel duality. The same potential lives

- on the **simplex**, as the generalised entropy $R=G$ — the *report space* of a
  scoring rule; and
- on **share space** $\mathbb{R}^n$, as the cost $C=R^{\ast}$ — the *inventory
  space* of a market maker.

The two gradients are inverse maps, $\nabla R:\Delta\to\mathbb{R}^n$ and
$\nabla C:\mathbb{R}^n\to\Delta$, which is why **market prices are implied
probabilities**: $\pi=\nabla C=(\nabla R)^{-1}$. For LMSR the entropy is
$R(p)=b\sum_i p_i\log p_i$ and the cost is the log-sum-exp
$C(\theta)=b\log\sum_i e^{\theta_i/b}$, with prices the softmax
$\pi(\theta)=\mathrm{softmax}(\theta/b)$ — see [`lmsr`](../mechanisms/lmsr.py).
The liquidity parameter $b$ scales the entropy and hence the curvature of $C$.

## 4. Three properties you get for free

Reading the market maker as "scoring rule in dual coordinates" hands you the
standard prediction-market facts as one-line corollaries.

- **Truthful trading.** Expected profit is $D_G(q,p)\ge 0$ with equality only at
  $q=p$ (strict convexity). A trader maximises expected profit by moving the
  price to her true belief. *Properness is incentive compatibility* — no
  separate game-theoretic argument needed.
- **Bounded subsidy.** Payments telescope: across any sequence of trades the
  market maker's total payout is $S(\theta_{\text{final}},i)-S(\theta_{0},i)$,
  bounded by the **range of $G$** — $b\log n$ for LMSR. The subsidy that buys
  liquidity equals the entropy you injected.
- **No regret.** Operating the market is Follow-the-Regularised-Leader with
  regulariser $R$; the price path is the mirror-descent play and the worst-case
  market loss is the online-convex-optimisation regret bound
  [chen2010newunderstanding; abernethy2013efficient; frongillo2012interpreting;
  hazan2016introduction]. The same $R$ is simultaneously the entropy, the
  scoring generator, the regulariser, and (conjugate) the cost.

## 5. Constant-function AMMs are the same animal

The DeFi constant-function market maker (Uniswap, Balancer, Curve) holds reserves
on a level set of a trading function $\varphi$ — see [`amm`](../mechanisms/amm.py).
Frongillo, Papireddygari & Waggoner [frongillo2024axiomatic] prove that, under
natural axioms, **every CFMM is equivalent to a cost-function prediction market**,
hence to a proper scoring rule; the trading function is the same convex potential
read in reserve coordinates (see also the axiomatic and geometric accounts
[schlegel2023axioms; angeris2024geometry; frongillo2018axiomatic]). So Uniswap
and LMSR are one mechanism in two coordinate systems — the concrete reason the
AMM and scoring-rule pages here share machinery. The earlier risk-measure and
exponential-family framing [abernethy2015risk] says the same thing in the
language of convex risk.

## 6. The rest of the web

The convex spine reaches the other mechanisms by changing *what plays the role of
the outcome* $i$.

- **Property elicitation.** The subgradient $G'(p)$ is precisely what the market
  "reads off". Only properties expressible as such a gradient are directly
  elicitable; means are gradients of $G$, while quantiles, variance and expected
  shortfall need auxiliary structure [lambert2008eliciting;
  abernethy2012characterization; fissler2016higher].
- **Peer prediction.** Replace the (unavailable) outcome with a peer's report and
  score against it; truthfulness now requires the score's Bregman geometry to
  line up with the information structure — the correlated-agreement and
  mutual-information mechanisms [shnayder2016informed; kong2019information].
  Same convexity, different "ground truth".
- **Local scoring.** On a continuous outcome space where normalisation is
  intractable, restrict the generator to depend on the density only through its
  log-derivatives: Hyvärinen score matching [hyvarinen2005score;
  parry2012proper] is the *local* form of the same proper-scoring generator —
  see [`scoring_rules`](../mechanisms/scoring_rules.py).
- **Parimutuel.** A self-financing limit in which the "cost" is the pooled stake
  and prices are stake shares; bounded loss is automatic because the operator
  never subsidises [pennock2004dynamic].

---

### One-line summary

Pick a convex function on the simplex. Its supporting hyperplanes are a proper
scoring rule (Savage); its conjugate is a market maker's cost function (Hanson);
its gradient is the price map; its Bregman divergence is the trader's profit and
the elicitation regret at once. Everything else in this repository is that
function wearing different clothes.
