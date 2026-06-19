# Composition and the algebra of mechanisms

The mechanisms in this repository are usually presented as a list, scoring
rules, market makers, parimutuels, auctions, aggregators. But the interesting
structure is not the list; it is how they **compose**. This note sketches an
"algebra": a small set of objects and operators under which the mechanisms are
closed, so that pipelines of mechanisms can be assembled the way an engineer
assembles a data pipeline.

The operational discipline is borrowed from
[`microprediction/skaters`](https://github.com/microprediction/skaters) (a
dependency-free successor to the `timemachines` *skater* interface). Skaters
showed that if every stage speaks one type and threads its own state, a tiny
operator set composes freely. The same is true of these mechanisms, with one
substitution: a skater threads *model* state; a mechanism threads *wealth*.

Cross-links: [`scoring_rules`](../mechanisms/scoring_rules.py),
[`cmm`](../mechanisms/cmm.py), [`amm`](../mechanisms/amm.py),
[`lmsr`](../mechanisms/lmsr.py), [`aggregation`](../mechanisms/aggregation.py),
[`calibration`](../mechanisms/calibration.py); runnable demo
[`examples/sim_pipeline.py`](../examples/sim_pipeline.py); the
[nearest-the-pin paper](../papers/nearest-the-pin-parimutuel.md).

---

## 1. One object type: a distributional belief

A skater has the contract

```python
dists, state = f(y, state)        # dists : list[Dist] for horizons 1..k
```

where every payload is a `Dist`, a weighted Gaussian mixture exposing `.mean`,
`.std`, `.quantile`, `.logpdf`, `.cdf`. Transforms, ensembles, and search all
*consume and emit that same type*. That closure is the whole reason a handful of
operators suffice.

Every mechanism here already speaks the same language:

| Mechanism | As a map on distributions |
|-----------|---------------------------|
| scoring rule (`scoring_rules`) | `(Dist, outcome) → reward` |
| LMSR / cost-function maker (`lmsr`, `cmm`) | state `→ Dist` (prices *are* implied probabilities) |
| parimutuel (`parimutuel`) | stakes `→ Dist` (pool fractions are implied probabilities) |
| opinion pool (`aggregation`) | `[Dist] → Dist` |
| PIT / calibration (`calibration`) | `(Dist F, x) → z = Φ⁻¹(F(x))` |

So the mechanisms are already a category whose objects are distributional
beliefs. What skaters supplies is the missing **operational signature** that
turns them into composable pipeline stages. The mechanism analogue of the skater
contract is the same shape, with **wealth as the threaded state**:

```python
payouts, aggregate, wealth = M(reports, wealth, outcome)
#   reports   : list[Dist]   participants' beliefs      ← skater's y
#   wealth    : state          incentive-bearing         ← skater's state
#   aggregate : Dist           price / implied prob       ← skater's dists
```

A market maker *is* a skater whose state is the inventory/share vector `q` and
whose emitted `Dist` is the price. Hanson's construction is exactly the skater
move: **sequentialising a proper scoring rule over a wealth state turns it into a
market maker.**

## 2. The operators

Skaters expose a small operator set; each lifts to a relationship between
mechanisms.

**Conjugation**, `conjugate(skater, transform)` runs a model in a transformed
coordinate and maps back (`difference`, `standardize`, `ema`, …; all
invertible). It appears *twice* in the mechanism world, and the coincidence of
names is not accidental:

- *Statistical.* "One market determines a transformation, another tests whether
  it is uniform" is literally conjugation. Market A elicits a predictive CDF
  `F`; that `F` **is** the transform; the z-stream `z = Φ⁻¹(F(x))` is the
  conjugated coordinate; Market B trades against the null "`z` is iid N(0,1)".
- *Convex-analytic.* The CMM↔CFMM duality (`cmm` ↔ `amm`) is the **Legendre
  conjugate** of the potential: the trading-function market is the cost-function
  market viewed in dual coordinates.

**Ensemble = aggregation.** Skaters' `precision_weighted_ensemble` (∝ 1/MSE) is
an inverse-variance linear pool; `bayesian_ensemble` (log-likelihood with a
complexity penalty) is a logarithmic opinion pool, both implemented in
[`aggregation`](../mechanisms/aggregation.py). The
[nearest-the-pin pool](../papers/nearest-the-pin-parimutuel.md) is precisely a
**wealth-weighted ensemble whose weights are updated by realised score**, an
ensemble whose mixing weights are themselves a market.

**Residual chaining**, one stage models what the previous stage got wrong; a
correction/boosting market on the residual stream.

**Sequentialise**, the Hanson functor `score → market maker` (state = wealth).

**Pool**, the batch functor `score → parimutuel` (elicitation in one shot
rather than sequentially).

**Spec / grammar**, skaters serialise a pipeline to data (`to_json`/`from_json`
/`build`) and search over it. The mechanism analogue is a *market over
pipelines*: the architecture itself becomes the thing being elicited.

## 3. The generator: convex functions under Legendre duality

Underneath the operators is a single generating object, **convex functions,
with Legendre duality as the central involution.**

- A convex `G` generates a strictly proper scoring rule; the induced divergence
  is `G`'s **Bregman divergence** (Savage 1971; Banerjee et al. 2005),
  `scoring_rules`.
- A convex potential `C` generates a no-arbitrage cost-function market maker
  with prices `∇C` (Abernethy–Chen–Wortman Vaughan 2013), `cmm`.
- The conjugate `C*` generates the trading function / CFMM, `amm`.

Operations on convex functions then become operations on mechanisms:

| Operation on the potential | Effect on the mechanism |
|---------------------------|--------------------------|
| Legendre conjugation `C ↦ C*` | cost-function maker ↦ CFMM (the dual reading) |
| scaling `C ↦ bC` | the liquidity knob `b` (depth vs subsidy) |
| inf-convolution `C₁ □ C₂` | merging two liquidity sources into one book |
| add a state monad over wealth | static elicitation ↦ dynamic, competitive market |

This is the sense in which there is an *algebra*: objects are distributions; the
generators are convex potentials; the morphisms are
{sequentialise, pool, conjugate, ensemble, residual}; and composition closes
because every stage emits the same `Dist` the next stage consumes.

## 3a. The classic results, stated (with proof sketches)

The claim "convex functions generate the mechanisms" is not a slogan; it is a
chain of theorems. The conventions here are reward-form (higher is better); the
repository's `scoring_rules` uses the loss form, i.e. the negation. Outcomes lie
in $\{1,\dots,n\}$, a report is $p$ in the simplex $\Delta$, and the expected
score of reporting $p$ when one believes $q$ is
$S(p;q)=\sum_i q_i\, S(p,i)$.

A scoring rule is **proper** if honesty is optimal, $S(q;q)\ge S(p;q)$ for all
$p,q$, and **strictly proper** if equality forces $p=q$.

### Theorem 1 (Savage 1971; McCarthy 1956). Characterisation of proper scoring rules.

A regular scoring rule $S$ is proper **iff** there is a convex function
$G:\Delta\to\mathbb{R}$ with

$$S(p,i) \;=\; G(p) + \langle\, G'(p),\; e_i - p\,\rangle,
\qquad G'(p)\in\partial G(p),$$

where $e_i$ is the $i$-th unit vector. It is *strictly* proper iff $G$ is
*strictly* convex, and then $G(p)=S(p;p)$ is the **expected score of an honest
forecaster** (the "generalised entropy") and $G'(p)$ is its gradient.

**Proof sketch.**
*(⇐)* Because $\sum_i q_i(e_i-p)=q-p$, the expected score collapses to an affine
function of the belief, $S(p;q)=G(p)+\langle G'(p),\,q-p\rangle$. Hence

$$S(q;q)-S(p;q) \;=\; G(q)-G(p)-\langle G'(p),\,q-p\rangle \;=\; D_G(q,p)\ \ge\ 0,$$

the **Bregman divergence** of $G$, which is non-negative by the supporting-
hyperplane inequality for convex $G$ and zero only at $q=p$ when $G$ is strictly
convex. So $S$ is (strictly) proper.
*(⇒)* Define $G(p):=S(p;p)$. Properness says $G(q)=\max_p S(p;q)$, a pointwise
maximum over $p$ of functions that are **affine in $q$**; a pointwise sup of
affine functions is convex, so $G$ is convex. The maximiser at $q=p$ exhibits
$S(p,\cdot)$ as a supporting hyperplane of $G$ at $p$, which is precisely the
displayed formula with $G'(p)\in\partial G(p)$. Uniqueness of the maximiser
(strict properness) is equivalent to strict convexity. $\qquad\blacksquare$

The content of Theorem 1 is the dictionary **proper scoring rule ⇄ convex
function ⇄ Bregman divergence** (Gneiting & Raftery 2007; Banerjee et al. 2005).
The three classics in `scoring_rules` are three choices of $G$:

| Score | Generator $G(p)=S(p;p)$ | Bregman divergence $D_G(q,p)$ |
|-------|--------------------------|-------------------------------|
| logarithmic | $\sum_i p_i\log p_i$ (negative entropy) | KL divergence $\mathrm{KL}(q\Vert p)$ |
| Brier (quadratic) | $\lVert p\rVert_2^2$ | squared Euclidean $\lVert q-p\rVert_2^2$ |
| spherical | $\lVert p\rVert_2$ | angular separation of $q,p$ |

### Theorem 2 (Hanson 2007; Abernethy–Chen–Wortman Vaughan 2013). Scoring rule ⇒ market maker.

Let $R=-G$ be the convex "cost" conjugate generator and define the cost function
as its **Legendre–Fenchel conjugate**

$$C(\mathbf q)=\sup_{p\in\Delta}\big(\langle p,\mathbf q\rangle - R(p)\big),
\qquad \text{prices } \ \nabla C(\mathbf q)=\arg\max_{p}(\cdot)\in\Delta .$$

Then a trader moving inventory $\mathbf q\to\mathbf q'$ pays $C(\mathbf q')-C(\mathbf q)$,
prices are a valid probability vector (sum to one, non-negative), the market is
arbitrage-free (because $C$ is convex), and the market maker's **worst-case loss
is the range of $R$**,

$$\text{worst-case loss}=\sup_p R(p)-\inf_p R(p).$$

Taking $R(p)=b\sum_i p_i\log p_i$ (scaled negative entropy, i.e. $b\times$ the
log-score generator of Theorem 1) gives $C(\mathbf q)=b\log\sum_i e^{q_i/b}$,
**Hanson's LMSR**, with worst-case loss $b\log n$.

**Proof sketch.** Conjugate duality gives $\nabla C(\mathbf q)=\arg\max_p$, which
lies in $\Delta$, and $\langle \nabla C,\mathbf 1\rangle=1$ by the simplex
constraint, so prices are probabilities. Telescoping costs over any trade
sequence returning to $\mathbf q$ sum to zero, ruling out arbitrage. The maker's
loss after the market settles on outcome $i$ is bounded by how far $R$ can move
between its extremes, $\sup R-\inf R$; for $b\times$negentropy on $\Delta$ that
range is $b\log n$. The conjugacy $C=R^{*}$ is the formal content of "the log
score, run sequentially, *is* LMSR." $\qquad\blacksquare$

### Proposition 3. Cost-function maker ⇄ CFMM is Legendre duality.

A constant-function market maker holds reserves and accepts trades preserving a
concave trading function $\varphi$; a cost-function maker prices via convex $C$.
The two are the same object read through $C\leftrightarrow C^{*}$: the reachable
reserve set of the CFMM is the (sub)gradient image of $C$, and the price the
CFMM quotes equals $\nabla C$ at the corresponding inventory. Thus
[`amm`](../mechanisms/amm.py) and [`cmm`](../mechanisms/cmm.py) are conjugate
views of one potential, the "convex dual" edge in the
[relationship map](https://mechanisms.microprediction.org/map.html).

### Proposition 4 (Rosenblatt 1952; Dawid 1984). The PIT, which the critic exploits.

If $X$ has continuous CDF $F$ then $U=F(X)\sim\mathrm{Uniform}(0,1)$, and hence
$z=\Phi^{-1}(U)\sim N(0,1)$.

**Proof.** $\;\Pr(F(X)\le u)=\Pr(X\le F^{-1}(u))=F(F^{-1}(u))=u.\ \blacksquare$

This is the formal warrant for §4: a *correctly* elicited predictive
distribution makes the z-stream standard normal, so any detectable departure is
exactly a miscalibration the critic can be paid to find. (For discrete forecasts
one uses the randomised or mid-PIT; the demo uses the latter.)

## 4. Worked composition: an elicitation market feeding a calibration critic

The cleanest non-trivial pipeline chains two mechanisms through a feedback loop
(runnable: [`examples/sim_pipeline.py`](../examples/sim_pipeline.py)):

1. **Elicitation market (the transformation).** A pool of forecasters each
   report a predictive distribution; the wealth-weighted *linear opinion pool*
   is a single predictive distribution `F_t`. `F_t` is the transformation, the
   coordinate change that should turn outcomes into noise.
2. **Calibration critic (test for uniformity).** Apply the probability-integral
   transform `u_t = F_t(x_t)` (Dawid's prequential PIT, 1984); if `F_t` is right
   the `u_t` are Uniform(0,1) and `z_t = Φ⁻¹(u_t)` is standard normal. The critic
   measures how far the z-stream departs from uniform, its detectable edge is
   exactly the aggregate's miscalibration.
3. **Feedback.** Each forecaster's wealth is updated by its log score, the
   proper-scoring "gradient", so wealth flows to the calibrated reports, `F_t`
   sharpens to the truth, and the critic's edge collapses.

This is a **market-native GAN**: a generator (the transform market) against a
critic (the uniformity market), with a proper score as the loss and wealth as
the gradient. It is exactly the **z-stream / nearest-the-pin** mechanism of the
microprediction vision, and z-streams were not only theory: they ran on a live
forecasting platform (Cotton, *Microprediction: Building an Open AI Network*,
MIT Press, 2022).

The demo makes the loop quantitative. Five forecasters (two calibrated, plus a
bull, a bear, and an overconfident report) against an iid N(0,1) stream:

- z-stream **uniformity error** (TV distance from uniform): equal-weighted
  aggregate `0.165` → converged aggregate `0.068`;
- **bias** (mean `z`) stays ≈ 0 throughout, the symmetric bull/bear cancel, so
  the *mean never reveals the problem*. It is the **shape** of the PIT that is
  wrong and then fixed, which is exactly why the critic tests uniformity of the
  whole z-stream rather than a moment or two;
- **wealth** on the two calibrated reports climbs `0.41 → 1.00` as the loop runs.

## 5. Why this matters for engineering pipelines

Read this way, market/scoring mechanisms are *composable pipeline stages* with a
property ordinary pipeline stages lack: each stage carries its own incentive, so
the pipeline is **self-correcting**. One market picks a transformation; a second
market is paid to find structure the first one left behind; wealth, not a hand-
tuned loss, routes credit between them. A forecasting stack, a feature
pipeline, or a data-quality monitor can each be posed as a chain of elicitation
and test markets glued by the PIT.

Open questions worth flagging:

- **Properness under composition.** Sequentialise and pool preserve strict
  properness; does an arbitrary conjugation? (PIT does, by Dawid; a learned,
  non-invertible transform may not.)
- **Incentive compatibility of the chain.** Each market is individually
  truthful; the *composite* can admit strategic play across stages (cf. the
  finite-`b` caveats in the nearest-the-pin paper).
- **Conservation.** Wealth is the threaded state; a well-formed pipeline should
  conserve it stage-to-stage (zero-sum), the analogue of a skater faithfully
  passing its posterior state forward.

## References

Savage (1971); Banerjee, Merugu, Dhillon & Ghosh (2005); Abernethy, Chen &
Wortman Vaughan (2013); Hanson (2007); Gneiting & Raftery (2007); Genest & Zidek
(1986); Dawid (1984); Cotton (2022); the `skaters` and `timemachines` packages.
Keys in [`bibliography.bib`](bibliography.bib).
