# Survey: mechanisms for eliciting, aggregating, and rewarding predictions

A reading guide to *Novel Statistical and Market-Inspired Mechanisms for the
Elicitation, Aggregation, and Rewarding of Predictive Contributions*
([PDF](../assets/pdf-literature/Market%20Mechanisms%20for%20Prediction%20Aggregation%20-%20Google%20Docs.pdf)),
the survey that organises this repository. It traces a single arc: from the
axioms of strictly proper scoring rules, through market-inspired mechanisms for
continuous aggregation, to the web-scale **microprediction** paradigm. Each
section below points to the corresponding reference implementation.

## 1. Axiomatic foundations: strictly proper scoring rules

A scoring rule $S(P, y)$ is **proper** if honest reporting maximises expected
reward, **strictly proper** if uniquely so (McCarthy–Savage: properness ⇔ the
expected-score function is convex and the score is its subgradient). For point
forecasts, *consistent scoring functions* target specific functionals — the
**pinball loss** for quantiles, asymmetric squared error for expectiles, the
**interval score** for prediction intervals. The choice of rule shapes the
optimisation landscape, so the reward dictates the model.
→ `mechanisms/scoring_rules.py`

## 2. Evaluating distributional forecasts: CRPS & energy score

The **CRPS** is the gold standard for real-valued probabilistic forecasts,
$\mathrm{CRPS}(F,x)=\mathbb E_F|X-x|-\tfrac12\mathbb E_F|X-X'|$, reducing to
absolute error for a point forecast. Its multivariate generalisation, the
**energy score**, captures dependence and admits a kernel representation —
making it the natural score for sample-based forecasts.
→ `mechanisms/scoring_rules.py` (`crps_ensemble`, `energy_score`)

## 3. Automated market makers: LMSR and LS-LMSR

Hanson's **LMSR** turns the log score into a market maker with cost
$C(q)=b\log\sum_i e^{q_i/b}$, prices summing to one, and bounded loss $b\log N$.
Its weakness is the fixed liquidity $b$ and the guaranteed subsidy; the
**liquidity-sensitive LMSR** scales $b$ with volume so the maker can run at a
profit.
→ `mechanisms/lmsr.py`, `mechanisms/cmm.py`

## 4. Continuous parimutuel markets & dynamic pricing

The **Dynamic Parimutuel Market** (Pennock) hybridises the continuous pricing of
a CDA with the zero-house-risk guarantee of a parimutuel pool: shares are priced
by a cost function, and the entire pot is redistributed to winning shares. The
*projection game* shows specific DPM forms are strategically equivalent to the
spherical scoring rule.
→ `mechanisms/parimutuel.py`

## 5. Binary options & DeFi: the pm-AMM

Generic constant-product AMMs are catastrophic for binary outcome tokens.
Paradigm's **pm-AMM** uses *Gaussian score dynamics* (the signal is Brownian, the
price is a Black-Scholes binary-option probability), with static invariant
$(y-x)\Phi(\tfrac{y-x}{L})+L\phi(\tfrac{y-x}{L})-y=0$. It concentrates liquidity
at the 0.50 mark, normalising loss-versus-rebalancing; the *dynamic* pm-AMM
withdraws liquidity as expiry approaches to keep the LP loss rate constant.
→ `mechanisms/pm_amm.py`

## 6. Microstructure: CDA vs. frequent batch auctions

The continuous limit order book rewards latency and invites a high-frequency
sniping arms race. **Frequent batch auctions** (Budish–Cramton–Shim) discretise
time and clear each batch at a uniform price, eliminating speed advantages and,
on chain, neutralising transaction-ordering (MEV) attacks.
→ `mechanisms/cda.py`, `mechanisms/fba.py`

## 7. Elicitation without ground truth: peer prediction

When truth is never observed, reward reports by their statistical relationship to
peers. Output-agreement collapses to a degenerate (everyone-agrees) equilibrium;
Prelec's **Bayesian Truth Serum** breaks it by scoring *surprisingly common*
answers (information score) plus a prediction score (negative KL to the empirical
distribution). The **Correlated Agreement** mechanism generalises this across
multiple tasks without eliciting full peer distributions.
→ `mechanisms/peer_prediction.py`

## 8. Aggregation: opinion pools and copulas

**Linear opinion pools** (weighted arithmetic mean) are robust but
over-dispersed; **logarithmic pools** (geometric mean) are sharper. Robust
variants use statistical depth to trim outliers before pooling. For multivariate
forecasts, **copulas** (e.g. regular vine copulas estimated by sequential Monte
Carlo) model the joint dependence that marginal pooling misses.
→ `mechanisms/aggregation.py`

## 9. The microprediction paradigm

The synthesis: a web-scale network of autonomous "micromanagers" that
continuously submit *distributional* forecasts — standardised as a fixed number
of Monte-Carlo samples (e.g. 225 quantiles) — combined and rewarded in real time.
Aggregated forecasts produce **Z-streams** ($z=\Phi^{-1}(F(x))$): if the market
is calibrated the $z$ values are standard normal, and any deviation is an
exploitable, self-correcting anomaly. Rewards follow a density-based pot-splitting
rule ("nearest-the-pin") drawing on both dynamic-parimutuel and proper-scoring
principles. See Cotton, *Microprediction: Building an Open AI Network* (MIT
Press, 2022) and [monteprediction.com](https://monteprediction.com).
→ `mechanisms/scoring_rules.py` (`energy_score`), `mechanisms/parimutuel.py`

---

The recurring thesis: **accurate intelligence is not freely volunteered — it must
be extracted, verified, and combined through engineered economic incentives.**
These mechanisms are the engineering. The full works-cited (69 sources) is in the
survey PDF; key entries are mirrored in [`bibliography.bib`](bibliography.bib).
