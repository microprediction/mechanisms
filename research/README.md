# Research notes

Academic background for the mechanisms implemented in this repository. Each note
links to the relevant reference implementation in [`../mechanisms`](../mechanisms)
and back to the consolidated [`bibliography.bib`](bibliography.bib).

| Note | Covers | Implementations |
|------|--------|-----------------|
| [parimutuel-and-scoring-rules.md](parimutuel-and-scoring-rules.md) | Pool betting, favourite–longshot bias, dynamic & combinatorial parimutuels; strictly proper scoring rules (log, Brier, spherical), CRPS | `parimutuel`, `scoring_rules` |
| [market-scoring-rules-and-amms.md](market-scoring-rules-and-amms.md) | Hanson's LMSR, cost-function market makers, liquidity sensitivity; DeFi CFMMs (constant product / mean / StableSwap), impermanent loss, LVR | `lmsr`, `cmm`, `amm` |
| [perps-cda-monteprediction.md](perps-cda-monteprediction.md) | Perpetual futures & funding; continuous double auctions; sample-based distributional forecasting & the energy score | `perp`, `cda`, `scoring_rules` |
| [perpetual-demand-lending-pools.md](perpetual-demand-lending-pools.md) | Full model of Chitra et al. (2025), funding/price-impact arbitrage, the target-weight mechanism, GMX's discount function, delta hedging | `pdlp` |
| [composition-and-the-algebra-of-mechanisms.md](composition-and-the-algebra-of-mechanisms.md) | How the mechanisms compose, a `skaters`-style operator algebra over distributional beliefs; Savage's characterisation and the convex-duality generator (with proof sketches); a worked elicitation→calibration pipeline | `scoring_rules`, `cmm`, `amm`, `aggregation`, `calibration` |
| [gaps-and-roadmap.md](gaps-and-roadmap.md) | Gap analysis & roadmap, an external audit of the microprediction ecosystem mapped onto this repo: what is already implemented vs genuinely missing (local scoring, CA/EA peer prediction, Kelly, combinatorial markets) | `local_scoring`, `peer_prediction` |

### Generalizations (candidate originality, calibrated)

These notes develop generalizations that may be original. Each carries a
calibrated novelty verdict and an exhaustive, web-verified prior-art appendix —
the prior art is the point.

| Note | Idea | Verdict |
|------|------|---------|
| [local-score-wagering-pool.md](local-score-wagering-pool.md) | A self-funding wagering pool scored by a local (Hyvärinen) proper rule, so participants submit unnormalized energy-based densities and no partition function is ever computed | Likely novel, narrow gap; explicit about score matching's mode-mass blindness |
| [anisotropic-sliced-scores.md](anisotropic-sliced-scores.md) | Schur-damped anisotropic slicing of the energy score with a reliability dial `γ` (develops a conjecture in the nearest-the-pin paper) | Likely novel — strictly proper only if the anisotropy uses a *fixed reference* covariance, not the forecast's own |
| [composing-mechanisms-conservation-and-boosting.md](composing-mechanisms-conservation-and-boosting.md) | Conservation laws for chained self-funding mechanisms (edge vs wealth) and boosting/residual markets | Two likely-novel narrow gaps; one sub-question retired as a known theorem |
| [proportional-fees-and-the-order-book.md](proportional-fees-and-the-order-book.md) | Fold a proportional fee into a cost-function maker and conjugate: the fee is a bid-ask spread (dead zone of half-width `f`), routing across fee-bearing makers is one monotone clearing-price root-find with lasso-sparse fills, the aggregate supply curve is a consolidated limit order book, and self-set fees are disciplined by the same minimization that clears the trade | Mathematics classical; assembly likely novel, narrow gaps (fees inside the costs vs beside them; routing-disciplined fee competition in the cost-function setting); implemented in `fee_routing.py` |
| [liquidity-is-precision.md](liquidity-is-precision.md) | Liquidity is precision: merging cost-function makers is Gaussian precision fusion, exactly for quadratic costs (clearing price = posterior mean, price impact = posterior covariance); a diagonal quoter makes the aggregate a Ledoit-Wolf blend with intensity equal to its capital share, tuned by P&L; unquoted entries default to the max-ent (Dempster) completion | Chiefly a connections map: the convex half is Bhaskara et al. (2023), the fusion half is textbook; capital-share intensity and coverage-induced sparsity framings unoccupied; corrects the source thread's sech²-damping claim (inventory shifts quotes, never weights); implemented in `liquidity_fusion.py` |
| [covariance-market.md](covariance-market.md) | A market whose price is a covariance matrix: PSD consistency is the no-arbitrage condition (free, not engineered); the maker is the Gaussian exponential-family MSR (log-det cost, inventory = precision matrix, payment rule = Stein's loss); restricted coverage quotes the Dempster completion natively; a nuclear-norm fee is a spectral bid-ask band (SVT responses); monteprediction is the batch/pool form; closes with a calibrated dictionary between regularized estimation and market design | Template published (Abernethy et al. 2014 exponential-family markets; d'Aspremont 2005 for PSD-as-arbitrage-test); the matrix instantiation, precision-inventory and Dempster-coverage semantics, Stein-loss identity, and spectral fee not found; implemented in `covariance_market.py` |
| [predictors-as-markets.md](predictors-as-markets.md) | Which predictors are markets: coherence is a chord condition (not convexity), rational flow trades the biconjugate (contact-set landing, gap pass-through, book holes as multimodal quotes), a fee of at least the chord excursion restores no-arbitrage (generalizing the linear-fee lemma to non-convex costs), and a deep quadratic co-quoter is Moreau smoothing; adaptive/path-dependent learners map to the adaptive-liquidity trilemma | Chord condition, contact/pass-through identities, generalized fee lemma and Moreau-by-merge not found (nearest: canonical concave CFMM trading functions, Angeris et al. 2023; Guasoni/GRS transaction-cost no-arbitrage; Frongillo-Waggoner 2018 fee repair); rendered as the working paper [predictors-as-markets.md](../papers/predictors-as-markets.md); implemented in `nonconvex_maker.py` |
| [mollified-scoring-and-the-heat-ladder.md](mollified-scoring-and-the-heat-ladder.md) | KDE-log-scored pools elicit the deconvolution of the belief (closed form: shave `h²` off the variance); the repair is to jitter the pin; run the proper rung at every smoothing scale and de Bruijn's identity splits the pot — rung differences pay Fisher (shape), the coarse rung pays mode mass, the total is the log score | Improperness known (Theis et al. 2016); closed form, strictness closure, and the heat-ladder pool assembly likely novel, narrow gaps; repaired in `nearest_the_pin.py` |

The PDF source sits in [`../assets/pdf-literature/`](../assets/pdf-literature/):
Chitra et al.'s *Perpetual Demand Lending Pools* (arXiv:2502.06028).

The recurring theme across all three notes is the duality between proper
scoring rules and market mechanisms: the same machinery that scores a
distributional forecast, wrapped as a sequential market maker, becomes a
prediction market (LMSR); a parimutuel pool is a batch elicitation of the same
beliefs; and a CFMM is the convex conjugate of a cost-function market maker.

> **Sourcing note.** The three topic notes were initially drafted from
> established knowledge of the literature (the background research agents ran
> without web access). Prominent and newer citations have since been
> web-verified, notably the pm-AMM (Moallemi & Robinson, 2024, Paradigm) and
> the LVR paper (Milionis, Moallemi, Roughgarden & Zhang, 2022). The canonical
> academic works are standard; for formal citation, a few page numbers/DOIs and
> the live monteprediction.com operational details are still worth a final check
> against primary sources.
