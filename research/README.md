# Research notes

Academic background for the mechanisms implemented in this repository. Each note
links to the relevant reference implementation in [`../mechanisms`](../mechanisms)
and back to the consolidated [`bibliography.bib`](bibliography.bib).

| Note | Covers | Implementations |
|------|--------|-----------------|
| [survey-prediction-aggregation.md](survey-prediction-aggregation.md) | **Reading guide to the survey PDF** — the full arc from scoring rules through AMMs, DPMs, pm-AMM, FBAs, and peer prediction to microprediction | all modules |
| [parimutuel-and-scoring-rules.md](parimutuel-and-scoring-rules.md) | Pool betting, favourite–longshot bias, dynamic & combinatorial parimutuels; strictly proper scoring rules (log, Brier, spherical), CRPS | `parimutuel`, `scoring_rules` |
| [market-scoring-rules-and-amms.md](market-scoring-rules-and-amms.md) | Hanson's LMSR, cost-function market makers, liquidity sensitivity; DeFi CFMMs (constant product / mean / StableSwap), impermanent loss, LVR | `lmsr`, `cmm`, `amm` |
| [perps-cda-monteprediction.md](perps-cda-monteprediction.md) | Perpetual futures & funding; continuous double auctions; sample-based distributional forecasting & the energy score | `perp`, `cda`, `scoring_rules` |
| [perpetual-demand-lending-pools.md](perpetual-demand-lending-pools.md) | **Full model of Chitra et al. (2025)** — funding/price-impact arbitrage, the target-weight mechanism, GMX's discount function, delta hedging | `pdlp` |
| [composition-and-the-algebra-of-mechanisms.md](composition-and-the-algebra-of-mechanisms.md) | **How the mechanisms compose** — a `skaters`-style operator algebra over distributional beliefs; Savage's characterisation and the convex-duality generator (with proof sketches); a worked elicitation→calibration pipeline | `scoring_rules`, `cmm`, `amm`, `aggregation`, `calibration` |
| [gaps-and-roadmap.md](gaps-and-roadmap.md) | **Gap analysis & roadmap** — an external audit of the microprediction ecosystem mapped onto this repo: what is already implemented vs genuinely missing (local scoring, CA/EA peer prediction, Kelly, combinatorial markets) | `local_scoring`, `peer_prediction` |

The PDF sources sit in [`../assets/pdf-literature/`](../assets/pdf-literature/):
the prediction-aggregation survey and Chitra et al.'s *Perpetual Demand Lending
Pools* (arXiv:2502.06028).

The recurring theme across all three notes is the **duality between proper
scoring rules and market mechanisms**: the same machinery that scores a
distributional forecast, wrapped as a sequential market maker, becomes a
prediction market (LMSR); a parimutuel pool is a batch elicitation of the same
beliefs; and a CFMM is the convex conjugate of a cost-function market maker.

> **Sourcing note.** The three topic notes were initially drafted from
> established knowledge of the literature (the background research agents ran
> without web access). Prominent and newer citations have since been
> web-verified — notably the pm-AMM (Moallemi & Robinson, 2024, Paradigm) and
> the LVR paper (Milionis, Moallemi, Roughgarden & Zhang, 2022). The canonical
> academic works are standard; for formal citation, a few page numbers/DOIs and
> the live monteprediction.com operational details are still worth a final check
> against primary sources.
