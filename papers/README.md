# Papers

Original write-ups developed in this repository (as opposed to the literature
[`research/`](../research) notes, which survey existing work). Rendered,
readable versions live on the site under
[mechanisms.microprediction.org/papers.html](https://mechanisms.microprediction.org/papers.html)
(generated from these markdown sources by `scripts/build_papers.py` — edit the
markdown, run the script, commit both).

| Paper | Status | Topic |
|-------|--------|-------|
| [scoring-point-cloud-distributional-submissions.md](scoring-point-cloud-distributional-submissions.md) | Preprint · July 2026 | The nearest-the-pin parimutuel (a continuous, density-pot-split pool, the reward engine behind monteprediction) and the incentive theory of cloud submissions: scoring a KDE at the raw outcome pays the deconvolution of the belief; jittering the pin by the bandwidth restores strict propriety; scoring across smoothing scales splits the log-score edge into shape and mass payments; in high dimensions the pool runs on random projections (the sliced energy score). Implemented and theorem-tested in [`mechanisms/nearest_the_pin.py`](../mechanisms/nearest_the_pin.py). |
| [composition-and-the-algebra-of-mechanisms.md](composition-and-the-algebra-of-mechanisms.md) | Working draft v0.4 (expository) | The single-stage dictionary: one object type (a distributional belief), convex functions under Legendre duality as the generator. Savage's characterisation and Hanson's conjugate market maker (with intuition and full proofs), the CFMM level-set duality, the two opinion pools as the two KL barycenters, merging makers as infimal convolution, and the PIT. Classical mathematics collected in one convention; the multi-stage story and the conformal instance are split out to the companions below. |
| [multi-stage-solicitation.md](multi-stage-solicitation.md) | Working draft v0.1 | The more novel companion: a report on the chained elicitation system the microprediction platform deployed (z-streams of community percentiles, bivariate/trivariate copula streams, stacked lotteries). Stages as transducers over one message type; sample clouds as the deployed message; margin+rank-copula factoring; and the residual chain behind a conformal predictor. Cites the algebra for the single-stage theory and the conformal note for the information-gap result. |
| [nearest-the-pin-parimutuel.md](nearest-the-pin-parimutuel.md) | Merged | Redirect stub; content lives in the point-cloud paper (§2, §6, §7). |

A source PDF that motivates the write-ups lives in
[`../assets/pdf-literature/`](../assets/pdf-literature): Chitra et al.,
*Perpetual Demand Lending Pools* (arXiv:2502.06028).
