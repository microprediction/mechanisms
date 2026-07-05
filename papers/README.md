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
| [multi-stage-solicitation.md](multi-stage-solicitation.md) | Working draft v0.3 | "Multi-Stage Solicitation of Probability Distributions: Experiments and Theory". Leads with the experiments (the microprediction platform's chained games: 225-sample nearest-the-pin pool, z1~ PIT streams, z2~/z3~ Morton z-curve copula streams, stacked lottery, monteprediction, MidOne — fact-checked against `microprediction/rediz`, `microconventions`, and the 2020 copula-contest post), then the theory of when a chain is proper, then a verdict on each game's validity (base pool improper without a jitter; z1~ a valid pool but a calibration diagnostic; z2~/z3~ proper elicitation but a curve-distorted metric). Cites the algebra for single-stage theory. |
| [nearest-the-pin-parimutuel.md](nearest-the-pin-parimutuel.md) | Merged | Redirect stub; content lives in the point-cloud paper (§2, §6, §7). |

A source PDF that motivates the write-ups lives in
[`../assets/pdf-literature/`](../assets/pdf-literature): Chitra et al.,
*Perpetual Demand Lending Pools* (arXiv:2502.06028).
