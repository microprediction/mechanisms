# Papers

Original write-ups developed in this repository (as opposed to the literature
[`research/`](../research) notes, which survey existing work). Rendered,
readable versions live on the site under
[mechanisms.microprediction.org/papers.html](https://mechanisms.microprediction.org/papers.html)
(generated from these markdown sources by `scripts/build_papers.py` — edit the
markdown, run the script, commit both).

| Paper | Status | Topic |
|-------|--------|-------|
| [scoring-point-cloud-distributional-submissions.md](scoring-point-cloud-distributional-submissions.md) | Working draft v0.3 | The nearest-the-pin parimutuel (a continuous, density-pot-split pool, the reward engine behind monteprediction) and the incentive theory of cloud submissions: scoring a KDE at the raw outcome pays the deconvolution of the belief; jittering the pin by the bandwidth restores strict propriety; scoring across smoothing scales splits the log-score edge into shape and mass payments; in high dimensions the pool runs on random projections (the sliced energy score). Implemented and theorem-tested in [`mechanisms/nearest_the_pin.py`](../mechanisms/nearest_the_pin.py). |
| [composition-and-the-algebra-of-mechanisms.md](composition-and-the-algebra-of-mechanisms.md) | Working draft v0.1 | The operator algebra of mechanisms: one object type (a distributional belief), wealth as threaded state, convex functions under Legendre duality as the generator. Savage's characterisation, Hanson's sequentialisation, the CFMM duality, and the PIT, with proof sketches; a worked elicitation-market-to-calibration-critic pipeline with numbers from `examples/sim_pipeline.py`. |
| [nearest-the-pin-parimutuel.md](nearest-the-pin-parimutuel.md) | Merged | Redirect stub; content lives in the point-cloud paper (§2, §6, §7). |

A source PDF that motivates the write-ups lives in
[`../assets/pdf-literature/`](../assets/pdf-literature): Chitra et al.,
*Perpetual Demand Lending Pools* (arXiv:2502.06028).
