# Papers

Original write-ups developed in this repository (as opposed to the literature
[`research/`](../research) notes, which survey existing work). Rendered,
readable versions live on the site under
[mechanisms.microprediction.org/papers.html](https://mechanisms.microprediction.org/papers.html)
(generated from these markdown sources by `scripts/build_papers.py` — edit the
markdown, run the script, commit both).

| Paper | Status | Topic |
|-------|--------|-------|
| [nearest-the-pin-parimutuel.md](nearest-the-pin-parimutuel.md) | Working draft v0.2 | The continuous, density-pot-split parimutuel behind monteprediction; its log-wealth truthfulness, the projection (sliced energy score) version, and the link to random projections and the Schur pseudo-likelihood. Implemented in [`mechanisms/nearest_the_pin.py`](../mechanisms/nearest_the_pin.py). |
| [scoring-point-cloud-distributional-submissions.md](scoring-point-cloud-distributional-submissions.md) | Working draft v0.1 | Scoring a KDE'd sample cloud at the raw outcome is improper — the optimal cloud is the **deconvolution** of the belief (Gaussian: shave `h²` off the variance). The repair: **jitter the pin** by the same kernel, strictly proper via injectivity of Gaussian convolution. Run the proper rung at every smoothing scale and de Bruijn's identity splits the pot: the **heat-ladder pool**. Includes the historical note that microprediction's jitter was intuition first. Implemented and theorem-tested in [`mechanisms/nearest_the_pin.py`](../mechanisms/nearest_the_pin.py). |

A source PDF that motivates the write-ups lives in
[`../assets/pdf-literature/`](../assets/pdf-literature): Chitra et al.,
*Perpetual Demand Lending Pools* (arXiv:2502.06028).
