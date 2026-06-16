# mechanisms

Reference implementations of **market mechanisms** — the rules by which markets
aggregate information, allocate risk, and price contingent claims — together with
the academic literature that explains them.

**Site:** [mechanisms.microprediction.org](https://mechanisms.microprediction.org)
*(after Pages is enabled)*
**Research notes:** [`research/`](research/) · **Code:** [`mechanisms/`](mechanisms/)

The unifying theme is the **duality between proper scoring rules and market
mechanisms**: the same machinery that scores a probabilistic forecast, wrapped as
a sequential market maker, *is* a prediction market (Hanson's LMSR); a parimutuel
pool is a batch elicitation of the same beliefs; and a DeFi constant-function
market maker is the convex conjugate of that cost-function market maker.

## What's here

| Module | Mechanism | Key references |
|--------|-----------|----------------|
| [`scoring_rules`](mechanisms/scoring_rules.py) | Proper scoring rules (log, Brier, spherical), pinball & interval scores, CRPS, energy score | Gneiting & Raftery (2007) |
| [`local_scoring`](mechanisms/local_scoring.py) | Local / m-local proper scoring rules — the Hyvärinen score, scored without a normalizing constant | Hyvärinen (2005); Parry, Dawid & Lauritzen (2012) |
| [`parimutuel`](mechanisms/parimutuel.py) | Pool betting; Pennock's dynamic parimutuel market | Pennock (2004) |
| [`lmsr`](mechanisms/lmsr.py) | Logarithmic Market Scoring Rule (cost-function market maker) | Hanson (2003, 2007) |
| [`cmm`](mechanisms/cmm.py) | Generic convex cost-function market maker | Abernethy, Chen & Wortman Vaughan (2013) |
| [`amm`](mechanisms/amm.py) | Constant-product / constant-mean AMMs, impermanent loss | Angeris et al. (2021) |
| [`pm_amm`](mechanisms/pm_amm.py) | Parimutuel AMM for binary markets (Gaussian score dynamics) | Moallemi & Robinson (2024) |
| [`cda`](mechanisms/cda.py) | Continuous double auction / limit order book | Smith (1962); Gode & Sunder (1993) |
| [`fba`](mechanisms/fba.py) | Frequent batch auction (uniform-price clearing) | Budish, Cramton & Shim (2015) |
| [`perp`](mechanisms/perp.py) | Perpetual futures funding rate, mark-price liquidation | Shiller (1993); BitMEX |
| [`pdlp`](mechanisms/pdlp.py) | Perpetual demand lending pools — funding/price-impact arbitrage, target-weight mechanism, delta hedge | Chitra et al. (2025) |
| [`peer_prediction`](mechanisms/peer_prediction.py) | Truth elicitation without ground truth (Bayesian Truth Serum) | Prelec (2004) |
| [`aggregation`](mechanisms/aggregation.py) | Linear / logarithmic / depth-trimmed opinion pools | Genest & Zidek (1986) |
| [`calibration`](mechanisms/calibration.py) | Reliability diagrams, ECE, Brier decomposition | Murphy (1973) |

Implementations are deliberately small, `numpy`-only, and written to be *read* —
clarity and faithful correspondence to the literature over performance.

## Install

```bash
pip install -e .          # editable install of the `mechanisms` package
pip install -e ".[test]"  # plus pytest
```

## Quick tour

```bash
python examples/quickstart.py
```

```python
from mechanisms import lmsr
import numpy as np

m = lmsr.LMSR(n_outcomes=2, b=100)      # a 2-outcome prediction market
m.prices()                               # -> array([0.5, 0.5])
cost = m.buy(outcome=0, shares=100)      # move the market
m.prices()                               # -> array([0.731, 0.269])
m.max_loss                               # bounded subsidy: b*log(n) = 69.31
```

```python
from mechanisms import scoring_rules as sr
import numpy as np

# A Monte-Carlo (monteprediction-style) distributional forecast, scored by the
# multivariate energy score — lower is better.
samples = np.random.default_rng(0).normal([0.5, -0.3], 1.0, size=(1000, 2))
sr.energy_score(samples, y=[0.5, -0.3])
```

## Example simulations

Each demo in [`examples/`](examples/) drives one mechanism through a short
Monte-Carlo story and prints the result as a terminal chart (`numpy`-only, no
plotting dependencies). They run anywhere:

| Demo | What it shows |
|------|---------------|
| [`sim_scoring_rules.py`](examples/sim_scoring_rules.py) | Honest reporting beats a confident misreport under the log, Brier & spherical rules — what *strictly proper* means |
| [`sim_lmsr.py`](examples/sim_lmsr.py) | Informed traders move an LMSR market to the true probability while the maker's loss stays under `b·log n` |
| [`sim_parimutuel.py`](examples/sim_parimutuel.py) | A pool aggregates noisy private beliefs, then reproduces the favourite–longshot bias |
| [`sim_cda.py`](examples/sim_cda.py) | Zero-intelligence traders drive a continuous double auction to competitive equilibrium (Gode & Sunder, 1993) |
| [`sim_amm.py`](examples/sim_amm.py) | A constant-product LP's fees raced against impermanent loss under a random-walk price |
| [`sim_perp.py`](examples/sim_perp.py) | The funding rate tethers a perpetual to its index; a 5× leveraged long gets liquidated |
| [`sim_pm_amm.py`](examples/sim_pm_amm.py) | The pm-AMM prices a binary market as the Gaussian CDF `Φ((y−x)/L)` — always a probability, outcomes sum to 1 |
| [`sim_fba.py`](examples/sim_fba.py) | A frequent batch auction clears everyone at one uniform price at the equilibrium volume (Budish, Cramton & Shim) |
| [`sim_peer_prediction.py`](examples/sim_peer_prediction.py) | The Bayesian Truth Serum out-scores a strategist with no ground truth (Prelec) |
| [`sim_aggregation.py`](examples/sim_aggregation.py) | Linear vs log vs depth-trimmed opinion pools under confidently-wrong nodes — trimming wins |
| [`sim_cmm.py`](examples/sim_cmm.py) | One convex potential subsumes LMSR (and a quadratic maker); the finite-difference price fallback matches the analytic gradient |
| [`sim_calibration.py`](examples/sim_calibration.py) | Diagnose an overconfident forecaster (reliability diagram, ECE, Brier decomposition) and fix it by temperature recalibration |
| [`sim_pipeline.py`](examples/sim_pipeline.py) | **Composing two mechanisms** — an elicitation market produces a forecast, a calibration critic PITs it for uniformity, and wealth flows to the calibrated reports until the critic's edge collapses |
| [`sim_local_scoring.py`](examples/sim_local_scoring.py) | Rank *unnormalised* models with the Hyvärinen score — no partition function — and confirm invariance to the normalizing constant |
| [`sim_correlated_agreement.py`](examples/sim_correlated_agreement.py) | Correlated Agreement elicits truth with no ground truth, defeats the constant-report exploit, and is stochastic-dominance (enforced) truthful |

```bash
python examples/sim_lmsr.py
```

## Tests

```bash
pytest -q     # properness, bounded loss, invariants, matching, funding, JS parity
```

## JavaScript ports, tested against Python

The core deterministic mechanisms are also ported to JavaScript in
[`docs/js/mechanisms.js`](docs/js/mechanisms.js), which powers the site's
interactive demos. That JS is checked for numerical parity against the Python
reference in [`tests/test_js_parity.py`](tests/test_js_parity.py): each function
runs in a Node subprocess via [`tests/js_parity_runner.js`](tests/js_parity_runner.js)
and its result is compared to Python's. The parity tests skip cleanly when `node`
is absent. (Pattern borrowed from [microprediction/humpday](https://github.com/microprediction/humpday).)

```bash
node tests/js_parity_runner.js lmsrPrices '[[10,-5,3],50]'
```

## Research

The [`research/`](research/) directory contains literature notes — parimutuels and
scoring rules, market scoring rules and AMMs, perpetuals and distributional
forecasting — each cross-linked to the code and to a consolidated
[`bibliography.bib`](research/bibliography.bib). A standalone note,
[composition-and-the-algebra-of-mechanisms](research/composition-and-the-algebra-of-mechanisms.md),
develops how the mechanisms compose: a `skaters`-style operator algebra over
distributional beliefs, Savage's characterisation and the convex-duality
generator (with proof sketches), and the worked elicitation→calibration
pipeline in [`examples/sim_pipeline.py`](examples/sim_pipeline.py).

## Site

The [`docs/`](docs/) directory is a static site (no Jekyll) deployed to GitHub
Pages by [`.github/workflows/pages.yml`](.github/workflows/pages.yml). To enable
it: **Settings → Pages → Source: GitHub Actions**. The custom domain
`mechanisms.microprediction.org` is set via [`docs/CNAME`](docs/CNAME); add the
matching `CNAME` record at the DNS provider pointing to
`microprediction.github.io`.

## License

MIT — see [LICENSE](LICENSE).
