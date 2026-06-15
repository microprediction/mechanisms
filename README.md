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
| [`parimutuel`](mechanisms/parimutuel.py) | Pool betting; Pennock's dynamic parimutuel market | Pennock (2004) |
| [`lmsr`](mechanisms/lmsr.py) | Logarithmic Market Scoring Rule (cost-function market maker) | Hanson (2003, 2007) |
| [`cmm`](mechanisms/cmm.py) | Generic convex cost-function market maker | Abernethy, Chen & Wortman Vaughan (2013) |
| [`amm`](mechanisms/amm.py) | Constant-product / constant-mean AMMs, impermanent loss | Angeris et al. (2021) |
| [`pm_amm`](mechanisms/pm_amm.py) | Parimutuel AMM for binary markets (Gaussian score dynamics) | Moallemi & Robinson (2024) |
| [`cda`](mechanisms/cda.py) | Continuous double auction / limit order book | Smith (1962); Gode & Sunder (1993) |
| [`fba`](mechanisms/fba.py) | Frequent batch auction (uniform-price clearing) | Budish, Cramton & Shim (2015) |
| [`perp`](mechanisms/perp.py) | Perpetual futures funding rate, mark-price liquidation | Shiller (1993); BitMEX |
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
[`bibliography.bib`](research/bibliography.bib).

## Site

The [`docs/`](docs/) directory is a static site (no Jekyll) deployed to GitHub
Pages by [`.github/workflows/pages.yml`](.github/workflows/pages.yml). To enable
it: **Settings → Pages → Source: GitHub Actions**. The custom domain
`mechanisms.microprediction.org` is set via [`docs/CNAME`](docs/CNAME); add the
matching `CNAME` record at the DNS provider pointing to
`microprediction.github.io`.

## License

MIT — see [LICENSE](LICENSE).
