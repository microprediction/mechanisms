"""JavaScript <-> Python parity tests.

For each core mechanism, the JavaScript port in `docs/js/mechanisms.js` must
produce the same number as the Python reference in `mechanisms/`. The JS side is
run in a Node subprocess via `tests/js_parity_runner.js`; tests skip cleanly if
`node` is not on PATH, so Python-only CI is unaffected.

This mirrors the microprediction/humpday setup: the website's interactive JS is
the very code that is checked against Python here, so the two never drift.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mechanisms import amm, lmsr, parimutuel, pm_amm
from mechanisms import scoring_rules as sr

RUNNER = Path(__file__).parent / "js_parity_runner.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")


def run_js(fn: str, args: list) -> float:
    out = subprocess.run(
        [NODE, str(RUNNER), fn, json.dumps(args)],
        capture_output=True, text=True, timeout=30,
    )
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["result"]


def assert_parity(fn, js_args, py_value, tol=1e-9):
    js_value = run_js(fn, js_args)
    assert js_value == pytest.approx(py_value, abs=tol, rel=tol), (
        f"{fn}: JS={js_value} Python={py_value}"
    )


# --------------------------------------------------------------------------- #
def test_scoring_rules_parity():
    p = [0.2, 0.5, 0.3]
    assert_parity("logScore", [p, 1], sr.log_score(p, 1))
    assert_parity("brierScore", [p, 0], sr.brier_score(p, 0))
    assert_parity("sphericalScore", [p, 2], sr.spherical_score(p, 2))
    assert_parity("pinballLoss", [0.3, 1.0, 0.75], sr.pinball_loss(0.3, 1.0, 0.75))
    assert_parity("intervalScore", [0, 10, 12, 0.1],
                  sr.interval_score(0, 10, 12, 0.1))


def test_crps_and_energy_parity():
    samples = [0.1, 0.5, -0.3, 1.2, 0.0]
    assert_parity("crpsEnsemble", [samples, 0.4], sr.crps_ensemble(samples, 0.4))
    es_samples = [[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]]
    assert_parity("energyScore", [es_samples, [0.2, 0.1], 1.0],
                  sr.energy_score(np.array(es_samples), [0.2, 0.1], beta=1.0))


def test_lmsr_parity():
    q = [10.0, -5.0, 3.0]
    b = 50.0
    assert_parity("lmsrCost", [q, b], lmsr.LMSR(3, b=b, q0=q).cost())
    js_prices = run_js("lmsrPrices", [q, b])
    np.testing.assert_allclose(js_prices, lmsr.LMSR(3, b=b, q0=q).prices(), atol=1e-9)
    m = lmsr.LMSR(3, b=b, q0=q)
    assert_parity("lmsrCostToTrade", [q, [5.0, 0.0, 0.0], b],
                  m.cost_to_trade([5.0, 0.0, 0.0]))
    assert_parity("lmsrMaxLoss", [3, b], lmsr.LMSR(3, b=b).max_loss)


def test_amm_parity():
    pool = amm.ConstantProductAMM(1000, 2000, fee=0.003)
    assert_parity("cpammAmountOut", [1000, 2000, 100, 0.003],
                  pool.amount_out(100, x_in=True))
    assert_parity("impermanentLoss", [2.5], amm.impermanent_loss(2.5))


def test_pm_amm_parity():
    # JS uses an erf approximation (A&S 7.1.26), accurate to ~1e-7.
    assert_parity("pmAmmPrice", [80, 120, 40], pm_amm.pm_amm_price(80, 120, 40), tol=1e-6)
    assert_parity("normCdf", [0.5], pm_amm.norm_cdf(0.5), tol=1e-6)


def test_parimutuel_parity():
    stakes = [150.0, 30.0, 20.0]
    pool = parimutuel.ParimutuelPool(3, takeout=0.1)
    pool.bet("a", 0, 150); pool.bet("b", 1, 30); pool.bet("c", 2, 20)
    js_probs = run_js("parimutuelProbabilities", [stakes])
    np.testing.assert_allclose(js_probs, pool.implied_probabilities(), atol=1e-12)
    assert_parity("parimutuelPayoutPerUnit", [stakes, 0, 0.1], pool.decimal_odds(0))
