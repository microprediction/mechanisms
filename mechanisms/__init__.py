"""mechanisms — reference implementations of market mechanisms.

This package collects small, dependency-light (numpy-only), heavily documented
reference implementations of the mechanisms used to aggregate information,
allocate risk, and price contingent claims. Each module is self-contained and
written to be *read* — the goal is clarity and a faithful correspondence to the
literature, not production performance.

Modules
-------
scoring_rules   Strictly proper scoring rules (log, Brier, spherical), pinball,
                interval score, CRPS and the multivariate energy score.
parimutuel      Pool betting and Pennock's dynamic parimutuel market.
lmsr            Hanson's Logarithmic Market Scoring Rule (cost-function maker).
cmm             Generic convex cost-function market maker.
amm             Constant-function automated market makers (constant product/mean).
pm_amm          Paradigm's pm-AMM for binary prediction markets (Gaussian scores).
cda             Continuous double auction / limit order book.
fba             Frequent batch auction (uniform-price discrete clearing).
perp            Perpetual futures funding-rate mechanism.
pdlp            Perpetual demand lending pools (GMX/Jupiter/Hyperliquid model).
peer_prediction Truth elicitation without ground truth (Bayesian Truth Serum).
aggregation     Opinion pools (linear, logarithmic, depth-trimmed).
calibration     Reliability, sharpness, and calibration diagnostics.

See https://mechanisms.microprediction.org for the accompanying literature.
"""

from . import scoring_rules
from . import parimutuel
from . import lmsr
from . import cmm
from . import amm
from . import pm_amm
from . import cda
from . import fba
from . import perp
from . import pdlp
from . import peer_prediction
from . import aggregation
from . import calibration

__version__ = "0.1.0"

__all__ = [
    "scoring_rules",
    "parimutuel",
    "lmsr",
    "cmm",
    "amm",
    "pm_amm",
    "cda",
    "fba",
    "perp",
    "pdlp",
    "peer_prediction",
    "aggregation",
    "calibration",
    "__version__",
]
