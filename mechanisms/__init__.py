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
nearest_the_pin Continuous / Monte-Carlo parimutuel with density pot-split and
                projection (sliced) scoring (the monteprediction reward rule).
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
kelly           Kelly (log-optimal) sizing: the agent-side dual of the log score,
                and the wealth-weighted ensemble update.
decision_market Conditional markets on a decision variable: ensembling
                action-conditional forecasts and selecting (the futarchy core).

See https://mechanisms.microprediction.org for the accompanying literature.
"""

from . import scoring_rules
from . import local_scoring
from . import parimutuel
from . import nearest_the_pin
from . import lmsr
from . import cmm
from . import amm
from . import pm_amm
from . import combinatorial
from . import cda
from . import fba
from . import hybrid_market
from . import perp
from . import pdlp
from . import peer_prediction
from . import aggregation
from . import calibration
from . import kelly
from . import decision_market

__version__ = "0.1.0"

__all__ = [
    "scoring_rules",
    "local_scoring",
    "parimutuel",
    "nearest_the_pin",
    "lmsr",
    "cmm",
    "amm",
    "pm_amm",
    "combinatorial",
    "cda",
    "fba",
    "hybrid_market",
    "perp",
    "pdlp",
    "peer_prediction",
    "aggregation",
    "calibration",
    "kelly",
    "decision_market",
    "__version__",
]
