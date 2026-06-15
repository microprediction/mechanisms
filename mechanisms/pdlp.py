"""Perpetual Demand Lending Pools (PDLPs).

A faithful, heavily-referenced implementation of the model in

    Chitra, Diamandis, Sheng, Sterle & Yusubov (2025),
    "Perpetual Demand Lending Pools", arXiv:2502.06028.

A PDLP is the liquidity pool behind a decentralised perpetuals exchange (GMX's
GLP, Jupiter's JLP, Hyperliquid's HLP, dYdX's MegaVault). Liquidity providers
deposit assets; traders borrow from the pool to open *levered* positions on the
exchange and pay a lending fee; arbitrageurs keep the pool near a target
portfolio and keep the perpetual's mark price tethered to spot. The paper's
contributions, all implemented below with equation numbers from the paper:

  §2.1  Perpetuals exchange: contract (L, S, p0), trades, the collateral and
        liquidation conditions, and the linear funding rate (eq 2).
  §2.3  Single-period arbitrage:
          - funding-rate arbitrage size (eq 3) and the fee upper bound (eq 4);
          - PDLP price-impact arbitrage via a forward exchange function G,
            optimal trade G'(x*) = 1/p, and the fee lower bound (eq 5), with the
            Uniswap-v2 instance and its LP loss R1 + p R2 - 2 sqrt(p R1 R2).
  §3    Target Weight Mechanism: pool weights w(p,R) = (p . R)/(p^T R), the
        rebalancing program (eq 6), GMX's GLP discount function (eq 11), and
        portfolio dilution after a TWM update.
  §4    Hedged PDLPs: the mean-variance delta hedge (eq 12-13) and the
        Sharpe-ratio improvement conditions (Claim 4.1).

numpy only. Functions are written to be read alongside the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

__all__ = [
    # §2.1 perpetuals exchange
    "linear_funding_rate",
    "collateral_ok",
    "min_collateral",
    "liquidation_price",
    # §2.3.1 funding-rate arbitrage
    "funding_arb_size",
    "funding_fee_upper_bound",
    # §2.3.2 PDLP price-impact arbitrage
    "UniswapV2Forward",
    "pdlp_arb_optimal_x",
    "fee_lower_bound",
    # §2.2 / §3 the pool and the target-weight mechanism
    "PDLP",
    "weights",
    "gmx_glp_discount",
    "target_weight_trade",
    "portfolio_dilution_value",
    # §4 hedged PDLPs
    "delta_hedge",
    "sharpe_improves",
]


# ===================================================================== #
# §2.1  Perpetuals exchange
# ===================================================================== #
def linear_funding_rate(L: float, S: float, p: float, p0: float, kappa: float) -> float:
    r"""Linear funding rate, paper eq (2):  ``gamma_L = kappa (L/S - p/p0)``.

    ``L``, ``S`` are cumulative long / short positions, ``p`` the underlying price,
    ``p0`` the mark price, ``kappa > 0`` a constant. If ``gamma_L > 0`` shorts pay
    longs; if ``< 0`` longs pay shorts. The rate incentivises the two sides toward
    equality.
    """
    if S <= 0 or p0 <= 0:
        raise ValueError("S and p0 must be positive")
    return float(kappa * (L / S - p / p0))


def collateral_ok(c: float, delta: float, eta: float, p0: float) -> bool:
    r"""Collateral condition, paper eq (1):  ``p0 * delta <= |eta| * c``.

    A trader posting collateral ``c`` may open a notional ``delta * p0`` up to
    ``|eta|`` times larger. ``eta`` is the leverage (``< 0`` for a short).
    """
    return p0 * abs(delta) <= abs(eta) * c + 1e-12


def min_collateral(delta: float, eta: float, p0: float) -> float:
    """Smallest collateral that satisfies eq (1): ``p0 |delta| / |eta|``."""
    return float(p0 * abs(delta) / abs(eta))


def liquidation_price(delta: float, eta: float, p0: float,
                      c: Optional[float] = None) -> float:
    r"""Liquidation price from the condition ``sign(eta) delta (p0 - p) >= c``.

    A long (``eta > 0``) is liquidated once the price falls below
    ``p0 - c/delta``; a short once it rises above ``p0 + c/|delta|``. If ``c`` is
    omitted the *minimal*-collateral case (eq 1 saturated) is used, giving the
    clean form ``p0 (1 - 1/eta)`` for a long (paper's worked ETH example:
    ``p0 = 2000, eta = 4`` -> ``1500``).
    """
    if c is None:
        c = min_collateral(delta, eta, p0)
    if eta > 0:  # long
        return float(p0 - c / abs(delta))
    else:        # short
        return float(p0 + c / abs(delta))


# ===================================================================== #
# §2.3.1  Funding-rate arbitrage
# ===================================================================== #
def funding_arb_size(L0: float, p: float, p0: float) -> float:
    r"""Largest funding-capturing long, paper eq (3):  ``ell = L0 (p/p0 - 1)``.

    Starting from balanced open interest ``L = S = L0`` with zero funding, a price
    move to ``p > p0`` makes the funding rate negative (shorts overpaid); an
    arbitrageur opens a long of size ``ell`` that drives ``gamma_L`` back to 0.
    """
    return float(L0 * (p / p0 - 1.0))


def funding_fee_upper_bound(kappa: float, L0: float, p: float, p0: float,
                            B: Optional[float] = None) -> float:
    r"""Upper bound on the PDLP fee for profitable funding arbitrage, eq (4).

    The arbitrageur profits when ``f <= (kappa/L0)(1 - p0/p)``. If a relative
    price bound ``1 < p/p0 <= B`` is supplied, returns the worst-case bound
    ``kappa (1 - B^{-1}) / L0`` instead. Note the fee must fall as open interest
    ``L0`` grows.
    """
    if B is not None:
        return float(kappa * (1.0 - 1.0 / B) / L0)
    return float((kappa / L0) * (1.0 - p0 / p))


# ===================================================================== #
# §2.3.2  PDLP price-impact arbitrage (forward exchange function G)
# ===================================================================== #
@dataclass
class UniswapV2Forward:
    r"""Uniswap-v2-style forward exchange function (paper's §2.3.3 example).

    ``G(x) = R2 x / (R1 + x)`` is the amount of the risky asset the PDLP sells for
    ``x`` units of numeraire; ``G'(x) = R1 R2 / (R1 + x)^2``. The pool initially
    quotes the old price ``p0`` via ``R2 / R1 = 1 / p0`` (so ``G'(0) = 1/p0``).
    """

    R1: float  # numeraire reserve
    R2: float  # risky-asset reserve

    def G(self, x: float) -> float:
        return float(self.R2 * x / (self.R1 + x))

    def G_prime(self, x: float) -> float:
        return float(self.R1 * self.R2 / (self.R1 + x) ** 2)

    @property
    def initial_price(self) -> float:
        """The price the pool initially quotes, ``1 / G'(0) = R1 / R2``."""
        return float(self.R1 / self.R2)

    def optimal_x(self, p: float) -> float:
        r"""Most profitable numeraire input, from ``G'(x*) = 1/p`` (closed form).

        ``x* = sqrt(p R1 R2) - R1``.
        """
        return float(math.sqrt(p * self.R1 * self.R2) - self.R1)

    def lp_loss(self, p: float) -> float:
        r"""Rebalancing loss to LPs after arbitrage at price ``p`` (paper §2.3.3).

        ``R1 + p R2 - 2 sqrt(p R1 R2)`` >= 0 — the absolute-value form of CFMM
        impermanent loss / loss-versus-rebalancing.
        """
        return float(self.R1 + p * self.R2 - 2.0 * math.sqrt(p * self.R1 * self.R2))


def pdlp_arb_optimal_x(forward, p: float, hi: float = 1e12, tol: float = 1e-10) -> float:
    r"""Generic optimal arbitrage trade solving ``G'(x*) = 1/p`` by bisection.

    Works for any forward exchange function exposing ``G_prime`` (concave,
    decreasing derivative). For :class:`UniswapV2Forward` prefer its closed-form
    ``optimal_x``; this is the fallback for arbitrary ``G``.
    """
    target = 1.0 / p
    lo = 0.0
    # G' is decreasing; find x where G'(x) = target.
    if forward.G_prime(0.0) <= target:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if forward.G_prime(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


def fee_lower_bound(x_star: float, L0: float) -> float:
    r"""Sufficient PDLP fee lower bound for LP profitability, paper eq (5) corollary.

    Using concavity of ``G`` the paper shows LPs are profitable whenever
    ``f >= x* / L0``. Combined with :func:`funding_fee_upper_bound` this brackets
    a sustainable fee regime, with ``f = Theta(1/L0)``.
    """
    return float(x_star / L0)


# ===================================================================== #
# §2.2 / §3  The pool and the Target Weight Mechanism
# ===================================================================== #
def weights(p, R) -> np.ndarray:
    r"""Price-weighted pool composition, ``w(p, R) = (p . R) / (p^T R)``."""
    p = np.asarray(p, float)
    R = np.asarray(R, float)
    val = float(p @ R)
    if val <= 0:
        raise ValueError("portfolio value must be positive")
    return (p * R) / val


@dataclass
class PDLP:
    """A perpetual demand lending pool (paper §2.2).

    Parameters
    ----------
    R       : reserves, an ``n``-vector (units of each asset).
    prices  : asset prices ``p`` (the numeraire has price 1).
    target  : target weights ``w*`` on the simplex.
    fee     : per-period lending fee ``f in (0, 1)`` charged on loan collateral.
    loans   : outstanding loan collaterals ``c_i`` (in value), used by positions.
    """

    R: np.ndarray
    prices: np.ndarray
    target: np.ndarray
    fee: float = 0.001
    loans: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.R = np.asarray(self.R, float)
        self.prices = np.asarray(self.prices, float)
        self.target = np.asarray(self.target, float)
        if not (0.0 < self.fee < 1.0):
            raise ValueError("fee must be in (0, 1)")
        if not np.isclose(self.target.sum(), 1.0):
            raise ValueError("target weights must sum to 1")

    def value(self) -> float:
        """Total pool value ``p^T R``."""
        return float(self.prices @ self.R)

    def weights(self) -> np.ndarray:
        return weights(self.prices, self.R)

    @property
    def total_loaned(self) -> float:
        return float(sum(self.loans))

    def available_value(self) -> float:
        r"""Unutilised value ``p^T R - sum_i c_i`` (must stay >= 0; eq Sum c_i <= R)."""
        return self.value() - self.total_loaned

    def fee_income(self) -> float:
        """Per-period fee revenue ``f * sum_i c_i`` paid by borrowers to LPs."""
        return float(self.fee * self.total_loaned)


def gmx_glp_discount(w_before, w_after, w_star,
                     gamma_b: float = 0.0, gamma_t: float = 0.1) -> float:
    r"""GMX GLP discount function, paper eq (11).

    ``F = max(0, gamma_b + max_i G_i(w_before, w_after, w*))`` where the per-asset
    term ``G_i`` rewards trades that move asset ``i``'s weight toward its target
    ``w*_i`` and penalises trades that move it away:

    .. math::
        G_i = \begin{cases}
          0 & w^a_i = w^b_i \\
          \gamma_t\,\bigl|\tfrac{w^b_i-w^*_i}{w^*_i}\bigr|
            & |w^a_i-w^*_i| < |w^b_i-w^*_i| \quad(\text{moved closer}) \\
          -\tfrac{\gamma_t}{2}\bigl(|\tfrac{w^b_i-w^*_i}{w^*_i}|
            + |\tfrac{w^a_i-w^*_i}{w^*_i}|\bigr) & \text{else}
        \end{cases}

    A positive ``F`` is a *discount* (subsidy) to an LP who improves the pool's
    balance; a negative term is a fee. Resembles a PID controller on the weights.
    """
    wb = np.asarray(w_before, float)
    wa = np.asarray(w_after, float)
    ws = np.asarray(w_star, float)
    gs = np.empty(len(ws))
    for i in range(len(ws)):
        before_gap = abs(wb[i] - ws[i])
        after_gap = abs(wa[i] - ws[i])
        if np.isclose(wa[i], wb[i]):
            gs[i] = 0.0
        elif after_gap < before_gap:  # moved closer to target
            gs[i] = gamma_t * abs((wb[i] - ws[i]) / ws[i])
        else:                          # moved away
            gs[i] = -0.5 * gamma_t * (abs((wb[i] - ws[i]) / ws[i])
                                      + abs((wa[i] - ws[i]) / ws[i]))
    return float(max(0.0, gamma_b + float(np.max(gs))))


def target_weight_trade(prices, R, available, target, new_value=None):
    r"""Solve the TWM rebalancing program (paper eq 6) to hit the target weights.

    Returns the reserve change ``Delta`` that makes ``w(p, R + Delta) = w*`` for a
    chosen post-trade portfolio value, subject to ``Delta >= -R_available`` (you
    cannot remove utilised, loan-backing assets). To reach the target exactly,
    each asset must hold ``R_i + Delta_i = w*_i * T' / p_i`` where ``T'`` is the
    new total value; we default ``T'`` to the current value (a pure rebalance) and
    clip removals at the available reserves.

    Parameters
    ----------
    prices, R   : current prices and reserves.
    available   : per-asset unutilised reserves ``R^A`` (max removable).
    target      : target weights ``w*``.
    new_value   : desired post-trade value ``T'`` (defaults to current value).
    """
    p = np.asarray(prices, float)
    R = np.asarray(R, float)
    avail = np.asarray(available, float)
    ws = np.asarray(target, float)
    T = float(p @ R) if new_value is None else float(new_value)
    desired_R = ws * T / p                 # reserves that realise w* at value T
    delta = desired_R - R
    # Constraint: cannot remove more than available (Delta_i >= -avail_i).
    delta = np.maximum(delta, -avail)
    return delta


def portfolio_dilution_value(prices, R, ell: float, fee: float, F: float) -> float:
    r"""PDLP-share portfolio value after one TWM update, paper §3.5.

    ``V_new = (1/(1+F)) * p^T R + f * p^T ell`` — the diluted value of the initial
    portfolio (existing LPs are diluted by ``1/(1+F)`` when a new LP receives a
    discount ``F > 0``) plus the fees earned on lent assets ``ell``.
    """
    p = np.asarray(prices, float)
    R = np.asarray(R, float)
    portfolio = float(p @ R)
    # `ell` may be a scalar lent value or a per-asset vector.
    lent = float(ell) if np.ndim(ell) == 0 else float(p @ np.asarray(ell, float))
    return float(portfolio / (1.0 + F) + fee * lent)


# ===================================================================== #
# §4  Hedged PDLPs
# ===================================================================== #
def delta_hedge(fee: float, gamma: float, Sigma, ell, Delta, costs=None) -> np.ndarray:
    r"""Mean-variance delta hedge for PDLP LPs, paper eq (12)-(13).

    Solves
    ``max_x  f * ell^T (x + R-ish) - 1/2 (x-pi)^T Diag(c) (x-pi) - gamma/2 (x+Delta)^T Sigma (x+Delta)``.
    With transaction costs ``c`` the optimum is
    ``(gamma Sigma + Diag(c))^{-1} [f ell + Diag(c) pi - gamma Sigma Delta]``;
    without costs (``costs=None``) this collapses to the closed form (eq 13)

    .. math::  \pi_{\text{new}} = (f/\gamma)\,\Sigma^{-1}\ell - \Delta.

    Parameters
    ----------
    fee    : loan fee ``f``.
    gamma  : risk-aversion ``gamma >= 0``.
    Sigma  : (n, n) covariance of the risky assets (SPD).
    ell    : (n,) lent amounts (loan demand) per asset.
    Delta  : (n,) current portfolio delta to offset.
    costs  : optional (n,) transaction costs; if given, uses the full eq (13).
    """
    Sigma = np.asarray(Sigma, float)
    ell = np.asarray(ell, float)
    Delta = np.asarray(Delta, float)
    if costs is None:
        return (fee / gamma) * np.linalg.solve(Sigma, ell) - Delta
    C = np.diag(np.asarray(costs, float))
    A = gamma * Sigma + C
    # `pi` (prior hedge) defaults to zero when only costs are supplied.
    b = fee * ell - gamma * Sigma @ Delta
    return np.linalg.solve(A, b)


def sharpe_improves(fee: float, ell, Delta, Sigma, prices, gamma: float):
    r"""Check the two sufficient conditions of Claim 4.1 for Sharpe improvement.

    Returns ``(expectation_ok, variance_ok)``:

    1. **Expectation non-decreasing** (eq 14):
       ``f * p^T ell  >=  gamma * lambda_max(Sigma) * p^T Delta``.
    2. **Variance non-increasing**: ``Var[p^T pi] <= 4 Var[p^T R]``. With the
       mean-variance hedge ``pi = (f/gamma) Sigma^{-1} ell - Delta`` and
       ``Var[p^T R] = p^T Sigma p``, we check ``p^T pi-variance <= 4 p^T Sigma p``.

    When both hold, the delta-hedged PDLP's Sharpe ratio is at least that of the
    unhedged pool.
    """
    p = np.asarray(prices, float)
    Sigma = np.asarray(Sigma, float)
    ell = np.asarray(ell, float)
    Delta = np.asarray(Delta, float)
    lam_max = float(np.linalg.eigvalsh(Sigma).max())

    expectation_ok = fee * float(p @ ell) >= gamma * lam_max * float(p @ Delta)

    pi = delta_hedge(fee, gamma, Sigma, ell, Delta)
    var_hedge = float(pi @ Sigma @ pi)
    var_pool = float(p @ Sigma @ p)
    variance_ok = var_hedge <= 4.0 * var_pool
    return expectation_ok, variance_ok
