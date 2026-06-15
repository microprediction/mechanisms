"""pm-AMM — a parimutuel automated market maker for prediction markets.

Paradigm's pm-AMM (Moallemi & Robinson, 2024) is a constant-function market maker
specialised for *binary outcome tokens* — tokens that pay \\$1 if an event occurs
and \\$0 otherwise. Generic constant-product AMMs are catastrophic for such
tokens because the price inevitably resolves to 0 or 1, maximising impermanent
loss. The pm-AMM instead derives its invariant from **Gaussian score dynamics**:
it assumes the underlying signal (a vote margin, a score differential) follows a
Brownian motion, and the token price is the probability that the signal finishes
above a threshold — exactly the Black-Scholes structure for a binary option.

The *static* pm-AMM invariant over the reserves ``(x, y)`` of the two opposing
outcome tokens, with liquidity scale ``L``, is

.. math::
    (y - x)\\,\\Phi\\!\\left(\\frac{y - x}{L}\\right)
        + L\\,\\phi\\!\\left(\\frac{y - x}{L}\\right) - y = 0,

where ``Phi`` and ``phi`` are the standard-normal CDF and PDF. This concentrates
liquidity around the 0.50 price (maximum uncertainty) and thins it at the
extremes, normalising LP risk (loss-versus-rebalancing) across price levels.

References
----------
- Moallemi, C. & Robinson, D. (2024). "pm-AMM: A Uniform AMM for Prediction
  Markets." Paradigm. https://www.paradigm.xyz/2024/11/pm-amm
"""

from __future__ import annotations

import math

__all__ = ["norm_cdf", "norm_pdf", "pm_amm_invariant", "pm_amm_price", "implied_price"]


def norm_pdf(x: float) -> float:
    """Standard-normal probability density ``phi(x)``."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF ``Phi(x)`` via ``erf`` (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pm_amm_invariant(x: float, y: float, L: float) -> float:
    r"""The static pm-AMM invariant value (zero on the trading curve).

    ``(y - x) Phi((y-x)/L) + L phi((y-x)/L) - y``. On the curve this is 0.
    """
    d = (y - x) / L
    return float((y - x) * norm_cdf(d) + L * norm_pdf(d) - y)


def pm_amm_price(x: float, y: float, L: float) -> float:
    r"""Marginal price of the YES token implied by the reserves.

    For the pm-AMM the price of one outcome is ``Phi((y - x) / L)`` — the
    Black-Scholes binary-option probability. Prices of the two outcomes sum to 1.
    """
    return float(norm_cdf((y - x) / L))


def implied_price(reserve_diff: float, L: float) -> float:
    """Convenience: price as a function of the reserve difference ``y - x``."""
    return float(norm_cdf(reserve_diff / L))
