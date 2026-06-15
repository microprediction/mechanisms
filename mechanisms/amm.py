"""Constant-function automated market makers (CFMMs).

A CFMM holds reserves ``R = (R_1, ..., R_n)`` and accepts any trade that leaves a
*trading function* ``phi(R)`` unchanged (up to a fee). The canonical instance is
the constant-product market maker (Uniswap v2): ``x * y = k``. This module
implements the constant-product and constant-weighted-geometric-mean (Balancer)
invariants, with fees, spot prices, and a closed-form impermanent-loss helper.

References
----------
- Angeris, G. et al. (2021). "An Analysis of Uniswap Markets." Cryptoeconomic Systems.
- Angeris, G. & Chitra, T. (2020). "Improved Price Oracles: CFMMs." ACM AFT.
- Martinelli, F. & Mushegian, N. (2019). Balancer whitepaper.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ConstantProductAMM", "ConstantMeanAMM", "impermanent_loss"]


class ConstantProductAMM:
    """Two-asset constant-product market maker (``x * y = k``).

    Parameters
    ----------
    x, y : initial reserves of asset X and asset Y.
    fee  : proportional trading fee on the input amount (e.g. 0.003 = 0.30%).
    """

    def __init__(self, x: float, y: float, fee: float = 0.003):
        if x <= 0 or y <= 0:
            raise ValueError("reserves must be positive")
        if not (0.0 <= fee < 1.0):
            raise ValueError("fee must be in [0, 1)")
        self.x = float(x)
        self.y = float(y)
        self.fee = float(fee)

    @property
    def k(self) -> float:
        return self.x * self.y

    @property
    def spot_price(self) -> float:
        """Spot price of X in units of Y (``y / x``), ignoring fees."""
        return self.y / self.x

    def amount_out(self, dx: float, x_in: bool = True) -> float:
        r"""Output for selling ``dx`` of one asset into the pool.

        With fee ``f`` on the input, selling ``dx`` of X for Y returns
        ``dy = y * dx*(1-f) / (x + dx*(1-f))`` (and symmetrically for Y->X).
        """
        if dx <= 0:
            raise ValueError("input amount must be positive")
        gamma = 1.0 - self.fee
        if x_in:
            dx_eff = dx * gamma
            return float(self.y * dx_eff / (self.x + dx_eff))
        else:
            dy_eff = dx * gamma
            return float(self.x * dy_eff / (self.y + dy_eff))

    def swap(self, dx: float, x_in: bool = True) -> float:
        """Execute a swap, mutating reserves; return the output amount."""
        out = self.amount_out(dx, x_in=x_in)
        if x_in:
            self.x += dx
            self.y -= out
        else:
            self.y += dx
            self.x -= out
        return out

    def price_impact(self, dx: float, x_in: bool = True) -> float:
        """Relative difference between execution price and pre-trade spot price."""
        out = self.amount_out(dx, x_in=x_in)
        exec_price = out / dx if x_in else dx / out
        spot = self.spot_price
        return float(abs(exec_price - spot) / spot)


class ConstantMeanAMM:
    """Constant weighted-geometric-mean market maker (Balancer).

    Invariant ``prod_i R_i^{w_i} = k`` with weights ``w_i`` summing to 1.
    Constant-product is the two-asset, equal-weight special case.
    """

    def __init__(self, reserves, weights, fee: float = 0.003):
        self.R = np.asarray(reserves, float)
        self.w = np.asarray(weights, float)
        if self.R.shape != self.w.shape:
            raise ValueError("reserves and weights must align")
        if np.any(self.R <= 0):
            raise ValueError("reserves must be positive")
        if not np.isclose(self.w.sum(), 1.0):
            raise ValueError("weights must sum to 1")
        self.fee = float(fee)

    def invariant(self) -> float:
        return float(np.prod(self.R ** self.w))

    def spot_price(self, i: int, j: int) -> float:
        """Spot price of asset ``i`` in units of asset ``j``."""
        return float((self.R[j] / self.w[j]) / (self.R[i] / self.w[i]))

    def amount_out(self, i_in: int, j_out: int, amount_in: float) -> float:
        r"""Output of asset ``j_out`` for selling ``amount_in`` of asset ``i_in``.

        ``dy = R_j * (1 - (R_i / (R_i + amount_in*(1-f)))^{w_i / w_j})``.
        """
        gamma = 1.0 - self.fee
        ai = amount_in * gamma
        ratio = self.R[i_in] / (self.R[i_in] + ai)
        return float(self.R[j_out] * (1.0 - ratio ** (self.w[i_in] / self.w[j_out])))


def impermanent_loss(price_ratio: float) -> float:
    r"""Impermanent (divergence) loss for a constant-product LP, vs. holding.

    If the external price ratio moves by a factor ``r`` from entry, an equal-value
    constant-product LP underperforms simply holding the initial basket by

    .. math::
        \\mathrm{IL}(r) = \\frac{2\\sqrt{r}}{1 + r} - 1 \\le 0,

    zero at ``r = 1`` and strictly negative otherwise. Returned as a (non-positive)
    fraction; e.g. ``-0.0572`` means a 5.72% shortfall.
    """
    r = float(price_ratio)
    if r <= 0:
        raise ValueError("price ratio must be positive")
    return float(2.0 * np.sqrt(r) / (1.0 + r) - 1.0)
