"""Fee-bearing bounded market makers and clearing-price routing.

A cost-function market maker charges ``C(q + s) - C(q)`` for a fill ``s``, so
any round trip is free: the charge telescopes to zero over any path returning
to the same state. The simplest convex charge that *can* price a round trip is
path-dependent: a proportional fee ``f * |s|`` per fill. This module works out
the consequences for a single (scalar) risk traded against several makers.

Under Fenchel conjugation the fee is exactly a bid-ask spread. The conjugate of
``f * |.|`` is the indicator of the band ``[-f, f]``, so the conjugate of the
effective cost ``C(q + s) - C(q) + f * |s|`` is the no-fee conjugate evaluated
at a soft-thresholded price: the maker quotes ``ask = C'(q) + f``,
``bid = C'(q) - f``, and takes no flow while the clearing price sits inside the
band. Aggregating several makers is the infimal convolution of their effective
costs, and its minimiser is found by a one-dimensional monotone root-find on
the clearing price: each maker supplies

    s_i(p) = (C_i')^{-1}(p - f_i * sign(s_i)) - q_i,   or 0 inside the band,

and ``p*`` solves ``sum_i s_i(p*) = size``. The ``|s|`` term is an L1 penalty,
so the optimal split is sparse — makers whose quote band contains ``p*``
contribute exactly zero, and a growing trade eats through makers in fee order
like walking the levels of a limit order book.

The bounded maker used here is the log-cosh family ``C(q) = b log cosh(q/b)``,
whose marginal price ``tanh(q/b)`` lives in ``(-1, 1)`` (a settlement in
``[-1, 1]`` gives worst-case loss ``b log 2``) and whose supply curve inverts
in closed form via ``arctanh``.

References
----------
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). "Efficient Market
  Making via Convex Optimization, and a Connection to Online Learning."
  ACM TEAC 1(2).
- Barrieu, P. & El Karoui, N. (2005). "Inf-Convolution of Risk Measures and
  Optimal Risk Transfer." Finance and Stochastics 9(2).
- Glosten, L. (1994). "Is the Electronic Open Limit Order Book Inevitable?"
  Journal of Finance 49(4).

See ``research/proportional-fees-and-the-order-book.md`` for the derivation
and a calibrated prior-art assessment.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["LogCoshMaker", "route", "consolidated_book"]

_PRICE_EPS = 1e-12


class LogCoshMaker:
    """Bounded scalar market maker ``C(q) = b log cosh(q / b)`` with fee ``f``.

    Parameters
    ----------
    b   : liquidity (depth) parameter, ``b > 0``.
    fee : proportional fee per unit traded, ``f >= 0`` (half-spread).
    q0  : initial inventory (signed).
    """

    def __init__(self, b: float, fee: float = 0.0, q0: float = 0.0):
        if b <= 0:
            raise ValueError("liquidity b must be positive")
        if fee < 0:
            raise ValueError("fee must be non-negative")
        self.b = float(b)
        self.fee = float(fee)
        self.q = float(q0)

    def cost(self, q: float) -> float:
        """Potential ``b log cosh(q/b)``, computed overflow-safely."""
        x = abs(q) / self.b
        # log cosh x = x + log(1 + exp(-2x)) - log 2
        return float(self.b * (x + np.log1p(np.exp(-2.0 * x)) - np.log(2.0)))

    @property
    def marginal_price(self) -> float:
        """Current no-fee marginal price ``tanh(q/b)``."""
        return float(np.tanh(self.q / self.b))

    @property
    def bid(self) -> float:
        return self.marginal_price - self.fee

    @property
    def ask(self) -> float:
        return self.marginal_price + self.fee

    def trade_cost(self, s: float) -> float:
        """Effective cost ``C(q+s) - C(q) + f |s|`` of a fill ``s``."""
        return self.cost(self.q + s) - self.cost(self.q) + self.fee * abs(s)

    def worst_case_loss(self) -> float:
        """Bound ``b log 2`` on the maker's loss for settlements in [-1, 1]."""
        return float(self.b * np.log(2.0))

    def supply(self, p: float) -> float:
        """Fill the maker optimally provides at clearing price ``p``.

        Zero inside the quote band ``[bid, ask]``; otherwise the closed-form
        inverse of the fee-shifted marginal price, ``b arctanh(p -+ f) - q``.
        Monotone non-decreasing in ``p``.
        """
        if p >= self.ask:
            target = min(p - self.fee, 1.0 - _PRICE_EPS)
            return float(self.b * np.arctanh(target) - self.q)
        if p <= self.bid:
            target = max(p + self.fee, -1.0 + _PRICE_EPS)
            return float(self.b * np.arctanh(target) - self.q)
        return 0.0

    def apply_fill(self, s: float) -> float:
        """Execute a fill, mutating inventory; return the charge collected."""
        charge = self.trade_cost(s)
        self.q += s
        return charge


def route(makers: Sequence[LogCoshMaker], size: float, tol: float = 1e-12):
    """Split ``size`` across ``makers`` at minimal total effective cost.

    Returns ``(fills, clearing_price)`` where ``fills[i]`` is maker ``i``'s
    (signed) fill, ``sum(fills) == size``, and the split minimises
    ``sum_i [C_i(q_i + s_i) - C_i(q_i) + f_i |s_i|]`` — the infimal convolution
    of the effective costs, evaluated by bisection on the clearing price.

    Does not mutate the makers; call ``apply_fill`` on each to execute.
    """
    if not makers:
        raise ValueError("need at least one maker")
    if size == 0.0:
        return np.zeros(len(makers)), float(np.mean([m.marginal_price for m in makers]))

    lo, hi = -1.0 + _PRICE_EPS, 1.0 - _PRICE_EPS
    total = sum(m.supply(hi) for m in makers)
    if total < size:
        raise ValueError("size exceeds the makers' aggregate capacity")
    total = sum(m.supply(lo) for m in makers)
    if total > size:
        raise ValueError("size exceeds the makers' aggregate capacity")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum(m.supply(mid) for m in makers) < size:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    p_star = 0.5 * (lo + hi)
    fills = np.array([m.supply(p_star) for m in makers])
    # Bisection leaves a rounding residual; assign it to a maker already in
    # the money at p* so no dead-zone maker is dragged into a tiny fill.
    residual = size - fills.sum()
    if abs(residual) > 0:
        active = np.flatnonzero(fills)
        idx = active[np.argmax(np.abs(fills[active]))] if active.size else int(
            np.argmin([abs(p_star - m.marginal_price) - m.fee for m in makers])
        )
        fills[idx] += residual
    return fills, float(p_star)


def consolidated_book(makers: Sequence[LogCoshMaker], prices: np.ndarray) -> np.ndarray:
    """Aggregate supply curve ``sum_i s_i(p)`` over a grid of prices.

    This is the consolidated limit order book implied by the makers: flat
    exactly where every maker's quote band covers ``p``, smooth and strictly
    increasing where at least one maker is in the money.
    """
    prices = np.asarray(prices, float)
    return np.array([sum(m.supply(p) for m in makers) for p in prices])
