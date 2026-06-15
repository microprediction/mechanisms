"""Perpetual futures: the funding-rate tether.

A perpetual future ("perp") is a derivative with no expiry. Because it never
settles at maturity, it is kept pinned to an external index price by a periodic
**funding** cash flow exchanged directly between longs and shorts. The
conceptual ancestor is Shiller's (1993) perpetual claim, anchored by a
dividend-like flow; crypto perps (BitMEX, 2016) instead anchor via a
market-driven funding rate keyed to the perp's premium over the index.

This module implements the standard BitMEX-style funding rate

.. math::
    F = P + \\mathrm{clamp}(I - P,\\ -c,\\ +c)

(``P`` premium index, ``I`` interest-rate term, ``c`` clamp), the funding
payment, and a minimal margined position with mark-price liquidation.

References
----------
- Shiller, R. (1993). "Measuring Asset Values for Cash Settlement..." J. Finance 48(3).
- BitMEX. "Perpetual Contracts Guide" / "Funding" / "Fair Price Marking."
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["funding_rate", "funding_payment", "PerpPosition"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def funding_rate(premium_index: float, interest_rate: float = 0.0001,
                 clamp: float = 0.0005) -> float:
    r"""BitMEX-style funding rate ``F = P + clamp(I - P, -c, +c)``.

    Parameters
    ----------
    premium_index : ``P``, the perp's time-weighted premium over the index.
    interest_rate : ``I``, the per-interval interest term (default 0.01%).
    clamp         : the symmetric clamp band ``c`` (default 0.05%).

    When the premium is small, ``F`` sits near the interest rate; when the
    premium is large, the premium term drives funding to push the perp back to
    the index.
    """
    return float(premium_index + _clamp(interest_rate - premium_index, -clamp, clamp))


def funding_payment(position_notional: float, rate: float) -> float:
    """Funding paid by a long (received by a short) over one interval.

    Positive ``rate`` => longs pay shorts; the long's cash flow is negative.
    Returned value is the amount the *long* pays (so a short receives its
    negation).
    """
    return float(position_notional * rate)


@dataclass
class PerpPosition:
    """A leveraged perpetual position with mark-price liquidation.

    Parameters
    ----------
    side        : +1 for long, -1 for short.
    size        : contract size in units of the base asset.
    entry_price : execution price at open.
    collateral  : margin posted (in quote currency).
    maintenance_margin_ratio : fraction of notional required to avoid liquidation.
    """

    side: int
    size: float
    entry_price: float
    collateral: float
    maintenance_margin_ratio: float = 0.005

    def __post_init__(self):
        if self.side not in (1, -1):
            raise ValueError("side must be +1 (long) or -1 (short)")
        if self.size <= 0 or self.entry_price <= 0 or self.collateral <= 0:
            raise ValueError("size, entry_price, collateral must be positive")

    def notional(self, mark_price: float) -> float:
        return float(self.size * mark_price)

    def unrealized_pnl(self, mark_price: float) -> float:
        """Mark-to-market PnL: ``side * size * (mark - entry)``."""
        return float(self.side * self.size * (mark_price - self.entry_price))

    def equity(self, mark_price: float) -> float:
        """Account equity = collateral + unrealised PnL."""
        return float(self.collateral + self.unrealized_pnl(mark_price))

    def leverage(self, mark_price: float) -> float:
        eq = self.equity(mark_price)
        return float("inf") if eq <= 0 else self.notional(mark_price) / eq

    def is_liquidated(self, mark_price: float) -> bool:
        """True when equity falls below maintenance margin on the notional."""
        return self.equity(mark_price) <= self.maintenance_margin_ratio * self.notional(mark_price)

    def liquidation_price(self) -> float:
        r"""Approximate mark price at which the position is liquidated.

        Solving ``collateral + side*size*(p - entry) = mmr*size*p`` for ``p``::

            p* = (mmr_term ... )  # closed form below

        For a long: ``p* = (size*entry - collateral) / (size*(1 - mmr))``.
        For a short: ``p* = (size*entry + collateral) / (size*(1 + mmr))``.
        """
        m = self.maintenance_margin_ratio
        if self.side == 1:
            return float((self.size * self.entry_price - self.collateral) /
                         (self.size * (1.0 - m)))
        else:
            return float((self.size * self.entry_price + self.collateral) /
                         (self.size * (1.0 + m)))

    def apply_funding(self, mark_price: float, rate: float) -> float:
        """Debit/credit funding for one interval; return the cash flow to this position.

        With ``rate > 0`` longs pay shorts. A position's cash flow is
        ``-side * notional * rate``: a long (side +1) pays (negative flow) and a
        short (side -1) receives (positive flow). The flow adjusts collateral and
        is returned (positive = credit to this position).
        """
        cash_flow = -self.side * funding_payment(self.notional(mark_price), rate)
        self.collateral += cash_flow
        return float(cash_flow)
