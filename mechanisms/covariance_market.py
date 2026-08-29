"""A cost-function market maker whose price is a covariance matrix.

Securities pay the products ``X_j X_k`` of a random vector's coordinates. The
convex hull of the payoff matrices ``x x^T`` is the positive-semidefinite
cone, so coherence (no arbitrage) of a price matrix is exactly positive
semidefiniteness: PSD consistency is not a constraint to engineer but the
market analogue of "prices lie in the simplex".

The maker here is the exponential-family market scoring rule instantiated for
the zero-mean Gaussian family, whose sufficient statistic is ``x x^T``. The
cost is the scaled log-partition function

    C(Q) = b A(Theta_0 + Q / b),      A(Theta) = -(1/2) log det(-2 Theta),

so the quoted price is the mean parameter ``Sigma = (-2 Theta)^{-1}``,
positive definite by construction, and the inventory is (minus half) a
precision matrix: traders literally deposit precision, tying this design to
``research/liquidity-is-precision.md``. The maker's Bregman divergence is the
Kullback-Leibler divergence between zero-mean Gaussians — Stein's loss, the
classical covariance loss — so a myopic trader maximizes expected profit by
moving the quote to their believed second-moment matrix and collects
``b`` times the Stein discrepancy of the previous quote.

An optional proportional fee on the nuclear norm of the fill conjugates to a
spectral-norm dead zone: no profitable trade exists while the spectral norm
of the mispricing is below the fee, the matrix form of the fee-spread lemma
in ``papers/combining-linear-fee-market-makers.md``.

See ``research/covariance-market.md`` for the derivations, the graphical
(restricted-coverage) reading, and the monteprediction connection.

References
----------
- Abernethy, J. D., Frongillo, R. M. & Kutty, S. (2015). "On Risk Measures,
  Market Making, and Exponential Families." ACM SIGecom Exchanges 13(2).
- Dempster, A. P. (1972). "Covariance Selection." Biometrics 28(1).
- Cai, J.-F., Candès, E. J. & Shen, Z. (2010). "A Singular Value
  Thresholding Algorithm for Matrix Completion." SIAM J. Optimization 20(4).
"""

from __future__ import annotations

import numpy as np

__all__ = ["GaussianCovarianceMaker"]


def _sym(S):
    S = np.asarray(S, float)
    return 0.5 * (S + S.T)


class GaussianCovarianceMaker:
    """Market maker quoting a covariance matrix via the Gaussian log-partition.

    Parameters
    ----------
    sigma0 : initial quote, a symmetric positive-definite matrix.
    b      : liquidity (depth); worst-case exposure scales with ``b``.
    fee    : proportional fee per unit nuclear norm of a fill (half-spread in
             spectral norm), ``>= 0``.
    """

    def __init__(self, sigma0, b: float = 1.0, fee: float = 0.0):
        sigma0 = _sym(sigma0)
        if np.any(np.linalg.eigvalsh(sigma0) <= 0):
            raise ValueError("sigma0 must be positive definite")
        if b <= 0:
            raise ValueError("liquidity b must be positive")
        if fee < 0:
            raise ValueError("fee must be non-negative")
        self.d = sigma0.shape[0]
        self.b = float(b)
        self.fee = float(fee)
        self.theta0 = -0.5 * np.linalg.inv(sigma0)  # natural parameter
        self.Q = np.zeros_like(sigma0)              # inventory (symmetric)

    # -- exponential-family machinery -------------------------------------

    def _theta(self, Q):
        return self.theta0 + _sym(Q) / self.b

    @staticmethod
    def _log_partition(theta):
        """``A(Theta) = -(1/2) log det(-2 Theta)``; +inf outside the domain."""
        m = -2.0 * theta
        vals = np.linalg.eigvalsh(m)
        if np.any(vals <= 0):
            return np.inf
        return -0.5 * float(np.sum(np.log(vals)))

    def cost(self, Q) -> float:
        """Potential ``b A(Theta_0 + Q / b)``."""
        return self.b * self._log_partition(self._theta(Q))

    @property
    def price(self) -> np.ndarray:
        """Quoted second-moment matrix ``(-2 Theta)^{-1}``, positive definite."""
        return np.linalg.inv(-2.0 * self._theta(self.Q))

    @property
    def precision(self) -> np.ndarray:
        """Implied precision ``-2 Theta`` — the inventory's natural habitat."""
        return -2.0 * self._theta(self.Q)

    # -- trading -----------------------------------------------------------

    def trade_cost(self, S) -> float:
        """Charge ``C(Q+S) - C(Q) + fee * ||S||_*`` for a (symmetric) fill.

        Infinite if the fill would push the quote out of the PSD cone —
        the maker simply cannot be traded outside coherent prices.
        """
        S = _sym(S)
        nuclear = float(np.abs(np.linalg.eigvalsh(S)).sum())
        return self.cost(self.Q + S) - self.cost(self.Q) + self.fee * nuclear

    def apply_fill(self, S) -> float:
        """Execute a fill, mutating inventory; return the charge collected."""
        charge = self.trade_cost(S)
        if not np.isfinite(charge):
            raise ValueError("fill would leave the coherent (PSD) price cone")
        self.Q = self.Q + _sym(S)
        return charge

    def fill_to(self, sigma_target) -> np.ndarray:
        """The fill that moves the quote to ``sigma_target``:
        ``S = b (Theta_target - Theta_current)``."""
        sigma_target = _sym(sigma_target)
        theta_target = -0.5 * np.linalg.inv(sigma_target)
        return self.b * (theta_target - self._theta(self.Q))

    # -- diagnostics -------------------------------------------------------

    @staticmethod
    def stein_divergence(m, sigma) -> float:
        """``KL(N(0, M) || N(0, Sigma))`` — Stein's loss of the quote."""
        m, sigma = _sym(m), _sym(sigma)
        d = m.shape[0]
        inv_sigma = np.linalg.inv(sigma)
        _, ld_m = np.linalg.slogdet(m)
        _, ld_s = np.linalg.slogdet(sigma)
        return 0.5 * float(np.trace(inv_sigma @ m) - d + ld_s - ld_m)
