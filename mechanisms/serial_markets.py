"""Serial composition of markets: chain-rule factors, filtering, cycles.

Parallel composition of makers is infimal convolution (liquidities add;
``liquidity_fusion.py``). Serial composition runs one market per factor of a
probabilistic factorization ``P(everything) = prod P(node | parents)``, each
market pricing its conditional given what is upstream. For linear-Gaussian
structure the correspondence is exact and this module demonstrates it:

- ``propagate`` is the computed (model) step: pushing a belief through the
  dynamics ``x' = a x + noise`` in mean/precision form.
- ``market_update`` is the market step: fusing the propagated belief with an
  observation quoted by a maker whose capital is the observation precision —
  precisely the update step of the Kalman filter, by the fusion identity of
  ``liquidity_fusion.py``. Alternating the two *is* the Kalman filter
  (theorem-tested against the classical recursion).
- ``chain_posterior`` computes an interior node's posterior on a Gaussian
  chain by market operations only (forward message, local observation,
  backward message, fused by precision addition) — Gaussian belief
  propagation with trades as messages, exact on trees.
- ``cycle_arbitrage`` is the loopy pathology: pairwise correlation quotes
  around a cycle that admit no joint (a non-PSD quote matrix) hand any
  trader a bundle with negative price and nonnegative payoff. Cycle
  inconsistency is arbitrage, and arbitrageurs are the loop correction.

See ``research/serial-markets-and-belief-propagation.md``.

References
----------
- Hanson, R. (2007). "Logarithmic Market Scoring Rules for Modular
  Combinatorial Information Aggregation." J. Prediction Markets 1(1).
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction
  Problems." J. Basic Engineering 82(1).
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems.*
  Morgan Kaufmann.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "propagate",
    "market_update",
    "market_kalman_filter",
    "chain_posterior",
    "cheapest_route",
    "cycle_arbitrage",
]


def propagate(mean: float, precision: float, a: float, q: float):
    """Push a scalar belief through ``x' = a x + N(0, q)``: the model step."""
    var = a * a / precision + q
    return a * mean, 1.0 / var


def market_update(mean: float, precision: float, y: float, obs_precision: float):
    """Fuse the belief with an observation maker quoting ``y`` with capital
    ``obs_precision``: the market step, equal to the Kalman update."""
    post_precision = precision + obs_precision
    post_mean = (precision * mean + obs_precision * y) / post_precision
    return post_mean, post_precision


def market_kalman_filter(ys, a: float, q: float, r: float, mean0: float, var0: float):
    """Filter observations ``ys`` of ``x_t`` (obs noise variance ``r``) by
    alternating the model step and the market step. Returns arrays of
    filtered means and variances, one per observation."""
    mean, precision = float(mean0), 1.0 / float(var0)
    means, variances = [], []
    for y in ys:
        mean, precision = propagate(mean, precision, a, q)
        mean, precision = market_update(mean, precision, float(y), 1.0 / r)
        means.append(mean)
        variances.append(1.0 / precision)
    return np.array(means), np.array(variances)


def chain_posterior(node: int, mus, taus, coeffs, noises, ys, rs):
    """Posterior of ``x_node`` on a Gaussian chain, by market operations only.

    Chain ``x_0 ~ N(mus, 1/taus)``, ``x_{i+1} = coeffs[i] x_i + N(0,
    noises[i])``, with observation ``ys[i] ~ N(x_i, rs[i])`` at every node.
    The forward message is propagate/market_update run up the chain to
    ``node``; the backward message pulls each downstream observation back
    through the dynamics (``y/coeff`` observed with precision
    ``coeff^2/(r + noise-adjusted)``) — Gaussian belief propagation with the
    fusion done by precision addition. Exact on the chain.
    """
    n = len(ys)
    # forward: prior at 0, then alternate model and market steps up to node
    mean, prec = float(mus), float(taus)
    mean, prec = market_update(mean, prec, ys[0], 1.0 / rs[0])
    for i in range(node):
        mean, prec = propagate(mean, prec, coeffs[i], noises[i])
        mean, prec = market_update(mean, prec, ys[i + 1], 1.0 / rs[i + 1])
    # backward: fold downstream observations back into node's coordinates
    b_mean, b_prec = 0.0, 0.0
    for j in range(n - 1, node, -1):
        if b_prec > 0:
            fused_mean, fused_prec = market_update(b_mean, b_prec, ys[j], 1.0 / rs[j])
        else:
            fused_mean, fused_prec = ys[j], 1.0 / rs[j]
        # invert x_j = c x_{j-1} + noise: x_{j-1} seen with mean fused/c
        c, s = coeffs[j - 1], noises[j - 1]
        b_mean = fused_mean / c
        b_prec = c * c / (1.0 + s * fused_prec) * fused_prec
    if b_prec > 0:
        mean, prec = market_update(mean, prec, b_mean, b_prec)
    return mean, 1.0 / prec


def cheapest_route(stage_costs):
    """Min-plus elimination over a finite-state chain: Viterbi by routing.

    ``stage_costs[t]`` is a ``(k_t, k_{t+1})`` array of leg costs
    ``phi_t(i, j)`` (for an HMM, ``-log P(x_{t+1}=j | x_t=i) - log
    P(y_{t+1} | x_{t+1}=j)`` plus an initial vector folded into stage 0).
    Returns ``(values, path)``: ``values[j]`` is the cheapest total cost of
    any route ending in state ``j``, and ``path`` the argmin route — the
    max-product (Viterbi) decoding of the corresponding HMM.
    """
    values = np.zeros(np.asarray(stage_costs[0], float).shape[0])
    back = []
    for phi in stage_costs:
        phi = np.asarray(phi, float)
        totals = values[:, None] + phi
        back.append(np.argmin(totals, axis=0))
        values = totals.min(axis=0)
    path = [int(np.argmin(values))]
    for pointers in reversed(back):
        path.append(int(pointers[path[-1]]))
    return values, path[::-1]


def cycle_arbitrage(corr):
    """Certificate that inconsistent pairwise quotes around a cycle are
    arbitrage.

    ``corr`` is a symmetric quote matrix with unit diagonal assembled from
    pairwise markets. If it admits no joint distribution (a negative
    eigenvalue), return ``(w, guaranteed_profit)``: the bundle with weights
    ``w w^T`` has price ``w' corr w < 0`` yet pays ``(w'x)^2 >= 0``, a sure
    profit of at least ``-lambda_min`` per unit. If the quotes are
    consistent, return ``(None, 0.0)``: no such bundle exists.
    """
    corr = np.asarray(corr, float)
    vals, vecs = np.linalg.eigh(corr)
    if vals[0] >= 0:
        return None, 0.0
    return vecs[:, 0], float(-vals[0])
