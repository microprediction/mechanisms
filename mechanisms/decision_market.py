"""decision_market — conditional markets on a decision variable.

A decision market turns *action-conditional* forecasts into a choice. For each
candidate action ``a`` a conditional market predicts the distribution of an
outcome **given that ``a`` is taken**; a decision rule maps the resulting
per-action value estimates ``v_a = E[value | a]`` to a choice (or a distribution
over choices); and only the chosen action's market settles against what actually
happens — the markets for the actions not taken are counterfactual and void.

Read as ensembling: each conditional market is itself an aggregate of its
traders' beliefs, and the decision market *ensembles across the action-conditional
aggregates and selects among them*. This is the mechanism core of "futarchy"
with the governance framing removed (Hanson 2013; Othman & Sandholm 2010;
Chen, Kash, Ruberry & Shnayder 2011) — a pricing-and-selection rule, not a
theory of how a polity should decide.

Incentives. Under a **deterministic** argmax rule the conditional markets are not
incentive compatible: a trader can move *which* action is chosen, and the markets
for unchosen actions never settle, so reports there go unscored. The fix is a
**stochastic** decision rule with full support — every action keeps positive
probability of being taken, so every conditional market settles with positive
probability and a risk-neutral trader maximises expected payoff by reporting its
true conditional belief in each one (Othman & Sandholm 2010; Chen-Kash et al.
2011). :func:`softmax_decision` and :func:`epsilon_greedy_decision` have full
support; :func:`argmax_decision` does not, and is offered only as the
(manipulable) baseline the stochastic rules repair.

Functions
---------
conditional_expected_values  per-action ``E[value | a]`` from conditional rows
argmax_decision              deterministic best action (one-hot; not full support)
softmax_decision             Boltzmann choice distribution (full support, ``T>0``)
epsilon_greedy_decision      epsilon-mixed argmax (full support, ``eps>0``)

Class
-----
DecisionMarket               one LMSR conditional market per action
"""

from __future__ import annotations

import numpy as np

from .lmsr import LMSR

__all__ = [
    "conditional_expected_values",
    "argmax_decision",
    "softmax_decision",
    "epsilon_greedy_decision",
    "DecisionMarket",
]


def conditional_expected_values(conditional_probs, outcome_values) -> np.ndarray:
    r"""Per-action expected value ``v_a = sum_o P(o | a) value(o)``.

    ``conditional_probs`` is an ``(A, K)`` array whose row ``a`` is the outcome
    distribution conditional on action ``a``; ``outcome_values`` is length ``K``.
    """
    P = np.atleast_2d(np.asarray(conditional_probs, float))
    v = np.asarray(outcome_values, float)
    if P.shape[1] != v.shape[0]:
        raise ValueError("outcome_values must align with the outcome axis")
    return P @ v


def argmax_decision(values) -> np.ndarray:
    r"""Deterministic rule: all weight on the best action (ties split evenly).

    Returns a one-hot (tie-uniform) distribution over actions. It has **no full
    support**, so the conditional markets it drives are manipulable — included as
    the baseline the stochastic rules fix, not a recommended settlement rule.
    """
    v = np.asarray(values, float)
    best = np.flatnonzero(v == v.max())
    d = np.zeros(v.shape[0])
    d[best] = 1.0 / best.size
    return d


def softmax_decision(values, temperature: float = 1.0) -> np.ndarray:
    r"""Boltzmann choice ``P(a) ∝ exp(v_a / T)``.

    Full support for ``T > 0``; as ``T -> 0`` it concentrates on the argmax.
    """
    v = np.asarray(values, float)
    if temperature <= 0:
        raise ValueError("temperature must be positive (use argmax_decision for T=0)")
    z = v / float(temperature)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def epsilon_greedy_decision(values, epsilon: float = 0.1) -> np.ndarray:
    r"""Epsilon-greedy choice: weight ``1-eps`` on the argmax (tie-split), with
    ``eps`` spread uniformly over all actions. Full support for ``eps > 0``.
    """
    v = np.asarray(values, float)
    n = v.shape[0]
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must lie in [0, 1]")
    d = np.full(n, epsilon / n)
    best = np.flatnonzero(v == v.max())
    d[best] += (1.0 - epsilon) / best.size
    return d


class DecisionMarket:
    """A decision variable backed by one LMSR conditional market per action."""

    def __init__(self, n_actions: int, n_outcomes: int, b: float = 100.0):
        if n_actions < 2:
            raise ValueError("need at least 2 actions")
        self.n_actions = int(n_actions)
        self.n_outcomes = int(n_outcomes)
        self.markets = [LMSR(self.n_outcomes, b=b) for _ in range(self.n_actions)]

    def conditional_prices(self, action: int) -> np.ndarray:
        """Outcome distribution implied by ``action``'s conditional market."""
        return self.markets[int(action)].prices()

    def conditional_matrix(self) -> np.ndarray:
        """``(A, K)`` matrix of conditional outcome distributions, one row per action."""
        return np.array([m.prices() for m in self.markets])

    def trade(self, action: int, outcome: int, shares: float) -> float:
        """Buy ``shares`` of ``outcome`` in ``action``'s conditional market; return cost."""
        return self.markets[int(action)].buy(int(outcome), float(shares))

    def values(self, outcome_values) -> np.ndarray:
        """Per-action ``E[value | a]`` under the current conditional markets."""
        return conditional_expected_values(self.conditional_matrix(), outcome_values)

    def decision(self, outcome_values, rule: str = "softmax", **kwargs) -> np.ndarray:
        """Choice distribution over actions from the conditional values.

        ``rule`` is ``"softmax"`` (default, full support), ``"epsilon_greedy"``
        (full support), or ``"argmax"`` (deterministic, manipulable baseline).
        Extra keyword arguments pass through to the chosen rule.
        """
        v = self.values(outcome_values)
        if rule == "softmax":
            return softmax_decision(v, **kwargs)
        if rule == "epsilon_greedy":
            return epsilon_greedy_decision(v, **kwargs)
        if rule == "argmax":
            return argmax_decision(v)
        raise ValueError(f"unknown decision rule {rule!r}")

    def recommend(self, outcome_values) -> int:
        """The best action under the current conditional values (plain argmax)."""
        return int(np.argmax(self.values(outcome_values)))
