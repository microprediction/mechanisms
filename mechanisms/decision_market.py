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
for unchosen actions never settle, so reports there go unscored. A **stochastic**
rule with full support is the first half of the repair — every action keeps a
positive probability of being taken, so every conditional market settles with
positive probability. :func:`softmax_decision` and :func:`epsilon_greedy_decision`
have full support; :func:`argmax_decision` does not.

Full support alone is not sufficient, because the selection probability is itself
a function of the reports, and a trader who moves a report moves the odds that it
is ever scored. What restores propriety is dividing the realised score by that
probability. Writing ``R(P, a) > 0`` for the chance action ``a`` is selected given
the submitted conditionals ``P``, and scoring only the action taken,

    S_R(P, a, w) = Sbar(P_a, w) / R(P, a),

so that the expectation telescopes,

    sum_a R(P,a) E[ Sbar(P_a, w) / R(P,a) ]  =  sum_a E[ Sbar(P_a, w) ],

a sum of ordinary proper scores, one per action, each optimised by reporting that
action's true conditional (Othman & Sandholm 2010; Chen, Kash, Ruberry & Shnayder
2011). :func:`inverse_selection_score` applies the factor and
:func:`expected_decision_score` is the sum on the right. Scores here are losses,
as everywhere in :mod:`mechanisms.scoring_rules`, so the agent minimises.

The factor is not free. When ``R(P, a)`` is small the realised transfer scales
like ``1 / R(P, a)``, so near-deterministic selection buys identification with
high-variance, potentially very large contingent payments.

Functions
---------
conditional_expected_values  per-action ``E[value | a]`` from conditional rows
argmax_decision              deterministic best action (one-hot; not full support)
softmax_decision             Boltzmann choice distribution (full support, ``T>0``)
epsilon_greedy_decision      epsilon-mixed argmax (full support, ``eps>0``)
inverse_selection_score      a realised score divided by its selection probability
expected_decision_score      expected score under the inverse-selection rule

Class
-----
DecisionMarket               one LMSR conditional market per action, with action
                             sampling (``select``), recorded trader positions,
                             and inverse-selection settlement (``settle``)
"""

from __future__ import annotations

import numpy as np

from .lmsr import LMSR
from .scoring_rules import log_score

__all__ = [
    "conditional_expected_values",
    "argmax_decision",
    "softmax_decision",
    "epsilon_greedy_decision",
    "inverse_selection_score",
    "expected_decision_score",
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
    support**, so unchosen actions are never scored and reports there can be
    pushed around for free. Included as the manipulable baseline, and note that
    swapping in a full-support rule is necessary but not sufficient: the
    ``1 / R(P, a)`` factor of :func:`inverse_selection_score` is what makes
    truthful reporting optimal.
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


def inverse_selection_score(raw_score: float, selection_prob: float) -> float:
    r"""The realised score divided by the probability the action was selected.

    This is the whole of the construction that makes a decision market proper:
    ``S_R(P, a, w) = Sbar(P_a, w) / R(P, a)``. Because only the taken action is
    scored, the endogenous selection probability would otherwise reweight the
    agent's objective; dividing by it cancels the reweighting exactly.
    """
    if not (0.0 < selection_prob <= 1.0):
        raise ValueError("selection probability must lie in (0, 1]")
    return float(raw_score) / float(selection_prob)


def expected_decision_score(reports, true_conditionals, score=log_score) -> float:
    r"""Expected inverse-selection score: ``sum_a E_{w ~ true(.|a)}[Sbar(P_a, w)]``.

    The selection probabilities have already cancelled, which is the point, so
    the decision rule does not appear. ``reports`` and ``true_conditionals`` are
    ``(A, K)`` arrays of per-action outcome distributions. Every summand is an
    ordinary expected proper score, so the whole is optimised report by report at
    ``reports == true_conditionals``, whatever the rule does the selecting. Scores
    here follow the module-wide loss convention, so *optimised* means minimised.
    """
    P = np.atleast_2d(np.asarray(reports, float))
    Q = np.atleast_2d(np.asarray(true_conditionals, float))
    if P.shape != Q.shape:
        raise ValueError("reports and true conditionals must have the same shape")
    return float(sum(sum(Q[a, k] * score(P[a], k) for k in range(P.shape[1]))
                     for a in range(P.shape[0])))


class DecisionMarket:
    """A decision variable backed by one LMSR conditional market per action."""

    def __init__(self, n_actions: int, n_outcomes: int, b: float = 100.0):
        if n_actions < 2:
            raise ValueError("need at least 2 actions")
        self.n_actions = int(n_actions)
        self.n_outcomes = int(n_outcomes)
        self.markets = [LMSR(self.n_outcomes, b=b) for _ in range(self.n_actions)]
        # positions[trader][action] is a length-K vector of shares held.
        self.positions: dict = {}

    def conditional_prices(self, action: int) -> np.ndarray:
        """Outcome distribution implied by ``action``'s conditional market."""
        return self.markets[int(action)].prices()

    def conditional_matrix(self) -> np.ndarray:
        """``(A, K)`` matrix of conditional outcome distributions, one row per action."""
        return np.array([m.prices() for m in self.markets])

    def trade(self, action: int, outcome: int, shares: float,
              trader: str = None) -> float:
        """Buy ``shares`` of ``outcome`` in ``action``'s conditional market; return cost.

        Naming a ``trader`` records the position, which is what settlement needs:
        an unattributed trade still moves the price but nobody is paid for it.
        """
        cost = self.markets[int(action)].buy(int(outcome), float(shares))
        if trader is not None:
            book = self.positions.setdefault(
                trader, np.zeros((self.n_actions, self.n_outcomes)))
            book[int(action), int(outcome)] += float(shares)
        return cost

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

    def select(self, outcome_values, rule: str = "softmax", rng=None, **kwargs):
        """Draw an action from the decision rule; return ``(action, decision)``.

        The decision distribution is returned alongside the draw because
        settlement needs the probability that *this* action was selected, and it
        must be the one standing at the moment of the draw. Recomputing it later
        from a moved market would divide by the wrong number.
        """
        decision = self.decision(outcome_values, rule=rule, **kwargs)
        rng = np.random.default_rng() if rng is None else rng
        if np.any(decision <= 0):
            raise ValueError(
                "decision rule has an action with zero probability; that market "
                "never settles, so reports in it go unscored (use softmax or "
                "epsilon_greedy)")
        return int(rng.choice(self.n_actions, p=decision)), decision

    def settle(self, action_taken: int, outcome: int, decision):
        r"""Settle the taken action's market; return ``(maker_pnl, trader_payouts)``.

        Only the taken action settles, so its payoffs carry the inverse-selection
        factor ``1 / R(P, a)``. ``decision`` is the choice distribution that was
        standing when the action was drawn, which :meth:`select` returns for
        exactly this purpose.

        A trader holding ``s`` shares of the realised outcome in that action's
        market collects ``s / R(P, a)``; holdings in the other actions' markets
        are void, which is the counterfactual structure the factor compensates
        for. The maker's figure is its own P&L on the settled market, scaled the
        same way, so an LMSR bounded at ``b log K`` is bounded at
        ``b log K / R(P, a)``: a rule that rarely picks an action needs
        correspondingly deeper pockets when it does.
        """
        a, k = int(action_taken), int(outcome)
        prob = float(np.asarray(decision, float)[a])
        maker = inverse_selection_score(self.markets[a].realized_pnl(k), prob)
        payouts = {t: inverse_selection_score(float(book[a, k]), prob)
                   for t, book in self.positions.items() if book[a, k] > 0}
        return maker, payouts
