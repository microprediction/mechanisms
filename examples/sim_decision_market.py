"""Simulation: a decision market ensembles action-conditional forecasts and picks.

For each candidate action a conditional LMSR market predicts the outcome *given
that the action is taken*; the per-action expected values ``v_a = E[value | a]``
are read off the markets, and a decision rule turns them into a choice. We push
two of three action-conditional markets toward the good outcome, read the implied
values, compare the decision rules, and settle. A deterministic argmax leaves the
unchosen markets unsettled and so manipulable; a full-support stochastic rule
keeps every conditional market live, which is necessary but not sufficient, and
the 1/R(P, a) settlement factor is what actually makes truthful reporting
optimal.

Run:  python examples/sim_decision_market.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars

from mechanisms.decision_market import (
    DecisionMarket,
    argmax_decision,
    softmax_decision,
    epsilon_greedy_decision,
)


def simulate(b=100.0):
    # Three actions, two outcomes (loss, win); value = payoff of each outcome,
    # so E[value | a] is just the conditional probability of the win.
    outcome_values = [0.0, 1.0]
    WIN = 1
    dm = DecisionMarket(n_actions=3, n_outcomes=2, b=b)

    section("Three action-conditional markets, initially uninformed")
    labeled_bars([(f"v(action {a})", dm.values(outcome_values)[a]) for a in range(3)],
                 vmax=1.0)
    print("  every conditional market is uniform, so E[value | a] = 0.5 for all actions.")

    section("Traders push two markets toward the good outcome")
    dm.trade(action=0, outcome=WIN, shares=40.0, trader="ann")
    dm.trade(action=1, outcome=WIN, shares=90.0, trader="bo")  # action 2 draws none
    v = dm.values(outcome_values)
    labeled_bars([(f"v(action {a})", v[a]) for a in range(3)], vmax=1.0)
    print(f"  action 1 now looks best: v = {v[1]:.3f}  (vs {v[0]:.3f} and {v[2]:.3f}).")

    section("Decision rules turn the values into a choice")
    rules = [
        ("argmax (deterministic, no full support)", argmax_decision(v)),
        ("softmax  T = 0.10", softmax_decision(v, temperature=0.10)),
        ("softmax  T = 0.30", softmax_decision(v, temperature=0.30)),
        ("epsilon-greedy  eps = 0.10", epsilon_greedy_decision(v, epsilon=0.10)),
    ]
    for name, dist in rules:
        print(f"\n  {name}")
        labeled_bars([(f"P(action {a})", dist[a]) for a in range(3)], vmax=1.0)

    section("Full support is necessary")
    argmax_d = argmax_decision(v)
    soft_d = softmax_decision(v, temperature=0.30)
    unsettled = [int(a) for a in np.flatnonzero(argmax_d == 0.0)]
    print(f"  Under argmax only action {int(np.argmax(v))} is ever taken, so the markets for")
    print(f"  actions {unsettled} never settle: reports there go unscored, hence manipulable.")
    print("  Under softmax every action keeps positive settlement probability:")
    labeled_bars([(f"settle P(action {a})", soft_d[a]) for a in range(3)], vmax=1.0)

    section("...but not sufficient: settle with the 1/R(P, a) factor")
    print("  The selection probability is itself a function of the reports, so a")
    print("  trader who shades a report moves the odds it is ever scored. Dividing")
    print("  the settled payoff by R(P, a) cancels that exactly (Othman & Sandholm")
    print("  2010; Chen, Kash, Ruberry & Shnayder 2011).")
    rng = np.random.default_rng(0)
    action, decision = dm.select(outcome_values, rule="softmax",
                                 temperature=0.30, rng=rng)
    maker, payouts = dm.settle(action, WIN, decision)
    print(f"\n  drawn action        : {action}   (selected with probability "
          f"R = {decision[action]:.3f})")
    print(f"  maker P&L, scaled   : {maker:+.2f}")
    for who, amount in sorted(payouts.items()):
        print(f"  pays {who:<16}: {amount:8.2f}")
    voided = [w for w in dm.positions if w not in payouts]
    print(f"  void (wrong action) : {voided if voided else 'none'}")
    print("\n  Note the size of the numbers. The factor buys identification with")
    print("  variance: the rarer the action, the larger the contingent payment.")


if __name__ == "__main__":
    simulate()
