"""Simulation: Correlated Agreement elicits truth with no ground truth.

Output agreement (reward = match a peer) has an uninformative equilibrium:
everyone can settle on a constant report and always "agree", revealing nothing.
Correlated Agreement (Shnayder, Agarwal, Frongillo & Parkes 2016, generalising
the binary mechanism of Dasgupta & Ghosh 2013) fixes that by rewarding agreement
on the SAME task only beyond the chance agreement expected from UNRELATED tasks.
A constant report agrees everywhere equally, so it nets zero. No prediction
report is needed (unlike BTS) — only signals on a shared pool of tasks.

Two things this demo is careful about, because both are easy to get wrong.

CA is *informed-truthful*, not stochastic-dominance truthful. The mechanism that
carries the SD guarantee is Enforced Agreement (Zhang, Xu, Pennock & Schoenebeck
2025), a different construction: output agreement with each agent's report
marginal forced to a target. Both are run below.

And CA's guarantee assumes the sign matrix is fixed, not estimated from the very
reports it scores. With only two agents that assumption fails loudly: a constant
reporter is half the data, inverts the estimated matrix, and profits.

Run:  python examples/sim_correlated_agreement.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars

from mechanisms.peer_prediction import (
    _ca_sign_matrix,
    correlated_agreement,
    enforced_agreement,
    output_agreement,
)


def simulate_signals(n_agents, n_tasks, q, rng):
    """Each task has a hidden binary state; an agent observes it correctly w.p. q."""
    state = rng.integers(0, 2, n_tasks)
    flip = rng.random((n_agents, n_tasks)) > q
    return state[None, :] ^ flip.astype(int)


def dominates(better, worse):
    """Does `better` first-order stochastically dominate `worse`, empirically?

    FOSD is F_better(t) <= F_worse(t) at every threshold. Also returns the
    largest violation, which is 0 when dominance holds on this sample.
    """
    better, worse = np.asarray(better), np.asarray(worse)
    grid = np.unique(np.concatenate([better, worse]))
    f_better = np.array([np.mean(better <= t) for t in grid])
    f_worse = np.array([np.mean(worse <= t) for t in grid])
    return bool(np.all(f_better <= f_worse + 1e-12)), float(np.max(f_better - f_worse))


def simulate(n_agents=8, n_tasks=400, q=0.82, seed=0):
    rng = np.random.default_rng(seed)
    sig = simulate_signals(n_agents, n_tasks, q, rng)

    # Three strategies for agent 0, against a truthful crowd (agents 1..n-1).
    truthful = sig.copy()
    constant = sig.copy(); constant[0, :] = 0          # the output-agreement exploit
    flipper = sig.copy(); flipper[0, :] = 1 - sig[0, :]  # adversarial anti-report

    section("Output agreement rewards the uninformative constant report")
    def mean_oa(reports):
        return float(np.mean([output_agreement(reports[0, t], reports[j, t])
                              for j in range(1, n_agents) for t in range(n_tasks)]))
    all_constant = np.zeros_like(sig)
    labeled_bars([("truthful", mean_oa(truthful)),
                  ("constant 0", mean_oa(constant)),
                  ("all settle on 0", mean_oa(all_constant))], vmax=1.0, fmt="{:.3f}")
    print("  a constant reporter still collects substantial reward for revealing")
    print("  nothing — and if everyone settles on a constant, agreement is a")
    print("  perfect 1.000. Output agreement cannot punish the empty strategy.")
    print("  Worse, truth is not always even a best response: positive correlation")
    print("  is not enough, only the stronger self-dominant condition is.")

    section("Correlated Agreement defeats it (agent 0's score by strategy)")
    s_truth = correlated_agreement(truthful, n_signals=2)[0]
    s_const = correlated_agreement(constant, n_signals=2)[0]
    s_flip = correlated_agreement(flipper, n_signals=2)[0]
    floor = min(s_truth, s_const, s_flip)
    labeled_bars([("truthful", s_truth - floor),
                  ("constant 0", s_const - floor),
                  ("flipper", s_flip - floor)],
                 vmax=(s_truth - floor) or 1.0, fmt="{:.3f}")
    print(f"  raw CA scores:  truthful {s_truth:+.3f}   constant {s_const:+.3f}"
          f"   flipper {s_flip:+.3f}")
    print("  truthful reporting pays the most; the constant exploit nets ~0.")
    print("  The guarantee behind this is informed truthfulness: no profile beats")
    print("  truthful, and ties are possible only against other informed")
    print("  strategies. It is not a dominant strategy, and truth is only weakly")
    print("  best — permutation strategies can tie.")

    section("Stochastic dominance belongs to Enforced Agreement, not to CA")
    R = 1500
    rows = []
    for name, score in (("CA", lambda s, r: correlated_agreement(s, n_signals=2)[0]),
                        ("EA", lambda s, r: enforced_agreement(
                            s, rng=np.random.default_rng(r))[0])):
        truth_scores, dev_scores = [], []
        for r in range(R):
            s = simulate_signals(n_agents, 120, q, np.random.default_rng(1000 + r))
            truth_scores.append(score(s, r))
            dev = s.copy(); dev[0, :] = 0
            dev_scores.append(score(dev, r))
        ok, viol = dominates(truth_scores, dev_scores)
        rows.append((name, np.mean(truth_scores), np.mean(dev_scores), ok, viol))
    for name, t, d, ok, viol in rows:
        verdict = "holds on this sample" if ok else f"fails, CDF gap {viol:.3f}"
        print(f"  {name}: mean truthful {t:+.3f}  constant {d:+.3f}   "
              f"empirical dominance {verdict}")
    print("\n  Read this narrowly. Checking one deviation over one signal structure")
    print("  is not stochastic dominance, which quantifies over every unilateral")
    print("  deviation at every threshold. Zhang et al. exhibit a world where a")
    print("  constant strategy is not stochastically dominated under CA, and")
    print("  prove the guarantee for EA instead (their Theorem 6.1, binary")
    print("  self-predicting signals). A simulation can refute dominance; it")
    print("  cannot establish it.")

    section("Why the sign matrix must not come from the reports it scores")
    two = simulate_signals(2, 400, q, np.random.default_rng(3))
    two_dev = two.copy(); two_dev[0, :] = 0
    print("  Sign(Delta) estimated with both agents truthful:")
    print("   ", _ca_sign_matrix(two, 2).tolist())
    print("  Sign(Delta) after agent 0 switches to a constant:")
    print("   ", _ca_sign_matrix(two_dev, 2).tolist())
    print(f"  agent 0's CA score: truthful "
          f"{correlated_agreement(two, n_signals=2)[0]:+.3f}   "
          f"constant {correlated_agreement(two_dev, n_signals=2)[0]:+.3f}")
    print("  The constant reporter is half the data, so they flip the estimated")
    print("  matrix from the identity to its opposite — and are then paid by the")
    print("  matrix they inverted. CA assumes Sign(Delta) is fixed; the detail-free")
    print("  version estimates it on a held-out half of the tasks, which does not")
    print("  help when a single agent is half the population.")


if __name__ == "__main__":
    simulate()
