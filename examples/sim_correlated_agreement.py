"""Simulation: Correlated Agreement elicits truth with no ground truth.

Output agreement (reward = match a peer) has a fatal degenerate equilibrium:
everyone can collude on a constant report and always "agree", revealing nothing.
Correlated Agreement (Dasgupta-Ghosh 2013; Shnayder et al. 2016) fixes this by
rewarding agreement on the SAME task only beyond the chance agreement expected
from UNRELATED tasks. A constant report agrees everywhere equally, so it nets
zero. No prediction report is needed (unlike BTS) — only signals on a shared
pool of tasks.

We also show the "Enforced Agreement" property: in the binary case truthful
reporting is *stochastically dominant* — its whole score distribution sits to the
right of a deviator's, so honesty wins for any risk-averse, monotone utility.

Run:  python examples/sim_correlated_agreement.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _viz import section, labeled_bars, sparkline

from mechanisms.peer_prediction import output_agreement, correlated_agreement


def simulate_signals(n_agents, n_tasks, q, rng):
    """Each task has a hidden binary state; an agent observes it correctly w.p. q."""
    state = rng.integers(0, 2, n_tasks)
    flip = rng.random((n_agents, n_tasks)) > q
    return state[None, :] ^ flip.astype(int)


def simulate(n_agents=8, n_tasks=400, q=0.82, seed=0):
    rng = np.random.default_rng(seed)
    sig = simulate_signals(n_agents, n_tasks, q, rng)

    # Three strategies for agent 0, against a truthful crowd (agents 1..n-1).
    truthful = sig.copy()
    constant = sig.copy(); constant[0, :] = 0          # the output-agreement exploit
    flipper = sig.copy(); flipper[0, :] = 1 - sig[0, :]  # adversarial anti-report

    section("Output agreement rewards the degenerate constant report")
    # mean agreement of agent 0 with the rest, per strategy
    def mean_oa(reports):
        return float(np.mean([output_agreement(reports[0, t], reports[j, t])
                              for j in range(1, n_agents) for t in range(n_tasks)]))
    all_constant = np.zeros_like(sig)
    labeled_bars([("truthful", mean_oa(truthful)),
                  ("constant 0", mean_oa(constant)),
                  ("all collude on 0", mean_oa(all_constant))], vmax=1.0, fmt="{:.3f}")
    print("  a constant reporter still collects substantial reward for revealing")
    print("  nothing — and if everyone colludes on a constant, agreement is a")
    print("  perfect 1.000. Output agreement cannot punish the empty strategy.")

    section("Correlated Agreement defeats it (agent 0's score by strategy)")
    s_truth = correlated_agreement(truthful, n_signals=2)[0]
    s_const = correlated_agreement(constant, n_signals=2)[0]
    s_flip = correlated_agreement(flipper, n_signals=2)[0]
    labeled_bars([("truthful", s_truth - min(s_truth, s_const, s_flip)),
                  ("constant 0", s_const - min(s_truth, s_const, s_flip)),
                  ("flipper", s_flip - min(s_truth, s_const, s_flip))],
                 vmax=(s_truth - min(s_truth, s_const, s_flip)) or 1.0, fmt="{:.3f}")
    print(f"  raw CA scores:  truthful {s_truth:+.3f}   constant {s_const:+.3f}   flipper {s_flip:+.3f}")
    print("  honest reporting pays the most; the constant exploit nets ~0.")

    section("Enforced Agreement: truthful reporting stochastically dominates")
    R = 300
    truth_scores, dev_scores = [], []
    for r in range(R):
        rg = np.random.default_rng(1000 + r)
        s = simulate_signals(n_agents, 120, q, rg)
        truth_scores.append(correlated_agreement(s, n_signals=2)[0])
        dev = s.copy(); dev[0, :] = 0
        dev_scores.append(correlated_agreement(dev, n_signals=2)[0])
    truth_scores = np.sort(truth_scores); dev_scores = np.sort(dev_scores)
    print(f"  truthful score quantiles  10/50/90%: "
          f"{np.quantile(truth_scores,0.1):+.3f} / {np.quantile(truth_scores,0.5):+.3f} / {np.quantile(truth_scores,0.9):+.3f}")
    print(f"  deviator score quantiles  10/50/90%: "
          f"{np.quantile(dev_scores,0.1):+.3f} / {np.quantile(dev_scores,0.5):+.3f} / {np.quantile(dev_scores,0.9):+.3f}")
    # fraction of the deviator distribution that the truthful 10th percentile beats
    dominance = float(np.mean(truth_scores[int(0.1 * R)] >= dev_scores))
    print(f"  truthful 10th-percentile beats {100*dominance:.0f}% of deviator outcomes "
          f"— the truthful CDF lies to the right (stochastic dominance).")


if __name__ == "__main__":
    simulate()
