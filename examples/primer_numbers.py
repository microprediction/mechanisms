"""Reproduce every number quoted in the running example of the primer
papers/composition-and-the-algebra-of-mechanisms.md.

The example: outcomes A, B, C; belief q = (0.5, 0.3, 0.2); false report
p = (0.7, 0.2, 0.1); LMSR liquidity b = 10. Each block prints the figures the
paper quotes and asserts the identity the paper states, so a change to the
implementation that broke a claim would fail here.

Run:  python examples/primer_numbers.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from mechanisms.aggregation import linear_opinion_pool, logarithmic_opinion_pool
from mechanisms.kelly import wealth_weighted_update
from mechanisms.lmsr import LMSR
from mechanisms.scoring_rules import brier_score, expected_score, log_score

q = np.array([0.5, 0.3, 0.2])
p = np.array([0.7, 0.2, 0.1])
b = 10.0


def section(title):
    print(f"\n== {title}")


# --- Section 2 and 3: the two gaps are the two Bregman divergences -----------
section("Scoring rules (scores are implemented in loss form; reward = -loss)")
log_truth = -expected_score(log_score, q, q)
log_false = -expected_score(log_score, q, p)
kl = float(np.sum(q * np.log(q / p)))
print(f"log score: truthful {log_truth:.3f}, false {log_false:.3f}, gap {log_truth - log_false:.3f}, KL(q||p) {kl:.3f}")
assert abs((log_truth - log_false) - kl) < 1e-12

brier_truth = -expected_score(brier_score, q, q)
brier_false = -expected_score(brier_score, q, p)
# The implementation's Brier loss is ||e_i - p||^2 = 1 - (2 p_i - ||p||^2),
# so reward form 2 p_i - ||p||^2 is 1 minus the loss.
brier_truth, brier_false = 1 + brier_truth, 1 + brier_false
sq = float(np.sum((q - p) ** 2))
print(f"Brier payments for p on A,B,C: {np.round(2 * p - np.sum(p**2), 2)}")
print(f"Brier: truthful {brier_truth:.2f}, false {brier_false:.2f}, gap {brier_truth - brier_false:.2f}, ||q-p||^2 {sq:.2f}")
assert abs((brier_truth - brier_false) - sq) < 1e-12

# --- Section 4: the LMSR with b = 10 -----------------------------------------
section("LMSR, b = 10")
m = LMSR(3, b=b)
print(f"initial prices {np.round(m.prices(), 3)}")
c1 = m.buy(0, 10.0)
p1 = m.prices().copy()
print(f"buy 10 of A: cost {c1:.2f}, prices {np.round(p1, 3)}")
c2 = m.buy(1, 5.0)
print(f"then 5 of B: cost {c2:.2f}, prices {np.round(m.prices(), 3)}")
m0 = LMSR(3, b=b)
round_trip = m0.buy(0, 10.0) + m0.buy(0, -10.0)
print(f"buy 10 of A then sell them back: net {round_trip:.2e}")
assert abs(round_trip) < 1e-9
loss_if_A = 10.0 - c1
print(f"loss if A occurs after the first trade {loss_if_A:.2f}; bound b log 3 = {m.max_loss:.2f}")
assert loss_if_A <= m.max_loss
seq = b * (np.log(p1[0]) - np.log(1 / 3))
print(f"sequential log-score reading b(log p'_A - log p_A) = {seq:.2f}")
assert abs(seq - loss_if_A) < 1e-9
# q_i = b log p_i + C(q)
mm = LMSR(3, b=b); mm.buy(0, 10.0)
assert np.allclose(mm.q, b * np.log(mm.prices()) + mm.cost())

# --- Section 4: constant product ---------------------------------------------
section("Constant product, k = 100")
V = lambda pr, k=100.0: 2 * np.sqrt(k * pr * (1 - pr))
print(f"value at even prices {V(0.5):.0f}, at (0.9, 0.1) {V(0.9):.0f}, worst-case loss sqrt(k) = {np.sqrt(100):.0f}")

# --- Section 5: the two pools ------------------------------------------------
section("Opinion pools")
q2 = np.array([0.2, 0.3, 0.5])
print(f"linear {np.round(linear_opinion_pool([q, q2]), 3)}, logarithmic {np.round(logarithmic_opinion_pool([q, q2]), 3)}")
q3 = np.array([0.0, 0.5, 0.5])
print(f"with a veto: linear {np.round(linear_opinion_pool([q, q3]), 3)}, logarithmic {np.round(logarithmic_opinion_pool([q, q3]), 3)}")
assert logarithmic_opinion_pool([q, q3])[0] < 1e-6  # the implementation floors zero mass at 1e-15 before the log

# --- Section 6: merging makers -----------------------------------------------
section("Merging LMSR(10) and LMSR(5) for 10 shares of A")
def cost(bb, t):
    mk = LMSR(3, b=bb)
    return mk.buy(0, t)
split = minimize_scalar(lambda t: cost(10, t) + cost(5, 10 - t), bounds=(0, 10), method="bounded")
merged = cost(15, 10.0)
print(f"optimal split {split.x:.2f} / {10 - split.x:.2f}, cost {split.fun:.2f}; single LMSR(15) cost {merged:.2f}")
assert abs(split.fun - merged) < 1e-6 and abs(split.x - 20 / 3) < 1e-4

# --- Section 8: clamps -------------------------------------------------------
section("Freeze entry: Kelly bettors and Bayesian model averaging")
w = wealth_weighted_update([0.5, 0.5], [q[0], q2[0]])
print(f"wealths after A occurs {np.round(w, 3)}; posterior under an equal prior {np.round([0.5 / 0.7, 0.2 / 0.7], 3)}")
assert np.allclose(w, [0.5 / 0.7, 0.2 / 0.7])

section("Freeze capital: least squares as a merge of quadratic makers")
obs, prec = np.array([1.0, 2.0]), np.array([1.0, 4.0])
est = float(np.sum(prec * obs) / np.sum(prec))
print(f"estimate {est:.1f}, precision {np.sum(prec):.0f}")
assert est == 1.8

section("Probability integral transform, N(0,1) reported, N(0,4) true")
tail = norm.cdf(norm.ppf(0.025) / 2)
middle = norm.cdf(norm.ppf(0.75) / 2) - norm.cdf(norm.ppf(0.25) / 2)
print(f"P(U < 0.025) = {tail:.3f}; P(0.25 < U < 0.75) = {middle:.3f}")

print("\nall identities hold")
