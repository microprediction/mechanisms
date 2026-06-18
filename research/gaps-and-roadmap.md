# Gaps and roadmap

This note responds to an external "gap analysis" of the **microprediction
ecosystem** and maps its claims onto *this repository*. The distinction matters:
the essay audits the live microprediction *platform* and the surrounding family
of packages (skaters, precise, humpday, schur, thurstone, the ε-window lottery,
the "French mechanism"). This repo is narrower — a set of faithful **reference
implementations of market mechanisms** — so several mechanisms the essay calls
"missing" are in fact already here, while others are genuine, well-scoped
additions.

## Already implemented here

| Essay's "gap" | Status in this repo |
|---------------|---------------------|
| **LMSR AMM, bounded `b log n` loss** (Gap 3) | [`lmsr.py`](../mechanisms/lmsr.py), [`cmm.py`](../mechanisms/cmm.py). The catalog page already argues the essay's point — constant-product is wrong for bounded probabilities — and [`pm_amm.py`](../mechanisms/pm_amm.py) is the Gaussian-score answer. |
| **Peer prediction** (Gap 1) | [`peer_prediction.py`](../mechanisms/peer_prediction.py): Bayesian Truth Serum + output agreement. *CA / EA variants were missing — see below.* |
| **Strictly proper scoring incl. CRPS, energy** | [`scoring_rules.py`](../mechanisms/scoring_rules.py). |
| **The "French mechanism"** (collaborative parimutuel paying KL-contribution to the consensus) | The wealth-weighted **nearest-the-pin** pool plus log-opinion aggregation, written up in [composition-and-the-algebra-of-mechanisms.md](composition-and-the-algebra-of-mechanisms.md). |

So the essay's headline financial-primitive claim — "replace constant-product
with LMSR" — is already the repository's position.

## Genuine gaps: clean reference-implementation candidates

These are well-defined additions that belong in a reference library and are
being added — all mechanisms except (3), which is an agent-side *sizing
primitive* the mechanisms compose with:

1. **Local / m-local proper scoring rules** (Gap 2). Proper rules that depend on
   the quoted density only through its value *and a finite number of
   derivatives* at the outcome, and so can be evaluated **without the
   normalizing constant**. The canonical instance is the **Hyvärinen score**
   (the `m=2` local rule), whose population minimiser is score matching and
   whose divergence is the Fisher divergence (Hyvärinen 2005; Parry, Dawid &
   Lauritzen 2012). *Implemented:* [`local_scoring.py`](../mechanisms/local_scoring.py).
2. **Correlated Agreement & Enforced Agreement peer prediction** (Gap 1). CA
   (Dasgupta–Ghosh 2013; Shnayder, Agarwal, Frongillo & Parkes 2016) is an
   *informed-truthful*, multi-task mechanism that needs no elicited peer
   distribution; EA / **stochastically-dominant** truthfulness (Schoenebeck et
   al.) hardens this against non-linear (risk-averse) utilities. *Implemented:*
   added to [`peer_prediction.py`](../mechanisms/peer_prediction.py).
3. **Kelly-optimal sizing** (Gap 5) — a *sizing primitive*, not a mechanism.
   Kelly is an agent's wealth-growth-optimal bet fraction; a mechanism is a
   *market* rule (aggregate / price / allocate). It earns its place here only in
   composition: it is the **agent-side dual of the log score**. A log-wealth
   maximiser betting against prices $\pi$ optimally bets its beliefs $p$, with
   growth rate $\sum_i p_i\log(p_i/\pi_i) = \mathrm{KL}(p\Vert\pi)$ — exactly the
   regret of the log score (the Bregman divergence of negentropy), and exactly
   the multiplicative wealth update $w_i \mathrel{*}= p_i(y)/\pi$ that already
   makes the wealth-weighted pool in
   [`sim_pipeline.py`](../examples/sim_pipeline.py) self-correcting. So the right
   addition is a small `kelly.py` primitive the mechanisms *compose with*, not a
   new market. (A protocol-level auto-sizer that allocates stakes from broadcast
   beliefs *would* be a mechanism — stake allocation with a Kelly objective.)
   *Implemented:* [`kelly.py`](../mechanisms/kelly.py) — binary and
   complete-market sizing, the `KL(p‖π)` growth rate, and the wealth-weighted
   ensemble update (the multiplicative-weights engine of a self-correcting pool).
4. **Combinatorial / conditional markets** (Gap 4). LMSR over a combinatorial
   outcome space and conditional ("if A then B") markets — the cost-function
   machinery already in [`cmm.py`](../mechanisms/cmm.py) extends to this.
   *Implemented:* [`combinatorial.py`](../mechanisms/combinatorial.py).
5. **Decision markets as a mechanism** (Gap 6). The *mechanism* layer —
   conditional markets on a decision variable, read as ensembling
   action-conditional forecasts and selecting among them — is in scope; the
   governance/futarchy framing is social theory and is not.
   *Implemented:* [`decision_market.py`](../mechanisms/decision_market.py) — one
   conditional market per action, the per-action value map, and argmax / softmax /
   epsilon-greedy decision rules, with the incentive result that a full-support
   stochastic rule keeps every conditional market truthful where deterministic
   argmax does not (Othman & Sandholm 2010; Chen, Kash, Ruberry & Shnayder 2011).
6. **Hybrid CLOB + AMM matching** (Gap 4). The repo has the
   [continuous double auction](../mechanisms/cda.py), the
   [frequent batch auction](../mechanisms/fba.py), and
   [CFMMs](../mechanisms/amm.py) separately; a matcher that fills against resting
   limit orders first and an AMM backstop second.
   *Implemented:* [`hybrid_market.py`](../mechanisms/hybrid_market.py).

## Out of scope for a mechanism library

The essay's remaining items are platform/infrastructure rather than market
primitives, and do not belong in a reference library of mechanisms:

- **Info-Finance / DeFi composability, decentralized oracles (Chainlink CCIP/CRE),
  on-chain settlement, yield-bearing collateral, TVL.** Integration concerns.
- **Sybil-resistance / insider-trading throttles.** Important operationally, but
  detection/policy rather than a clean elicitation or pricing mechanism.
- **"Morphological-matching" algorithmic routing** (the plant–hummingbird
  analogy). This is *orchestration* — matching agents to streams — not a market
  mechanism. It could make an interesting standalone note.

## m-local scoring: a note on existence

The essay states m-local proper scoring rules "exist for all even values of m
(though not for odd m)". That is the Parry–Dawid–Lauritzen (2012) result: a
genuinely local (derivative-based) proper scoring rule of order m on the real
line exists iff m is even. The `m=2` case is the Hyvärinen score implemented
here; higher even orders trade more derivative information for more flexibility,
all while remaining free of the normalizing constant.

## Priority

Every gap above is now closed: **local scoring rules**, **CA/EA peer
prediction**, **combinatorial/conditional markets**, a **hybrid CLOB+AMM
matcher**, a **Kelly sizing** primitive (the agent-side dual of the log score,
with the wealth-weighted ensemble update), and a **decision market** (conditional
markets on a decision variable as an ensembling-and-selection meta-mechanism) —
all mathematically clean, unit-tested, and extending modules already here. The
governance/futarchy reading of decision markets is deliberately left out as
social theory rather than a market primitive.

## References

Hyvärinen (2005); Parry, Dawid & Lauritzen (2012); Dasgupta & Ghosh (2013);
Shnayder, Agarwal, Frongillo & Parkes (2016); Prelec (2004); Hanson (2007);
Cotton (2022). Keys in [`bibliography.bib`](bibliography.bib).
