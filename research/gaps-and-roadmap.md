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

These are well-defined mechanisms that belong in a reference library and are
being added:

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
3. **Kelly-optimal sizing** (Gap 5). The log-optimal bet size; abstracts the
   exact wealth dynamics already driving [`sim_pipeline.py`](../examples/sim_pipeline.py).
   *Planned.*
4. **Combinatorial / conditional markets** (Gap 4). LMSR over a combinatorial
   outcome space and conditional ("if A then B") markets — the cost-function
   machinery already in [`cmm.py`](../mechanisms/cmm.py) extends to this.
   *Planned.*
5. **Decision markets / futarchy as a mechanism** (Gap 6). The *mechanism* layer
   (conditional markets on a decision variable) is in scope even though the
   execution/governance plumbing is not. *Planned.*
6. **Hybrid CLOB + AMM matching** (Gap 4). The repo has the
   [continuous double auction](../mechanisms/cda.py), the
   [frequent batch auction](../mechanisms/fba.py), and
   [CFMMs](../mechanisms/amm.py) separately; a matcher that fills against resting
   limit orders first and an AMM backstop second is a small, clean addition.
   *Planned.*

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

The highest-value, lowest-risk additions — because they are mathematically
clean, unit-testable, and extend modules already here — are **(1) local scoring
rules** and **(2) CA/EA peer prediction**, both delivered alongside this note.
Kelly sizing, combinatorial/conditional markets, and a hybrid matcher follow.

## References

Hyvärinen (2005); Parry, Dawid & Lauritzen (2012); Dasgupta & Ghosh (2013);
Shnayder, Agarwal, Frongillo & Parkes (2016); Prelec (2004); Hanson (2007);
Cotton (2022). Keys in [`bibliography.bib`](bibliography.bib).
