"""Independent exact oracles for the numerical claims in the paper.

The point of these is *oracle independence*. Every function here computes a
quantity the library also computes, from the definition, by an algorithm that
shares no code and no strategy with the implementation under test. Checking a
discretisation against itself, or against a refinement of itself, cannot detect
an error in what the discretisation is discretising; the errors that survived
the previous test suite were of exactly that kind.

They are exact for the *piecewise-linear interpolant* of a sampled grid, which
is the object the library's grid routines are really operating on. Where the
implementation minimises over sampled points and the oracle minimises over the
continuum, the two disagree by a known discretisation error, and the tests say
which they expect.
"""

from .generators import (
    binary_support,
    controlled_violation_cost,
    multi_well_cost,
    near_singular_correlation,
    non_attained_threshold_cost,
    tied_makers,
)
from .finite_lp import (
    ArbitrageCertificate,
    arbitrage_depth_lp,
    arbitrage_lp,
)
from .piecewise_linear import (
    exact_chord_bounds,
    exact_lower_convex_envelope,
    exact_max_sure_profit,
    exact_moreau_envelope,
)

__all__ = [
    "ArbitrageCertificate",
    "binary_support",
    "controlled_violation_cost",
    "multi_well_cost",
    "near_singular_correlation",
    "non_attained_threshold_cost",
    "tied_makers",
    "arbitrage_depth_lp",
    "arbitrage_lp",
    "exact_chord_bounds",
    "exact_lower_convex_envelope",
    "exact_max_sure_profit",
    "exact_moreau_envelope",
]
