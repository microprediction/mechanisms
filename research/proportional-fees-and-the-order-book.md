# Proportional fees, infimal convolution, and the order book

*Status: a worked generalization with a calibrated novelty assessment. The
conjugate identities are classical; the assembly is assessed in §8. Reference
implementation: [`fee_routing.py`](../mechanisms/fee_routing.py), theorem-tested
in [`test_fee_routing.py`](../tests/test_fee_routing.py).*

This note extends the merge operator of the composition paper (merging
cost-function makers is infimal convolution; Proposition 6 of
[composition-and-the-algebra-of-mechanisms.md](../papers/composition-and-the-algebra-of-mechanisms.md))
to makers that charge a proportional fee, and lets each maker set its own fee.
Three things follow. Under conjugation the fee is exactly a bid-ask spread: a
dead zone of half-width `f` around the maker's marginal price. Routing a trade
across fee-bearing makers remains a one-dimensional clearing-price computation,
and the optimal split is sparse, in the same way lasso solutions are sparse.
The aggregate of fee-bearing bounded makers is a consolidated limit order book
with smooth curves for levels: the book is derived from convex duality rather
than assumed as an institution. Read as producer theory (§5), the aggregate
is Marshall's industry supply curve, and the fee is the irreversibility that
widget production has physically and share production lacks.

## 1. The gap: a state-dependent cost cannot charge for a round trip

A cost-function market maker (Hanson 2003; Abernethy, Chen & Wortman Vaughan
2013) charges `C(q + s) - C(q)` for a fill `s` from inventory state `q`. The
charge telescopes: over any sequence of fills returning the state to where it
started, the maker collects exactly zero. This is the path-independence that
makes the maker arbitrage-free, and it has a converse cost: *no* charging rule
that depends only on the endpoints of the state path can price a round trip.
A trader who buys and immediately sells pays nothing, so the maker earns
nothing on uninformed flow and cannot recover adverse-selection losses from
volume.

This observation is not new. Othman & Sandholm (2012) state it verbatim for
cost-function makers and repair it by charging profit on *cumulative traded
volume*, a path-dependent state variable; the liquidity-sensitive vig of
Othman, Pennock, Sandholm & Reeves (2013) is by contrast state-dependent and
leaves round trips free. The linear special case of the volume charge is the
proportional fee `f |s|` per fill, chosen here because it conjugates in closed
form. The effective cost of a fill from state `q` is

    C~_q(s) = C(q + s) - C(q) + f |s|.

A round trip of size `s` now costs `2 f s` exactly (the `C` terms cancel, the
fee terms add), which is the first test in
[`test_fee_routing.py`](../tests/test_fee_routing.py). Who sets `f` is
deferred to §6.

## 2. Conjugation: the fee is a bid-ask spread

Write `g(s) = C(q + s) - C(q)` for the no-fee cost from the current state and
`m = C'(q)` for the marginal price. The conjugate `g*(p) = sup_s [p s - g(s)]`
is the maximum profit extractable by trading against the maker at external
price `p`; it is non-negative, convex, and zero exactly at `p = m`.

Two classical identities give the fee its meaning. The conjugate of a sum is
the infimal convolution of the conjugates, and the conjugate of `f |.|` is the
indicator of the band `[-f, f]`. Hence

    C~_q*(p) = (g + f|.|)*(p) = inf_{|u| <= f} g*(p - u) = min over [p-f, p+f] of g*.

Since `g*` is convex with minimum zero at `m`, the minimum over the interval
is attained at the projection of `m` onto `[p - f, p + f]`:

    C~_q*(p) = g*( m + T_f(p - m) ),      T_f(x) = sign(x) max(|x| - f, 0),

a *soft-threshold* of the price gap. In particular `C~_q*(p) = 0` if and only
if `|p - m| <= f`. Economically: the fee-bearing maker quotes `ask = m + f`
and `bid = m - f`, trades nothing while the external price sits inside the
band, and behaves like the no-fee maker with its price shifted by `f` once
outside. A proportional fee *is* a bid-ask spread, recovered by the algebra
rather than bolted on. (For a vector market with an `l1` fee `f ||s||_1`, the
band becomes the `l_infinity` ball: a per-coordinate spread.)

## 3. Routing: one clearing price, sparse fills

Let makers `i = 1..n` hold inventories `q_i` with costs `C_i`, liquidities of
their choosing, and fees `f_i` of their choosing. A trade of size `Delta`
routed optimally pays the infimal convolution of the effective costs,

    (C~_1 [] C~_2 [] ... [] C~_n)(Delta) = min { sum_i C~_i(s_i) : sum_i s_i = Delta },

which is the merge operator of the composition paper applied to the
fee-bearing costs. (Each `C~_i` is finite convex with recession slopes
`+-(1 + f_i)` for the log-cosh family, so the minimum is attained and the
convolution exact.) The first-order condition says every fill is chosen so
that fee-adjusted marginal prices equalise at a common clearing price `p*`:

    C_i'(q_i + s_i) + f_i sign(s_i) = p*   if s_i != 0,
    |p* - C_i'(q_i)| <= f_i                if s_i = 0.

Each maker's supply at price `p` is therefore

    s_i(p) = (C_i')^{-1}(p - f_i) - q_i    if p >= ask_i,
             (C_i')^{-1}(p + f_i) - q_i    if p <= bid_i,
             0                             inside the band,

monotone non-decreasing in `p`, and `p*` solves the scalar monotone equation
`sum_i s_i(p*) = Delta`. The fee costs nothing computationally: it is a
horizontal shift of each supply curve plus a flat segment. For the log-cosh
family `C(q) = b log cosh(q/b)` the supply curves invert in closed form,
`s(p) = b arctanh(p -+ f) - q`.

The `|s|` term is an `l1` penalty, so the optimal split is sparse for the same
reason lasso solutions are: any maker whose quote band contains `p*`
contributes exactly zero. A small trade routes entirely to the single cheapest
quoter; a larger trade pushes that maker's fee-adjusted marginal price out
through the next maker's band and spills over, consuming makers in fee order.
This is walking the levels of an order book, produced with no routing logic
written anywhere.

Two boundary checks tie back to the composition paper. With all fees zero the
convolution reduces to Proposition 6: for the log-cosh family, a perspective
family `C_b(q) = b C_1(q/b)`, liquidity adds, `C_{b1} [] C_{b2} = C_{b1+b2}`,
and the split is proportional to depth. Both are tested in
[`test_fee_routing.py`](../tests/test_fee_routing.py), along with optimality
of the routed split against random feasible alternatives.

## 4. The aggregate is a limit order book

Fix the makers' states and plot aggregate supply `S(p) = sum_i s_i(p)`. The
curve is flat at zero on the intersection of the quote bands (the inside
spread), flat wherever every maker is in a dead zone, and smooth and strictly
increasing wherever at least one maker is in the money. Read as a market: the
best bid and ask are the tightest quotes among the makers, depth at each price
is the sum of the active makers' closed-form supply curves, and large orders
walk through successive bands. This is a consolidated limit order book whose
"levels" are smooth curves, obtained as a theorem about infimal convolutions
of fee-bearing convex costs. The economic content, that competition among
liquidity suppliers assembles a book, is Glosten's (1994) inevitability
argument, and the primal unifications are recent: Milionis, Moallemi &
Roughgarden (2023a) aggregate AMMs and order books as summed demand curves,
and Diamandis et al. (2023) traverse Uniswap v3 as an aggregate of bounded
pieces in sorted price order, operationally a book. The dual derivation via
`(g + f|.|)* = g* [] indicator` is the part §8 finds unclaimed.

The bounded family matters here. Log-cosh makers have worst-case loss
`b log 2` for settlements in `[-1, 1]` (tested), so each maker's exposure is
capped by its own liquidity choice; fees price the flow, the bound prices the
tail.

## 5. The producer-theory reading

The machinery of §§2-4 has an older name. Read `C(q)` as a firm's cost of
supplying `q` units, shares instead of widgets, and the dictionary is
producer theory:

- price `= C'(q)` is marginal-cost pricing; the maker supplies along its
  marginal-cost curve, which is competitive firm behaviour;
- convexity is increasing marginal cost, the upward-sloping supply curve;
- `b` is the scale of the firm, and log-cosh saturation is a capacity
  constraint (the price pins at the boundary like a firm at full capacity);
- the conjugate `g*(p)` of §2 is the profit function, and Hotelling's lemma,
  supply `= dg*/dp`, is why the supply curves of §3 are derivatives of
  conjugates;
- the merge is Marshall's construction of industry supply: horizontal
  summation of firm supply curves. Adding supplies is adding conjugate
  derivatives, which is adding conjugates, which is the infimal convolution
  on the primal side. The industry cost function
  `C(Delta) = min { sum_i C_i(s_i) : sum_i s_i = Delta }` is a textbook
  object (Mas-Colell, Whinston & Green, ch. 5), and Proposition 6 of the
  composition paper is its conjugate form.

Two places where the market maker departs from the widget firm, and both are
load-bearing:

*Production is reversible.* A widget firm cannot un-produce a widget and
reclaim full cost; irreversibility is physical, so round trips were never
free and Marshall never needed a fee. A maker's production is un-doable at
zero cost by construction, which is the free-round-trip problem of §1. The
proportional fee is the missing irreversibility, added back by hand, and §2
shows the price of adding it is a spread.

*It is a multiproduct firm with economies of scope.* A vector-outcome maker
whose cost couples coordinates is a multiproduct cost function with cost
complementarities in the sense of Baumol, Panzar & Willig (1982): supplying
claims on one outcome changes the marginal cost of supplying claims on a
correlated one. Classical multiproduct theory never had to compute such a
cost function over very many products; the vector-fee and specialisation
questions of §9 are that computation.

## 6. Self-set fees: the fee is a price, not a parameter

If the fee must be estimated centrally, the design has an unresolved regress:
a device good enough to set `f` correctly would already solve the
adverse-selection estimation problem the fee exists to cover. The alternative
is to let each maker quote its own `f_i` and let routing do the rest. A maker
quoting too high sits inside the aggregate spread and receives no flow; one
quoting too low is picked off by informed flow until it widens or withdraws.
Undercutting happens inside the same operation that clears the trade, since
the `inf` in the convolution is a minimum over quotes. The surviving fee on a
node is the market's estimate of adverse selection there: the Glosten-Milgrom
(1985) zero-profit spread, whose emergence from competition among strategic
suppliers of convex schedules is Biais, Martimort & Rochet (2000), recovered
here as the fixed point of routing-disciplined fee quotes. Alternative
endogenizations exist: am-AMM auctions the right to set the pool fee (Adams
et al. 2025), and Baggiani, Herdegen & Sánchez-Betancourt (2026) model
dynamic fee competition between pools as a stochastic game.

Two caveats. The algebra loses one closed form: with heterogeneous self-set
fees the merged cost is no longer a named family member, though §3 shows the
computation stays a scalar root-find. And a node served by a single maker
carries a monopoly fee, not a competitive one; but excess fee revenue is
precisely the entry signal that attracts a second maker, and a monopoly fee on
a node nobody else will warehouse is payment for bearing its risk alone.

## 7. Implementation

[`fee_routing.py`](../mechanisms/fee_routing.py) provides `LogCoshMaker`
(bounded cost, closed-form supply, quoted bid/ask), `route` (the clearing-price
bisection), and `consolidated_book` (the aggregate supply curve). The tests
verify, in order: round trips cost exactly `2 f s` and are free at `f = 0`;
the routed split beats two thousand random feasible splits; active makers
equalise fee-adjusted marginals while inactive makers hold `p*` in their band;
zero-fee routing reproduces `C_{b1+b2}` and depth-proportional splits; small
trades route sparsely to the cheapest quote and large ones spill over; the
worst-case-loss bound holds over simulated fill sequences; the aggregate book
is monotone and exactly flat on the inside spread.

## 8. Prior art and novelty (calibrated)

Compiled from three web-verified searches (prediction-market cost functions,
CFMM fees and routing, transaction-cost duality and microstructure). Claim by
claim:

*Round trips require path dependence (§1).* Known. Othman & Sandholm (EC
2012) state that every cost-function maker is path independent, that "buying
a contract and then immediately selling it is without cost" for such makers,
and introduce volume-based (path-dependent) profit charging to fix it; path
independence as a design axiom goes back to Hanson (2003), Chen & Pennock
(2007) and Abernethy, Chen & Wortman Vaughan (2013). Dudík et al. (2014)
subsume the profit-charging construction in a volume-parameterized framework.
No novelty is claimed here.

*The fee is a spread, by conjugation (§2).* Essentially known; the packaging
is the contribution, if any. The convex-analysis identities are textbook
(Rockafellar 1970). The dual fact that proportional costs constrain the dual
price to the bid-ask band is classical in transaction-cost asset pricing
(Jouini & Kallal 1995; Kabanov's consistent price systems; Pennanen 2017).
The fee-induced no-trade band around an AMM's marginal price is established
in DeFi (Angeris et al. 2019; Angeris & Chitra 2020; Milionis, Moallemi &
Roughgarden 2024), and Diamandis et al. (2023) name the interval a bid-ask
spread. On the prediction-market side, Othman & Sandholm (2012) exhibit the
fee additively in quoted prices, and Abernethy, Chen & Wortman Vaughan (2013,
§3.2) state the mirror image: relaxing the feasible price region "is akin to
introducing a transaction cost", at the bundle level. Not found anywhere: the
one-line identity `(C + f|.|)* = C* ∘ soft-threshold` stated for per-security
cost-function makers with the threshold operator named. Verdict: known in
substance; the compact statement is expository, not mathematical, novelty.

*Routing is a scalar clearing-price problem with sparse fills (§3).*
Partially known, in two disjoint halves. The identity "aggregate market =
infimal convolution of the LPs' cost functions" is published: Bhaskara,
Frongillo, Lindgren & Papireddygari (2023) prove it with five equivalent
interpretations and recover Uniswap v2/v3 as special cases, and Angeris et
al. (2023) read Minkowski sums of trade sets the same way; their fees,
however, are side payments outside the cost function, and they prove natural
fee axioms incompatible for three or more securities (a scalar market
sidesteps the hypothesis). The clearing-price decomposition with zero flow to
makers whose fee band contains the dual price is established for CFMM routing
(Angeris, Evans, Chitra & Boyd 2022; Diamandis et al. 2023, who note "most
trades in the original problem will be 0" and solve two-asset subproblems by
monotone root-finds). Transaction cost as L1 regularization, with sparsity,
is published in portfolio choice (Olivares-Nadal & DeMiguel 2018; Brodie et
al. 2009), and frictions inside inf-convolution risk sharing appear in
Ludkovski & Young (2009). Not found: the combination — fees folded into each
cost so the inf-convolution itself carries the bands, one monotone root-find
on a single clearing price for the whole multi-maker problem, and the lasso
reading of which makers trade. Verdict: likely novel as an assembly, narrow
gap; every ingredient is published.

*The aggregate is an order book (§4).* Partially known. The economics is
Glosten (1994): the consolidated book assembled from competing risk-neutral
liquidity suppliers, including an aggregation theorem across exchanges.
Primal mathematical unifications exist: Agrawal et al. (2011) clear limit
orders and cost-function makers in one convex program; Milionis, Moallemi &
Roughgarden (2023a) make AMMs and books special cases of summed demand
curves; Uniswap v3 positions as approximate limit orders is whitepaper
folklore; Menz & Voß (2023) give a utility-book correspondence. Not found:
the book, with its fee-induced spread and dead zones, derived as the convex
dual of an aggregate fee-bearing cost. Verdict: the phenomenon is known, the
derivation is the candidate contribution; narrow gap.

*Self-set fees are disciplined by routing (§6).* The economics is classical:
dealer competition drives spreads to adverse-selection cost (Glosten &
Milgrom 1985; Glosten 1994; Biais, Martimort & Rochet 2000). Endogenous fee
setting exists in DeFi: fee-tier competition with routing (Lehar, Parlour &
Zoican 2023), dynamic fee competition as a stochastic game (Baggiani,
Herdegen & Sánchez-Betancourt 2026), auction-set fees (Adams et al. 2025),
equilibrium fee levels (Hasbrouck, Rivera & Saleh 2025), and a single maker
learning its spread as a bandit (Della Penna & Reid 2011). Not found: fee
competition expressed inside the inf-convolution algebra of cost-function
prediction markets, where the undercutting and the clearing are the same
minimization. Verdict: the mechanism-design framing in this setting is
unoccupied; the economic conclusion it reaches is a century of microstructure
and must be presented as such.

*The producer-theory reading (§5).* Textbook, deliberately. Marginal-cost
supply, the profit function as conjugate, Hotelling's lemma, and the industry
cost function as a minimization over firm allocations are standard producer
theory (Hotelling 1932; Mas-Colell, Whinston & Green 1995, ch. 5); the
multiproduct cost-complementarity language is Baumol, Panzar & Willig (1982);
the risk-measure reading of cost-function makers is Abernethy, Frongillo &
Kutty (2015). The two departure points, reversibility as the reason fees
exist here and not in Marshall, and the maker as a multiproduct firm at
scale, are interpretive rather than mathematical claims.

Overall: the mathematics is classical. The candidate contribution is one of
assembly: fees placed inside the costs rather than
beside them, so that a single conjugate calculation yields the spread, the
sparse routing, the consolidated book, and the venue for fee competition.
Positioning must cite Bhaskara et al. (2023), Diamandis et al. (2023),
Othman & Sandholm (2012), Glosten (1994) and Biais, Martimort & Rochet (2000)
as the five nearest neighbours.

## 9. Open questions

- *Equilibrium of the fee game.* §5 argues discipline, not equilibrium. Under
  what flow model does the quote game among makers have an equilibrium, and
  does it converge to the Glosten-Milgrom spread on thick nodes?
- *Beyond proportional.* Convex path-dependent charges `phi(|s|)` conjugate to
  general dead-zone shapes; which `phi` correspond to order-book shapes seen
  empirically (e.g. iceberg-like depth)?
- *Vector fees.* The `l1` fee gives an `l_infinity` band per maker. Makers
  specialising in subsets of outcomes correspond to fees infinite off a
  coordinate subspace; the routing problem becomes a monotone system rather
  than a scalar root-find.
- *Dynamic inventories.* Supply curves here are conditional on current `q_i`;
  a fill moves the state, so the book refreshes. What invariants of the
  refreshed book (resilience, spread dynamics) follow from the convex algebra?

## References

*Cost-function market makers, fees and vig.*
- Hanson, R. (2003). "Combinatorial Information Market Design." *Information
  Systems Frontiers* 5(1), 107–119.
- Chen, Y. & Pennock, D. M. (2007). "A Utility Framework for Bounded-Loss
  Market Makers." *UAI*. arXiv:1206.5252.
- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). "Efficient Market
  Making via Convex Optimization, and a Connection to Online Learning."
  *ACM TEAC* 1(2). arXiv:1011.1941.
- Othman, A., Pennock, D. M., Sandholm, T. & Reeves, D. M. (2013). "A Practical
  Liquidity-Sensitive Automated Market Maker." *ACM TEAC* 1(3).
- Othman, A. & Sandholm, T. (2012). "Profit-Charging Market Makers with Bounded
  Loss, Vanishing Bid/Ask Spreads, and Unlimited Market Depth." *EC*, 790–807.
- Othman, A. (2012). *Automated Market Making: Theory and Practice.* PhD
  thesis, CMU.
- Dudík, M., Lahaie, S., Pennock, D. M. & Rothschild, D. (2014). "A General
  Volume-Parameterized Market Making Framework." *EC*, 413–430.
- Della Penna, N. & Reid, M. D. (2011). "Bandit Market Makers."
  arXiv:1112.0076.
- Agrawal, S., Delage, E., Peters, M., Wang, Z. & Ye, Y. (2011). "A Unified
  Framework for Dynamic Prediction Market Design." *Operations Research*
  59(3), 550–568.

*Aggregation, infimal convolution, risk sharing.*
- Barrieu, P. & El Karoui, N. (2005). "Inf-Convolution of Risk Measures and
  Optimal Risk Transfer." *Finance and Stochastics* 9(2), 269–298.
- Jouini, E., Schachermayer, W. & Touzi, N. (2008). "Optimal Risk Sharing for
  Law Invariant Monetary Utility Functions." *Math. Finance* 18(2), 269–292.
- Ludkovski, M. & Young, V. R. (2009). "Optimal Risk Sharing under Distorted
  Probabilities." *Math. Financ. Econ.* 2(2), 87–105. arXiv:0809.3778.
- Bhaskara, A., Frongillo, R., Lindgren, E. & Papireddygari, M. (2023). "A
  General Theory of Liquidity Provisioning for Prediction Markets."
  arXiv:2311.08725.
- Frongillo, R., Papireddygari, M. & Waggoner, B. (2024). "An Axiomatic
  Characterization of CFMMs and Equivalence to Prediction Markets." *ITCS*.
  arXiv:2302.00196.
- Angeris, G., Chitra, T., Diamandis, T., Evans, A. & Kulkarni, K. (2023).
  "The Geometry of Constant Function Market Makers." arXiv:2308.08066.

*CFMM fees and routing.*
- Angeris, G., Kao, H.-T., Chiang, R., Noyes, C. & Chitra, T. (2019). "An
  Analysis of Uniswap Markets." *Cryptoeconomic Systems*. arXiv:1911.03380.
- Angeris, G. & Chitra, T. (2020). "Improved Price Oracles: Constant Function
  Market Makers." *ACM AFT*. arXiv:2003.10001.
- Angeris, G., Evans, A., Chitra, T. & Boyd, S. (2022). "Optimal Routing for
  Constant Function Market Makers." *EC*. arXiv:2204.05238.
- Diamandis, T., Resnick, M., Chitra, T. & Angeris, G. (2023). "An Efficient
  Algorithm for Optimal Routing Through Constant Function Market Makers."
  *FC*. arXiv:2302.04938.
- Milionis, J., Moallemi, C. C. & Roughgarden, T. (2023a).
  "Complexity-Approximation Trade-offs in Exchange Mechanisms: AMMs vs.
  LOBs." *FC*.
- Milionis, J., Moallemi, C. C. & Roughgarden, T. (2024). "Automated Market
  Making and Arbitrage Profits in the Presence of Fees." *FC*.
  arXiv:2305.14604.

*Order books from competing liquidity suppliers.*
- Glosten, L. R. & Milgrom, P. R. (1985). "Bid, Ask and Transaction Prices in
  a Specialist Market with Heterogeneously Informed Traders." *J. Financial
  Economics* 14(1), 71–100.
- Glosten, L. R. (1994). "Is the Electronic Open Limit Order Book
  Inevitable?" *J. Finance* 49(4), 1127–1161.
- Biais, B., Martimort, D. & Rochet, J.-C. (2000). "Competing Mechanisms in a
  Common Value Environment." *Econometrica* 68(4), 799–837.
- Klemperer, P. D. & Meyer, M. A. (1989). "Supply Function Equilibria in
  Oligopoly under Uncertainty." *Econometrica* 57(6), 1243–1277.
- Budish, E., Cramton, P., Kyle, A. S., Lee, J. & Malec, D. (2023). "Flow
  Trading." NBER w31098.
- Ramseyer, G., Goel, A. & Mazières, D. (2024). "Augmenting Batch Exchanges
  with Constant Function Market Makers." *EC*. arXiv:2210.04929.
- Menz, M. & Voß, J. (2023). "Aggregation of Financial Markets."
  arXiv:2309.04116.

*Fee setting and competition.*
- Lehar, A., Parlour, C. A. & Zoican, M. (2023). "Fragmentation and Optimal
  Liquidity Supply on Decentralized Exchanges." arXiv:2307.13772.
- Hasbrouck, J., Rivera, T. J. & Saleh, F. (2025). "The Need for Fees at a
  DEX: How Increases in Fees Can Increase DEX Trading Volume." *Management
  Science*.
- Baggiani, L., Herdegen, M. & Sánchez-Betancourt, L. (2026). "Competition
  between DEXs through Dynamic Fees." arXiv:2603.09669.
- Adams, A., Moallemi, C. C., Reynolds, S. & Robinson, D. (2025). "am-AMM: An
  Auction-Managed Automated Market Maker." *FC*. arXiv:2403.03367.
- Evans, A., Angeris, G. & Chitra, T. (2021). "Optimal Fees for Geometric
  Mean Market Makers." *FC WTSC*.

*Producer theory.*
- Marshall, A. (1890). *Principles of Economics.* Macmillan.
- Hotelling, H. (1932). "Edgeworth's Taxation Paradox and the Nature of
  Demand and Supply Functions." *J. Political Economy* 40(5), 577–616.
- Baumol, W. J., Panzar, J. C. & Willig, R. D. (1982). *Contestable Markets
  and the Theory of Industry Structure.* Harcourt Brace Jovanovich.
- Mas-Colell, A., Whinston, M. D. & Green, J. R. (1995). *Microeconomic
  Theory.* Oxford University Press. Ch. 5.
- Abernethy, J. D., Frongillo, R. M. & Kutty, S. (2015). "On Risk Measures,
  Market Making, and Exponential Families." *ACM SIGecom Exchanges* 13(2).

*Transaction costs, duality, sparsity.*
- Rockafellar, R. T. (1970). *Convex Analysis.* Princeton University Press.
- Magill, M. J. P. & Constantinides, G. M. (1976). "Portfolio Selection with
  Transactions Costs." *J. Econ. Theory* 13(2), 245–263.
- Davis, M. H. A. & Norman, A. R. (1990). "Portfolio Selection with
  Transaction Costs." *Math. Oper. Res.* 15(4), 676–713.
- Jouini, E. & Kallal, H. (1995). "Martingales and Arbitrage in Securities
  Markets with Transaction Costs." *J. Econ. Theory* 66(1), 178–197.
- Pennanen, T. (2017). "Convex Duality in Optimal Investment and Contingent
  Claim Valuation in Illiquid Markets." *Finance and Stochastics*.
  arXiv:1603.02867.
- Kühn, C. (2025). "The Fundamental Theorem of Asset Pricing with and without
  Transaction Costs" (survey). *Math. Finance*. arXiv:2307.00571.
- Brodie, J., Daubechies, I., De Mol, C., Giannone, D. & Loris, I. (2009).
  "Sparse and Stable Markowitz Portfolios." *PNAS* 106(30), 12267–12272.
- Olivares-Nadal, A. V. & DeMiguel, V. (2018). "A Robust Perspective on
  Transaction Costs in Portfolio Optimization." *Operations Research* 66(3).
- de Lataillade, J., Deremble, C., Lempérière, Y. & Bouchaud, J.-P. (2012).
  "Optimal Trading with Linear Costs." arXiv:1203.5957.
