# Perpetual Demand Lending Pools (PDLPs)

A reading guide to **Chitra, Diamandis, Sheng, Sterle & Yusubov (2025),
"Perpetual Demand Lending Pools"** (arXiv:2502.06028), the paper that formalises
the liquidity pools behind decentralised perpetuals exchanges, GMX's GLP,
Jupiter's JLP, Hyperliquid's HLP, dYdX's MegaVault, which together held ~\$2.5B
and earned ~\$700M+ in fees by the end of 2024.
The full PDF is in [`../assets/pdf-literature/lending-pools.pdf`](../assets/pdf-literature/lending-pools.pdf);
the reference implementation is [`../mechanisms/pdlp.py`](../mechanisms/pdlp.py)
and the narrated walk-through is [`../examples/pdlp_demo.py`](../examples/pdlp_demo.py).

## What a PDLP is

A PDLP is a liquidity pool that makes *under-collateralised, single-purpose*
loans: liquidity providers (LPs) deposit a basket of assets; traders borrow from
the pool to open **levered** positions on the associated perpetuals exchange, and
only for that purpose; LPs earn a fee proportional to position size; arbitrageurs
keep the pool at its target composition and keep the perpetual tethered to spot.
LPs, not the protocol, bear losses when the pool cannot rebalance or liquidate
fast enough. The empirical puzzle the paper explains: **why are PDLP LP positions
so much easier to delta-hedge than CFMM positions?**

## §2 The model

**Perpetuals exchange.** A contract is a triple $(L, S, p_0)$, cumulative long,
cumulative short, mark price. A trade is $(c, \delta, \eta, p_0)$, collateral,
size, leverage, entry price, subject to the **collateral condition** (eq 1)
$p_0\delta \le |\eta|c$ and the **liquidation condition**
$\operatorname{sign}(\eta)\delta(p_0-p)\ge c$; a minimal-collateral long liquidates
at $p_0(1-1/\eta)$. The **linear funding rate** (eq 2) is
$\gamma_L = \kappa(L/S - p/p_0)$: when positive, shorts pay longs.
→ `linear_funding_rate`, `collateral_ok`, `liquidation_price`

**PDLP.** A pool is $(R, w^\star, f, \{c_i\})$, reserves, target weights, lending
fee, outstanding loans, with $\sum_i c_i \le p^\top R$ and *available* (un-lent)
reserves $R^A = R - \sum_i c_i$. The pool weight is $w(p,R)=(p\odot R)/(p^\top R)$.
→ `PDLP`, `weights`

## §2.3 Single-period arbitrage

A price move from $p_0$ to $p$ opens two arbitrages whose profitability bounds the
fee from both sides:

- **Funding-rate arbitrage.** The largest funding-capturing long is
  $\ell = L_0(p/p_0-1)$ (eq 3); it is profitable while the fee
  $f \le \tfrac{\kappa}{L_0}(1-p_0/p)$, i.e. $f \le \kappa(1-B^{-1})/L_0$ under a
  price bound $p/p_0 \le B$ (eq 4).
  → `funding_arb_size`, `funding_fee_upper_bound`
- **PDLP price-impact arbitrage.** With a concave forward exchange function $G$
  (the amount of asset the pool sells for $x$ numeraire), the best trade solves
  $G'(x^\star)=1/p$ and the net pool value change is $x^\star - pG(x^\star)$,
  exactly CFMM loss-versus-rebalancing. For the Uniswap-v2 instance
  $G(x)=R_2x/(R_1+x)$ the LP loss is $R_1 + pR_2 - 2\sqrt{pR_1R_2}$, and LPs profit
  when $f \ge x^\star/L_0$ (eq 5).
  → `UniswapV2Forward`, `pdlp_arb_optimal_x`, `fee_lower_bound`

Together these give a **sustainable fee** $f = \Theta(1/L_0)$: it must fall as open
interest grows. Dynamic fees that track open interest are more likely to sustain
the LP↔trader equilibrium.

## §3 The Target Weight Mechanism (TWM)

Unlike a CFMM, a PDLP lets LPs deposit/withdraw *any subset* of assets. The TWM
keeps the pool near $w^\star$ by paying a **discount** $F$ to LPs whose deposits
move the weights toward target, solving (approximately) the program
$\min_\Delta \lVert w(p,R+\Delta)-w^\star\rVert$ s.t. $\Delta \ge -R^A$ (eq 6).
GMX's GLP discount function (eq 11) is an explicit, PID-controller-like instance
that rewards rebalancing trades and penalises imbalancing ones. A discount $F>0$
dilutes existing LPs by $1/(1+F)$; the post-update share value is
$V_{\text{new}} = p^\top R/(1+F) + f\,p^\top\ell$ (§3.5).
→ `target_weight_trade`, `gmx_glp_discount`, `portfolio_dilution_value`

**Why PDLPs hedge well.** Claim 3.4 shows the TWM *bounds* an LP's delta exposure:
the new portfolio's delta is at most $(\tfrac12-f)$ times the raw holding's, and
the bound tightens as the fee rises. This is the paper's explanation for the
proliferation of live, delta-hedged PDLP strategies (and why JLP-style assets make
good collateral elsewhere in DeFi).

## §4 Hedged PDLPs

Under a mean-variance objective (eq 12) the optimal delta hedge, ignoring
transaction costs, is $\pi_{\text{new}} = (f/\gamma)\Sigma^{-1}\ell - \Delta$
(eq 13). Claim 4.1 gives two sufficient conditions for the hedged pool's Sharpe
ratio to beat the unhedged pool's: the fee revenue must dominate the delta
($f\,\mathbb E[p^\top\ell] \ge \gamma\lambda_{\max}\,\mathbb E[p^\top\Delta]$) and
the hedge variance must be at most $4\times$ the pool variance.
→ `delta_hedge`, `sharpe_improves`

## Connection to the rest of this repo

PDLPs sit at the intersection of three mechanisms implemented here: the
**perpetual-futures** funding tether (`perp`), the **constant-function market
maker** whose loss-versus-rebalancing the PDLP price-impact arbitrage reproduces
(`amm`), and a **portfolio target-weight** controller. The lending-pools paper is
also cited in the [schur](https://github.com/microprediction/schur) bibliography,
its Appendix-B-style single-vs-multiple-pool question is a Schur-complement
decision on the pool covariance.
