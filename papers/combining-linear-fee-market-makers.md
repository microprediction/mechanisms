# Combining Linear-Fee Market Makers

### The infimal convolution of cost-function makers with proportional fees: spreads, sparse routing, and a derived order book

Peter Cotton · *Working draft v0.1* · 2026

---

## Abstract

A cost-function market maker charging a proportional fee $f\lvert s\rvert$ per
fill quotes, after Fenchel conjugation, exactly a bid-ask spread: the conjugate
of the fee-bearing cost is the no-fee conjugate evaluated at a soft-thresholded
price, zero on a band of half-width $f$ around the marginal price. Combining
$n$ such makers is the infimal convolution of their effective costs, and the
optimal split equalises fee-adjusted marginal prices at a single clearing
price, found by a monotone scalar root-find; any maker whose quote band
contains the clearing price receives exactly zero flow, so the split is sparse
in the manner of lasso solutions. The aggregate supply curve is a consolidated
limit order book whose levels are smooth curves. The identities are classical
and the near misses are catalogued; the assembly, fees inside the costs rather
than beside them, appears unoccupied, and it leaves each maker free to quote
its own fee with routing as the discipline.

---

## 1. Setup

A single risk settles in $[-1, 1]$. Maker $i \in \{1,\dots,n\}$ posts a
closed, finite, differentiable, strictly convex cost $C_i:\mathbb R \to
\mathbb R$, holds inventory $q_i$, and charges a proportional fee $f_i \ge 0$,
so a fill $s$ (signed) costs the trader

$$\tilde C_i(s) \;=\; C_i(q_i + s) - C_i(q_i) + f_i \lvert s\rvert .$$

Without the fee this is the standard cost-function maker
[@hanson2003combinatorial; @chen2007utility; @abernethy2013efficient], whose
charge telescopes over any path returning to the same state: round trips are
free, so uninformed flow pays nothing and adverse selection cannot be
recovered from volume. That no state-dependent cost can charge a round trip,
and that a path-dependent volume charge repairs it, is due to
@othman2012profitcharging; the linear charge $f_i\lvert s\rvert$ is their
volume levy in its simplest convex form, chosen here because it conjugates in
closed form. Write $g_i(s) = C_i(q_i+s) - C_i(q_i)$ and
$m_i = C_i'(q_i)$ for the marginal price.

## 2. A linear fee is a bid-ask spread

**Lemma 1 (fee–spread duality).** *Let
$T_f(x) = \operatorname{sign}(x)\max(\lvert x\rvert - f,\, 0)$ denote the
soft-threshold. Then*

$$\tilde C_i^*(p) \;=\; g_i^*\!\big( m_i + T_{f_i}(p - m_i) \big),$$

*and in particular $\tilde C_i^*(p) = 0$ if and only if
$\lvert p - m_i\rvert \le f_i$.*

**Proof.** The conjugate of a sum of closed proper convex functions with
overlapping relative interiors of domains is the infimal convolution of the
conjugates [@rockafellar1970convex, Thm. 16.4], and the conjugate of
$f\lvert\cdot\rvert$ is the indicator of $[-f, f]$. Hence
$\tilde C_i^*(p) = \min_{\lvert u\rvert \le f_i} g_i^*(p - u)$, the minimum of
the convex function $g_i^*$ over the interval $[p - f_i,\, p + f_i]$. Since
$g_i^* \ge 0$ with equality exactly at $m_i$, the minimum is attained at the
projection of $m_i$ onto the interval, which is $m_i + T_{f_i}(p - m_i)$.
$\blacksquare$

$g_i^*(p)$ is the profit extractable from maker $i$ at external price $p$ (in
producer-theory terms the profit function, with supply its derivative by
Hotelling's lemma [@hotelling1932edgeworth]). The lemma says the fee-bearing
maker quotes $\mathrm{ask}_i = m_i + f_i$ and $\mathrm{bid}_i = m_i - f_i$ and
is untouchable in between: a proportional fee is not like a spread, it is one,
the same duality by which a proportional transaction cost confines the pricing
functional to the bid-ask band [@jouini1995transaction].

## 3. Combining makers

**Theorem 2 (combination).** *Let
$\tilde C = \tilde C_1 \,\square\, \cdots \,\square\, \tilde C_n$ and define
each maker's supply*

$$s_i(p) \;=\;
\begin{cases}
(C_i')^{-1}(p - f_i) - q_i, & p \ge \mathrm{ask}_i,\\[2pt]
0, & \mathrm{bid}_i < p < \mathrm{ask}_i,\\[2pt]
(C_i')^{-1}(p + f_i) - q_i, & p \le \mathrm{bid}_i,
\end{cases}$$

*each non-decreasing in $p$. Fix a demand $\Delta$ for which a clearing price
$p^*$ with $\sum_i s_i(p^*) = \Delta$ exists. Then:*

*(i) $\tilde C^* = \sum_i \tilde C_i^*$, a sum of soft-thresholded profit
functions;*

*(ii) the split $s_i = s_i(p^*)$ attains $\tilde C(\Delta)$, and any optimal
split satisfies $C_i'(q_i + s_i) + f_i \operatorname{sign}(s_i) = p^*$ for
$s_i \ne 0$ and $\lvert p^* - m_i \rvert \le f_i$ for $s_i = 0$;*

*(iii) the split is sparse: every maker whose quote band strictly contains
$p^*$ trades exactly zero.*

**Proof.** (i) is the conjugate-sum identity applied to the convolution
[@rockafellar1970convex]. For (ii), the split is feasible by choice of $p^*$,
and $p^* \in \partial \tilde C_i(s_i(p^*))$ for every $i$: when
$s_i(p^*) \ne 0$ the subgradient is $C_i'(q_i+s_i) + f_i\operatorname{sign}(s_i)
= p^*$, and when $s_i(p^*) = 0$ it is the interval
$[m_i - f_i,\, m_i + f_i] \ni p^*$. A common multiplier certifying every
coordinate is exactly the optimality condition for
$\min\{\sum_i \tilde C_i(s_i) : \sum_i s_i = \Delta\}$. (iii) restates the
zero branch. $\blacksquare$

The computation is a scalar monotone root-find whatever $n$ is, and the fee
costs nothing beyond a horizontal shift of each supply curve. The
$\lvert s\rvert$ terms are an $\ell_1$ penalty, so sparsity arrives for the
same reason it does in the lasso, and for the same reason proportional
transaction costs produce no-trade regions and sparse portfolios
[@olivaresnadal2018robust]. A small trade routes entirely to the tightest
quote; a growing trade pushes that maker's fee-adjusted marginal price through
the next band and spills over, consuming makers in fee order.

**Corollary 3 (zero fees).** *With $f_i \equiv 0$ the convolution reduces to
the fee-free merge: conjugate regularisers add, and for a perspective family
$C_b(q) = b\,C_1(q/b)$ liquidity adds, $C_{b_1} \square C_{b_2} =
C_{b_1+b_2}$.* This is the merge law of the companion composition paper and
of @bhaskara2023general.

**Corollary 4 (the order book).** *The aggregate supply
$S(p) = \sum_i s_i(p)$ is non-decreasing, identically zero on
$\big(\max_i \mathrm{bid}_i,\ \min_i \mathrm{ask}_i\big)$ when that interval
is non-empty, flat wherever every maker's band covers $p$, and smooth and
strictly increasing wherever some maker is in the money.* Read as a market:
best bid and ask are the tightest quotes, depth at each price is the sum of
the active makers' closed-form supplies, and large orders walk the levels.
The aggregate of linear-fee makers is a consolidated limit order book, and in
producer-theory terms Theorem 2 is Marshall's horizontal summation of firm
supply curves [@marshall1890principles; @mascolell1995microeconomic] with the
reversibility of share production patched by the fee.

## 4. Discussion

Nothing requires the fees to be administered. Each maker may quote its own
$f_i$: a quote inside the aggregate spread earns nothing, a quote too tight is
picked off by informed flow, and the undercutting happens inside the same
minimisation that clears the trade, since the $\inf$ in the convolution is a
minimum over quotes. The surviving fee is the competitive adverse-selection
charge of classical microstructure [@glosten1985bidask; @biais2000competing],
reached here through routing rather than through a dealer game.

The ingredients are published. Path-dependent profit charging is
@othman2012profitcharging; aggregation of cost functions by infimal
convolution is @bhaskara2023general, with the geometric reading in
@angeris2024geometry; clearing-price routing across fee-bearing venues, with
zero flow to venues whose band contains the dual price, is
@angeris2022routing and @diamandis2023routing; the fee-induced no-trade
region is @milionis2024fees; the order book assembled from competing
liquidity suppliers is @glosten1994limit, with convergence of strategic
schedules to it in @biais2000competing; frictions inside inf-convolution risk
sharing appear in @ludkovski2009distorted. What was not found in any of these
is the present packaging: the fee folded into the cost so that one conjugate
identity (Lemma 1) yields the spread, the sparsity, the scalar clearing
computation, and the book, with per-maker fees as free competitive variables.
A fuller per-claim assessment, with the searches that ground it, is in the
repository note
[proportional-fees-and-the-order-book.md](https://github.com/microprediction/mechanisms/blob/main/research/proportional-fees-and-the-order-book.md);
a reference implementation with theorem tests is in
[`fee_routing.py`](https://github.com/microprediction/mechanisms/blob/main/mechanisms/fee_routing.py).

## References

::: {#refs}
:::
