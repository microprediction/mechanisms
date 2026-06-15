"""Perpetual Demand Lending Pools — a full walk-through of arXiv:2502.06028.

Demonstrates every mechanism in Chitra, Diamandis, Sheng, Sterle & Yusubov
(2025), "Perpetual Demand Lending Pools", using the paper's own worked examples:

  §2.1  the perpetuals exchange, collateral & liquidation, the linear funding rate
  §2.3  funding-rate arbitrage and PDLP price-impact arbitrage (a sustainable fee)
  §3    the target-weight mechanism and GMX's GLP discount function
  §3.5  portfolio dilution after a rebalance
  §4    the mean-variance delta hedge and the Sharpe-improvement conditions

Run:  python examples/pdlp_demo.py
"""

import numpy as np

from mechanisms import pdlp


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    # ---------------------------------------------------------------- #
    section("§2.1  Perpetuals exchange — a levered ETH long")
    # Paper's example, self-consistent reading: notional 8000, eta=4, p0=2000.
    p0, eta = 2000.0, 4.0
    c = 2000.0                      # collateral
    delta = eta * c / p0            # position size so eq (1) is saturated -> 4 ETH
    print(f"collateral c        = ${c:,.0f}")
    print(f"leverage eta        = {eta:g}x")
    print(f"mark price p0       = ${p0:,.0f}")
    print(f"position size delta = {delta:g} ETH  (notional ${delta*p0:,.0f})")
    print(f"collateral condition p0*delta <= |eta|*c : "
          f"{pdlp.collateral_ok(c, delta, eta, p0)}")
    liq = pdlp.liquidation_price(delta, eta, p0, c)
    print(f"liquidation price   = ${liq:,.0f}   (= p0(1 - 1/eta) = ${p0*(1-1/eta):,.0f})")

    section("§2.1  Linear funding rate (eq 2)")
    kappa = 0.01
    L, S = 1000.0, 250.0
    g = pdlp.linear_funding_rate(L, S, p=p0, p0=p0, kappa=kappa)
    print(f"L={L:g}, S={S:g}, p=p0=${p0:,.0f}, kappa={kappa}")
    print(f"gamma_L = kappa(L/S - p/p0) = {g:.4f}  (= 3*kappa = {3*kappa})  -> longs pay shorts")

    # ---------------------------------------------------------------- #
    section("§2.3.1  Funding-rate arbitrage — size and a profitable fee")
    L0 = 1000.0
    p = 2200.0                      # price rises 10%
    ell = pdlp.funding_arb_size(L0, p, p0)
    f_max = pdlp.funding_fee_upper_bound(kappa, L0, p, p0)
    print(f"balanced open interest L0={L0:g}; price moves ${p0:,.0f} -> ${p:,.0f}")
    print(f"arb long size ell = L0(p/p0 - 1) = {ell:.2f}")
    print(f"fee upper bound for profitable funding arb: f <= {f_max:.6f}")
    print(f"with price bound B=1.10: f <= {pdlp.funding_fee_upper_bound(kappa, L0, p, p0, B=1.10):.6f}")

    # ---------------------------------------------------------------- #
    section("§2.3.2-3  PDLP price-impact arbitrage (Uniswap-v2 forward function)")
    # Pool quotes the old price p0 via R2/R1 = 1/p0.
    R1 = 1_000_000.0                # numeraire reserve
    R2 = R1 / p0                    # risky reserve so initial price = p0
    fwd = pdlp.UniswapV2Forward(R1, R2)
    print(f"forward G(x)=R2 x/(R1+x), R1={R1:,.0f}, R2={R2:,.2f}")
    print(f"pool initially quotes ${fwd.initial_price:,.2f}  (= p0)")
    x_star = fwd.optimal_x(p)
    x_star_num = pdlp.pdlp_arb_optimal_x(fwd, p)   # generic bisection agrees
    print(f"optimal arb input x* = {x_star:,.2f}  (bisection check {x_star_num:,.2f})")
    print(f"LP rebalancing loss  = R1 + p R2 - 2 sqrt(p R1 R2) = {fwd.lp_loss(p):,.2f}")
    f_min = pdlp.fee_lower_bound(x_star, L0=R1)
    print(f"fee lower bound for LP profit: f >= x*/L0 = {f_min:.6f}")
    print("=> a sustainable fee lives between this lower bound and the funding-arb")
    print("   upper bound; the paper shows f = Theta(1/L0).")

    # ---------------------------------------------------------------- #
    section("§3  Target Weight Mechanism — pool weights & GMX GLP discount")
    prices = np.array([1.0, 2000.0, 60000.0])     # USDC, ETH, BTC
    R = np.array([500_000.0, 150.0, 5.0])          # reserves
    target = np.array([1/3, 1/3, 1/3])             # equal-weight target
    pool = pdlp.PDLP(R=R, prices=prices, target=target, fee=0.001,
                     loans=[120_000.0, 80_000.0])
    w = pool.weights()
    print(f"pool value      = ${pool.value():,.0f}")
    print(f"current weights = {np.round(w, 3)}")
    print(f"target weights  = {np.round(target, 3)}")
    print(f"utilised (loans)= ${pool.total_loaned:,.0f}; available = ${pool.available_value():,.0f}")
    print(f"per-period fee income = ${pool.fee_income():,.2f}")

    # An LP deposits ETH (underweight asset) -> should earn a discount.
    deposit = np.array([0.0, 50.0, 0.0])           # add 50 ETH
    w_after = pdlp.weights(prices, R + deposit)
    F = pdlp.gmx_glp_discount(w, w_after, target, gamma_b=0.0, gamma_t=0.1)
    print(f"\nLP adds 50 ETH; weights {np.round(w,3)} -> {np.round(w_after,3)}")
    print(f"GMX GLP discount F = {F:.4f}  ({'subsidy to the LP' if F>0 else 'no discount'})")

    # The TWM trade that exactly reaches the target.
    delta_star = pdlp.target_weight_trade(prices, R, available=R, target=target)
    print(f"trade to hit target weights, Delta = {np.round(delta_star, 2)}")
    print(f"weights after Delta = {np.round(pdlp.weights(prices, R + delta_star), 3)}")

    section("§3.5  Portfolio dilution after a TWM update")
    v_new = pdlp.portfolio_dilution_value(prices, R, ell=200_000.0, fee=pool.fee, F=F)
    print(f"V_new = p^T R/(1+F) + f * (lent value) = ${v_new:,.2f}")
    print("(existing LPs diluted by 1/(1+F); they keep the fee income on loans)")

    # ---------------------------------------------------------------- #
    section("§4  Hedged PDLP — mean-variance delta hedge (eq 13) & Sharpe (Claim 4.1)")
    # Two risky assets (ETH, BTC); numeraire is delta-neutral so excluded here.
    p_risky = np.array([2000.0, 60000.0])
    Sigma = np.array([[0.04, 0.018], [0.018, 0.05]])   # covariance of returns
    ell_v = np.array([80_000.0, 50_000.0]) / p_risky    # lent amounts (units)
    Delta = np.array([150.0, 5.0])                      # pool delta to offset
    fee, gamma = 0.001, 2.0
    pi = pdlp.delta_hedge(fee, gamma, Sigma, ell_v, Delta)
    print(f"covariance Sigma =\n{Sigma}")
    print(f"hedge position pi = (f/gamma) Sigma^-1 ell - Delta = {np.round(pi, 3)}")
    exp_ok, var_ok = pdlp.sharpe_improves(fee, ell_v, Delta, Sigma, p_risky, gamma)
    print(f"Claim 4.1 condition 1 (expectation non-decreasing): {exp_ok}")
    print(f"Claim 4.1 condition 2 (variance non-increasing):    {var_ok}")
    print("When both hold, the delta-hedged PDLP's Sharpe ratio >= the unhedged pool's.")


if __name__ == "__main__":
    main()
