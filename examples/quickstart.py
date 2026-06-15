"""A quick tour of the reference mechanisms.

Run with:  python examples/quickstart.py
"""

import numpy as np

from mechanisms import amm, cda, lmsr, parimutuel, perp, scoring_rules as sr


def section(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def main():
    section("Proper scoring rules — truthfulness is optimal")
    belief = np.array([0.2, 0.5, 0.3])
    print("belief             :", belief)
    print("expected log-loss of truthful report :",
          round(sr.expected_score(sr.log_score, belief, belief), 4))
    print("expected log-loss of a misreport      :",
          round(sr.expected_score(sr.log_score, belief, [0.4, 0.3, 0.3]), 4),
          "(higher — properness)")

    section("Energy score — a Monte-Carlo (monteprediction-style) forecast")
    rng = np.random.default_rng(0)
    truth = np.array([0.5, -0.3])
    good = rng.normal(truth, 1.0, size=(1000, 2))
    bad = rng.normal([4, 4], 1.0, size=(1000, 2))
    print("energy score, well-aimed cloud :", round(sr.energy_score(good, truth), 4))
    print("energy score, mis-aimed cloud  :", round(sr.energy_score(bad, truth), 4))

    section("Parimutuel pool — endogenous odds, self-funding")
    pool = parimutuel.ParimutuelPool(n_outcomes=3, takeout=0.1)
    pool.bet("alice", 0, 100)
    pool.bet("bob", 1, 50)
    pool.bet("carol", 2, 50)
    print("implied probabilities :", np.round(pool.implied_probabilities(), 3))
    print("decimal odds outcome 0:", round(pool.decimal_odds(0), 3))
    print("payouts if 0 wins     :", {k: round(v, 2) for k, v in pool.settle(0).items()})

    section("LMSR — bounded-loss prediction market")
    m = lmsr.LMSR(n_outcomes=2, b=100)
    print("start prices :", np.round(m.prices(), 3))
    cost = m.buy(0, 100)
    print(f"bought 100 of outcome 0 for {cost:.2f}; prices -> {np.round(m.prices(), 3)}")
    print("max market-maker loss (b*log n):", round(m.max_loss, 3))

    section("Constant-product AMM — x*y=k and impermanent loss")
    p = amm.ConstantProductAMM(x=1000, y=1000, fee=0.003)
    out = p.swap(100, x_in=True)
    print(f"sold 100 X, received {out:.3f} Y; new spot price {p.spot_price:.4f}")
    print("impermanent loss at 2x price move:", round(amm.impermanent_loss(2.0), 4))

    section("Continuous double auction — price-time priority")
    book = cda.LimitOrderBook()
    book.submit(cda.Order("sell", 101, 5, "mm"))
    book.submit(cda.Order("buy", 99, 5, "mm"))
    trades = book.submit(cda.Order("buy", 102, 3, "taker"))
    print("trade :", trades[0])
    print("book  : bids", book.depth()[0], "asks", book.depth()[1])

    section("Perpetual future — funding tether & liquidation")
    rate = perp.funding_rate(premium_index=0.002)
    print("funding rate (premium 0.2%):", round(rate, 5), "(longs pay shorts)")
    pos = perp.PerpPosition(side=1, size=1, entry_price=100, collateral=10)
    print("10x long, entry 100, liquidation price:", round(pos.liquidation_price(), 2))
    fc = pos.apply_funding(mark_price=100, rate=rate)
    print(f"funding cash flow to the long: {fc:.4f}")


if __name__ == "__main__":
    main()
