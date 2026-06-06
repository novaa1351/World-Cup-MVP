"""Compare model probabilities against market-implied probabilities.

Outputs an edge table and flags two kinds of opportunity:
  * VALUE   -> |model_prob - market_prob| beyond a threshold (model thinks the
               market is mispriced). Reported with Kelly-style sizing.
  * ARBITRAGE -> the same outcome priced differently on two platforms such that
               you could (in principle) lock a profit. Real-world frictions
               (fees, spread, settlement timing, eligibility) usually erase
               these, so we report them as *analytical* signals only.

This module is for ANALYSIS / EDUCATION. It does not place trades and is not
financial advice.
"""
from __future__ import annotations
import pandas as pd


def edge_table(model: dict[str, float], market: dict[str, float]) -> pd.DataFrame:
    """One row per outcome present in both dicts."""
    rows = []
    for k in sorted(set(model) & set(market)):
        mp, mk = model[k], market[k]
        rows.append(
            {
                "outcome": k,
                "model_prob": mp,
                "market_prob": mk,
                "edge": mp - mk,                       # +ve => model sees value on YES
                "edge_pct": (mp - mk) * 100,
                "fair_odds": (1 / mp) if mp > 0 else float("inf"),
                "market_odds": (1 / mk) if mk > 0 else float("inf"),
            }
        )
    return pd.DataFrame(rows).sort_values("edge", ascending=False, key=abs)


def flag_value(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Rows where the model disagrees with the market by > threshold."""
    return df[df["edge"].abs() >= threshold].copy()


def kelly_fraction(p_model: float, market_prob: float, cap: float = 0.25) -> float:
    """Fractional-Kelly stake for a binary YES bet priced at `market_prob`.

    Decimal odds b = (1/market_prob) - 1. Kelly f* = (b*p - q) / b.
    We cap and floor at 0 (no shorting here) -- position sizing borrowed
    straight from the options-trading risk mindset.
    """
    if not 0 < market_prob < 1:
        return 0.0
    b = (1 / market_prob) - 1
    q = 1 - p_model
    f = (b * p_model - q) / b
    return max(0.0, min(cap, f))


def cross_platform_arb(prob_a: dict[str, float], prob_b: dict[str, float],
                       fee: float = 0.0) -> pd.DataFrame:
    """Flag outcomes where platform A and B disagree enough to suggest arb.

    Buying YES on the cheaper side and NO on the dearer side locks profit when
    price_yes_cheap + price_no_dear < 1 - fees. Here we surface the raw gap;
    refine with real order-book depth and fees before believing it.
    """
    rows = []
    for k in sorted(set(prob_a) & set(prob_b)):
        pa, pb = prob_a[k], prob_b[k]
        # cost to own the outcome on the cheaper platform + the complement on
        # the other; < 1 (minus fees) => theoretical lock.
        lock_cost = min(pa, pb) + (1 - max(pa, pb))
        rows.append(
            {
                "outcome": k,
                "prob_platform_a": pa,
                "prob_platform_b": pb,
                "gap": abs(pa - pb),
                "lock_cost": lock_cost,
                "theo_arb": lock_cost < (1 - fee),
            }
        )
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


if __name__ == "__main__":
    model = {"Brazil": 0.22, "France": 0.15, "Argentina": 0.17}
    market = {"Brazil": 0.18, "France": 0.16, "Argentina": 0.19}
    t = edge_table(model, market)
    print(t.to_string(index=False))
    print("\nvalue flags:\n", flag_value(t).to_string(index=False))
    print("\nKelly on Brazil:", round(kelly_fraction(0.22, 0.18), 3))
