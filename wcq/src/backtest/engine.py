"""Minimal backtest harness for the model-vs-market strategy.

MVP scope: given historical markets with KNOWN outcomes plus the model probs you
would have produced, simulate flat or fractional-Kelly staking and report P&L,
hit rate, ROI, and a calibration check (Brier score). Wire in real resolved
markets (2018/2022 archives, or early-2026 group-stage results) as you collect
them. Start with the synthetic example in __main__ to see the shapes.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.markets.edges import kelly_fraction


def brier_score(p_model: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts (lower = better)."""
    return float(np.mean((p_model - outcome) ** 2))


def run_backtest(df: pd.DataFrame, bankroll: float = 1000.0,
                 staking: str = "kelly", flat_stake: float = 10.0,
                 edge_threshold: float = 0.03) -> dict:
    """`df` needs columns: model_prob, market_prob, outcome (1 win / 0 loss).

    Bets YES whenever model_prob - market_prob >= edge_threshold.
    Returns summary stats + the per-bet ledger.
    """
    bank = bankroll
    ledger = []
    for row in df.itertuples(index=False):
        edge = row.model_prob - row.market_prob
        if edge < edge_threshold:
            continue
        if staking == "kelly":
            stake = kelly_fraction(row.model_prob, row.market_prob) * bank
        else:
            stake = flat_stake
        payout = (stake / row.market_prob) if row.outcome == 1 else 0.0
        pnl = payout - stake
        bank += pnl
        ledger.append({"outcome_label": row.outcome, "edge": edge,
                       "stake": stake, "pnl": pnl, "bankroll": bank})
    led = pd.DataFrame(ledger)
    if led.empty:
        return {"n_bets": 0, "ledger": led}
    return {
        "n_bets": len(led),
        "final_bankroll": bank,
        "roi": bank / bankroll - 1,
        "hit_rate": float((df.loc[df["model_prob"] - df["market_prob"]
                                  >= edge_threshold, "outcome"]).mean()),
        "brier": brier_score(df["model_prob"].to_numpy(),
                             df["outcome"].to_numpy()),
        "ledger": led,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    market = rng.uniform(0.1, 0.6, n)
    model = np.clip(market + rng.normal(0.02, 0.05, n), 0.01, 0.99)
    outcome = (rng.uniform(size=n) < model).astype(int)  # model is "true" here
    demo = pd.DataFrame({"model_prob": model, "market_prob": market,
                         "outcome": outcome})
    res = run_backtest(demo)
    print({k: v for k, v in res.items() if k != "ledger"})
