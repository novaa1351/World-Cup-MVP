"""Compare model predictions against real Polymarket odds for 2026 matches
that have since resolved.

Usage: python run_market_comparison.py
"""
import config
import pandas as pd
from src.data.historical import load_results
from src.data.markets_historical import fetch_polymarket_matches_resolved
from src.markets.implied import implied_from_book
from src.models.elo import compute_elo
from src.models.match_model import match_probs
from src.models.confederations import CONFEDERATION, fit_confederation_offsets
from src.models.tournament_form import tournament_goal_diff_so_far, fit_goal_diff_weight


def brier(probs: dict, outcome_label: str) -> float:
    """Brier score for a 3-way prediction against the actual outcome."""
    return sum(
        (p - (1.0 if label == outcome_label else 0.0)) ** 2
        for label, p in probs.items()
    )


def main():
    matches = load_results()
    market_df = fetch_polymarket_matches_resolved()

    if market_df.empty:
        print("No market data available.")
        return

    # Build model ratings using ALL data before the 2026 WC started
    cutoff = pd.Timestamp("2026-06-11")
    train = matches[matches["date"] < cutoff]
    elo = compute_elo(train)

    offsets = fit_confederation_offsets(train, elo, CONFEDERATION)
    confed_adjust = {t: offsets.get(CONFEDERATION.get(t), 0.0) for t in elo}

    wc_2026 = matches[
        (matches["date"] >= cutoff) & (matches["tournament"] == "FIFA World Cup")
    ].sort_values("date").reset_index(drop=True)

    prep_rows = []
    for row in wc_2026.itertuples(index=False):
        gd = tournament_goal_diff_so_far(wc_2026, row.date)
        prep_rows.append({
            "home": row.home_team, "away": row.away_team,
            "home_score": int(row.home_score), "away_score": int(row.away_score),
            "neutral": bool(row.neutral),
            "gd_home": gd.get(row.home_team, 0), "gd_away": gd.get(row.away_team, 0),
        })
    weight = fit_goal_diff_weight(prep_rows, elo, confed_adjust)

    # Group market rows by match (event_slug groups the 3 outcome rows together)
    brier_model, brier_market, n = 0.0, 0.0, 0
    print(f"{'Match':<32} {'Model H/D/A':<24} {'Market H/D/A':<24} Actual")
    print("-" * 100)

    for slug, group in market_df.groupby("event_slug"):
        home = group["home_team"].iloc[0]
        away = group["away_team"].iloc[0]

        book = {row["outcome"]: row["price"] for _, row in group.iterrows()}
        if len(book) < 2:
            continue
        market_probs_raw = implied_from_book(book)

        real_match = wc_2026[(wc_2026["home_team"] == home) & (wc_2026["away_team"] == away)]
        if real_match.empty:
            continue
        real_row = real_match.iloc[0]

        gd = tournament_goal_diff_so_far(wc_2026, real_row["date"])
        r_home = elo.get(home, config.ELO_BASE) + confed_adjust.get(home, 0.0) + weight * gd.get(home, 0)
        r_away = elo.get(away, config.ELO_BASE) + confed_adjust.get(away, 0.0) + weight * gd.get(away, 0)
        model_probs = match_probs(r_home, r_away, neutral=bool(real_row["neutral"]))

        if real_row["home_score"] > real_row["away_score"]:
            actual = "home_win"
        elif real_row["home_score"] == real_row["away_score"]:
            actual = "draw"
        else:
            actual = "away_win"

        model_dict = {"home_win": model_probs["home"], "draw": model_probs["draw"], "away_win": model_probs["away"]}
        market_dict = {"home_win": market_probs_raw.get("home_win", 0), "draw": market_probs_raw.get("draw", 0), "away_win": market_probs_raw.get("away_win", 0)}

        brier_model += brier(model_dict, actual)
        brier_market += brier(market_dict, actual)
        n += 1

        print(f"{home} vs {away:<18} "
              f"{model_dict['home_win']:.0%}/{model_dict['draw']:.0%}/{model_dict['away_win']:.0%}   "
              f"{market_dict['home_win']:.0%}/{market_dict['draw']:.0%}/{market_dict['away_win']:.0%}   "
              f"{actual}")

    print("-" * 100)
    print(f"Matches compared: {n}")
    print(f"Mean Brier — MODEL:  {brier_model/n:.4f}")
    print(f"Mean Brier — MARKET: {brier_market/n:.4f}")


if __name__ == "__main__":
    main()