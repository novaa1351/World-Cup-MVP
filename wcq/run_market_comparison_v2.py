"""Compare model vs. market predictions using the bot's OWN live-logged
predictions (genuine pre-match odds, captured in real time during the
tournament) — not reconstructed after-the-fact prices.

Usage: python run_market_comparison_v2.py
"""
import sqlite3
import pandas as pd


def brier(p_home, p_draw, p_away, winner: str) -> float:
    actual = {"home": 1.0 if winner == "home" else 0.0,
              "draw": 1.0 if winner == "draw" else 0.0,
              "away": 1.0 if winner == "away" else 0.0}
    return (p_home - actual["home"])**2 + (p_draw - actual["draw"])**2 + (p_away - actual["away"])**2


def main():
    conn = sqlite3.connect("wcq_bot.db")

    query = """
    SELECT p.match_id, p.home_team, p.away_team,
           p.p_home_model, p.p_draw_model, p.p_away_model,
           p.p_home_market, p.p_draw_market, p.p_away_market,
           r.home_score, r.away_score, r.winner
    FROM predictions p
    JOIN match_results r ON p.match_id = r.match_id
    WHERE p.p_home_market IS NOT NULL
      AND r.winner IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No joined rows found — check match_id formats match between tables.")
        return

    print(f"{'Match':<32} {'Model H/D/A':<20} {'Market H/D/A':<20} Winner")
    print("-" * 95)

    brier_model_total, brier_market_total = 0.0, 0.0
    for _, row in df.iterrows():
        bm = brier(row["p_home_model"], row["p_draw_model"], row["p_away_model"], row["winner"])
        bk = brier(row["p_home_market"], row["p_draw_market"], row["p_away_market"], row["winner"])
        brier_model_total += bm
        brier_market_total += bk

        print(f"{row['home_team']} vs {row['away_team']:<18} "
              f"{row['p_home_model']:.0%}/{row['p_draw_model']:.0%}/{row['p_away_model']:.0%}   "
              f"{row['p_home_market']:.0%}/{row['p_draw_market']:.0%}/{row['p_away_market']:.0%}   "
              f"{row['winner']}")

    n = len(df)
    print("-" * 95)
    print(f"Matches compared: {n}")
    print(f"Mean Brier — MODEL:  {brier_model_total/n:.4f}")
    print(f"Mean Brier — MARKET: {brier_market_total/n:.4f}")


if __name__ == "__main__":
    main()
