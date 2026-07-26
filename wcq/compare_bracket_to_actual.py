"""Reconstruct the REAL 2026 knockout progression from historical.py's data
(no explicit round column, so we infer depth from match count per team),
and compare it against predict_full_bracket.py's deterministic prediction.

Usage: python compare_bracket_to_actual.py
"""
import pandas as pd
from src.data.historical import load_results

ROUND_BY_MATCH_COUNT = {
    3: "Group stage exit",
    4: "R32 exit",
    5: "R16 exit",
    6: "QF exit",
    7: "SF exit / Finalist",
}


def main():
    matches = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    end = pd.Timestamp("2026-07-19")

    wc = matches[
        (matches["date"] >= cutoff) & (matches["date"] <= end)
        & (matches["tournament"] == "FIFA World Cup")
    ].sort_values("date").reset_index(drop=True)

    print(f"Total 2026 WC matches in dataset: {len(wc)}")

    # Count matches played per team
    counts = {}
    for _, row in wc.iterrows():
        counts[row["home_team"]] = counts.get(row["home_team"], 0) + 1
        counts[row["away_team"]] = counts.get(row["away_team"], 0) + 1

    # The final is the LAST match chronologically
    final_match = wc.iloc[-1]
    if final_match["home_score"] > final_match["away_score"]:
        champion, runner_up = final_match["home_team"], final_match["away_team"]
    else:
        champion, runner_up = final_match["away_team"], final_match["home_team"]

    print(f"\nActual final: {final_match['home_team']} {final_match['home_score']}-"
          f"{final_match['away_score']} {final_match['away_team']}")
    print(f"ACTUAL CHAMPION: {champion}")
    print(f"ACTUAL RUNNER-UP: {runner_up}")

    print("\nTeams by matches played (deepest run first):")
    for team, n in sorted(counts.items(), key=lambda x: -x[1])[:12]:
        label = ROUND_BY_MATCH_COUNT.get(n, f"{n} matches")
        print(f"  {team:<20} {n} matches  ({label})")


if __name__ == "__main__":
    main()