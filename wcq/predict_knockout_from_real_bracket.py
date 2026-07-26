"""Two-part prediction, using the REAL, complete 2026 tournament:

PART 1 — Group-stage survival odds (soft probabilities, via Monte Carlo),
         using only pre-tournament data (no lookahead).
PART 2 — Knockout-stage picks: for every REAL knockout match (as it actually
         happened), predict the winner using only information available
         before that match (pre-tournament Elo + confederation offset +
         goal-difference from real tournament matches played so far).
         Compares each pick against the real result.

Usage: python predict_knockout_from_real_bracket.py
"""
import pandas as pd
from src.data.historical import load_results
from src.models.elo import compute_elo
from src.models.match_model import win_prob_knockout
from src.models.confederations import CONFEDERATION, fit_confederation_offsets
from src.models.tournament_form import tournament_goal_diff_so_far, fit_goal_diff_weight
from src.models.tournament import (
    GROUPS_2026, simulate_tournament, survival_table, _FIFA_TO_HIST,
)
import config


def build_adjusted_ratings(base_ratings, confed_adjust):
    """Fold confederation offsets into a ratings dict keyed by historical
    team names, so it can be passed straight into simulate_tournament()."""
    adjusted = dict(base_ratings)
    for team in [t for g in GROUPS_2026.values() for t in g]:
        hist_name = _FIFA_TO_HIST.get(team, team)
        off = confed_adjust.get(team, 0.0)
        adjusted[hist_name] = adjusted.get(hist_name, config.ELO_BASE) + off
    return adjusted


def elo_lookup(team, ratings, confed_adjust, base):
    hist_name = _FIFA_TO_HIST.get(team, team)
    return ratings.get(hist_name, base) + confed_adjust.get(team, 0.0)


def main():
    matches = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    train = matches[matches["date"] < cutoff]

    print("Training Elo (pre-tournament data only, no lookahead)...")
    ratings = compute_elo(train)
    offsets = fit_confederation_offsets(train, ratings, CONFEDERATION)
    confed_adjust = {t: offsets.get(CONFEDERATION.get(t), 0.0) for t in
                      [tm for g in GROUPS_2026.values() for tm in g]}
    base = config.ELO_BASE

    # ============== PART 1: Group-stage survival odds ==============
    print("\n=== PART 1: Predicted odds to advance from the group stage ===")
    adjusted_ratings = build_adjusted_ratings(ratings, confed_adjust)
    survival = simulate_tournament(adjusted_ratings, n_sims=5000, seed=42)
    table = survival_table(survival, sort_by="group", top_n=48)
    print(table[["group"]].to_string(float_format="{:.1%}".format))

    # ============== PART 2: Real knockout bracket, no-lookahead picks ==============
    print("\n\n=== PART 2: Knockout-stage picks (using REAL bracket as it happened) ===")
    end = pd.Timestamp("2026-07-19")
    wc = matches[
        (matches["date"] >= cutoff) & (matches["date"] <= end)
        & (matches["tournament"] == "FIFA World Cup")
    ].sort_values("date").reset_index(drop=True)

    group_stage = wc.iloc[:72]
    knockout = wc.iloc[72:].reset_index(drop=True)

    round_slices = {
        "R32":         knockout.iloc[0:16],
        "R16":         knockout.iloc[16:24],
        "QF":          knockout.iloc[24:28],
        "SF":          knockout.iloc[28:30],
        "3rd place":   knockout.iloc[30:31],
        "Final":       knockout.iloc[31:32],
    }

    # Fit goal-diff weight using the REAL, completed group-stage matches
    prep_rows = []
    for row in group_stage.itertuples(index=False):
        gd = tournament_goal_diff_so_far(group_stage, row.date)
        prep_rows.append({
            "home": row.home_team, "away": row.away_team,
            "home_score": int(row.home_score), "away_score": int(row.away_score),
            "neutral": bool(row.neutral),
            "gd_home": gd.get(row.home_team, 0), "gd_away": gd.get(row.away_team, 0),
        })
    ratings_hist_keyed = {t: elo_lookup(t, ratings, confed_adjust, base)
                           for t in set(group_stage["home_team"]) | set(group_stage["away_team"])}
    weight = fit_goal_diff_weight(prep_rows, ratings_hist_keyed, confed_adjust={})
    # (confed_adjust already folded into ratings_hist_keyed above, so pass empty here)

    correct, total = 0, 0
    for round_name, round_df in round_slices.items():
        print(f"\n--- {round_name} ---")
        for row in round_df.itertuples(index=False):
            gd = tournament_goal_diff_so_far(wc, row.date)
            r_home = elo_lookup(row.home_team, ratings, confed_adjust, base) + weight * gd.get(row.home_team, 0)
            r_away = elo_lookup(row.away_team, ratings, confed_adjust, base) + weight * gd.get(row.away_team, 0)

            p_home = win_prob_knockout(r_home, r_away)
            pick = row.home_team if p_home >= 0.5 else row.away_team
            pick_prob = p_home if pick == row.home_team else 1 - p_home

            actual_winner = row.home_team if row.home_score > row.away_score else row.away_team
            hit = "✓" if pick == actual_winner else "✗"

            total += 1
            if pick == actual_winner:
                correct += 1

            print(f"  {row.home_team} vs {row.away_team:<18} "
                  f"model picks {pick} ({pick_prob:.0%})   actual: {actual_winner}   {hit}")

    print(f"\nKnockout-stage accuracy: {correct}/{total} ({correct/total:.0%})")


if __name__ == "__main__":
    main()