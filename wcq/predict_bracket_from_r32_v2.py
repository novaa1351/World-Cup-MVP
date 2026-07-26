"""TRUE bracket prediction using the REAL bracket TREE structure (derived
from actual match participants, not chronological order), with the model's
OWN picks propagating through — no mid-bracket correction.

Usage: python predict_bracket_from_r32_v2.py
"""
import pandas as pd
from src.data.historical import load_results
from src.models.elo import compute_elo
from src.models.match_model import win_prob_knockout
from src.models.confederations import CONFEDERATION, fit_confederation_offsets
from src.models.tournament_form import tournament_goal_diff_so_far, fit_goal_diff_weight
from src.models.tournament import _FIFA_TO_HIST
import config


def elo_lookup(team, ratings, confed_adjust, base):
    hist_name = _FIFA_TO_HIST.get(team, team)
    return ratings.get(hist_name, base) + confed_adjust.get(team, 0.0)


def team_to_slot_map(round_df):
    """Map each team in this round to the match-index (slot) they played in."""
    mapping = {}
    for idx, row in enumerate(round_df.itertuples(index=False)):
        mapping[row.home_team] = idx
        mapping[row.away_team] = idx
    return mapping


def build_structure(prev_slot_map, next_round_df):
    """For each match in next_round_df, find which PREVIOUS-round slots its
    two participants actually came from — this IS the true bracket tree."""
    return [(prev_slot_map[row.home_team], prev_slot_map[row.away_team])
            for row in next_round_df.itertuples(index=False)]


def main():
    matches = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    end = pd.Timestamp("2026-07-19")
    train = matches[matches["date"] < cutoff]

    print("Training Elo (pre-tournament data only, no lookahead)...")
    ratings = compute_elo(train)
    offsets = fit_confederation_offsets(train, ratings, CONFEDERATION)

    wc = matches[
        (matches["date"] >= cutoff) & (matches["date"] <= end)
        & (matches["tournament"] == "FIFA World Cup")
    ].sort_values("date").reset_index(drop=True)
    all_teams = set(wc["home_team"]) | set(wc["away_team"])
    confed_adjust = {t: offsets.get(CONFEDERATION.get(t), 0.0) for t in all_teams}
    base = config.ELO_BASE

    group_stage = wc.iloc[:72]
    knockout = wc.iloc[72:].reset_index(drop=True)

    r32_matches = knockout.iloc[0:16]
    r16_matches = knockout.iloc[16:24]
    qf_matches  = knockout.iloc[24:28]
    sf_matches  = knockout.iloc[28:30]
    final_match = knockout.iloc[31:32]   # skip 30:31, that's the 3rd-place match

    # --- Fit goal-diff weight from real group stage; freeze at final group values ---
    prep_rows = []
    for row in group_stage.itertuples(index=False):
        gd = tournament_goal_diff_so_far(group_stage, row.date)
        prep_rows.append({
            "home": row.home_team, "away": row.away_team,
            "home_score": int(row.home_score), "away_score": int(row.away_score),
            "neutral": bool(row.neutral),
            "gd_home": gd.get(row.home_team, 0), "gd_away": gd.get(row.away_team, 0),
        })
    weight = fit_goal_diff_weight(
        prep_rows, {t: elo_lookup(t, ratings, confed_adjust, base) for t in all_teams},
        confed_adjust={})
    final_group_gd = tournament_goal_diff_so_far(group_stage, pd.Timestamp("2026-12-31"))

    def rating(team):
        return elo_lookup(team, ratings, confed_adjust, base) + weight * final_group_gd.get(team, 0)

    def pick_winner(a, b):
        p_a = win_prob_knockout(rating(a), rating(b))
        return (a, p_a) if p_a >= 0.5 else (b, 1 - p_a)

    # --- Build the TRUE bracket tree from real match participants ---
    r32_slot = team_to_slot_map(r32_matches)
    r16_structure = build_structure(r32_slot, r16_matches)
    r16_slot = team_to_slot_map(r16_matches)
    qf_structure = build_structure(r16_slot, qf_matches)
    qf_slot = team_to_slot_map(qf_matches)
    sf_structure = build_structure(qf_slot, sf_matches)
    sf_slot = team_to_slot_map(sf_matches)
    final_structure = build_structure(sf_slot, final_match)

    # --- Run the model's own bracket through the real tree ---
    print("\n=== R32 (real matchups) ===")
    r32_winners = []
    for row in r32_matches.itertuples(index=False):
        winner, prob = pick_winner(row.home_team, row.away_team)
        r32_winners.append(winner)
        print(f"  {row.home_team} vs {row.away_team:<18} -> {winner} ({prob:.0%})")

    print("\n=== R16 (true bracket structure, model's own R32 winners) ===")
    r16_winners = []
    for i, j in r16_structure:
        winner, prob = pick_winner(r32_winners[i], r32_winners[j])
        r16_winners.append(winner)
        print(f"  {r32_winners[i]} vs {r32_winners[j]:<18} -> {winner} ({prob:.0%})")

    print("\n=== QF ===")
    qf_winners = []
    for i, j in qf_structure:
        winner, prob = pick_winner(r16_winners[i], r16_winners[j])
        qf_winners.append(winner)
        print(f"  {r16_winners[i]} vs {r16_winners[j]:<18} -> {winner} ({prob:.0%})")

    print("\n=== SF ===")
    sf_winners = []
    for i, j in sf_structure:
        winner, prob = pick_winner(qf_winners[i], qf_winners[j])
        sf_winners.append(winner)
        print(f"  {qf_winners[i]} vs {qf_winners[j]:<18} -> {winner} ({prob:.0%})")

    print("\n=== FINAL ===")
    i, j = final_structure[0]
    champion, prob = pick_winner(sf_winners[i], sf_winners[j])
    print(f"  {sf_winners[i]} vs {sf_winners[j]:<18} -> {champion} ({prob:.0%})")

    print(f"\nPREDICTED CHAMPION (true bracket path): {champion}")
    print(f"ACTUAL CHAMPION: Spain")


if __name__ == "__main__":
    main()