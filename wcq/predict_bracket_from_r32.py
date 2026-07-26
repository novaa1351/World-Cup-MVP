"""TRUE bracket prediction: start from the REAL R32 matchups (the actual draw
outcome), then let the model's OWN picks determine who advances at every
subsequent round — no correction using real results mid-bracket. Only the
final predicted path is compared against what actually happened.

Usage: python predict_bracket_from_r32.py
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


def sequential_pairs(teams):
    return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]


def main():
    matches = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    train = matches[matches["date"] < cutoff]

    print("Training Elo (pre-tournament data only, no lookahead)...")
    ratings = compute_elo(train)
    offsets = fit_confederation_offsets(train, ratings, CONFEDERATION)
    all_2026_teams_query = matches[
        (matches["date"] >= cutoff) & (matches["date"] <= "2026-07-19")
        & (matches["tournament"] == "FIFA World Cup")
    ]
    all_teams = set(all_2026_teams_query["home_team"]) | set(all_2026_teams_query["away_team"])
    confed_adjust = {t: offsets.get(CONFEDERATION.get(t), 0.0) for t in all_teams}
    base = config.ELO_BASE

    end = pd.Timestamp("2026-07-19")
    wc = matches[
        (matches["date"] >= cutoff) & (matches["date"] <= end)
        & (matches["tournament"] == "FIFA World Cup")
    ].sort_values("date").reset_index(drop=True)

    group_stage = wc.iloc[:72]
    knockout = wc.iloc[72:].reset_index(drop=True)

    # Fit goal-diff weight from the REAL, completed group stage
    prep_rows = []
    for row in group_stage.itertuples(index=False):
        gd = tournament_goal_diff_so_far(group_stage, row.date)
        prep_rows.append({
            "home": row.home_team, "away": row.away_team,
            "home_score": int(row.home_score), "away_score": int(row.away_score),
            "neutral": bool(row.neutral),
            "gd_home": gd.get(row.home_team, 0), "gd_away": gd.get(row.away_team, 0),
        })
    weight = fit_goal_diff_weight(prep_rows, {t: elo_lookup(t, ratings, confed_adjust, base)
                                               for t in all_teams}, confed_adjust={})

    # Goal-diff FROZEN at real group-stage results (can't fabricate scores
    # for hypothetical model-only knockout wins in later rounds)
    final_group_gd = tournament_goal_diff_so_far(group_stage, pd.Timestamp("2026-12-31"))

    def rating(team):
        return elo_lookup(team, ratings, confed_adjust, base) + weight * final_group_gd.get(team, 0)

    def pick_winner(a, b):
        p_a = win_prob_knockout(rating(a), rating(b))
        return (a, p_a) if p_a >= 0.5 else (b, 1 - p_a)

    # REAL R32 matchups (actual draw — fixed, not predicted)
    r32_matches = knockout.iloc[0:16]
    r32_pairs = [(row.home_team, row.away_team) for row in r32_matches.itertuples(index=False)]

    print("\n=== R32 (real matchups, model picks winners) ===")
    r32_winners = []
    for a, b in r32_pairs:
        winner, prob = pick_winner(a, b)
        r32_winners.append(winner)
        print(f"  {a} vs {b:<18} -> {winner} ({prob:.0%})")

    print("\n=== R16 (MODEL'S OWN R32 winners advance — no correction) ===")
    r16_winners = []
    for a, b in sequential_pairs(r32_winners):
        winner, prob = pick_winner(a, b)
        r16_winners.append(winner)
        print(f"  {a} vs {b:<18} -> {winner} ({prob:.0%})")

    print("\n=== QF ===")
    qf_winners = []
    for a, b in sequential_pairs(r16_winners):
        winner, prob = pick_winner(a, b)
        qf_winners.append(winner)
        print(f"  {a} vs {b:<18} -> {winner} ({prob:.0%})")

    print("\n=== SF ===")
    sf_winners = []
    for a, b in sequential_pairs(qf_winners):
        winner, prob = pick_winner(a, b)
        sf_winners.append(winner)
        print(f"  {a} vs {b:<18} -> {winner} ({prob:.0%})")

    print("\n=== FINAL ===")
    champion, prob = pick_winner(*sf_winners)
    print(f"  {sf_winners[0]} vs {sf_winners[1]:<18} -> {champion} ({prob:.0%})")

    print(f"\nPREDICTED CHAMPION (model's own bracket path): {champion}")
    print(f"ACTUAL CHAMPION: Spain")


if __name__ == "__main__":
    main()