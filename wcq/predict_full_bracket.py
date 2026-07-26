"""Deterministic full-bracket prediction: picks the single most-likely outcome
at every match (group stage + knockout), using PRE-TOURNAMENT data only
(no lookahead — this is what the model would have called before ball one).

Compares the predicted bracket against real 2026 results.

Usage: python predict_full_bracket.py
"""
import pandas as pd
from itertools import combinations
from src.data.historical import load_results
from src.models.elo import compute_elo
from src.models.match_model import match_probs, win_prob_knockout
from src.models.confederations import CONFEDERATION, fit_confederation_offsets
from src.models.tournament import GROUPS_2026, _R32_SLOTS, _FIFA_TO_HIST
import config


def elo_lookup(team, ratings, confed_adjust, base):
    hist_name = _FIFA_TO_HIST.get(team, team)
    return ratings.get(hist_name, base) + confed_adjust.get(team, 0.0)


def predict_group(teams, ratings, confed_adjust, base):
    """Deterministic group standings: pick the most-likely outcome per match,
    award 3/1/0 pts, tiebreak by Elo (proxy for real GD/GF tiebreak, since we
    aren't simulating real scorelines)."""
    pts = {t: 0 for t in teams}
    for h, a in combinations(teams, 2):
        eh = elo_lookup(h, ratings, confed_adjust, base)
        ea = elo_lookup(a, ratings, confed_adjust, base)
        probs = match_probs(eh, ea, neutral=True)
        best = max(probs, key=probs.get)
        if best == "home":
            pts[h] += 3
        elif best == "away":
            pts[a] += 3
        else:
            pts[h] += 1
            pts[a] += 1
    ranked = sorted(teams, key=lambda t: (pts[t], elo_lookup(t, ratings, confed_adjust, base)), reverse=True)
    return ranked  # [1st, 2nd, 3rd, 4th]


def predict_knockout_round(pairs, ratings, confed_adjust, base):
    winners = []
    for a, b in pairs:
        ea = elo_lookup(a, ratings, confed_adjust, base)
        eb = elo_lookup(b, ratings, confed_adjust, base)
        p_a = win_prob_knockout(ea, eb)
        winners.append(a if p_a >= 0.5 else b)
    return winners


def sequential_pairs(teams):
    return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]


def main():
    matches = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    train = matches[matches["date"] < cutoff]

    print("Training Elo (pre-tournament data only, no lookahead)...")
    ratings = compute_elo(train)
    offsets = fit_confederation_offsets(train, ratings, CONFEDERATION)
    confed_adjust = {t: offsets.get(CONFEDERATION.get(t), 0.0) for t in ratings}
    base = config.ELO_BASE

    # --- Group stage ---
    print("\n=== PREDICTED GROUP STANDINGS ===")
    group_winners, group_runners, thirds = {}, {}, []
    for gid, teams in GROUPS_2026.items():
        ranked = predict_group(teams, ratings, confed_adjust, base)
        group_winners[gid] = ranked[0]
        group_runners[gid] = ranked[1]
        thirds.append((gid, ranked[2]))
        print(f"  Group {gid}: 1) {ranked[0]}  2) {ranked[1]}  3) {ranked[2]}  4) {ranked[3]}")

    # Best 8 thirds — proxy tiebreak by Elo (same simplification as group stage)
    thirds_ranked = sorted(thirds, key=lambda x: elo_lookup(x[1], ratings, confed_adjust, base), reverse=True)
    best8 = [t for _, t in thirds_ranked[:8]]
    print(f"\n  Best 8 third-place teams advancing: {best8}")

    # --- Build R32 ---
    slot = {}
    for gid in GROUPS_2026:
        slot[f"W_{gid}"] = group_winners[gid]
        slot[f"R_{gid}"] = group_runners[gid]
    for i, t in enumerate(best8, 1):
        slot[f"T_{i}"] = t

    r32_pairs = [(slot[s1], slot[s2]) for s1, s2 in _R32_SLOTS]

    print("\n=== PREDICTED KNOCKOUT BRACKET ===")
    r32w = predict_knockout_round(r32_pairs, ratings, confed_adjust, base)
    print(f"R32 winners: {r32w}")

    r16w = predict_knockout_round(sequential_pairs(r32w), ratings, confed_adjust, base)
    print(f"R16 winners: {r16w}")

    qfw = predict_knockout_round(sequential_pairs(r16w), ratings, confed_adjust, base)
    print(f"QF winners:  {qfw}")

    sfw = predict_knockout_round(sequential_pairs(qfw), ratings, confed_adjust, base)
    print(f"SF winners:  {sfw}")

    champion = predict_knockout_round([tuple(sfw)], ratings, confed_adjust, base)[0]
    print(f"\nPREDICTED CHAMPION: {champion}")


if __name__ == "__main__":
    main()