"""Quick check: how well does the current Elo model predict recent 2026 matches?

Usage: python check_recent_predictions.py
"""
from src.data.historical import load_results
from src.models.elo import compute_elo
from src.models.match_model import match_probs
from src.models.confederations import CONFEDERATION, fit_confederation_offsets
from collections import Counter


def main():
    matches = load_results()

    recent = matches.tail(5)
    history = matches.iloc[:-5]

    ratings = compute_elo(history)

    print("\n=== Elo Leaderboard (top 25) ===")
    for i, (team, elo) in enumerate(sorted(ratings.items(), key=lambda x: -x[1])[:25], 1):
        print(f"{i:>3}. {team:<20} {elo:.1f}")

    pair_counts = Counter()
    for _, row in matches.iterrows():
        ca = CONFEDERATION.get(row["home_team"])
        cb = CONFEDERATION.get(row["away_team"])
        if ca and cb and ca != cb:
            pair_counts[tuple(sorted([ca, cb]))] += 1
    print("\nCross-confederation match counts:")
    for pair, n in sorted(pair_counts.items(), key=lambda x: -x[1]):
        print(f"  {pair[0]} vs {pair[1]}: {n}")

    offsets_all = fit_confederation_offsets(matches, ratings, CONFEDERATION)
    print("\nFitted confederation offsets, ALL history (relative to UEFA=0):")
    for c, v in sorted(offsets_all.items(), key=lambda x: -x[1]):
        print(f"  {c:<10} {v:+.1f}")

    offsets_recent = fit_confederation_offsets(matches, ratings, CONFEDERATION, min_year=2010)
    print("\nFitted confederation offsets, 2010+ only (relative to UEFA=0):")
    for c, v in sorted(offsets_recent.items(), key=lambda x: -x[1]):
        print(f"  {c:<10} {v:+.1f}")

    print(f"\nMexico Elo: {ratings.get('Mexico')}")
    print(f"England Elo: {ratings.get('England')}\n")

    print(f"{'Match':<28} {'Model (H/D/A)':<28} Actual")
    print("-" * 70)
    for _, row in recent.iterrows():
        home, away = row["home_team"], row["away_team"]
        if home not in ratings or away not in ratings:
            print(f"{home} vs {away:<15} (unknown team, skipped)")
            continue

        probs = match_probs(ratings[home], ratings[away], neutral=row["neutral"])
        actual = f"{row['home_score']}-{row['away_score']}"

        pred_str = f"{probs['home']:.0%} / {probs['draw']:.0%} / {probs['away']:.0%}"
        print(f"{home} vs {away:<15} {pred_str:<28} {actual}")

    # Apply the fitted 2010+ offset to the Mexico/England matchup directly
    mex_conf = CONFEDERATION["Mexico"]
    eng_conf = CONFEDERATION["England"]
    mex_adjusted = ratings["Mexico"] + offsets_recent[mex_conf]
    eng_adjusted = ratings["England"] + offsets_recent[eng_conf]

    print(f"\n=== Mexico vs England, WITH confederation offset applied ===")
    print(f"Mexico: raw {ratings['Mexico']:.1f} + offset {offsets_recent[mex_conf]:+.1f} = {mex_adjusted:.1f}")
    print(f"England: raw {ratings['England']:.1f} + offset {offsets_recent[eng_conf]:+.1f} = {eng_adjusted:.1f}")

    probs_adjusted = match_probs(mex_adjusted, eng_adjusted, neutral=True)
    print(f"Model (offset-adjusted): {probs_adjusted['home']:.0%} / {probs_adjusted['draw']:.0%} / {probs_adjusted['away']:.0%}")

    probs_raw = match_probs(ratings["Mexico"], ratings["England"], neutral=True)
    print(f"Model (raw, no offset):  {probs_raw['home']:.0%} / {probs_raw['draw']:.0%} / {probs_raw['away']:.0%}")

    print(f"Actual result: Mexico 2-3 England")


if __name__ == "__main__":
    main()