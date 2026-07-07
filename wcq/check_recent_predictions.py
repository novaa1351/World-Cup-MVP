"""Quick check: how well does the current Elo model predict recent 2026 matches?

Usage: python check_recent_predictions.py
"""
from src.data.historical import load_results
from src.models.elo import compute_elo
from src.models.match_model import match_probs


def main():
    matches = load_results()

    recent = matches.tail(5)
    history = matches.iloc[:-5]

    ratings = compute_elo(history, recent_weight=1.5, recent_days=365)

    # Debug: show raw ratings for the Mexico/England game specifically
    print(f"\nMexico Elo (with recent-form weighting): {ratings.get('Mexico')}")
    print(f"England Elo (with recent-form weighting): {ratings.get('England')}\n")

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


if __name__ == "__main__":
    main()