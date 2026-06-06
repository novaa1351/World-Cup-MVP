"""Elo ratings computed from full match history.

Standard chess-style Elo adapted for football: K-factor, home advantage, and an
expected-score formula. Returns a dict {team: rating} after replaying history.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import defaultdict
import pandas as pd

import config


def expected_score(r_a: float, r_b: float) -> float:
    """Logistic expected score for A vs B (0..1)."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def compute_elo(
    matches: pd.DataFrame,
    k: float = config.ELO_K,
    home_adv: float = config.ELO_HOME_ADV,
    base: float = config.ELO_BASE,
) -> dict[str, float]:
    """Replay every match in chronological order and return final ratings."""
    ratings: dict[str, float] = defaultdict(lambda: base)
    for row in matches.itertuples(index=False):
        ra, rb = ratings[row.home_team], ratings[row.away_team]
        adv = 0 if row.neutral else home_adv
        exp_home = expected_score(ra + adv, rb)
        # actual score from home perspective: win=1, draw=0.5, loss=0
        if row.home_score > row.away_score:
            actual = 1.0
        elif row.home_score == row.away_score:
            actual = 0.5
        else:
            actual = 0.0
        delta = k * (actual - exp_home)
        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta
    return dict(ratings)


def top_n(ratings: dict[str, float], n: int = 20) -> pd.DataFrame:
    s = pd.Series(ratings).sort_values(ascending=False).head(n)
    return s.rename("elo").rename_axis("team").reset_index()


if __name__ == "__main__":
    from src.data.historical import load_results
    r = compute_elo(load_results())
    print(top_n(r, 15).to_string(index=False))
