"""Match-outcome probabilities from Elo ratings.

Soccer needs a 3-way (win/draw/loss) model, not the 2-way Elo expected score.
We map the Elo rating gap to a win/draw/loss triple. The draw share is highest
when teams are evenly matched and shrinks as the gap grows -- a simple but
well-behaved approximation. (Upgrade path: Dixon-Coles bivariate-Poisson on
goals, which also yields exact scorelines. See suggested prompts in README.)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

import config
from src.models.elo import expected_score


def match_probs(
    elo_home: float,
    elo_away: float,
    neutral: bool = True,
    home_adv: float = config.ELO_HOME_ADV,
    draw_base: float = 0.28,
) -> dict[str, float]:
    """Return {'home', 'draw', 'away'} probabilities summing to 1."""
    adv = 0 if neutral else home_adv
    p_home_2way = expected_score(elo_home + adv, elo_away)  # ignores draws

    gap = abs(elo_home + adv - elo_away)
    # Draws are most likely for evenly matched sides; decay with the gap.
    draw = draw_base * math.exp(-gap / 400.0)

    # Split the remaining (1 - draw) mass using the 2-way expectation.
    home = (1 - draw) * p_home_2way
    away = (1 - draw) * (1 - p_home_2way)
    total = home + draw + away
    return {"home": home / total, "draw": draw / total, "away": away / total}


def win_prob_knockout(elo_a: float, elo_b: float) -> float:
    """P(A beats B) in a knockout tie (draw resolved by ET/penalties ~ coin)."""
    p = match_probs(elo_a, elo_b)
    return p["home"] + 0.5 * p["draw"]


if __name__ == "__main__":
    # Example: a 200-point favourite on neutral ground
    print(match_probs(1900, 1700))
    print("knockout P(favourite advances):", round(win_prob_knockout(1900, 1700), 3))
