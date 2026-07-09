"""Elo ratings computed from full match history.

Standard chess-style Elo adapted for football with two evidence-backed
enhancements:

1. Tournament K-weighting (eloratings.net methodology, widely used in the
   academic football-rating literature):
     FIFA World Cup finals          K × 1.5   → 60 at base K=40
     Major confederation finals     K × 1.25  → 50
     Qualifiers / Nations Leagues   K × 1.0   → 40 (base)
     Other competitive matches      K × 0.75  → 30
     Friendly internationals        K × 0.5   → 20
   Source: eloratings.net FAQ; Hvattum & Arntzen (2010) JQAS.

2. Annual mean-reversion (FiveThirtyEight SPI methodology):
   At every year boundary the ratings are pulled `reversion` fraction of the
   way back toward ELO_BASE. This prevents early-history matches from
   dominating forever and gives more weight to recent form.
     r_new = r + reversion × (base − r)
   At reversion=0.2 a result from 5 years ago retains (0.8)^5 ≈ 33 % of
   its original weight; a result from 1 year ago retains 80 %.
   Source: FiveThirtyEight "How Our Club Soccer Predictions Work" (2019).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import defaultdict

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Tournament tier patterns (matched in priority order)
# ---------------------------------------------------------------------------

# Pattern matches must be checked in order: qualifier first, then finals, so
# "FIFA World Cup qualification" is not confused with "FIFA World Cup".
_QUALIFIER_RE = re.compile(r"qualif|qualifying|nations league", re.I)
_WC_RE = re.compile(r"world cup", re.I)
_CONF_RE = re.compile(
    r"\beuro\b|copa am[eé]rica|african? cup of nations|asia[n]? cup|gold cup"
    r"|concacaf championship|olympic",
    re.I,
)
_FRIENDLY_RE = re.compile(r"friendly", re.I)


def tournament_k(tournament: str, base_k: float) -> float:
    """Return the K-factor for a match based on its tournament importance.

    Follows the five-tier scale used by eloratings.net and supported by the
    academic football Elo literature (Hvattum & Arntzen 2010).
    """
    if _FRIENDLY_RE.search(tournament):
        return base_k * 0.5                          # friendlies → 20 at K=40
    if _QUALIFIER_RE.search(tournament):
        return base_k                                 # qualifiers → 40
    if _WC_RE.search(tournament):
        return base_k * 1.5                           # WC finals → 60
    if _CONF_RE.search(tournament):
        return base_k * 1.25                          # major confederation → 50
    return base_k * 0.75                              # everything else → 30


def expected_score(r_a: float, r_b: float) -> float:
    """Logistic expected score for A vs B (0..1)."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def compute_elo(
    matches: pd.DataFrame,
    k: float = config.ELO_K,
    home_adv: float = config.ELO_HOME_ADV,
    base: float = config.ELO_BASE,
    reversion: float = config.ELO_MEAN_REVERSION,
    use_tournament_k: bool = True,
    return_history: bool = False,
    recent_weight: float = 1.0,
    recent_days: int = 365,
    current_wc_boost: float = 1.0,
    current_wc_year: int = 2026,
) -> "dict[str, float] | tuple[dict[str, float], list]":
    """Replay every match in chronological order and return final Elo ratings.

    Args:
        matches:          Historical match DataFrame from historical.load_results().
        k:                Base K-factor (applied to qualifier-level matches by
                          default). WC matches receive k × 1.5.
        home_adv:         Elo points added to the home team when not neutral.
        base:             Starting rating for teams with no prior history.
        reversion:        Annual mean-reversion fraction toward `base`
                          (0 = off, 0.2 = 20 % pull per year — FiveThirtyEight
                          uses 0.33 for clubs).
        use_tournament_k: Apply tournament tier K-scaling (eloratings.net).
        recent_weight:    Multiplier applied to matches within `recent_days`
                          of the dataset's most recent match (1.0 = off,
                          no extra weighting; e.g. 1.5 = 50% more weight).
        recent_days:      Size of the "recent" window in days (default 365,
                          i.e. the last 12 months).
        current_wc_boost: Extra multiplier applied specifically to matches from
                          `current_wc_year`'s World Cup, on top of the existing
                          tier boost (1.0 = off). These are the only matches
                          that directly connect otherwise-disconnected
                          confederation rating pools, so they carry unusually
                          strong signal about cross-confederation strength.
        current_wc_year:  Year identifying "the current World Cup" (default 2026).
        return_history:   If True, also return a list of
                          (elo_diff_before_match, home_goals, away_goals)
                          for every match in chronological order. Used by
                          fit_dc_params() in match_model.py.

    Returns:
        dict[team → rating]  (if return_history=False, the default)
        (dict, list)         (if return_history=True)
    """
    sorted_m = matches.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = defaultdict(lambda: base)
    history: list[tuple[float, int, int]] = []
    last_year: int | None = None

    recent_cutoff = sorted_m["date"].max() - pd.Timedelta(days=recent_days)

    for row in sorted_m.itertuples(index=False):
        cur_year: int = row.date.year

        if reversion > 0 and last_year is not None and cur_year > last_year:
            for team in list(ratings.keys()):
                ratings[team] += reversion * (base - ratings[team])
        last_year = cur_year

        ra: float = ratings[row.home_team]
        rb: float = ratings[row.away_team]
        adv: float = 0.0 if row.neutral else home_adv
        elo_diff: float = (ra + adv) - rb

        if return_history:
            history.append((elo_diff, int(row.home_score), int(row.away_score)))

        k_eff = tournament_k(str(row.tournament), k) if use_tournament_k else k

        if row.date >= recent_cutoff:
            k_eff *= recent_weight

        if cur_year == current_wc_year and _WC_RE.search(str(row.tournament)):
            k_eff *= current_wc_boost

        exp_home = expected_score(ra + adv, rb)
        if row.home_score > row.away_score:
            actual = 1.0
        elif row.home_score == row.away_score:
            actual = 0.5
        else:
            actual = 0.0

        delta = k_eff * (actual - exp_home)
        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta

    ratings_dict = dict(ratings)
    if return_history:
        return ratings_dict, history
    return ratings_dict


def top_n(ratings: dict[str, float], n: int = 20) -> pd.DataFrame:
    s = pd.Series(ratings).sort_values(ascending=False).head(n)
    return s.rename("elo").rename_axis("team").reset_index()


if __name__ == "__main__":
    from src.data.historical import load_results
    r = compute_elo(load_results())
    print(top_n(r, 15).to_string(index=False))
