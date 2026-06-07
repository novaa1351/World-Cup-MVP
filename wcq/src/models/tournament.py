"""Monte Carlo simulation of the 2026 FIFA World Cup tournament.

Format: 48 teams in 12 groups (A–L) of 4.
  - Group stage: full round-robin (6 matches per group)
  - 3 pts / win, 1 pt / draw, 0 pts / loss; tiebreak: GD → GF → random
  - Top 2 from each group advance directly (24 teams)
  - 8 best third-placed teams also advance → 32 total enter the knockout
  - Knockout: R32 (16 matches) → R16 → QF → SF → Final → Champion

Output: dict[team -> dict[round -> float]]

Probability semantics (each is cumulative, unconditional):
  "group"    : P(advance from the group stage)
  "R32"      : P(win the R32 match)
  "R16"      : P(win the R16 match)
  "QF"       : P(win the QF match)
  "SF"       : P(win the SF match — equivalently, play in the Final)
  "final"    : P(play in the Final)  [equals "SF" by construction]
  "champion" : P(win the Final)

"SF" and "final" carry the same number so the SVI-surface has coverage
at every point on the round-depth axis without a redundant unique level.
These probabilities are the anchor inputs for SurvivalSurface.fit().
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from itertools import combinations
from typing import NamedTuple

import numpy as np
import pandas as pd

import config
from src.models.match_model import match_probs, win_prob_knockout

# ---------------------------------------------------------------------------
# 2026 World Cup bracket — official draw held 5 Dec 2025 in Washington D.C.
# ---------------------------------------------------------------------------

# FIFA official team names as used in the draw.
GROUPS_2026: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia-Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Prediction-market platforms (Polymarket, Kalshi) use their own English spellings
# that differ from the FIFA draw names stored as keys in GROUPS_2026 / mc_survival.
# Used by the streamlit edge tab to align market team names with MC survival keys.
MARKET_TO_FIFA: dict[str, str] = {
    # --- Polymarket ---
    "USA":                   "United States",
    "Ivory Coast":           "Côte d'Ivoire",
    "Cape Verde":            "Cabo Verde",
    "Turkiye":               "Türkiye",        # Polymarket drops the umlaut
    # --- Kalshi ---
    "Korea Republic":        "South Korea",    # FIFA/Kalshi official name
    "IR Iran":               "Iran",
    "Bosnia and Herzegovina":"Bosnia-Herzegovina",
    "Curacao":               "Curaçao",        # Kalshi drops the cedilla
}

# Six FIFA names differ from the spelling used in martj42's historical results CSV.
# This map is applied inside simulate_tournament when looking up Elo ratings so
# that teams that appear in the draw under a modern FIFA name still get their
# historically-derived rating rather than falling back to ELO_BASE.
_FIFA_TO_HIST: dict[str, str] = {
    "Czechia":            "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Türkiye":            "Turkey",
    "Côte d'Ivoire":      "Ivory Coast",
    "Cabo Verde":         "Cape Verde",
    "Congo DR":           "DR Congo",
}

# R32 bracket — 16 slot-pairs describing which group positions meet.
# W_X = winner of group X, R_X = runner-up, T_i = i-th best third (ranked by pts/GD/GF).
# After R32, winners advance through the bracket in sequential pairs (0&1, 2&3, ...).
_R32_SLOTS: list[tuple[str, str]] = [
    ("W_A", "R_B"), ("W_C", "R_D"), ("W_E", "R_F"), ("W_G", "R_H"),
    ("W_I", "R_J"), ("W_K", "R_L"), ("W_B", "R_A"), ("W_D", "R_C"),
    ("W_F", "R_E"), ("W_H", "R_G"), ("W_J", "R_I"), ("W_L", "R_K"),
    # 4 matches for the 8 best third-place qualifiers
    ("T_1", "T_2"), ("T_3", "T_4"), ("T_5", "T_6"), ("T_7", "T_8"),
]


class _GroupRecord(NamedTuple):
    team: str
    pts: int
    gd: int
    gf: int
    group: str


def _sample_scoreline(
    p_win: float, p_draw: float, rng: np.random.Generator
) -> tuple[int, int]:
    """Sample (goals_home, goals_away) from the three-way outcome distribution."""
    u = rng.random()
    if u < p_win:
        gh = int(rng.integers(1, 4))
        ga = int(rng.integers(0, gh))
    elif u < p_win + p_draw:
        g = int(rng.integers(0, 3))
        gh = ga = g
    else:
        ga = int(rng.integers(1, 4))
        gh = int(rng.integers(0, ga))
    return gh, ga


def _simulate_group(
    teams: list[str],
    precomp: dict[tuple[str, str], dict[str, float]],
    rng: np.random.Generator,
    group_id: str,
) -> list[_GroupRecord]:
    """Simulate one 4-team round-robin. Returns records sorted 1st → 4th."""
    pts = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}

    for home, away in combinations(teams, 2):
        probs = precomp[(home, away)]
        gh, ga = _sample_scoreline(probs["home"], probs["draw"], rng)
        gf[home] += gh
        gf[away] += ga
        gd[home] += gh - ga
        gd[away] += ga - gh
        if gh > ga:
            pts[home] += 3
        elif gh == ga:
            pts[home] += 1
            pts[away] += 1
        else:
            pts[away] += 3

    # Sort by pts → GD → GF → random tiebreak (covers the very rare exact tie)
    ranking = sorted(
        teams,
        key=lambda t: (pts[t], gd[t], gf[t], rng.random()),
        reverse=True,
    )
    return [_GroupRecord(team=t, pts=pts[t], gd=gd[t], gf=gf[t], group=group_id)
            for t in ranking]


def _select_best_thirds(thirds: list[_GroupRecord], n: int = 8) -> list[_GroupRecord]:
    """Pick the n best third-placed teams by pts → GD → GF."""
    return sorted(thirds, key=lambda r: (r.pts, r.gd, r.gf), reverse=True)[:n]


def _run_knockout_round(
    pairs: list[tuple[str, str]],
    p_ko: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> list[str]:
    """Simulate one knockout round; return the list of winners in bracket order."""
    winners = []
    for a, b in pairs:
        winners.append(a if rng.random() < p_ko[a][b] else b)
    return winners


def _sequential_pairs(teams: list[str]) -> list[tuple[str, str]]:
    return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]


def _elo(team: str, ratings: dict[str, float], base: float) -> float:
    """Return the Elo rating for `team`, normalising FIFA draw names to the
    spellings used in the martj42 historical dataset before falling back to `base`."""
    hist_name = _FIFA_TO_HIST.get(team, team)
    return ratings.get(hist_name, base)


def simulate_tournament(
    elo: dict[str, float],
    groups: dict[str, list[str]] | None = None,
    n_sims: int = config.MC_SIMS,
    seed: int | None = None,
    draw_base: float | None = None,
    scale: float | None = None,
) -> dict[str, dict[str, float]]:
    """Run a Monte Carlo bracket simulation of the 2026 World Cup.

    Args:
        elo:       Team Elo ratings (output of elo.compute_elo). Teams absent
                   from this dict fall back to config.ELO_BASE.
        groups:    Override the default GROUPS_2026 bracket.
        n_sims:    Number of Monte Carlo iterations (default config.MC_SIMS).
        seed:      RNG seed for reproducibility.
        draw_base: Draw model override — passed through to match_probs().
                   None = use calibrated file or hard default.
        scale:     Elo-gap decay constant override. None = same resolution.

    Returns:
        dict[team -> dict[round -> float]] — survival probability at each
        round in config.ROUNDS for every team in the bracket.
    """
    if groups is None:
        groups = GROUPS_2026

    base = config.ELO_BASE
    rng = np.random.default_rng(seed)
    all_teams = [t for g in groups.values() for t in g]

    # ------------------------------------------------------------------
    # Precompute match probabilities once (same ELOs every sim).
    # Group matches use the 3-way model; knockout uses the 2-way with
    # draw resolved by extra time / penalties (coin-flip).
    # ------------------------------------------------------------------
    group_probs: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for gid, teams in groups.items():
        group_probs[gid] = {
            (h, a): match_probs(_elo(h, elo, base), _elo(a, elo, base),
                                draw_base=draw_base, scale=scale)
            for h, a in combinations(teams, 2)
        }

    # Pairwise knockout win-probabilities (a beats b in a ko tie)
    p_ko: dict[str, dict[str, float]] = {a: {} for a in all_teams}
    for a in all_teams:
        for b in all_teams:
            if a != b:
                p_ko[a][b] = win_prob_knockout(_elo(a, elo, base), _elo(b, elo, base),
                                               draw_base=draw_base, scale=scale)

    counts: dict[str, dict[str, int]] = {
        t: {r: 0 for r in config.ROUNDS} for t in all_teams
    }

    for _ in range(n_sims):
        # ---- Group stage ------------------------------------------------
        group_winners: dict[str, str] = {}
        group_runners: dict[str, str] = {}
        thirds: list[_GroupRecord] = []

        for gid, teams in groups.items():
            records = _simulate_group(teams, group_probs[gid], rng, gid)
            group_winners[gid] = records[0].team
            group_runners[gid] = records[1].team
            thirds.append(records[2])

        best8 = _select_best_thirds(thirds, n=8)
        third_teams = [r.team for r in best8]

        qualifiers = (set(group_winners.values())
                      | set(group_runners.values())
                      | set(third_teams))
        for t in qualifiers:
            counts[t]["group"] += 1

        # ---- Build R32 slot map -----------------------------------------
        slot: dict[str, str] = {}
        for gid in groups:
            slot[f"W_{gid}"] = group_winners[gid]
            slot[f"R_{gid}"] = group_runners[gid]
        for i, t in enumerate(third_teams, start=1):
            slot[f"T_{i}"] = t

        r32_pairs = [(slot[s1], slot[s2]) for s1, s2 in _R32_SLOTS]

        # ---- Knockout rounds --------------------------------------------
        r32w = _run_knockout_round(r32_pairs, p_ko, rng)
        for t in r32w:
            counts[t]["R32"] += 1

        r16w = _run_knockout_round(_sequential_pairs(r32w), p_ko, rng)
        for t in r16w:
            counts[t]["R16"] += 1

        qfw = _run_knockout_round(_sequential_pairs(r16w), p_ko, rng)
        for t in qfw:
            counts[t]["QF"] += 1

        sfw = _run_knockout_round(_sequential_pairs(qfw), p_ko, rng)
        for t in sfw:
            counts[t]["SF"] += 1
            counts[t]["final"] += 1  # won SF = plays in the Final

        champion = _run_knockout_round([tuple(sfw)], p_ko, rng)[0]  # type: ignore[arg-type]
        counts[champion]["champion"] += 1

    return {
        t: {r: counts[t][r] / n_sims for r in config.ROUNDS}
        for t in all_teams
    }


def survival_table(
    survival: dict[str, dict[str, float]],
    sort_by: str = "champion",
    top_n: int | None = None,
) -> pd.DataFrame:
    """Convert the survival dict to a sorted DataFrame for display.

    Args:
        survival: Output of simulate_tournament.
        sort_by:  Round column to sort by (default: champion odds).
        top_n:    Return only the top-n teams by `sort_by` (None = all).
    """
    rows = [{"team": t, **probs} for t, probs in survival.items()]
    df = pd.DataFrame(rows).set_index("team")[config.ROUNDS]
    df = df.sort_values(sort_by, ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    return df


if __name__ == "__main__":
    from src.data.historical import download_results
    from src.models.elo import compute_elo

    print("Loading Elo ratings...")
    elo_ratings = compute_elo(download_results())

    print(f"Running {config.MC_SIMS:,} Monte Carlo simulations...")
    survival = simulate_tournament(elo_ratings, seed=42)

    print("\nTop-16 champion probabilities:")
    print(survival_table(survival, sort_by="champion", top_n=16).to_string(float_format="{:.3f}".format))
