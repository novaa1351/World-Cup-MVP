"""Match-outcome probabilities from Elo ratings.

Model
-----
P(home win), P(draw), P(away win) are derived from two components:

  1. Elo expected score — the standard logistic win probability for the
     home team, treating draws as half-wins (the classic Elo formula).

  2. Draw probability — a calibrated exponential decay in the Elo gap:

         P(draw | Δelo) = draw_base · exp(−|Δelo| / scale)

     where draw_base is the draw probability when teams are evenly matched,
     and scale controls how quickly that probability falls as the gap widens.
     The remaining (1 − P(draw)) mass is split by the 2-way Elo expectation.

Both draw_base and scale are fitted by maximum-likelihood on competitive
historical matches (no friendlies, 1990+). This replaces the earlier
hardcoded draw_base = 0.28 with a data-driven estimate.

Fitting
-------
Run once (fast — ~5 s):
    python src/models/match_model.py

Parameters are saved to config.DRAW_PARAMS_PATH and loaded at import time.
If that file does not exist the module falls back to draw_base=0.28, scale=400.

Public interface
----------------
match_probs(elo_home, elo_away, neutral=True, ...) → {'home', 'draw', 'away'}
win_prob_knockout(elo_a, elo_b)                   → float
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.models.elo import compute_elo, expected_score


# ---------------------------------------------------------------------------
# Draw-model parameter I/O  (loaded once per process, cached at module level)
# ---------------------------------------------------------------------------

_DRAW_PARAMS: dict | None = None

# Hard defaults used when no fitted file is present
_DEFAULT_DRAW_BASE = 0.28
_DEFAULT_SCALE = 400.0


def _load_draw_params() -> dict | None:
    """Load calibrated draw parameters from disk; cache after first call."""
    global _DRAW_PARAMS
    if _DRAW_PARAMS is not None:
        return _DRAW_PARAMS
    path = config.DRAW_PARAMS_PATH
    if path.exists():
        with open(path) as f:
            _DRAW_PARAMS = json.load(f)
    return _DRAW_PARAMS


def draw_params_available() -> bool:
    """Return True if calibrated draw parameters exist on disk."""
    return config.DRAW_PARAMS_PATH.exists()


# ---------------------------------------------------------------------------
# Core three-way probability calculation
# ---------------------------------------------------------------------------

def _three_way_probs(
    elo_diff: float,
    draw_base: float,
    scale: float,
) -> tuple[float, float, float]:
    """Return (P_home, P_draw, P_away) given a home-adjusted Elo difference.

    The draw probability peaks at draw_base when teams are equal and decays
    exponentially as the rating gap grows. Remaining probability is split
    proportionally to the 2-way Elo expectation.
    """
    # Logistic 2-way win probability (ignores draws)
    p_2way = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    # Draw falls off with the absolute Elo gap
    draw = draw_base * math.exp(-abs(elo_diff) / scale)
    home = (1.0 - draw) * p_2way
    away = (1.0 - draw) * (1.0 - p_2way)
    # Normalise for floating-point safety
    total = home + draw + away
    return home / total, draw / total, away / total


# ---------------------------------------------------------------------------
# MLE parameter fitting
# ---------------------------------------------------------------------------

def _neg_loglik(
    params: list[float],
    records: list[tuple[float, int]],
) -> float:
    """Negative log-likelihood of observed outcomes under the draw model.

    records: list of (elo_diff, outcome) where outcome ∈ {0=home, 1=draw, 2=away}.
    """
    draw_base, scale = params
    if draw_base <= 0.0 or draw_base >= 1.0 or scale <= 0.0:
        return 1e12
    ll = 0.0
    for elo_diff, outcome in records:
        ph, pd_, pa = _three_way_probs(elo_diff, draw_base, scale)
        p = (ph, pd_, pa)[outcome]
        if p <= 0.0:
            return 1e12
        ll += math.log(p)
    return -ll


def fit_draw_params(matches: "pd.DataFrame") -> dict:
    """Fit (draw_base, scale) by MLE on competitive historical match data.

    Uses Elo differences computed incrementally (pre-match ratings) to avoid
    any lookahead bias. Filters to non-friendly matches from 1990 onwards.

    Saves the result to config.DRAW_PARAMS_PATH as JSON and returns the dict.

    Parameters
    ----------
    matches : Full historical DataFrame from historical.load_results().

    Returns
    -------
    dict with keys: draw_base, scale, n_matches
    """
    import pandas as pd
    from scipy.optimize import minimize

    print(f"  Computing incremental Elo history over {len(matches):,} matches…")
    _, history = compute_elo(matches, return_history=True)

    sorted_m = matches.sort_values("date").reset_index(drop=True)

    records: list[tuple[float, int]] = []
    for i, row in enumerate(sorted_m.itertuples(index=False)):
        if row.date.year < 1990:
            continue
        if "friendly" in str(row.tournament).lower():
            continue
        elo_diff, hg, ag = history[i]
        hg, ag = int(hg), int(ag)
        if hg > ag:
            outcome = 0   # home win
        elif hg == ag:
            outcome = 1   # draw
        else:
            outcome = 2   # away win
        records.append((float(elo_diff), outcome))

    draw_rate = sum(1 for _, o in records if o == 1) / len(records)
    print(f"  Fitting draw model on {len(records):,} competitive matches (1990+)…")
    print(f"  Observed draw rate: {draw_rate:.3f}")

    result = minimize(
        _neg_loglik,
        x0=[0.28, 400.0],
        args=(records,),
        method="L-BFGS-B",
        bounds=[(0.05, 0.60), (100.0, 2000.0)],
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not result.success:
        print(f"  Warning: optimizer did not converge — {result.message}")

    draw_base, scale = result.x.tolist()
    params = {
        "draw_base":  draw_base,
        "scale":      scale,
        "n_matches":  len(records),
    }

    config.DRAW_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DRAW_PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved to {config.DRAW_PARAMS_PATH}")

    # Invalidate in-process cache
    global _DRAW_PARAMS
    _DRAW_PARAMS = params

    return params


# ---------------------------------------------------------------------------
# Public match-probability interface
# ---------------------------------------------------------------------------

def match_probs(
    elo_home: float,
    elo_away: float,
    neutral: bool = True,
    home_adv: float = config.ELO_HOME_ADV,
    draw_base: float | None = None,
    scale: float | None = None,
) -> dict[str, float]:
    """Return {'home', 'draw', 'away'} probabilities summing to 1.

    Resolution order for draw_base / scale:
      1. Explicit argument (not None) — used directly.
      2. Fitted file at config.DRAW_PARAMS_PATH — loaded once per process.
      3. Hard defaults: draw_base=0.28, scale=400.

    Args:
        elo_home:  Pre-match Elo rating of the home/first team.
        elo_away:  Pre-match Elo rating of the away/second team.
        neutral:   True if played at a neutral venue.
        home_adv:  Elo-point boost for the home team when neutral=False.
        draw_base: Override for the equal-strength draw probability.
        scale:     Override for the Elo-gap decay constant.
    """
    adv = 0.0 if neutral else home_adv
    elo_diff = (elo_home + adv) - elo_away

    # Resolve parameters: explicit arg > fitted file > hard default
    params = _load_draw_params()
    _db = draw_base if draw_base is not None else (
        params["draw_base"] if params else _DEFAULT_DRAW_BASE)
    _sc = scale if scale is not None else (
        params["scale"] if params else _DEFAULT_SCALE)

    ph, pd_, pa = _three_way_probs(elo_diff, _db, _sc)
    return {"home": ph, "draw": pd_, "away": pa}


def win_prob_knockout(
    elo_a: float,
    elo_b: float,
    draw_base: float | None = None,
    scale: float | None = None,
) -> float:
    """P(A beats B) in a knockout tie; draws resolved by ET/penalties ≈ coin."""
    p = match_probs(elo_a, elo_b, draw_base=draw_base, scale=scale)
    return p["home"] + 0.5 * p["draw"]


# ---------------------------------------------------------------------------
# CLI — fit and demonstrate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.historical import load_results

    print("Loading match history…")
    matches = load_results()

    print("\nFitting draw model parameters…")
    params = fit_draw_params(matches)

    print(f"\n  draw_base = {params['draw_base']:.4f}  "
          f"(P(draw) at equal strength on neutral ground)")
    print(f"  scale     = {params['scale']:.1f}  "
          f"(Elo points for draw prob to fall to draw_base/e ≈ {params['draw_base']/math.e:.3f})")
    print(f"  n_matches = {params['n_matches']:,}  competitive matches used")

    print("\nSample predictions (calibrated draw model):")
    for diff in [0, 50, 100, 200]:
        p = match_probs(1500 + diff // 2, 1500 - diff // 2)
        print(f"  Δelo={diff:+4d}: home={p['home']:.3f}  draw={p['draw']:.3f}  away={p['away']:.3f}")
    print(f"\n  Knockout P(+200-Elo fav advances): "
          f"{win_prob_knockout(1600, 1400):.3f}")
