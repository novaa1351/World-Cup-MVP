"""Match-outcome probabilities using a Dixon-Coles bivariate-Poisson model.

Dixon & Coles (1997) treat home and away goals as near-independent Poisson
random variables with rates λ and μ, then apply a low-score correlation
correction ρ to the four scorelines (0,0), (1,0), (0,1), (1,1) that football
systematically over-produces relative to independent Poisson.

This implementation maps Elo ratings to goal rates via two calibrated global
parameters (μ₀, γ) fitted by maximum-likelihood on competitive historical
matches:

    λ (home team expected goals) = exp(μ₀ + γ · Δelo)
    μ (away team expected goals) = exp(μ₀ − γ · Δelo)

where Δelo = elo_home − elo_away (neutral ground) or
      Δelo = elo_home + ELO_HOME_ADV − elo_away (home soil).

The three parameters (μ₀, γ, ρ) are fitted by MLE and saved to
config.DC_PARAMS_PATH.  If that file does not exist at import time the module
falls back to the original exponential draw-rate model so that the rest of the
pipeline keeps working without re-fitting.

Public interface (unchanged)
----------------------------
match_probs(elo_home, elo_away, neutral=True, ...) → {'home', 'draw', 'away'}
win_prob_knockout(elo_a, elo_b)                   → float

Fitting
-------
Run once (takes ~20–40 s on a full 50 k-match history):
    python src/models/match_model.py

References
----------
Dixon, M. J. & Coles, S. G. (1997). "Modelling association football scores
    and inefficiencies in the football betting market." Applied Statistics
    46(2) 265-280.
Hvattum, L. M. & Arntzen, H. (2010). "Using ELO ratings for match result
    prediction in association football." JQAS 6(1).
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import config
from src.models.elo import compute_elo, expected_score


# ---------------------------------------------------------------------------
# DC parameter I/O  (loaded once per process, cached at module level)
# ---------------------------------------------------------------------------

_DC_PARAMS: dict | None = None


def _load_dc_params() -> dict | None:
    """Load fitted DC parameters from disk; cache after first call."""
    global _DC_PARAMS
    if _DC_PARAMS is not None:
        return _DC_PARAMS
    path = config.DC_PARAMS_PATH
    if path.exists():
        with open(path) as f:
            _DC_PARAMS = json.load(f)
    return _DC_PARAMS


def dc_params_available() -> bool:
    """Return True if fitted Dixon-Coles parameters are present on disk."""
    return config.DC_PARAMS_PATH.exists()


# ---------------------------------------------------------------------------
# Dixon-Coles low-score correction
# ---------------------------------------------------------------------------

def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score correlation correction τ(x, y, λ, μ, ρ) — DC (1997) eq. (1).

    Applied only to scorelines where x ≤ 1 AND y ≤ 1; equals 1 elsewhere.
    rho < 0 increases P(0,0) relative to independent Poisson (the typical
    football case where 0-0 draws are more common than the model would predict).
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _dc_joint_matrix(
    lam: float, mu: float, rho: float, max_goals: int = 10
) -> np.ndarray:
    """P[i, j] = DC-corrected joint probability P(home scores i, away scores j).

    Sums over all i ∈ [0..max_goals], j ∈ [0..max_goals] and renormalises so
    the mass outside this window does not distort the three outcome shares.
    """
    from scipy.stats import poisson

    i = np.arange(max_goals + 1)
    j = np.arange(max_goals + 1)

    # Independent Poisson joint probability matrix
    P = np.outer(poisson.pmf(i, lam), poisson.pmf(j, mu))

    # Apply DC correction to the four low-score cells
    P[0, 0] *= max(0.0, 1.0 - lam * mu * rho)   # clip avoids negative prob
    P[1, 0] *= max(0.0, 1.0 + mu * rho)
    P[0, 1] *= max(0.0, 1.0 + lam * rho)
    P[1, 1] *= max(0.0, 1.0 - rho)

    total = P.sum()
    return P / total if total > 0 else P


def _dc_outcome_probs(lam: float, mu: float, rho: float) -> dict[str, float]:
    """Aggregate joint goal matrix into {home, draw, away} probabilities."""
    P = _dc_joint_matrix(lam, mu, rho)
    home = float(np.sum(np.tril(P, k=-1)))   # i > j: home scores more
    draw = float(np.sum(np.diag(P)))          # i = j
    away = float(np.sum(np.triu(P, k=1)))     # i < j: away scores more
    # Re-normalise for floating-point safety
    total = home + draw + away
    return {"home": home / total, "draw": draw / total, "away": away / total}


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------

def _dc_loglik_match(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Log P(home=x, away=y) under the Dixon-Coles model."""
    from math import lgamma
    tau = _tau(x, y, lam, mu, rho)
    if tau <= 0 or lam <= 0 or mu <= 0:
        return -1e10
    return (
        math.log(tau)
        + x * math.log(lam) - lam - lgamma(x + 1)
        + y * math.log(mu)  - mu  - lgamma(y + 1)
    )


def _neg_loglik(
    params: list[float],
    records: list[tuple[float, int, int]],
) -> float:
    """Negative total log-likelihood over all (elo_diff, home_goals, away_goals)."""
    mu_0, gamma, rho = params
    ll = 0.0
    for elo_diff, x, y in records:
        lam = math.exp(mu_0 + gamma * elo_diff)
        mu  = math.exp(mu_0 - gamma * elo_diff)
        ll += _dc_loglik_match(x, y, lam, mu, rho)
    return -ll


def fit_dc_params(matches: "pd.DataFrame") -> dict:
    """Fit (μ₀, γ, ρ) by MLE on competitive historical match data.

    Uses Elo differences computed incrementally (pre-match ratings) to avoid
    any lookahead bias. Filters to non-friendly matches from 1990 onwards,
    where both goal data and Elo ratings are most reliable for modern
    international football.

    Saves the result to config.DC_PARAMS_PATH as JSON and returns the dict.

    Parameters
    ----------
    matches : Full historical DataFrame from historical.load_results().

    Returns
    -------
    dict with keys: mu_0, gamma, rho, n_matches, avg_goals_per_team
    """
    import pandas as pd
    from scipy.optimize import minimize

    print(f"  Computing incremental Elo history over {len(matches):,} matches…")
    _, history = compute_elo(matches, return_history=True)
    # history[i] = (elo_diff_before_match_i, home_goals, away_goals)
    # aligned row-for-row with matches.sort_values("date")

    sorted_m = matches.sort_values("date").reset_index(drop=True)

    records: list[tuple[float, int, int]] = []
    for i, row in enumerate(sorted_m.itertuples(index=False)):
        if row.date.year < 1990:
            continue
        if "friendly" in str(row.tournament).lower():
            continue
        elo_diff, hg, ag = history[i]
        records.append((float(elo_diff), int(hg), int(ag)))

    print(f"  Fitting DC model on {len(records):,} competitive matches (1990+)…")
    avg_goals = sum(hg + ag for _, hg, ag in records) / (2 * len(records))

    # Initial guess: average goals ≈ 1.25/team, neutral equal teams
    x0 = [math.log(avg_goals), 0.0008, -0.10]
    bounds = [
        (math.log(0.5), math.log(3.5)),  # mu_0: avg goals 0.5–3.5 per team
        (0.0, 0.005),                     # gamma: Elo sensitivity (positive only)
        (-0.5, 0.3),                      # rho: low-score correlation
    ]

    result = minimize(
        _neg_loglik,
        x0=x0,
        args=(records,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-9},
    )

    if not result.success:
        print(f"  Warning: optimizer did not converge — {result.message}")

    mu_0, gamma, rho = result.x.tolist()
    params = {
        "mu_0":               mu_0,
        "gamma":              gamma,
        "rho":                rho,
        "n_matches":          len(records),
        "avg_goals_per_team": avg_goals,
    }

    config.DC_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DC_PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved to {config.DC_PARAMS_PATH}")

    # Invalidate the in-process cache so next match_probs call uses new params
    global _DC_PARAMS
    _DC_PARAMS = params

    return params


# ---------------------------------------------------------------------------
# Public match-probability interface (same signature as the original)
# ---------------------------------------------------------------------------

def match_probs(
    elo_home: float,
    elo_away: float,
    neutral: bool = True,
    home_adv: float = config.ELO_HOME_ADV,
    draw_base: float = 0.28,   # unused when DC params are loaded; kept for compat
) -> dict[str, float]:
    """Return {'home', 'draw', 'away'} probabilities summing to 1.

    When Dixon-Coles parameters are available (config.DC_PARAMS_PATH exists),
    uses the calibrated bivariate-Poisson model. Otherwise falls back to the
    simple exponential draw-rate approximation.

    Args:
        elo_home: Pre-match Elo rating for the home/first team.
        elo_away: Pre-match Elo rating for the away/second team.
        neutral:  True if the match is played at a neutral venue.
        home_adv: Elo-point boost for the home team when neutral=False.
        draw_base: Ignored when DC is active (draws emerge from the model).
                   Kept for backward compatibility.
    """
    adv = 0.0 if neutral else home_adv
    elo_diff = (elo_home + adv) - elo_away

    params = _load_dc_params()
    if params is not None:
        lam = math.exp(params["mu_0"] + params["gamma"] * elo_diff)
        mu  = math.exp(params["mu_0"] - params["gamma"] * elo_diff)
        return _dc_outcome_probs(lam, mu, params["rho"])

    # --- Fallback: original exponential draw-rate model ---------------------
    p_home_2way = expected_score(elo_home + adv, elo_away)
    gap = abs(elo_diff)
    draw = draw_base * math.exp(-gap / 400.0)
    home = (1 - draw) * p_home_2way
    away = (1 - draw) * (1 - p_home_2way)
    total = home + draw + away
    return {"home": home / total, "draw": draw / total, "away": away / total}


def win_prob_knockout(elo_a: float, elo_b: float) -> float:
    """P(A beats B) in a knockout tie; draws resolved by ET/penalties ≈ coin."""
    p = match_probs(elo_a, elo_b)
    return p["home"] + 0.5 * p["draw"]


# ---------------------------------------------------------------------------
# CLI — fit and demonstrate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.historical import load_results

    print("Loading match history…")
    matches = load_results()

    print("\nFitting Dixon-Coles parameters…")
    params = fit_dc_params(matches)

    avg_g = math.exp(params["mu_0"])
    print(f"\n  μ₀   = {params['mu_0']:.4f}  → avg goals/team/match = {avg_g:.3f}")
    print(f"  γ    = {params['gamma']:.6f}  (Elo→goal-rate sensitivity)")
    print(f"  ρ    = {params['rho']:.4f}  (low-score correlation; <0 = more 0-0 than Poisson)")
    print(f"  n    = {params['n_matches']:,} competitive matches used for fitting")

    print("\nSample predictions (DC model):")
    print(f"  Equal teams, neutral:        {match_probs(1500, 1500)}")
    print(f"  +200 Elo favourite, neutral: {match_probs(1600, 1400)}")
    print(f"  +400 Elo favourite, neutral: {match_probs(1700, 1300)}")
    print(f"\n  Knockout P(+200 fav advances): {win_prob_knockout(1600, 1400):.3f}")
