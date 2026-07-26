"""Backtest harness for the model-vs-market strategy.

Goal: given a historical World Cup whose outcomes are now known, rebuild the
Elo ratings from only the data available BEFORE that tournament started, run
the match-probability model on every WC game, then measure how well the
predictions calibrate against real results.

Two independent trials:
  - 2018 FIFA World Cup (Russia): Elo trained on all matches < 2018-06-14
  - 2022 FIFA World Cup (Qatar):  Elo trained on all matches < 2022-11-20

Because we strip all data from the WC start date onwards, there is zero
lookahead bias — the model only knew what was public knowledge the morning
the tournament kicked off.

For each WC match we generate three rows — one per possible outcome (home
win, draw, away win) — so the full 3-way model can be evaluated fairly:
  model_prob   : our draw-adjusted probability for that specific outcome
  market_prob  : 1/3 flat (equal-odds baseline, stands in for absent archival
                 market data; swap in real odds if you have them)
  outcome      : 1 if that outcome actually occurred, else 0

Brier score measures calibration (lower = better).
The staking simulation bets fractional-Kelly whenever model_prob exceeds the
flat 1/3 baseline by at least `edge_threshold`.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import config
from src.models.elo import compute_elo, expected_score
from src.models.match_model import match_probs
from src.markets.edges import kelly_fraction

# ---------------------------------------------------------------------------
# World Cup date windows
# ---------------------------------------------------------------------------
WC_CUTOFFS: dict[int, tuple[str, str]] = {
    2002: ("2002-05-31", "2002-06-30"),
    2006: ("2006-06-09", "2006-07-09"),
    2010: ("2010-06-11", "2010-07-11"),
    2014: ("2014-06-12", "2014-07-13"),
    2018: ("2018-06-14", "2018-07-15"),
    2022: ("2022-11-20", "2022-12-18"),
}


def brier_score(p_model: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts (lower = better)."""
    return float(np.mean((p_model - outcome) ** 2))


def build_wc_backtest(
    year: int,
    all_matches: pd.DataFrame,
    elo_reversion: float = config.ELO_MEAN_REVERSION,
    use_tournament_k: bool = True,
    draw_base: float | None = None,
    scale: float | None = None,
) -> pd.DataFrame:
    """Build a per-outcome backtest DataFrame for a historical World Cup.

    Args:
        year:             2018 or 2022.
        all_matches:      Full historical match DataFrame from load_results().
                          Matches from the WC start date onwards are excluded
                          from Elo training — no lookahead bias.
        elo_reversion:    Annual mean-reversion rate for Elo (0.0 = off).
        use_tournament_k: Apply tournament tier K-scaling (eloratings.net).
        draw_base:        Draw model override passed through to match_probs().
        scale:            Elo-gap decay constant override.

    Returns:
        DataFrame with one row per (match × outcome) and columns:
          date, home, away, outcome_label, model_prob, market_prob, outcome,
          home_score, away_score, elo_home (pre-WC), elo_away (pre-WC),
          baseline_prob (raw 2-way Elo expected score, for reference).
    """
    if year not in WC_CUTOFFS:
        raise ValueError(f"year must be one of {list(WC_CUTOFFS)}")

    start, end = WC_CUTOFFS[year]
    cutoff = pd.Timestamp(start)

    # --- Elo trained strictly before the WC started -----------------------
    train = all_matches[all_matches["date"] < cutoff]
    elo = compute_elo(train, reversion=elo_reversion, use_tournament_k=use_tournament_k)
    base = config.ELO_BASE

    # --- Pull actual WC match rows ----------------------------------------
    wc = all_matches[
        (all_matches["date"] >= cutoff)
        & (all_matches["date"] <= end)
        & (all_matches["tournament"] == "FIFA World Cup")
    ].copy()

    if wc.empty:
        raise RuntimeError(f"No FIFA World Cup matches found for {year}.")

    rows = []
    for row in wc.itertuples(index=False):
        r_home = elo.get(row.home_team, base)
        r_away = elo.get(row.away_team, base)
        probs = match_probs(r_home, r_away, neutral=bool(row.neutral),
                            draw_base=draw_base, scale=scale)
        baseline = expected_score(r_home, r_away)  # raw 2-way Elo, ignores draws

        actual_home = int(row.home_score)
        actual_away = int(row.away_score)

        outcomes = {
            "home_win": (probs["home"], 1 if actual_home > actual_away else 0),
            "draw":     (probs["draw"], 1 if actual_home == actual_away else 0),
            "away_win": (probs["away"], 1 if actual_home < actual_away else 0),
        }
        for label, (mp, oc) in outcomes.items():
            rows.append({
                "date":         row.date,
                "home":         row.home_team,
                "away":         row.away_team,
                "outcome_label": label,
                "model_prob":   mp,
                "market_prob":  1 / 3,   # flat equal-odds baseline
                "outcome":      oc,
                "home_score":   actual_home,
                "away_score":   actual_away,
                "elo_home":     r_home,
                "elo_away":     r_away,
                "baseline_prob": baseline if label == "home_win" else (1 - baseline) / 2,
            })

    return pd.DataFrame(rows)

def build_wc_backtest_with_offset(
    year: int,
    all_matches: pd.DataFrame,
    elo_reversion: float = config.ELO_MEAN_REVERSION,
    use_tournament_k: bool = True,
    draw_base: float | None = None,
    scale: float | None = None,
    offset_min_year: int | None = None,
) -> pd.DataFrame:
    """Same as build_wc_backtest, but applies fitted confederation offsets.

    Offsets are fit using ONLY pre-cutoff data (same matches Elo itself is
    trained on) to avoid lookahead bias — this is a fair, apples-to-apples
    comparison against build_wc_backtest().

    Args:
        offset_min_year: passed through to fit_confederation_offsets' min_year.
                         None = fit on all available pre-cutoff history.
    """
    from src.models.confederations import CONFEDERATION, fit_confederation_offsets

    if year not in WC_CUTOFFS:
        raise ValueError(f"year must be one of {list(WC_CUTOFFS)}")

    start, end = WC_CUTOFFS[year]
    cutoff = pd.Timestamp(start)

    train = all_matches[all_matches["date"] < cutoff]
    elo = compute_elo(train, reversion=elo_reversion, use_tournament_k=use_tournament_k)
    base = config.ELO_BASE

    # Fit offsets using ONLY pre-cutoff data — no lookahead
    offsets = fit_confederation_offsets(train, elo, CONFEDERATION, min_year=offset_min_year)

    wc = all_matches[
        (all_matches["date"] >= cutoff)
        & (all_matches["date"] <= end)
        & (all_matches["tournament"] == "FIFA World Cup")
    ].copy()

    if wc.empty:
        raise RuntimeError(f"No FIFA World Cup matches found for {year}.")

    rows = []
    for row in wc.itertuples(index=False):
        conf_home = CONFEDERATION.get(row.home_team)
        conf_away = CONFEDERATION.get(row.away_team)
        off_home = offsets.get(conf_home, 0.0) if conf_home else 0.0
        off_away = offsets.get(conf_away, 0.0) if conf_away else 0.0

        r_home = elo.get(row.home_team, base) + off_home
        r_away = elo.get(row.away_team, base) + off_away

        probs = match_probs(r_home, r_away, neutral=bool(row.neutral),
                            draw_base=draw_base, scale=scale)
        baseline = expected_score(r_home, r_away)

        actual_home = int(row.home_score)
        actual_away = int(row.away_score)

        outcomes = {
            "home_win": (probs["home"], 1 if actual_home > actual_away else 0),
            "draw":     (probs["draw"], 1 if actual_home == actual_away else 0),
            "away_win": (probs["away"], 1 if actual_home < actual_away else 0),
        }
        for label, (mp, oc) in outcomes.items():
            rows.append({
                "date":         row.date,
                "home":         row.home_team,
                "away":         row.away_team,
                "outcome_label": label,
                "model_prob":   mp,
                "market_prob":  1 / 3,
                "outcome":      oc,
                "home_score":   actual_home,
                "away_score":   actual_away,
                "elo_home":     r_home,
                "elo_away":     r_away,
                "baseline_prob": baseline if label == "home_win" else (1 - baseline) / 2,
            })

    return pd.DataFrame(rows)

def build_wc_backtest_full(
    year: int,
    all_matches: pd.DataFrame,
    elo_reversion: float = config.ELO_MEAN_REVERSION,
    use_tournament_k: bool = True,
    draw_base: float | None = None,
    scale: float | None = None,
) -> pd.DataFrame:
    """Same as build_wc_backtest, but applies BOTH confederation offset AND
    in-tournament goal-difference form. Both are fit with no lookahead:
    confederation offset from pre-cutoff history, goal_diff_weight from this
    tournament's own matches (using only goals scored strictly before each
    match being predicted).
    """
    from src.models.confederations import CONFEDERATION, fit_confederation_offsets
    from src.models.tournament_form import tournament_goal_diff_so_far, fit_goal_diff_weight

    if year not in WC_CUTOFFS:
        raise ValueError(f"year must be one of {list(WC_CUTOFFS)}")

    start, end = WC_CUTOFFS[year]
    cutoff = pd.Timestamp(start)

    train = all_matches[all_matches["date"] < cutoff]
    elo = compute_elo(train, reversion=elo_reversion, use_tournament_k=use_tournament_k)
    base = config.ELO_BASE

    offsets = fit_confederation_offsets(train, elo, CONFEDERATION)

    wc = all_matches[
        (all_matches["date"] >= cutoff)
        & (all_matches["date"] <= end)
        & (all_matches["tournament"] == "FIFA World Cup")
    ].copy().sort_values("date").reset_index(drop=True)

    if wc.empty:
        raise RuntimeError(f"No FIFA World Cup matches found for {year}.")

    # First pass: build (home, away, scores, goal-diff-so-far) for every match,
    # so we can fit goal_diff_weight against the whole tournament at once.
    prep_rows = []
    for row in wc.itertuples(index=False):
        gd = tournament_goal_diff_so_far(wc, row.date)
        prep_rows.append({
            "home": row.home_team, "away": row.away_team,
            "home_score": int(row.home_score), "away_score": int(row.away_score),
            "neutral": bool(row.neutral),
            "gd_home": gd.get(row.home_team, 0),
            "gd_away": gd.get(row.away_team, 0),
        })

    confed_adjust = {}
    for team in elo:
        conf = CONFEDERATION.get(team)
        if conf:
            confed_adjust[team] = offsets.get(conf, 0.0)

    weight = fit_goal_diff_weight(prep_rows, elo, confed_adjust)

    # Second pass: build final backtest rows using fitted weight
    rows = []
    for m in prep_rows:
        r_home = elo.get(m["home"], base) + confed_adjust.get(m["home"], 0.0) + weight * m["gd_home"]
        r_away = elo.get(m["away"], base) + confed_adjust.get(m["away"], 0.0) + weight * m["gd_away"]

        probs = match_probs(r_home, r_away, neutral=m["neutral"],
                            draw_base=draw_base, scale=scale)
        baseline = expected_score(r_home, r_away)

        outcomes = {
            "home_win": (probs["home"], 1 if m["home_score"] > m["away_score"] else 0),
            "draw":     (probs["draw"], 1 if m["home_score"] == m["away_score"] else 0),
            "away_win": (probs["away"], 1 if m["home_score"] < m["away_score"] else 0),
        }
        for label, (mp, oc) in outcomes.items():
            rows.append({
                "date": None, "home": m["home"], "away": m["away"],
                "outcome_label": label, "model_prob": mp, "market_prob": 1 / 3,
                "outcome": oc, "home_score": m["home_score"], "away_score": m["away_score"],
                "elo_home": r_home, "elo_away": r_away,
                "baseline_prob": baseline if label == "home_win" else (1 - baseline) / 2,
            })

    return pd.DataFrame(rows)

def run_backtest(
    df: pd.DataFrame,
    bankroll: float = 1000.0,
    staking: str = "kelly",
    flat_stake: float = 10.0,
    edge_threshold: float = 0.03,
) -> dict:
    """`df` needs columns: model_prob, market_prob, outcome (1=yes / 0=no).

    Bets YES whenever model_prob - market_prob >= edge_threshold.
    Returns summary stats and the per-bet ledger DataFrame.
    """
    bank = bankroll
    ledger = []
    for row in df.itertuples(index=False):
        edge = row.model_prob - row.market_prob
        if edge < edge_threshold:
            continue
        if staking == "kelly":
            stake = kelly_fraction(row.model_prob, row.market_prob) * bank
        else:
            stake = flat_stake
        payout = (stake / row.market_prob) if row.outcome == 1 else 0.0
        pnl = payout - stake
        bank += pnl
        ledger.append({
            "outcome_label": getattr(row, "outcome_label", "—"),
            "home": getattr(row, "home", ""),
            "away": getattr(row, "away", ""),
            "edge": edge,
            "stake": stake,
            "pnl": pnl,
            "bankroll": bank,
        })

    led = pd.DataFrame(ledger)
    if led.empty:
        return {"n_bets": 0, "ledger": led}

    bet_rows = df[df["model_prob"] - df["market_prob"] >= edge_threshold]
    return {
        "n_bets":         len(led),
        "final_bankroll": round(bank, 2),
        "roi":            round(bank / bankroll - 1, 4),
        "hit_rate":       round(float(bet_rows["outcome"].mean()), 4),
        "brier_model":    round(brier_score(df["model_prob"].to_numpy(),
                                             df["outcome"].to_numpy()), 4),
        "brier_baseline": round(brier_score(df["baseline_prob"].to_numpy(),
                                             df["outcome"].to_numpy()), 4),
        "ledger":         led,
    }


def run_wc_backtest(
    year: int,
    all_matches: pd.DataFrame,
    elo_reversion: float = config.ELO_MEAN_REVERSION,
    use_tournament_k: bool = True,
    draw_base: float | None = None,
    scale: float | None = None,
    **kwargs,
) -> dict:
    """Convenience wrapper: build the dataset and run the backtest in one call.

    Returns the run_backtest() dict plus the raw `data` DataFrame and `year`.
    """
    data = build_wc_backtest(year, all_matches,
                             elo_reversion=elo_reversion,
                             use_tournament_k=use_tournament_k,
                             draw_base=draw_base,
                             scale=scale)
    result = run_backtest(data, **kwargs)
    result["data"] = data
    result["year"] = year
    return result


if __name__ == "__main__":
    from src.data.historical import load_results

    matches = load_results()

    for yr in [2018, 2022]:
        print(f"\n{'='*50}")
        print(f"  {yr} World Cup backtest")
        print(f"{'='*50}")
        res = run_wc_backtest(yr, matches, edge_threshold=0.05)
        for k, v in res.items():
            if k not in ("ledger", "data"):
                print(f"  {k:18s}: {v}")
        print(f"  Sample bets:")
        if not res["ledger"].empty:
            print(res["ledger"][["home", "away", "outcome_label",
                                  "edge", "pnl", "bankroll"]].head(6).to_string(index=False))
        else:
            print("  (no bets above threshold)")
