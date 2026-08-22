"""Reproduce the README's headline comparison.

Scores three model variants on the SAME set of real 2026 World Cup matches
that the bot logged predictions for, so the comparison is like-for-like:

  (a) what the bot actually predicted live, read back from wcq_bot.db
  (b) plain Elo, trained only on pre-tournament data
  (c) plain Elo + confederation offsets + goal-difference form, same cutoff

(b) versus (c) is the question that matters: do the corrections added in this
project beat plain Elo on live data? Beating a uniform 1/3 prior is table
stakes and is reported only for scale.

Every variant is strictly walk-forward: Elo and the confederation offsets are
fit on matches before 2026-06-11, the goal-difference weight is fit on prior
World Cup editions only, and each match's goal-difference-so-far counts only
matches played before that match's kickoff.

Usage: python run_live_baseline_comparison.py
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from src.data.historical import load_results
from src.models.confederations import CONFEDERATION, fit_confederation_offsets
from src.models.elo import compute_elo
from src.models.match_model import match_probs
from src.models.tournament import _FIFA_TO_HIST
from src.models.tournament_form import fit_goal_diff_weight, tournament_goal_diff_so_far

CUTOFF = pd.Timestamp("2026-06-11")
_OUTCOME_IDX = {"home": 0, "draw": 1, "away": 2}

# fixturedownload.com vs martj42 spelling differences not already covered by
# _FIFA_TO_HIST (which maps the other direction for some of these).
_EXTRA_ALIAS = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina", "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Türkiye": "Turkey",
}


def _hist_name(team: str) -> str:
    return _FIFA_TO_HIST.get(team, _EXTRA_ALIAS.get(team, team))


def _brier_per_match(prob_vectors: list[list[float]], winners: list[str]) -> np.ndarray:
    """Three-outcome Brier, averaged over the 3 outcomes so it is on the
    conventional 0-to-2 scale divided by 3 (i.e. comparable to a binary Brier)."""
    out = []
    for probs, winner in zip(prob_vectors, winners):
        ai = _OUTCOME_IDX[winner]
        out.append(sum((probs[i] - (1.0 if i == ai else 0.0)) ** 2 for i in range(3)) / 3)
    return np.array(out)


def _hit_rate(prob_vectors: list[list[float]], winners: list[str]) -> float:
    return float(np.mean([
        int(max(range(3), key=lambda i: p[i]) == _OUTCOME_IDX[w])
        for p, w in zip(prob_vectors, winners)
    ]))


def run(db_path: str = "wcq_bot.db", n_boot: int = 10_000, seed: int = 42) -> None:
    conn = sqlite3.connect(db_path)
    preds = pd.read_sql_query(
        """SELECT p.home_team, p.away_team, p.kickoff_utc,
                  p.p_home_model, p.p_draw_model, p.p_away_model, r.winner
           FROM predictions p JOIN match_results r ON p.match_id = r.match_id
           WHERE r.winner IS NOT NULL AND p.p_home_model IS NOT NULL""",
        conn,
    )
    conn.close()
    if preds.empty:
        print("No settled predictions in the DB. Run jobs/backfill_results.py first.")
        return

    matches = load_results()
    train = matches[matches["date"] < CUTOFF]
    wc26 = matches[(matches["tournament"] == "FIFA World Cup") & (matches["date"] >= CUTOFF)]

    elo = compute_elo(train)
    offsets = fit_confederation_offsets(train, elo, CONFEDERATION)
    confed = {t: offsets.get(CONFEDERATION.get(t), 0.0) for t in elo if CONFEDERATION.get(t)}

    prior = matches[(matches["tournament"] == "FIFA World Cup") & (matches["date"] < CUTOFF)].copy()
    prior["edition_year"] = prior["date"].dt.year
    prior_rows = []
    for _, edition in prior.groupby("edition_year"):
        for row in edition.itertuples(index=False):
            if row.home_team not in elo or row.away_team not in elo:
                continue
            gd = tournament_goal_diff_so_far(edition, row.date)
            prior_rows.append({
                "home": row.home_team, "away": row.away_team,
                "home_score": int(row.home_score), "away_score": int(row.away_score),
                "neutral": bool(row.neutral),
                "gd_home": gd.get(row.home_team, 0), "gd_away": gd.get(row.away_team, 0),
            })
    weight = fit_goal_diff_weight(prior_rows, elo, confed)

    winners, logged, plain, full = [], [], [], []
    for p in preds.itertuples(index=False):
        kickoff = pd.Timestamp(p.kickoff_utc).tz_localize(None)
        h, a = _hist_name(p.home_team), _hist_name(p.away_team)
        gd = tournament_goal_diff_so_far(wc26, kickoff)

        e_h = elo.get(h, config.ELO_BASE)
        e_a = elo.get(a, config.ELO_BASE)
        pl = match_probs(e_h, e_a, neutral=True)
        fu = match_probs(
            e_h + confed.get(h, 0.0) + weight * gd.get(h, 0),
            e_a + confed.get(a, 0.0) + weight * gd.get(a, 0),
            neutral=True,
        )

        winners.append(p.winner)
        logged.append([p.p_home_model, p.p_draw_model, p.p_away_model])
        plain.append([pl["home"], pl["draw"], pl["away"]])
        full.append([fu["home"], fu["draw"], fu["away"]])

    n = len(winners)
    uniform = [[1 / 3, 1 / 3, 1 / 3]] * n
    b_logged = _brier_per_match(logged, winners)
    b_plain = _brier_per_match(plain, winners)
    b_full = _brier_per_match(full, winners)
    b_unif = _brier_per_match(uniform, winners)

    print(f"Scored on {n} real 2026 matches, all variants walk-forward from {CUTOFF.date()}")
    print(f"goal-difference weight, fit on prior WC editions only: {weight:.2f}\n")
    print(f"{'variant':<48}{'Brier':<10}{'hit rate'}")
    print("-" * 68)
    print(f"{'uniform 1/3 (for scale only)':<48}{b_unif.mean():<10.4f}{'':>8}")
    print(f"{'plain Elo, pre-tournament only':<48}{b_plain.mean():<10.4f}{_hit_rate(plain, winners):.1%}")
    print(f"{'+ confed offsets + goal-diff form':<48}{b_full.mean():<10.4f}{_hit_rate(full, winners):.1%}")
    print(f"{'(what the bot actually logged live)':<48}{b_logged.mean():<10.4f}{_hit_rate(logged, winners):.1%}")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = b_plain[idx].mean() - b_full[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nBrier improvement of corrections over plain Elo, {n_boot:,} bootstrap resamples:")
    print(f"  mean {diffs.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"{'excludes zero' if lo > 0 or hi < 0 else 'INCLUDES ZERO'}")


if __name__ == "__main__":
    run()
