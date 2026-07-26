"""In-tournament goal-difference form adjustment.

Unlike confederation offsets (static per-team-group correction), this signal
changes match-by-match as a tournament progresses — it only uses goals scored
in matches STRICTLY BEFORE the one being predicted, within the same tournament,
so it's safe to use in a no-lookahead backtest.
"""
import numpy as np
from scipy.optimize import minimize
from src.models.match_model import match_probs


def tournament_goal_diff_so_far(tournament_matches: "pd.DataFrame", as_of_date) -> dict:
    """Goal difference for each team from matches strictly before `as_of_date`,
    within the given tournament's matches only.

    Args:
        tournament_matches: rows already filtered to one tournament (e.g. one
                            World Cup's matches).
        as_of_date:         only matches with date < this are counted.

    Returns:
        dict[team -> goal_diff] (goals_for - goals_against so far).
    """
    prior = tournament_matches[tournament_matches["date"] < as_of_date]
    diff: dict = {}
    for row in prior.itertuples(index=False):
        diff[row.home_team] = diff.get(row.home_team, 0) + (row.home_score - row.away_score)
        diff[row.away_team] = diff.get(row.away_team, 0) + (row.away_score - row.home_score)
    return diff


def fit_goal_diff_weight(wc_rows: list, ratings: dict, confed_adjust: dict | None = None) -> float:
    """Fit the Elo-points-per-goal-difference weight via MLE against real outcomes.

    Args:
        wc_rows: list of dicts, each with keys: home, away, home_score,
                 away_score, neutral, gd_home, gd_away (goal diff so far
                 for each team, computed via tournament_goal_diff_so_far).
        ratings: dict[team -> base elo] (pre-tournament, no lookahead).
        confed_adjust: optional dict[team -> extra points] to add first
                       (e.g. confederation offsets), so this fits ON TOP of
                       whatever correction is already applied.

    Returns:
        Fitted weight (Elo points per goal of tournament goal-difference).
    """
    confed_adjust = confed_adjust or {}

    def neg_log_likelihood(w):
        weight = w[0]
        total = 0.0
        for m in wc_rows:
            eh = ratings[m["home"]] + confed_adjust.get(m["home"], 0.0) + weight * m["gd_home"]
            ea = ratings[m["away"]] + confed_adjust.get(m["away"], 0.0) + weight * m["gd_away"]
            probs = match_probs(eh, ea, neutral=m["neutral"])
            if m["home_score"] > m["away_score"]:
                p = probs["home"]
            elif m["home_score"] == m["away_score"]:
                p = probs["draw"]
            else:
                p = probs["away"]
            total -= np.log(max(p, 1e-9))
        total += 0.01 * weight**2  # mild regularization, same as confed offsets
        return total

    result = minimize(neg_log_likelihood, x0=[0.0], method="Nelder-Mead",
                       options={"maxiter": 2000})
    print(f"Fitted goal_diff_weight={result.x[0]:.2f}, "
          f"loss={result.fun:.2f}, converged={result.success}")
    return result.x[0]