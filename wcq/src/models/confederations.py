"""Team → confederation mapping for the 2026 World Cup field.

Used to fit cross-confederation Elo offsets (see elo.py: fit_confederation_offsets).
"""

CONFEDERATION = {
    # UEFA (Europe)
    "Czechia": "UEFA", "Bosnia-Herzegovina": "UEFA", "Switzerland": "UEFA",
    "Scotland": "UEFA", "Türkiye": "UEFA", "Germany": "UEFA",
    "Sweden": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA",
    "Spain": "UEFA", "France": "UEFA", "Norway": "UEFA",
    "Austria": "UEFA", "Portugal": "UEFA", "England": "UEFA",
    "Croatia": "UEFA",

    # CONMEBOL (South America)
    "Brazil": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Argentina": "CONMEBOL", "Colombia": "CONMEBOL", "Uruguay": "CONMEBOL",

    # CONCACAF (North/Central America, Caribbean)
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "United States": "CONCACAF",
    "Curaçao": "CONCACAF", "Haiti": "CONCACAF", "Panama": "CONCACAF",

    # CAF (Africa)
    "South Africa": "CAF", "Morocco": "CAF", "Côte d'Ivoire": "CAF",
    "Egypt": "CAF", "Senegal": "CAF", "Algeria": "CAF",
    "Congo DR": "CAF", "Ghana": "CAF", "Cabo Verde": "CAF",
    "Tunisia": "CAF",

    # AFC (Asia)
    "South Korea": "AFC", "Qatar": "AFC", "Australia": "AFC",
    "Japan": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Iraq": "AFC", "Uzbekistan": "AFC", "Jordan": "AFC",

    # OFC (Oceania)
    "New Zealand": "OFC",
}


import numpy as np
from scipy.optimize import minimize
from src.models.match_model import match_probs

_FITTED_CONFEDS = ["CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]  # UEFA = reference (0.0)


def fit_confederation_offsets(matches, ratings, confederation=CONFEDERATION,
                                min_year=None):
    """Fit per-confederation Elo offsets from real cross-confederation matches.

    UEFA is fixed as the reference confederation (offset = 0.0); all other
    offsets are relative to it — only the differences between offsets are
    identifiable, so a free reference point isn't meaningful.

    LIMITATION: uses a single static `ratings` dict (e.g. from compute_elo())
    rather than each team's rating at the time of each historical match, so
    this doesn't correct for rating drift over time — only for confederation-
    level bias present in final ratings. A more rigorous version would use
    return_history=True snapshots instead.

    Args:
        matches:       DataFrame from historical.load_results().
        ratings:       dict[team -> elo] (e.g. from compute_elo(matches)).
        confederation: dict[team -> confederation code].
        min_year:      If set, only use matches from this year onward.
                       Restricting to recent eras avoids blending very
                       different historical periods of relative
                       confederation strength into one fit.

    Returns:
        dict[confederation -> offset], with "UEFA": 0.0 always included.
    """
    cross = []
    for row in matches.itertuples(index=False):
        if min_year is not None and row.date.year < min_year:
            continue
        ca = confederation.get(row.home_team)
        cb = confederation.get(row.away_team)
        if ca is None or cb is None or ca == cb:
            continue
        if row.home_team not in ratings or row.away_team not in ratings:
            continue
        cross.append((row.home_team, row.away_team, ca, cb,
                       row.home_score, row.away_score, bool(row.neutral)))

    if not cross:
        raise ValueError("No cross-confederation matches found to fit against.")

    def unpack(x):
        offsets = {"UEFA": 0.0}
        for c, v in zip(_FITTED_CONFEDS, x):
            offsets[c] = v
        return offsets

    def neg_log_likelihood(x):
        offsets = unpack(x)
        total = 0.0
        for home, away, ca, cb, hs, aw, neutral in cross:
            eh = ratings[home] + offsets[ca]
            ea = ratings[away] + offsets[cb]
            probs = match_probs(eh, ea, neutral=neutral)
            if hs > aw:
                p = probs["home"]
            elif hs == aw:
                p = probs["draw"]
            else:
                p = probs["away"]
            total -= np.log(max(p, 1e-9))
        # Regularization: mild penalty on offset magnitude, discourages
        # extreme values not well-supported by the data (esp. small-sample pairs)
        total += 0.01 * sum(v**2 for v in x)
        return total

    x0 = np.zeros(len(_FITTED_CONFEDS))
    result = minimize(neg_log_likelihood, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-4})
    print(f"Fitted on {len(cross)} cross-confederation matches, "
          f"final loss={result.fun:.2f}, converged={result.success}")
    return unpack(result.x)