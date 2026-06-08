"""Multi-source results feed for post-match scoring.

Priority order (as suggested by the project spec):
  1. Market resolution — when a Polymarket per-match binary market settles to
     ~$1 / ~$0, the winning side is the result. Zero new dependencies.
  2. football-data.org free API — covers the WC and allows ~10 req/min on the
     free tier; needs FOOTBALL_DATA_API_KEY env var.
  3. martj42 historical CSV nightly re-fetch — catches anything missed.

All three sources return the same shape: {"home_score", "away_score", "winner"}.
A 20-30 minute delay after full-time is fine for Brier scoring, so we don't
optimise for low latency.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

FOOTBALL_DATA_KEY: str = os.environ.get("FOOTBALL_DATA_API_KEY", "")
_FDORG_BASE = "https://api.football-data.org/v4"

# football-data.org WC 2026 competition code (confirm at tournament start)
_WC_COMPETITION = "WC"


# ---------------------------------------------------------------------------
# Source 1: Polymarket market resolution
# ---------------------------------------------------------------------------

def _outcome_from_polymarket_resolution(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Infer W/D/L from a settled Polymarket per-match market.

    A settled binary market has the winning outcome at price ≈ 1.0 (> 0.95).
    We look for outcome titles containing "home wins", "draw", "away wins", or
    team names with "win" — Polymarket title formats vary.
    """
    try:
        from src.bot.market_discovery import find_polymarket_match, get_polymarket_current_prices
        mkt = find_polymarket_match(home, away, kickoff_utc)
        if not mkt:
            return None
        prices = get_polymarket_current_prices(mkt["market_id"])
        if not prices:
            return None

        home_l = home.lower()
        away_l = away.lower()

        for outcome_label, price in prices.items():
            if price < 0.95:
                continue
            ol = outcome_label.lower()
            if "draw" in ol or "tie" in ol:
                return {"home_score": 0, "away_score": 0, "winner": "draw", "source": "polymarket"}
            if home_l in ol or "home" in ol:
                return {"home_score": 1, "away_score": 0, "winner": "home", "source": "polymarket"}
            if away_l in ol or "away" in ol:
                return {"home_score": 0, "away_score": 1, "winner": "away", "source": "polymarket"}
    except Exception as e:
        print(f"[results] Polymarket resolution: {e}")
    return None


# ---------------------------------------------------------------------------
# Source 2: football-data.org
# ---------------------------------------------------------------------------

def _result_from_football_data(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Fetch result from football-data.org.

    Requires FOOTBALL_DATA_API_KEY. Free tier: ~10 req/min, covers WC.
    """
    if not FOOTBALL_DATA_KEY:
        return None

    date_str = kickoff_utc[:10]
    try:
        r = requests.get(
            f"{_FDORG_BASE}/competitions/{_WC_COMPETITION}/matches",
            headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
            params={"dateFrom": date_str, "dateTo": date_str},
            timeout=15,
        )
        r.raise_for_status()
        matches = r.json().get("matches", [])
    except Exception as e:
        print(f"[results] football-data.org: {e}")
        return None

    home_l = home.lower()
    away_l = away.lower()

    for m in matches:
        ht = str(m.get("homeTeam", {}).get("name", "") or m.get("homeTeam", {}).get("shortName", "")).lower()
        at = str(m.get("awayTeam", {}).get("name", "") or m.get("awayTeam", {}).get("shortName", "")).lower()

        # Loose name match (handles spelling differences)
        home_words = [w for w in home_l.split() if len(w) > 2]
        away_words = [w for w in away_l.split() if len(w) > 2]
        if not (any(w in ht for w in home_words) and any(w in at for w in away_words)):
            continue

        status = m.get("status", "")
        if status not in ("FINISHED", "AWARDED"):
            return None  # match not finished yet

        score = m.get("score", {})
        full = score.get("fullTime", score.get("regularTime", {}))
        hs = full.get("home") if full else None
        as_ = full.get("away") if full else None
        if hs is None or as_ is None:
            return None

        winner_map = {"HOME_TEAM": "home", "AWAY_TEAM": "away", "DRAW": "draw"}
        winner = winner_map.get(score.get("winner", ""), None)
        if winner is None:
            winner = "home" if hs > as_ else ("away" if as_ > hs else "draw")

        return {
            "home_score": int(hs),
            "away_score": int(as_),
            "winner": winner,
            "source": "football-data.org",
        }
    return None


# ---------------------------------------------------------------------------
# Source 3: martj42 nightly CSV re-fetch
# ---------------------------------------------------------------------------

def _result_from_historical_csv(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Check the martj42 historical results CSV for a completed match.

    The CSV is typically updated within a few hours of full time.
    Loads via the project's existing historical.load_results() function.
    """
    date_str = kickoff_utc[:10]
    try:
        from src.data.historical import load_results
        df = load_results(force_download=True)  # always re-fetch for post-match use
        if df is None or df.empty:
            return None

        home_words = [w.lower() for w in home.split() if len(w) > 2]
        away_words = [w.lower() for w in away.split() if len(w) > 2]

        mask = df["date"].astype(str).str.startswith(date_str)
        day_matches = df[mask]

        for _, row in day_matches.iterrows():
            ht = str(row.get("home_team", "")).lower()
            at = str(row.get("away_team", "")).lower()
            if any(w in ht for w in home_words) and any(w in at for w in away_words):
                hs = int(row.get("home_score", 0))
                as_ = int(row.get("away_score", 0))
                winner = "home" if hs > as_ else ("away" if as_ > hs else "draw")
                return {
                    "home_score": hs,
                    "away_score": as_,
                    "winner": winner,
                    "source": "martj42",
                }
    except Exception as e:
        print(f"[results] historical CSV: {e}")
    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_result(
    home: str,
    away: str,
    kickoff_utc: str,
    use_polymarket: bool = True,
    use_football_data: bool = True,
    use_historical: bool = True,
) -> dict | None:
    """Try all sources in priority order. Returns first non-None result.

    Shape: {"home_score": int, "away_score": int, "winner": str, "source": str}
    where winner ∈ {"home", "draw", "away"}.
    Returns None if the match hasn't finished or no source has the result yet.
    """
    if use_polymarket:
        r = _outcome_from_polymarket_resolution(home, away, kickoff_utc)
        if r:
            return r

    if use_football_data and FOOTBALL_DATA_KEY:
        r = _result_from_football_data(home, away, kickoff_utc)
        if r:
            return r

    if use_historical:
        r = _result_from_historical_csv(home, away, kickoff_utc)
        if r:
            return r

    return None


def fetch_and_store_result(
    match_id: str,
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Fetch the result and persist it to the DB if found."""
    from src.bot.storage import save_result, get_result

    existing = get_result(match_id)
    if existing:
        return existing

    result = fetch_result(home, away, kickoff_utc)
    if result:
        save_result(match_id, home, away, result["home_score"], result["away_score"])
        print(f"[results] Stored {home} {result['home_score']}-{result['away_score']} {away} (source: {result['source']})")
    return result


if __name__ == "__main__":
    # Demo: check if a known historical match can be found
    print("Testing result lookup (Brazil vs Germany, 2014-07-08)...")
    r = fetch_result("Brazil", "Germany", "2014-07-08T17:00:00Z",
                     use_polymarket=False, use_football_data=False)
    if r:
        print(f"  {r['home_score']}-{r['away_score']} winner={r['winner']} source={r['source']}")
    else:
        print("  Not found (historical CSV not yet downloaded)")
