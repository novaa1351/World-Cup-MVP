"""Runtime discovery of per-match Polymarket and Kalshi markets.

Per-match markets roll out only a few days before kickoff, so nothing is
hardcoded. Every pre-match job re-runs discovery to pick up newly listed
markets.

Polymarket (primary, in-play repricing):
  GET /events?slug=world-cup-matches  → all WC per-match markets
  Match by: both team names appear in the market question/title + kickoff date
  close time within ±36hr of fixture kickoff.

Kalshi (cross-platform spread; may close at kickoff, not in-play):
  Discover series dynamically via /series (search soccer/WC 2026 entries).
  For each candidate series, list events and match by team names.
  Always check close_time and status before treating as a live signal.

Results are cached in a JSON sidecar at WCQ_MARKET_CACHE_PATH (default:
data/market_cache.json) with a TTL of MARKET_CACHE_TTL_MIN minutes.
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
import config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_PATH = Path(os.environ.get(
    "WCQ_MARKET_CACHE_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "market_cache.json"),
))
CACHE_TTL_MIN: int = int(os.environ.get("MARKET_CACHE_TTL_MIN", "30"))

_POLY_EVENTS_URL = f"{config.POLYMARKET_GAMMA}/events"
_POLY_MARKETS_URL = f"{config.POLYMARKET_GAMMA}/markets"
_KALSHI_SERIES_URL = f"{config.KALSHI_BASE}/series"
_KALSHI_EVENTS_URL = f"{config.KALSHI_BASE}/events"
_KALSHI_MARKETS_URL = f"{config.KALSHI_BASE}/markets"

# Team name aliases: market spelling → FIFA/model spelling
# (extends MARKET_TO_FIFA from tournament.py)
try:
    from src.models.tournament import MARKET_TO_FIFA as _M2F
    _ALIASES: dict[str, str] = dict(_M2F)
except ImportError:
    _ALIASES = {}

_ALIASES.update({
    "United States": "United States",
    "USA": "United States",
    "US": "United States",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Côte D'Ivoire": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",
    "South Korea": "South Korea",
    "Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Curacao": "Curaçao",
    "DR Congo": "Congo DR",
    "Congo, DR": "Congo DR",
    "Iran": "Iran",
    "IR Iran": "Iran",
})


def _norm(name: str) -> str:
    """Normalise a team name: alias map, lower-case, strip whitespace."""
    name = name.strip()
    name = _ALIASES.get(name, name)
    return name.lower()


def _teams_match(market_title: str, home: str, away: str) -> bool:
    """Return True if both team names appear in the market title (case-insensitive)."""
    title_l = market_title.lower()
    home_l = _norm(home)
    away_l = _norm(away)
    # Accept partial match (e.g. "Brazil" in "Brazil vs Morocco")
    home_words = [w for w in home_l.split() if len(w) > 2]
    away_words = [w for w in away_l.split() if len(w) > 2]
    home_ok = any(w in title_l for w in home_words)
    away_ok = any(w in title_l for w in away_words)
    return home_ok and away_ok


def _kickoff_close_enough(market_end: str | None, kickoff_utc: str, window_hr: float = 36.0) -> bool:
    """Return True if the market close time is within window_hr of the kickoff."""
    if not market_end:
        return True  # no close time — assume valid
    try:
        from dateutil import parser as du
        close = du.parse(market_end)
        ko = du.parse(kickoff_utc)
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        return abs((close - ko).total_seconds()) < window_hr * 3600
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _cache_get(key: str) -> Any | None:
    cache = _load_cache()
    entry = cache.get(key)
    if not entry:
        return None
    age_min = (time.time() - entry.get("ts", 0)) / 60
    if age_min > CACHE_TTL_MIN:
        return None
    return entry.get("data")


def _cache_set(key: str, data: Any) -> None:
    cache = _load_cache()
    cache[key] = {"ts": time.time(), "data": data}
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Polymarket discovery
# ---------------------------------------------------------------------------

def _fetch_polymarket_wc_markets(limit: int = 500) -> list[dict]:
    """Fetch all active Polymarket markets under the world-cup-matches event hub."""
    cached = _cache_get("poly_wc_markets")
    if cached is not None:
        return cached

    markets: list[dict] = []

    # Try the events slug endpoint first
    try:
        r = requests.get(
            _POLY_EVENTS_URL,
            params={"slug": "world-cup-matches", "limit": 100},
            timeout=20,
        )
        r.raise_for_status()
        events = r.json()
        if isinstance(events, list):
            for ev in events:
                for m in ev.get("markets", []):
                    markets.append(m)
        elif isinstance(events, dict):
            for m in events.get("markets", []):
                markets.append(m)
    except Exception as e:
        print(f"[market_discovery] Polymarket events endpoint: {e}")

    # Fallback: generic markets search filtered by WC tag/keyword
    if not markets:
        try:
            r = requests.get(
                _POLY_MARKETS_URL,
                params={"active": "true", "closed": "false", "limit": limit,
                        "tag_slug": "world-cup"},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            markets = data if isinstance(data, list) else data.get("markets", [])
        except Exception as e:
            print(f"[market_discovery] Polymarket markets fallback: {e}")

    _cache_set("poly_wc_markets", markets)
    return markets


def find_polymarket_match(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Find the Polymarket per-match market for a specific fixture.

    Returns a dict with keys: market_id, question, outcomes, prices, end_date_iso
    suitable for polling, or None if not yet listed.
    """
    key = f"poly_match_{_norm(home)}_{_norm(away)}_{kickoff_utc[:10]}"
    cached = _cache_get(key)
    if cached is not None:
        return cached if cached else None

    markets = _fetch_polymarket_wc_markets()
    for m in markets:
        title = str(m.get("question") or m.get("title") or "")
        if not _teams_match(title, home, away):
            continue
        end_date = m.get("endDate") or m.get("end_date") or m.get("endDateIso")
        if not _kickoff_close_enough(end_date, kickoff_utc):
            continue

        # Parse outcomes and prices
        try:
            import json as _json
            outcomes = _json.loads(m.get("outcomes", "[]"))
            prices = [float(p) for p in _json.loads(m.get("outcomePrices", "[]"))]
        except Exception:
            outcomes, prices = [], []

        result = {
            "market_id": m.get("id") or m.get("conditionId", ""),
            "question": title,
            "outcomes": outcomes,
            "prices": prices,
            "end_date": end_date,
            "active": m.get("active", True),
            "closed": m.get("closed", False),
            "platform": "polymarket",
        }
        _cache_set(key, result)
        return result

    _cache_set(key, {})  # cache miss so we don't hammer the API
    return None


def get_polymarket_current_prices(market_id: str) -> dict[str, float] | None:
    """Poll the current YES prices for a Polymarket market by ID."""
    try:
        r = requests.get(
            f"{_POLY_MARKETS_URL}/{market_id}",
            timeout=10,
        )
        r.raise_for_status()
        m = r.json()
        if isinstance(m, list):
            m = m[0] if m else {}
        import json as _json
        outcomes = _json.loads(m.get("outcomes", "[]"))
        prices = [float(p) for p in _json.loads(m.get("outcomePrices", "[]"))]
        return dict(zip(outcomes, prices)) if outcomes and prices else None
    except Exception as e:
        print(f"[market_discovery] price fetch for {market_id}: {e}")
        return None


def get_polymarket_champion_prices() -> dict[str, float]:
    """Current Polymarket champion (WC winner) prices. Always available."""
    try:
        from src.data.markets import fetch_polymarket, winner_probs
        df = fetch_polymarket()
        return winner_probs(df)
    except Exception as e:
        print(f"[market_discovery] champion prices: {e}")
        return {}


def get_kalshi_survival_prices() -> dict[str, dict[str, float]]:
    """Current Kalshi round-survival prices (KXWCROUND). Always available."""
    try:
        from src.data.markets import fetch_kalshi, kalshi_survival_probs
        df = fetch_kalshi()
        return kalshi_survival_probs(df)
    except Exception as e:
        print(f"[market_discovery] Kalshi survival: {e}")
        return {}


# ---------------------------------------------------------------------------
# Kalshi per-match discovery
# ---------------------------------------------------------------------------

def _discover_kalshi_wc_series() -> list[str]:
    """Return Kalshi series tickers likely to contain WC 2026 per-match markets."""
    cached = _cache_get("kalshi_wc_series")
    if cached is not None:
        return cached

    tickers: list[str] = []
    try:
        r = requests.get(
            _KALSHI_SERIES_URL,
            params={"limit": 200},
            timeout=20,
        )
        r.raise_for_status()
        series_list = r.json().get("series", [])
        for s in series_list:
            title = str(s.get("title", "") or s.get("name", "")).lower()
            ticker = str(s.get("ticker", ""))
            # Match any WC/World Cup soccer series that isn't the survival series we already know
            if ("world cup" in title or "wc" in ticker.lower()) and "soccer" in title or "wc2026" in ticker.lower():
                if ticker not in ("KXWCROUND", "KXMENWORLDCUP"):
                    tickers.append(ticker)
    except Exception as e:
        print(f"[market_discovery] Kalshi series discovery: {e}")

    _cache_set("kalshi_wc_series", tickers)
    return tickers


def find_kalshi_match(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Find the Kalshi per-match market for a fixture, if listed.

    Checks whether the market is open at discovery time and whether close_time
    suggests it trades in-play — callers must recheck status before each poll.

    Returns None if not listed or if already closed.
    """
    key = f"kalshi_match_{_norm(home)}_{_norm(away)}_{kickoff_utc[:10]}"
    cached = _cache_get(key)
    if cached is not None:
        return cached if cached else None

    # Try known WC series first, then dynamically discovered ones
    candidate_tickers = ["KXWCMATCH", "KXWCSOCCER"] + _discover_kalshi_wc_series()

    for ticker in candidate_tickers:
        try:
            r = requests.get(
                _KALSHI_MARKETS_URL,
                params={"series_ticker": ticker, "limit": 200},
                timeout=20,
            )
            if r.status_code == 404:
                continue
            r.raise_for_status()
            mkts = r.json().get("markets", [])
        except Exception as e:
            print(f"[market_discovery] Kalshi series {ticker}: {e}")
            continue

        for m in mkts:
            title = str(m.get("title", ""))
            if not _teams_match(title, home, away):
                continue
            close_time = m.get("close_time") or m.get("expiration_time")
            if not _kickoff_close_enough(close_time, kickoff_utc):
                continue

            status = m.get("status", "open").lower()
            if status in ("settled", "closed"):
                _cache_set(key, {})
                return None

            result = {
                "ticker": m.get("ticker", ""),
                "title": title,
                "close_time": close_time,
                "status": status,
                "yes_bid": m.get("yes_bid_dollars"),
                "yes_ask": m.get("yes_ask_dollars"),
                "platform": "kalshi",
                # True only if close_time is AFTER kickoff (i.e., trades in-play)
                "trades_inplay": _close_is_after_kickoff(close_time, kickoff_utc),
            }
            _cache_set(key, result)
            return result

    _cache_set(key, {})
    return None


def get_kalshi_match_price(ticker: str) -> float | None:
    """Current mid-price for a Kalshi binary YES market."""
    try:
        r = requests.get(
            f"{_KALSHI_MARKETS_URL}/{ticker}",
            timeout=10,
        )
        r.raise_for_status()
        m = r.json().get("market", r.json())
        bid = m.get("yes_bid_dollars")
        ask = m.get("yes_ask_dollars")
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2
    except Exception as e:
        print(f"[market_discovery] Kalshi price {ticker}: {e}")
    return None


def _close_is_after_kickoff(close_time: str | None, kickoff_utc: str) -> bool:
    """True if the market closes after kickoff (i.e., it trades in-play)."""
    if not close_time:
        return False
    try:
        from dateutil import parser as du
        close = du.parse(close_time)
        ko = du.parse(kickoff_utc)
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        return close > ko
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Convenience: resolve all markets for a fixture
# ---------------------------------------------------------------------------

def resolve_fixture_markets(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict[str, Any]:
    """Return all available markets for a fixture, tiered by priority.

    Returns:
        {
          "polymarket_match":   dict | None,   # per-match, primary live signal
          "kalshi_match":       dict | None,   # per-match, cross-platform spread
          "polymarket_champion": dict[team, price],  # always available
          "kalshi_survival":    dict[team, dict[round, price]],  # always available
        }
    """
    return {
        "polymarket_match":    find_polymarket_match(home, away, kickoff_utc),
        "kalshi_match":        find_kalshi_match(home, away, kickoff_utc),
        "polymarket_champion": get_polymarket_champion_prices(),
        "kalshi_survival":     get_kalshi_survival_prices(),
    }


if __name__ == "__main__":
    print("Resolving markets for Brazil vs Morocco 2026-06-15...")
    mkts = resolve_fixture_markets("Brazil", "Morocco", "2026-06-15T18:00:00Z")
    for k, v in mkts.items():
        print(f"  {k}: {type(v).__name__} len={len(v) if v else 0}")
