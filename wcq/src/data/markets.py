"""Fetch live World Cup prediction-market data.

Both endpoints are PUBLIC and read-only (no API key). If the network is blocked
(e.g. a sandbox) or no WC markets exist yet, every function fails gracefully and
returns an empty frame, so the dashboard never crashes. Swap in `mock_markets()`
to develop the UI offline.

Sources verified June 2026:
  Polymarket Gamma : https://gamma-api.polymarket.com/markets
  Kalshi (public)  : https://external-api.kalshi.com/trade-api/v2/markets
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import re
import pandas as pd
import requests

import config

# Matches "Will <team> win the [FIFA] World Cup[?]" — handles minor punctuation variations.
_WINNER_PATTERN = re.compile(
    r"will\s+(.+?)\s+win\s+the\s+\d{4}\s+(?:FIFA\s+)?World\s+Cup",
    re.IGNORECASE,
)


def winner_probs(df: pd.DataFrame) -> dict[str, float]:
    """Parse binary "Will X win the WC?" markets into {team: yes_price}.

    Polymarket (and potentially Kalshi) represent each team as a separate
    binary market with "Yes" and "No" outcomes.  This function:
      1. Keeps only Yes/YES rows.
      2. Extracts the team name from the market title via regex.
      3. Returns a raw {team: yes_price} dict ready for implied_from_book().

    The caller is responsible for de-vigging and matching team names to the
    MC survival dict (see MARKET_TO_FIFA in src/models/tournament.py).
    """
    result: dict[str, float] = {}
    for _, row in df.iterrows():
        if str(row["outcome"]).lower() != "yes":
            continue
        m = _WINNER_PATTERN.search(str(row.get("market", "")))
        if m:
            result[m.group(1).strip()] = float(row["price"])
    return result


def fetch_polymarket(query: str = "World Cup", limit: int = 100) -> pd.DataFrame:
    """Search Gamma markets by keyword. Returns outcome-level rows."""
    try:
        r = requests.get(
            f"{config.POLYMARKET_GAMMA}/markets",
            params={"active": "true", "closed": "false", "limit": limit},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # network blocked, rate-limited, schema change...
        print(f"[polymarket] {e}")
        return pd.DataFrame()

    rows = []
    for m in data:
        title = m.get("question") or m.get("title") or ""
        if query.lower() not in title.lower():
            continue
        # Gamma stores outcomes and prices as JSON-encoded strings.
        try:
            outcomes = json.loads(m.get("outcomes", "[]"))
            prices = [float(p) for p in json.loads(m.get("outcomePrices", "[]"))]
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        for o, p in zip(outcomes, prices):
            rows.append({"platform": "polymarket", "market": title,
                         "outcome": o, "price": p})
    return pd.DataFrame(rows)


def fetch_kalshi(query: str = "World Cup", limit: int = 200) -> pd.DataFrame:
    """List Kalshi markets and keep those whose title matches the query."""
    try:
        r = requests.get(
            f"{config.KALSHI_BASE}/markets",
            params={"status": "open", "limit": limit},
            timeout=20,
        )
        r.raise_for_status()
        markets = r.json().get("markets", [])
    except Exception as e:
        print(f"[kalshi] {e}")
        return pd.DataFrame()

    rows = []
    for m in markets:
        title = m.get("title", "")
        if query.lower() not in title.lower():
            continue
        # yes_bid/yes_ask are in cents; use mid as the implied price.
        bid, ask = m.get("yes_bid"), m.get("yes_ask")
        if bid is None or ask is None:
            continue
        mid_cents = (bid + ask) / 2
        rows.append({"platform": "kalshi", "market": title,
                     "outcome": m.get("yes_sub_title") or "YES",
                     "price": mid_cents / 100.0})
    return pd.DataFrame(rows)


def mock_markets() -> pd.DataFrame:
    """Offline sample so the UI works with no network. Numbers are made up."""
    return pd.DataFrame(
        [
            {"platform": "polymarket", "market": "2026 World Cup Winner",
             "outcome": t, "price": p}
            for t, p in {"Brazil": 0.18, "France": 0.16, "Argentina": 0.15,
                         "England": 0.10, "Spain": 0.12}.items()
        ]
        + [
            {"platform": "kalshi", "market": "2026 World Cup Winner",
             "outcome": t, "price": p}
            for t, p in {"Brazil": 0.20, "France": 0.15, "Argentina": 0.14,
                         "England": 0.11, "Spain": 0.13}.items()
        ]
    )


def get_all(query: str = "World Cup", use_mock_fallback: bool = True) -> pd.DataFrame:
    df = pd.concat([fetch_polymarket(query), fetch_kalshi(query)], ignore_index=True)
    if df.empty and use_mock_fallback:
        print("[markets] live fetch empty -> using mock data")
        return mock_markets()
    return df


if __name__ == "__main__":
    print(get_all().to_string(index=False))
