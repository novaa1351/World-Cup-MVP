"""Fetch live World Cup prediction-market data.

Both endpoints are PUBLIC and read-only (no API key). If the network is
blocked or no WC markets exist, every function fails gracefully and returns
an empty frame so the dashboard never crashes.

Sources verified June 2026:
  Polymarket Gamma : https://gamma-api.polymarket.com/markets
  Kalshi (public)  : https://external-api.kalshi.com/trade-api/v2/markets

Data shapes
-----------
Polymarket  — one market per team: "Will X win the 2026 FIFA World Cup?"
              Outcomes: Yes / No.  price = YES mid (already 0-1).
              winner_probs() extracts {team: yes_price} for the edge tab.

Kalshi      — KXWCROUND series: "Will X qualify for FIFA World Cup [Round]?"
              Four round types: 26RO16→R16, 26QUAR→QF, 26SEMI→SF, 26FINAL→final.
              API v2 uses yes_bid_dollars / yes_ask_dollars (0-1 range).
              kalshi_survival_probs() builds a {team: {round: mid_price}} dict
              that plugs directly into the SVI surface and edge comparison.
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

# ---------------------------------------------------------------------------
# Kalshi round-code → our ROUNDS labels
# ---------------------------------------------------------------------------
_KALSHI_ROUND_MAP: dict[str, str] = {
    "26RO16":  "R16",
    "26QUAR":  "QF",
    "26SEMI":  "SF",
    "26FINAL": "final",
}

# Regex to extract team name from Kalshi market titles:
#   "Will <team> qualify for FIFA World Cup [Round]?"
_KALSHI_TEAM_PATTERN = re.compile(
    r"Will\s+(.+?)\s+qualify\s+for\s+(?:FIFA\s+)?World\s+Cup",
    re.IGNORECASE,
)

# Regex to extract team name from Polymarket winner titles:
#   "Will <team> win the 2026 FIFA World Cup?"
_WINNER_PATTERN = re.compile(
    r"will\s+(.+?)\s+win\s+the\s+\d{4}\s+(?:FIFA\s+)?World\s+Cup",
    re.IGNORECASE,
)


def winner_probs(df: pd.DataFrame) -> dict[str, float]:
    """Parse Polymarket binary "Will X win the WC?" markets → {team: yes_price}.

    Keeps only Yes/YES rows, extracts the team name from the market title, and
    returns a raw dict ready for implied_from_book(). The caller maps names via
    MARKET_TO_FIFA in src/models/tournament.py before looking up mc_survival.
    """
    result: dict[str, float] = {}
    for _, row in df.iterrows():
        if str(row["outcome"]).lower() != "yes":
            continue
        m = _WINNER_PATTERN.search(str(row.get("market", "")))
        if m:
            result[m.group(1).strip()] = float(row["price"])
    return result


def kalshi_survival_probs(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Convert the Kalshi DataFrame into {team: {round: mid_price}}.

    Only rows with platform=='kalshi' and a recognised round are included.
    Mid-price = (bid + ask) / 2 — already in 0-1 range.
    Round labels match config.ROUNDS: 'R16', 'QF', 'SF', 'final'.
    """
    result: dict[str, dict[str, float]] = {}
    kalshi_rows = df[df["platform"] == "kalshi"] if "platform" in df.columns else df
    for _, row in kalshi_rows.iterrows():
        team = str(row.get("team", ""))
        rnd = str(row.get("round", ""))
        price = row.get("price")
        if team and rnd and price is not None:
            result.setdefault(team, {})[rnd] = float(price)
    return result


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_polymarket(query: str = "World Cup", limit: int = 100) -> pd.DataFrame:
    """Search Polymarket Gamma markets by keyword. Returns outcome-level rows."""
    try:
        r = requests.get(
            f"{config.POLYMARKET_GAMMA}/markets",
            params={"active": "true", "closed": "false", "limit": limit},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[polymarket] {e}")
        return pd.DataFrame()

    rows = []
    for m in data:
        title = m.get("question") or m.get("title") or ""
        if query.lower() not in title.lower():
            continue
        try:
            outcomes = json.loads(m.get("outcomes", "[]"))
            prices = [float(p) for p in json.loads(m.get("outcomePrices", "[]"))]
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        for o, p in zip(outcomes, prices):
            rows.append({"platform": "polymarket", "market": title,
                         "outcome": o, "price": p,
                         "team": None, "round": "champion"})
    return pd.DataFrame(rows)


def fetch_kalshi() -> pd.DataFrame:
    """Fetch the KXWCROUND survival markets from Kalshi.

    Targets the series_ticker=KXWCROUND directly — the generic /markets
    endpoint's default sort never surfaces WC markets in 200 rows.

    API v2 price fields are yes_bid_dollars / yes_ask_dollars (0-1 range,
    not cents). The old yes_bid / yes_ask fields are deprecated and return None.

    Returns one row per (team, round) with the mid-price.
    """
    try:
        r = requests.get(
            f"{config.KALSHI_BASE}/markets",
            params={"series_ticker": "KXWCROUND", "limit": 200},
            timeout=20,
        )
        r.raise_for_status()
        markets = r.json().get("markets", [])
    except Exception as e:
        print(f"[kalshi] {e}")
        return pd.DataFrame()

    rows = []
    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")

        # Identify which round this market is for
        round_code = None
        for code in _KALSHI_ROUND_MAP:
            if f"-{code}-" in ticker:
                round_code = code
                break
        if round_code is None:
            continue

        # Extract team name from title
        team_match = _KALSHI_TEAM_PATTERN.search(title)
        if not team_match:
            continue
        team = team_match.group(1).strip()

        # API v2: prices are in yes_bid_dollars / yes_ask_dollars (0-1 range)
        bid = m.get("yes_bid_dollars")
        ask = m.get("yes_ask_dollars")
        if bid is None or ask is None:
            continue
        bid, ask = float(bid), float(ask)
        mid = (bid + ask) / 2

        rows.append({
            "platform": "kalshi",
            "market":   title,
            "outcome":  team,     # team name (for compatibility with live-markets tab)
            "price":    mid,
            "team":     team,
            "round":    _KALSHI_ROUND_MAP[round_code],
            "bid":      bid,
            "ask":      ask,
        })

    return pd.DataFrame(rows)


def fetch_polymarket_matches(series_slug: str = "soccer-fifwc") -> pd.DataFrame:
    """Fetch 3-way (home win / draw / away win) Polymarket markets for WC 2026 matches.

    Uses the Gamma API events endpoint filtered to the FIFA WC series.
    Skips halftime, spread, total, and any other non-3-way events.

    Returns one row per outcome with columns:
        platform, event_slug, home_team, away_team, date, outcome, price, market
    where outcome is one of: 'home_win', 'draw', 'away_win'.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        try:
            r = requests.get(
                f"{config.POLYMARKET_GAMMA}/events",
                params={"series_slug": series_slug, "limit": 100,
                        "active": "true", "closed": "false", "offset": offset},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[polymarket_matches] {e}")
            break

        for event in data:
            slug  = event.get("slug", "") or ""
            title = event.get("title", "") or ""

            if not slug.startswith("fifwc"):
                continue
            # Keep only full-time W/D/W events; skip halftime, totals, spreads, etc.
            if any(kw in title for kw in ("Halftime", "More Markets", "Spread", "Total")):
                continue
            if not any(m.get("slug", "").endswith("-draw") for m in event.get("markets", [])):
                continue

            parts = title.split(" vs. ")
            if len(parts) != 2:
                continue
            home_team = parts[0].strip()
            away_team = parts[1].strip()
            date = (event.get("eventDate") or event.get("startDate") or "")[:10]

            for market in event.get("markets", []):
                question = market.get("question", "") or ""
                try:
                    prices   = [float(p) for p in json.loads(market.get("outcomePrices", "[]"))]
                    outcomes = json.loads(market.get("outcomes", "[]"))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not prices or not outcomes:
                    continue

                # Identify outcome from question text — more reliable than slug suffix
                q_lower = question.lower()
                if "draw" in q_lower:
                    outcome = "draw"
                elif home_team.lower() in q_lower:
                    outcome = "home_win"
                elif away_team.lower() in q_lower:
                    outcome = "away_win"
                else:
                    continue

                yes_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "yes"), 0)
                price = prices[yes_idx] if yes_idx < len(prices) else prices[0]

                rows.append({
                    "platform":  "polymarket",
                    "event_slug": slug,
                    "home_team":  home_team,
                    "away_team":  away_team,
                    "date":       date,
                    "outcome":    outcome,
                    "price":      price,
                    "market":     question,
                })

        if len(data) < 100:
            break
        offset += 100

    return pd.DataFrame(rows)


def mock_markets() -> pd.DataFrame:
    """Offline sample so the UI works with no network. Numbers are made up."""
    pm_rows = [
        {"platform": "polymarket", "market": "Will Brazil win the 2026 FIFA World Cup?",
         "outcome": "Yes", "price": 0.18, "team": "Brazil", "round": "champion"},
        {"platform": "polymarket", "market": "Will France win the 2026 FIFA World Cup?",
         "outcome": "Yes", "price": 0.16, "team": "France", "round": "champion"},
        {"platform": "polymarket", "market": "Will Spain win the 2026 FIFA World Cup?",
         "outcome": "Yes", "price": 0.15, "team": "Spain", "round": "champion"},
    ]
    ks_rows = [
        {"platform": "kalshi", "market": f"Will {t} qualify for FIFA World Cup {r_name}?",
         "outcome": t, "price": p, "team": t, "round": r_label, "bid": p - 0.005, "ask": p + 0.005}
        for t, r_label, r_name, p in [
            ("Spain",     "R16",   "Round of 16", 0.77),
            ("Brazil",    "R16",   "Round of 16", 0.69),
            ("France",    "R16",   "Round of 16", 0.78),
            ("Spain",     "QF",    "Quarterfinals", 0.55),
            ("Brazil",    "QF",    "Quarterfinals", 0.47),
            ("France",    "QF",    "Quarterfinals", 0.52),
            ("Spain",     "SF",    "Semifinals", 0.30),
            ("Brazil",    "SF",    "Semifinals", 0.25),
            ("France",    "SF",    "Semifinals", 0.28),
            ("Spain",     "final", "Final", 0.17),
            ("Brazil",    "final", "Final", 0.14),
            ("France",    "final", "Final", 0.15),
        ]
    ]
    return pd.DataFrame(pm_rows + ks_rows)


def get_all(use_mock_fallback: bool = True) -> pd.DataFrame:
    """Fetch and combine Polymarket + Kalshi WC market data.

    Returns a unified DataFrame with columns:
      platform, market, outcome, price, team, round
    where `round` is:
      'champion'         — Polymarket winner markets
      'R16','QF','SF','final' — Kalshi KXWCROUND survival markets
    """
    pm = fetch_polymarket()
    ks = fetch_kalshi()
    df = pd.concat([pm, ks], ignore_index=True)
    if df.empty and use_mock_fallback:
        print("[markets] live fetch empty -> using mock data")
        return mock_markets()
    return df


if __name__ == "__main__":
    df = get_all()
    print(f"Total rows: {len(df)}")
    print()
    for plat in df["platform"].unique():
        sub = df[df["platform"] == plat]
        print(f"--- {plat} ---")
        if "round" in sub.columns:
            for rnd in sub["round"].unique():
                rnd_sub = sub[sub["round"] == rnd]
                print(f"  {rnd}: {len(rnd_sub)} markets, "
                      f"price range [{rnd_sub['price'].min():.3f}, {rnd_sub['price'].max():.3f}]")
                # Show top 3 by price
                for _, row in rnd_sub.nlargest(3, "price").iterrows():
                    name = row.get("team") or row.get("outcome") or "?"
                    print(f"    {str(name):<25} {row['price']:.3f}")
        print()
