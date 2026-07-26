"""Fetch RESOLVED (closed) 2026 World Cup match markets from Polymarket.

Unlike markets.py's fetch_polymarket_matches() (which only sees currently-
OPEN markets), this targets closed=true events, so it works after the
tournament has ended — needed for a post-tournament market-baseline backtest.

CAVEAT: uses `lastTradePrice` as a proxy for the market's implied probability.
This is the LAST traded price before the market closed/resolved, which for a
liquid market close to kickoff is a reasonable proxy for the pre-match
implied probability — but it is not guaranteed to be the exact price at
kickoff, and could be influenced by trades placed after the match started
(if the market didn't freeze trading at kickoff). This is a known limitation,
not a hidden one.
"""
from __future__ import annotations
import json
import requests
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config


def fetch_polymarket_matches_resolved(series_slug: str = "soccer-fifwc") -> pd.DataFrame:
    rows: list[dict] = []
    offset = 0
    while True:
        try:
            r = requests.get(
                f"{config.POLYMARKET_GAMMA}/events",
                params={"series_slug": series_slug, "limit": 100,
                        "closed": "true", "offset": offset},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[markets_historical] {e}")
            break

        if not data:
            break

        for event in data:
            slug = event.get("slug", "") or ""
            title = event.get("title", "") or ""

            if not slug.startswith("fifwc"):
                continue
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
                last_price = market.get("lastTradePrice")
                if last_price is None:
                    continue
                last_price = float(last_price)

                q_lower = question.lower()
                if "draw" in q_lower:
                    outcome = "draw"
                elif home_team.lower() in q_lower:
                    outcome = "home_win"
                elif away_team.lower() in q_lower:
                    outcome = "away_win"
                else:
                    continue

                rows.append({
                    "platform": "polymarket",
                    "event_slug": slug,
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": date,
                    "outcome": outcome,
                    "price": last_price,
                    "market": question,
                })

        if len(data) < 100:
            break
        offset += 100

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch_polymarket_matches_resolved()
    print(f"Total rows: {len(df)}")
    print(df.head(20))