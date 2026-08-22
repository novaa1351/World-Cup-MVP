"""Pre-match briefing job — posts ~1 hour before kickoff.

Designed to run every 30 minutes via GitHub Actions cron. Each run:
  1. Finds fixtures kicking off 45–90 minutes from now (UTC).
  2. For each, fetches model probs + re-runs market discovery.
  3. Posts a pre-match briefing embed and places paper bets.
  4. Deduplication: a briefing is only posted once per match_id.

Env vars required:
  DISCORD_WEBHOOK_URL
  WCQ_DB_PATH
  WCQ_SCHEDULE_PATH

Optional:
  DRY_RUN=1
  API_FOOTBALL_KEY
  WCQ_KELLY_EDGE_THRESHOLD (default 0.03)
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.storage import init_db, make_match_id, save_prediction, is_alert_sent, mark_alert_sent
from src.bot.fixtures import load_schedule, _parse_dt
from src.bot.notify import post_pre_match_briefing
from src.bot.market_discovery import find_polymarket_match, find_kalshi_match
from src.bot.paper_trader import consider_bets_for_match
from src.markets.implied import implied_from_book

_PRE_MATCH_WINDOW_MIN = (45, 90)  # post briefing when kickoff is 45–90 min away


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _get_model_data() -> tuple[dict, object, object]:
    from src.data.historical import load_results
    from src.models.elo import production_elo
    from src.models.match_model import match_probs
    from src.models.tournament import _FIFA_TO_HIST
    ratings = production_elo(load_results())
    return ratings, match_probs, _FIFA_TO_HIST


def _parse_match_market_probs(mkt: dict, home: str, away: str) -> dict[str, float] | None:
    outcomes = mkt.get("outcomes", [])
    prices = mkt.get("prices", [])
    if not outcomes or not prices:
        return None
    raw: dict[str, float] = {}
    home_l = home.lower()
    away_l = away.lower()
    for label, p in zip(outcomes, prices):
        ll = label.lower()
        if "draw" in ll or "tie" in ll:
            raw["draw"] = p
        elif any(w in ll for w in home_l.split() if len(w) > 2):
            raw["home"] = p
        elif any(w in ll for w in away_l.split() if len(w) > 2):
            raw["away"] = p
    if len(raw) >= 2:
        return implied_from_book(raw)
    return None


def run() -> None:
    init_db()

    now = _now()
    lo = now + timedelta(minutes=_PRE_MATCH_WINDOW_MIN[0])
    hi = now + timedelta(minutes=_PRE_MATCH_WINDOW_MIN[1])

    upcoming = [
        f for f in load_schedule()
        if lo <= _parse_dt(f["kickoff_utc"]) <= hi
    ]
    if not upcoming:
        print(f"[pre_match] No fixtures kicking off in {_PRE_MATCH_WINDOW_MIN[0]}–{_PRE_MATCH_WINDOW_MIN[1]} min")
        return

    print(f"[pre_match] {len(upcoming)} fixture(s) in window")

    try:
        ratings, match_probs, FIFA_TO_HIST = _get_model_data()
    except Exception as e:
        print(f"[pre_match] Model load failed: {e}")
        ratings = {}
        match_probs = None
        FIFA_TO_HIST = {}

    for f in upcoming:
        home, away = f["home_team"], f["away_team"]
        kickoff = f["kickoff_utc"]
        mid = make_match_id(home, away, kickoff)

        # Dedup — only post once per match
        dedup_key = f"pre_match:{mid}"
        if is_alert_sent(dedup_key):
            print(f"[pre_match] Already posted briefing for {home} vs {away}")
            continue

        # Model probabilities + Elo ratings for the embed
        model: dict[str, float] | None = None
        elo_h = elo_a = 1500.0
        if ratings and match_probs:
            try:
                home_hist = FIFA_TO_HIST.get(home, home)
                away_hist = FIFA_TO_HIST.get(away, away)
                elo_h = ratings.get(home_hist, ratings.get(home, 1500.0))
                elo_a = ratings.get(away_hist, ratings.get(away, 1500.0))
                model = match_probs(elo_h, elo_a, neutral=True)
            except Exception as e:
                print(f"[pre_match] match_probs error: {e}")

        # Market probabilities — try Polymarket per-match, fall back to None
        market: dict[str, float] | None = None
        platform = "Polymarket"
        poly_mkt = find_polymarket_match(home, away, kickoff)
        if poly_mkt:
            market = _parse_match_market_probs(poly_mkt, home, away)

        if not market:
            kalshi_mkt = find_kalshi_match(home, away, kickoff)
            if kalshi_mkt and kalshi_mkt.get("yes_bid") is not None:
                bid = float(kalshi_mkt["yes_bid"])
                ask = float(kalshi_mkt.get("yes_ask", bid))
                mid_price = (bid + ask) / 2
                # Kalshi binary = home win; synthesise W/D/L from model ratios
                market = {"home": mid_price}
                platform = "Kalshi"

        # Persist prediction for calibration scoring later
        if model:
            save_prediction(mid, home, away, kickoff, model, market, platform.lower())

        # Post briefing embed
        post_pre_match_briefing(
            home, away, kickoff,
            model or {},
            market,
            elo_h, elo_a,
            platform=platform,
        )
        mark_alert_sent(dedup_key)
        print(f"[pre_match] Posted briefing for {home} vs {away}")

        # Paper bets — only place when we have a full W/D/L market
        if model and market and len(market) == 3:
            bets = consider_bets_for_match(mid, home, away, model, market, platform.lower())
            for b in bets:
                print(f"[pre_match] Paper bet: {b['outcome']} ${b['stake']:.2f} (edge {b['edge']*100:+.1f}pp)")


if __name__ == "__main__":
    run()
