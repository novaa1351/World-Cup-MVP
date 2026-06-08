"""Post-match scorecard job — posts result, calibration update, and settles paper bets.

Designed to run every 30 minutes via GitHub Actions cron. Each run:
  1. Finds fixtures whose match window ended 10–90 minutes ago.
  2. Tries to fetch the result (market resolution → football-data.org → martj42).
  3. Posts the scorecard embed with running Brier/log-loss comparison.
  4. Settles any open paper bets for the match.
  5. Deduplication: scorecard is posted at most once per match_id.

Env vars required:
  DISCORD_WEBHOOK_URL
  WCQ_DB_PATH
  WCQ_SCHEDULE_PATH

Optional:
  DRY_RUN=1
  FOOTBALL_DATA_API_KEY      (enables football-data.org source)
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.storage import (
    init_db, get_prediction, get_result,
    compute_and_save_calibration, get_latest_calibration,
    is_alert_sent, mark_alert_sent,
)
from src.bot.fixtures import load_schedule, _parse_dt, _MATCH_DURATION
from src.bot.results import fetch_and_store_result
from src.bot.notify import post_post_match_scorecard
from src.bot.paper_trader import settle_match

_RESULT_WINDOW_MIN = (10, 90)  # look for results 10–90 min after full time


def _now() -> datetime:
    return datetime.now(timezone.utc)


def run() -> None:
    init_db()

    now = _now()

    recently_finished = [
        f for f in load_schedule()
        if (
            timedelta(minutes=_RESULT_WINDOW_MIN[0])
            <= (now - (_parse_dt(f["kickoff_utc"]) + _MATCH_DURATION))
            <= timedelta(minutes=_RESULT_WINDOW_MIN[1])
        )
    ]

    if not recently_finished:
        print(f"[post_match] No fixtures finished in the last {_RESULT_WINDOW_MIN[1]} min")
        return

    print(f"[post_match] {len(recently_finished)} recently finished fixture(s)")

    for f in recently_finished:
        home, away = f["home_team"], f["away_team"]
        kickoff = f["kickoff_utc"]
        from src.bot.storage import make_match_id
        mid = make_match_id(home, away, kickoff)

        # Dedup — only post one scorecard per match
        dedup_key = f"post_match:{mid}"
        if is_alert_sent(dedup_key):
            print(f"[post_match] Already posted scorecard for {home} vs {away}")
            continue

        # Try to get the result
        result = fetch_and_store_result(mid, home, away, kickoff)
        if not result:
            print(f"[post_match] Result not available yet for {home} vs {away} — will retry")
            continue

        print(f"[post_match] Result: {home} {result['home_score']}-{result['away_score']} {away} "
              f"(source: {result.get('source', '?')})")

        # Retrieve stored prediction (may be None if briefing job didn't run)
        pred = get_prediction(mid)
        model_probs: dict[str, float] | None = None
        market_probs: dict[str, float] | None = None
        if pred:
            model_probs = {
                "home": pred.get("p_home_model") or 0.0,
                "draw": pred.get("p_draw_model") or 0.0,
                "away": pred.get("p_away_model") or 0.0,
            }
            if pred.get("p_home_market") is not None:
                market_probs = {
                    "home": pred.get("p_home_market") or 0.0,
                    "draw": pred.get("p_draw_market") or 0.0,
                    "away": pred.get("p_away_market") or 0.0,
                }

        # Recompute calibration including this new result
        calibration = compute_and_save_calibration()
        if not calibration:
            calibration = get_latest_calibration() or {}

        # Post scorecard (safe even if model_probs is None — embed degrades gracefully)
        post_post_match_scorecard(
            home, away,
            result["home_score"], result["away_score"],
            model_probs or {},
            market_probs,
            calibration,
        )
        mark_alert_sent(dedup_key)
        print(f"[post_match] Posted scorecard for {home} vs {away}")

        # Settle paper bets
        settled = settle_match(mid, result["winner"])
        for b in settled:
            sign = "+" if b["pnl"] > 0 else ""
            print(f"[post_match] Settled bet {b['id']}: {b['outcome']} → {b['status']} "
                  f"({sign}${b['pnl']:.2f})")

        if settled:
            from src.bot.paper_trader import post_pnl_update
            post_pnl_update()


if __name__ == "__main__":
    run()
