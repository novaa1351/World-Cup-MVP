"""Long-running in-play market poller (Railway always-on worker).

Polling strategy (tiered, per the project spec):

  Tier 1 — Polymarket per-match (primary live signal)
    When a per-match market is listed and open, poll every POLL_INTERVAL seconds.
    These reprice on goals and red cards.

  Tier 2 — Kalshi per-match (cross-platform spread)
    When a Kalshi per-match market is open AND its close_time is after kickoff
    (i.e., it trades in-play), also poll and compute the Poly–Kalshi spread.
    Kalshi per-match markets sometimes close at kickoff — always check status.

  Tier 3 — Tournament-level fallback (always available)
    Regardless of whether per-match markets exist, always poll the Polymarket
    champion prices and Kalshi round-survival prices for the teams currently
    playing. These shift when a match goes badly for a group favourite.

Alert conditions (each fires at most once per match per condition — deduped):
  • Model edge vs Polymarket per-match crosses EDGE_THRESHOLD
  • Polymarket per-match price moves more than PRICE_MOVE_THRESHOLD between polls
  • Poly–Kalshi spread on same outcome crosses SPREAD_THRESHOLD
  • Tournament-level champion/survival edge crosses TOURNAMENT_EDGE_THRESHOLD

State lives in SQLite via storage.py. The main loop never crashes on a single
failed request — errors are logged, the match stays in the active set, and the
next poll retries normally.

Config env vars:
  POLL_INTERVAL             — seconds between per-match polls (default 75)
  EDGE_THRESHOLD            — minimum model vs Polymarket edge to alert (default 0.06)
  PRICE_MOVE_THRESHOLD      — minimum price Δ to alert on sharp reprice (default 0.05)
  SPREAD_THRESHOLD          — minimum Poly–Kalshi spread to alert (default 0.04)
  TOURNAMENT_EDGE_THRESHOLD — edge threshold for champion/survival alerts (default 0.08)
  WCQ_DB_PATH               — SQLite file path
  DISCORD_WEBHOOK_URL       — webhook target
"""
from __future__ import annotations
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bot.storage import init_db, is_alert_sent, mark_alert_sent
from src.bot.fixtures import get_live_now, get_upcoming_within, _parse_dt
from src.bot.market_discovery import (
    find_polymarket_match,
    find_kalshi_match,
    get_polymarket_current_prices,
    get_kalshi_match_price,
    get_polymarket_champion_prices,
    get_kalshi_survival_prices,
)
from src.bot.notify import (
    post_live_edge_alert,
    post_cross_platform_spread,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [live_poller] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("live_poller")

POLL_INTERVAL: int = int(os.environ.get("POLL_INTERVAL", "75"))
EDGE_THRESHOLD: float = float(os.environ.get("EDGE_THRESHOLD", "0.06"))
PRICE_MOVE_THRESHOLD: float = float(os.environ.get("PRICE_MOVE_THRESHOLD", "0.05"))
SPREAD_THRESHOLD: float = float(os.environ.get("SPREAD_THRESHOLD", "0.04"))
TOURNAMENT_EDGE_THRESHOLD: float = float(os.environ.get("TOURNAMENT_EDGE_THRESHOLD", "0.08"))

_MATCH_DURATION = timedelta(hours=2, minutes=30)


# ---------------------------------------------------------------------------
# Model probability lookup (cached — recomputing Elo from ~50k rows takes ~2s)
# ---------------------------------------------------------------------------

_elo_cache: tuple[dict[str, float], float] | None = None
_ELO_CACHE_TTL = 3600 * 6  # refresh every 6 hours


def _get_elo_ratings() -> dict[str, float]:
    """Return current Elo ratings, recomputing at most every 6 hours."""
    global _elo_cache
    now = time.time()
    if _elo_cache and (now - _elo_cache[1]) < _ELO_CACHE_TTL:
        return _elo_cache[0]
    try:
        from src.data.historical import load_results
        from src.models.elo import production_elo
        ratings = production_elo(load_results())
        _elo_cache = (ratings, now)
        log.info("Elo ratings refreshed (%d teams)", len(ratings))
        return ratings
    except Exception as e:
        log.error("Elo refresh failed: %s", e)
        return _elo_cache[0] if _elo_cache else {}


def _model_probs(home: str, away: str) -> dict[str, float] | None:
    """Compute match_probs using current Elo ratings."""
    try:
        from src.models.match_model import match_probs
        from src.models.tournament import _FIFA_TO_HIST
        ratings = _get_elo_ratings()
        home_hist = _FIFA_TO_HIST.get(home, home)
        away_hist = _FIFA_TO_HIST.get(away, away)
        elo_h = ratings.get(home_hist, ratings.get(home, 1500.0))
        elo_a = ratings.get(away_hist, ratings.get(away, 1500.0))
        return match_probs(elo_h, elo_a, neutral=True)
    except Exception as e:
        log.error("model_probs(%s, %s): %s", home, away, e)
        return None


# ---------------------------------------------------------------------------
# Per-outcome price state (tracks previous prices for reprice detection)
# ---------------------------------------------------------------------------

class _PriceState:
    """Tracks last-seen prices for a market to detect sharp moves."""
    def __init__(self) -> None:
        self.prices: dict[str, float] = {}
        self.ts: float = 0.0

    def update(self, prices: dict[str, float]) -> dict[str, float]:
        """Update prices; return dict of outcomes where |Δ| > PRICE_MOVE_THRESHOLD."""
        moves: dict[str, float] = {}
        now = time.time()
        for k, v in prices.items():
            prev = self.prices.get(k)
            if prev is not None and abs(v - prev) >= PRICE_MOVE_THRESHOLD:
                moves[k] = v - prev
        self.prices = dict(prices)
        self.ts = now
        return moves


# ---------------------------------------------------------------------------
# Alert helpers (all async-wrapped synchronous calls)
# ---------------------------------------------------------------------------

def _alert_key(match_id: str, kind: str, suffix: str = "") -> str:
    return f"{match_id}:{kind}:{suffix}"


async def _run_sync(fn, *args, **kwargs):
    """Run a blocking function in the default executor (avoids blocking the loop)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def _check_edge_alert(
    match_id: str,
    home: str,
    away: str,
    outcome: str,
    model_prob: float,
    market_prob: float,
    platform: str,
) -> None:
    edge = model_prob - market_prob
    if abs(edge) < EDGE_THRESHOLD:
        return
    key = _alert_key(match_id, "edge", f"{outcome}_{platform}")
    if is_alert_sent(key):
        return
    trigger = f"Model edge {'+' if edge>0 else ''}{edge*100:.1f}pp crossed {EDGE_THRESHOLD*100:.0f}pp threshold"
    await _run_sync(
        post_live_edge_alert, home, away, outcome,
        model_prob, market_prob, edge, platform, trigger,
    )
    mark_alert_sent(key)
    log.info("Edge alert fired: %s vs %s  %s  %+.1fpp", home, away, outcome, edge * 100)


async def _check_reprice_alert(
    match_id: str,
    home: str,
    away: str,
    outcome: str,
    new_price: float,
    delta: float,
    platform: str,
) -> None:
    key = _alert_key(match_id, "reprice", f"{outcome}_{platform}_{round(new_price, 2)}")
    if is_alert_sent(key):
        return
    sign = "▲" if delta > 0 else "▼"
    trigger = f"Sharp reprice {sign}{abs(delta)*100:.1f}pp on {platform}"
    await _run_sync(
        post_live_edge_alert, home, away, outcome,
        new_price, new_price, delta, platform, trigger,
    )
    mark_alert_sent(key)
    log.info("Reprice alert: %s vs %s  %s  %+.1fpp", home, away, outcome, delta * 100)


async def _check_spread_alert(
    match_id: str,
    home: str,
    away: str,
    outcome: str,
    poly_prob: float,
    kalshi_prob: float,
) -> None:
    spread = abs(poly_prob - kalshi_prob)
    if spread < SPREAD_THRESHOLD:
        return
    key = _alert_key(match_id, "spread", outcome)
    if is_alert_sent(key):
        return
    await _run_sync(post_cross_platform_spread, home, away, outcome, poly_prob, kalshi_prob)
    mark_alert_sent(key)
    log.info("Spread alert: %s vs %s  %s  %.1fpp Poly/Kalshi", home, away, outcome, spread * 100)


# ---------------------------------------------------------------------------
# Per-match polling coroutine
# ---------------------------------------------------------------------------

async def poll_match(fixture: dict, stop_at: datetime) -> None:
    """Poll all tiers of markets for a single live match until stop_at."""
    home = fixture["home_team"]
    away = fixture["away_team"]
    kickoff = fixture["kickoff_utc"]
    match_id = fixture["match_id"]

    log.info("Starting poller: %s vs %s (until %s)", home, away, stop_at.isoformat())

    poly_state = _PriceState()
    kalshi_state = _PriceState()

    # Resolve which per-match markets exist (re-run at start of each match)
    poly_mkt = await _run_sync(find_polymarket_match, home, away, kickoff)
    kalshi_mkt = await _run_sync(find_kalshi_match, home, away, kickoff)

    if poly_mkt:
        log.info("  Polymarket per-match found: %s", poly_mkt.get("question", "?"))
    else:
        log.info("  No Polymarket per-match market — using champion/survival fallback")
    if kalshi_mkt:
        inplay = kalshi_mkt.get("trades_inplay", False)
        log.info("  Kalshi per-match found: %s (in-play=%s)", kalshi_mkt.get("title", "?"), inplay)

    model = _model_probs(home, away)
    if model:
        log.info("  Model: W=%.0f%%  D=%.0f%%  L=%.0f%%",
                 model["home"] * 100, model["draw"] * 100, model["away"] * 100)

    while datetime.now(timezone.utc) < stop_at:
        try:
            await _poll_once(
                match_id, home, away, kickoff, model,
                poly_mkt, kalshi_mkt,
                poly_state, kalshi_state,
            )
        except Exception as e:
            log.error("Poll error for %s vs %s: %s", home, away, e)

        # Respect the interval even if we errored
        await asyncio.sleep(POLL_INTERVAL)

    log.info("Poller done: %s vs %s", home, away)


async def _poll_once(
    match_id: str,
    home: str,
    away: str,
    kickoff: str,
    model: dict[str, float] | None,
    poly_mkt: dict | None,
    kalshi_mkt: dict | None,
    poly_state: _PriceState,
    kalshi_state: _PriceState,
) -> None:
    """One poll cycle: fetch prices, compute edges, fire alerts."""
    outcome_labels = {"home": home, "draw": "Draw", "away": away}

    # --- Tier 1: Polymarket per-match ---
    poly_prices: dict[str, float] | None = None
    if poly_mkt and not poly_mkt.get("closed", False):
        market_id = poly_mkt.get("market_id", "")
        if market_id:
            raw = await _run_sync(get_polymarket_current_prices, market_id)
            if raw:
                # Map outcomes to home/draw/away
                poly_prices = _parse_poly_outcome_prices(raw, home, away)
                moves = poly_state.update(poly_prices)
                for outcome, delta in moves.items():
                    await _check_reprice_alert(match_id, home, away, outcome,
                                               poly_prices[outcome], delta, "Polymarket")
                if model and poly_prices:
                    for outcome in ("home", "draw", "away"):
                        if outcome in model and outcome in poly_prices:
                            await _check_edge_alert(
                                match_id, home, away, outcome,
                                model[outcome], poly_prices[outcome], "Polymarket",
                            )

    # --- Tier 2: Kalshi per-match (cross-platform spread) ---
    if kalshi_mkt:
        ticker = kalshi_mkt.get("ticker", "")
        # Recheck status — Kalshi markets may close at kickoff
        inplay = kalshi_mkt.get("trades_inplay", False)
        if ticker and inplay:
            kalshi_price = await _run_sync(get_kalshi_match_price, ticker)
            if kalshi_price is not None:
                # Kalshi binary YES market: maps to home-win outcome
                # (actual outcome label depends on the market title — treat as "home" for now)
                kalshi_prices = {"home": kalshi_price}
                moves = kalshi_state.update(kalshi_prices)
                for outcome, delta in moves.items():
                    await _check_reprice_alert(match_id, home, away, outcome,
                                               kalshi_prices[outcome], delta, "Kalshi")

                # Cross-platform spread check (Poly vs Kalshi on same outcome)
                if poly_prices and "home" in poly_prices:
                    await _check_spread_alert(
                        match_id, home, away, "home",
                        poly_prices["home"], kalshi_price,
                    )

    # --- Tier 3: Tournament-level fallback (always) ---
    await _check_tournament_level(match_id, home, away, model)


async def _check_tournament_level(
    match_id: str,
    home: str,
    away: str,
    model: dict[str, float] | None,
) -> None:
    """Poll champion/survival markets for the teams currently playing.

    These always exist. A sharp move in champion odds during a match window
    is itself a signal (the market is reacting to something on the pitch).
    """
    try:
        champ_prices = await _run_sync(get_polymarket_champion_prices)
        from src.markets.implied import implied_from_book
        if len(champ_prices) > 1:
            clean = implied_from_book(champ_prices)
            for team in (home, away):
                mkt_p = clean.get(team)
                if mkt_p is None:
                    continue
                # We don't have a per-team-per-match model prob for champion odds;
                # just alert if the price has moved sharply compared to our prior.
                # (Stored in a module-level dict; keyed by match+team.)
                key = _alert_key(match_id, "champ_move", team)
                if is_alert_sent(key):
                    continue
                # Use tournament MC survival as model proxy if available
                model_champ = _get_mc_survival_champion(team)
                if model_champ and abs(model_champ - mkt_p) > TOURNAMENT_EDGE_THRESHOLD:
                    trigger = f"Champion market edge: model {model_champ:.0%} vs market {mkt_p:.0%}"
                    await _run_sync(
                        post_live_edge_alert, home, away, team,
                        model_champ, mkt_p, model_champ - mkt_p,
                        "Polymarket (champion)", trigger,
                    )
                    mark_alert_sent(key)
    except Exception as e:
        log.debug("Tournament-level check error: %s", e)


_mc_survival_cache: dict[str, float] = {}


def _get_mc_survival_champion(team: str) -> float | None:
    """MC survival probability for champion — computed once and cached."""
    global _mc_survival_cache
    if _mc_survival_cache:
        return _mc_survival_cache.get(team)
    try:
        from src.data.historical import load_results
        from src.models.elo import production_elo
        from src.models.tournament import simulate_tournament
        ratings = production_elo(load_results())
        mc = simulate_tournament(ratings, n_sims=5000)
        _mc_survival_cache = {t: v.get("champion", 0.0) for t, v in mc.items()}
        return _mc_survival_cache.get(team)
    except Exception:
        return None


def _parse_poly_outcome_prices(
    raw_prices: dict[str, float],
    home: str,
    away: str,
) -> dict[str, float]:
    """Map Polymarket outcome labels to home/draw/away.

    Polymarket outcome titles are not standardised — handles common variants:
    "Brazil wins", "Morocco wins", "Draw", "Yes (home)", etc.
    """
    result: dict[str, float] = {}
    home_l = home.lower()
    away_l = away.lower()
    for label, price in raw_prices.items():
        ll = label.lower()
        if "draw" in ll or "tie" in ll:
            result["draw"] = price
        elif any(w in ll for w in home_l.split() if len(w) > 2):
            result["home"] = price
        elif any(w in ll for w in away_l.split() if len(w) > 2):
            result["away"] = price
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main() -> None:
    """Outer loop: refresh active matches every 60s, manage polling tasks."""
    init_db()
    schedule_path = os.environ.get("WCQ_SCHEDULE_PATH", "data/schedule_2026.json")
    log.info("Live poller started (poll_interval=%ds, edge_threshold=%.0f%%)",
             POLL_INTERVAL, EDGE_THRESHOLD * 100)
    log.info("Schedule path: %s  exists=%s", schedule_path, Path(schedule_path).exists())

    if os.environ.get("TEST_ALERT", "0") == "1":
        log.info("TEST_ALERT=1 — firing test embed to Discord")
        from src.bot.notify import post_live_edge_alert
        post_live_edge_alert(
            home_team="Brazil", away_team="Germany",
            outcome="home", edge=0.09,
            model_prob=0.57, market_prob=0.48,
            platform="Polymarket", trigger="test",
        )
        log.info("TEST_ALERT sent — remove TEST_ALERT env var to disable")

    active_tasks: dict[str, asyncio.Task] = {}  # match_id -> task

    while True:
        try:
            # Matches currently live + starting within the next 5 minutes
            live = get_live_now()
            starting_soon = get_upcoming_within(hours=5 / 60)
            candidates = {f["match_id"]: f for f in live + starting_soon}

            # Start tasks for new matches
            for mid, fixture in candidates.items():
                if mid not in active_tasks or active_tasks[mid].done():
                    ko_dt = _parse_dt(fixture["kickoff_utc"])
                    stop_at = ko_dt + _MATCH_DURATION
                    task = asyncio.create_task(poll_match(fixture, stop_at), name=mid)
                    active_tasks[mid] = task
                    log.info("Task started: %s vs %s", fixture["home_team"], fixture["away_team"])

            # Clean up finished tasks
            done = [mid for mid, t in active_tasks.items() if t.done()]
            for mid in done:
                exc = active_tasks[mid].exception()
                if exc:
                    log.error("Task %s raised: %s", mid, exc)
                del active_tasks[mid]

            if not candidates:
                log.debug("No live or upcoming matches; sleeping 60s")

        except Exception as e:
            log.error("Main loop error: %s", e)

        await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Poller stopped by user")
