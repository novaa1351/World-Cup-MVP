"""Daily digest job — posts today's fixtures with model W/D/L vs market probabilities.

Designed to run once per day via GitHub Actions cron (~07:00 UTC).
Reads all fixtures scheduled for today's UTC date, runs the model, fetches
live market probabilities where available, and posts the digest embed.

Env vars required:
  DISCORD_WEBHOOK_URL
  WCQ_DB_PATH              (default: wcq_bot.db)
  WCQ_SCHEDULE_PATH        (default: data/schedule_2026.json)

Optional:
  DRY_RUN=1                prints payload instead of posting
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.storage import init_db, make_match_id, save_prediction
from src.bot.fixtures import get_todays_fixtures
from src.bot.notify import post_daily_digest
from src.bot.market_discovery import (
    get_polymarket_champion_prices,
    get_kalshi_survival_prices,
    find_polymarket_match,
)
from src.markets.implied import implied_from_book
from src.markets.edges import edge_table, flag_value


def _get_elo_ratings() -> dict[str, float]:
    from src.data.historical import load_results
    from src.models.elo import production_elo
    return production_elo(load_results())


def _market_probs_for_match(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict[str, float] | None:
    """Try to get per-match market W/D/L probabilities from Polymarket.

    Falls back to None if no per-match market is listed yet — the digest
    will show model-only probabilities in that case.
    """
    try:
        mkt = find_polymarket_match(home, away, kickoff_utc)
        if not mkt:
            return None
        outcomes = mkt.get("outcomes", [])
        prices = mkt.get("prices", [])
        if not outcomes or not prices:
            return None
        raw = dict(zip(outcomes, prices))
        # Map to home/draw/away
        home_l = home.lower()
        away_l = away.lower()
        mapped: dict[str, float] = {}
        for label, p in raw.items():
            ll = label.lower()
            if "draw" in ll or "tie" in ll:
                mapped["draw"] = p
            elif any(w in ll for w in home_l.split() if len(w) > 2):
                mapped["home"] = p
            elif any(w in ll for w in away_l.split() if len(w) > 2):
                mapped["away"] = p
        if len(mapped) >= 2:
            return implied_from_book(mapped)
    except Exception as e:
        print(f"[daily_digest] market_probs for {home} vs {away}: {e}")
    return None


def run() -> None:
    init_db()

    # Load Elo ratings (re-computes from full history — ~2s on GitHub Actions)
    print("[daily_digest] Computing Elo ratings...")
    try:
        ratings = _get_elo_ratings()
        from src.models.tournament import _FIFA_TO_HIST
        from src.models.match_model import match_probs
    except Exception as e:
        print(f"[daily_digest] Failed to load model: {e}")
        ratings = {}

    fixtures = get_todays_fixtures()
    if not fixtures:
        print("[daily_digest] No fixtures today — skipping digest")
        return

    print(f"[daily_digest] {len(fixtures)} fixtures today")

    # Fetch champion prices for edge computation fallback
    champ_prices = get_polymarket_champion_prices()
    champ_implied = implied_from_book(champ_prices) if len(champ_prices) > 1 else {}

    enriched_fixtures = []
    all_edges: list[dict] = []

    for f in fixtures:
        home, away = f["home_team"], f["away_team"]
        kickoff = f["kickoff_utc"]
        mid = make_match_id(home, away, kickoff)

        # Model probabilities
        model: dict[str, float] | None = None
        elo_h = elo_a = None
        if ratings:
            try:
                from src.models.tournament import _FIFA_TO_HIST
                from src.models.match_model import match_probs
                home_hist = _FIFA_TO_HIST.get(home, home)
                away_hist = _FIFA_TO_HIST.get(away, away)
                elo_h = ratings.get(home_hist, ratings.get(home, 1500.0))
                elo_a = ratings.get(away_hist, ratings.get(away, 1500.0))
                model = match_probs(elo_h, elo_a, neutral=True)
            except Exception as e:
                print(f"[daily_digest] match_probs {home} vs {away}: {e}")

        # Market probabilities (per-match when available)
        market = _market_probs_for_match(home, away, kickoff)

        # Save prediction for later calibration scoring
        if model:
            save_prediction(mid, home, away, kickoff, model, market, "polymarket")

        # Compute edges for today's "biggest edges" section
        if model and market:
            et = edge_table(
                {k: model[k] for k in ("home", "draw", "away")},
                {k: market.get(k, 0.0) for k in ("home", "draw", "away")},
            )
            for _, row in et.iterrows():
                all_edges.append({
                    "outcome": f"{home if row['outcome'] == 'home' else (away if row['outcome'] == 'away' else 'Draw')} ({home} vs {away})",
                    "model_prob": row["model_prob"],
                    "market_prob": row["market_prob"],
                    "edge": row["edge"],
                    "edge_pct": row["edge_pct"],
                })

        enriched_fixtures.append({
            **f,
            "model_probs": model,
            "market_probs": market,
        })

    # Sort edges by |edge|, deduplicated across fixtures
    all_edges.sort(key=lambda x: abs(x["edge"]), reverse=True)

    post_daily_digest(enriched_fixtures, all_edges[:5])
    print("[daily_digest] Done")


if __name__ == "__main__":
    run()
