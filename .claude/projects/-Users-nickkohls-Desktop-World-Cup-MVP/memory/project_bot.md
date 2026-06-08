---
name: bot-architecture
description: Discord bot system added to wcq/ — modules, job scripts, deployment targets, and key design decisions
metadata:
  type: project
---

WC 2026 Discord bot built June 2026. All files live in wcq/.

**Why:** Educational bot for a private dev-friends server, posting model vs market analysis for WC 2026 matches.

**Module map:**
- `src/bot/storage.py` — SQLite persistence (WCQ_DB_PATH env var, :memory: supported)
- `src/bot/fixtures.py` — Schedule JSON cache at data/schedule_2026.json; download once with `python src/bot/fixtures.py --download`
- `src/bot/notify.py` — Discord webhook embeds (DISCORD_WEBHOOK_URL env var, DRY_RUN=1 for testing)
- `src/bot/market_discovery.py` — Runtime Polymarket/Kalshi per-match market discovery (no hardcoded slugs; market_cache.json with 30-min TTL)
- `src/bot/results.py` — Multi-source results: Polymarket resolution → football-data.org (FOOTBALL_DATA_API_KEY) → martj42 CSV
- `src/bot/paper_trader.py` — Fractional Kelly, 25% cap, per-match W/D/L only, correlated-bet guard prevents stacking same team
- `src/bot/live_poller.py` — Asyncio long-running worker; tiered: Poly per-match > Kalshi per-match > champion/survival fallback

**Job scripts** in `wcq/jobs/`: daily_digest.py, pre_match.py (45-90min before kickoff), post_match.py (10-90min after full time). All use dedup keys in sent_alerts table.

**Deployment:**
- GitHub Actions: `.github/workflows/wc_daily_digest.yml`, `wc_pre_match.yml`, `wc_post_match.yml`. Commits wcq_bot.db back to repo for persistence.
- Railway: `wcq/Procfile` + `wcq/railway.json`, volume at /data for SQLite.
- Full setup guide: `wcq/DEPLOY.md`

**Key design choices:**
- Polymarket per-match markets discovered at runtime via /events?slug=world-cup-matches (roll out days before kickoff)
- Kalshi per-match: search /series dynamically; always check close_time before treating as in-play
- Calibration scoreboard: Brier + log-loss for both model and market; deliberately candid when model loses
- Paper trader excludes tournament-level champion/survival bets (settlement horizon too long)
- All times stored/computed in UTC; convert only for display

**Why:** See [[user_role]] for user context. Project is a CS freshman portfolio piece, so pragmatism over over-engineering.

**How to apply:** When adding features to the bot, follow the existing import structure (sys.path.insert at top of each file), use storage.py for all persistence, and keep modules independently runnable via `if __name__ == "__main__"`.
