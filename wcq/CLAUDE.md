# World Cup Quant Dashboard

Python quant/data project (CS freshman portfolio piece). Be practical and modular.
Explain tradeoffs briefly when making design choices.

## What this project does

Pulls live prediction-market odds (Polymarket + Kalshi) for the 2026 World Cup,
generates independent model probabilities from historical match data, and surfaces
where the two disagree (value edges, arbitrage). Educational/analytical only —
no actual trading.

## Architecture

```
results.csv (50k matches)          Polymarket / Kalshi (live odds)
     ↓                                      ↓
historical.py (load/clean)           markets.py (fetch prices)
     ↓                                      ↓
elo.py (team ratings)              implied.py (de-vig → clean probs)
     ↓                                      ↓
match_model.py (win/draw/loss)             |
     ↓                                      |
tournament.py (MC bracket sim)             |
     ↓                                      |
svi_surface.py (survival curves)           |
     ↓_____________________________________|
                   edges.py (model vs market)
                         ↓
               streamlit_app.py (dashboard)
                         ↓
              src/bot/  (Discord notification layer)
```

Supporting files: `config.py` (shared settings/paths, no dependencies),
`backtest/engine.py` (staking sim), `viz/charts.py` (plotly charts).

## Current state (as of end of June 8 2026 session)

### Dashboard (fully built and wired)

| Tab | Contents |
|-----|----------|
| Live markets | Raw Polymarket + Kalshi prices; pivot table of Kalshi round survival prices |
| Model forecast | Top-20 Elo ratings; MC survival probability table; group tables |
| Edge detection | Polymarket champion edges (bar chart, scatter); Kalshi round-survival edges with correct round-label remapping |
| Survival surface | SVI-style survival curves per team |
| Backtest | Six historical WC backtests (2002–2022); per-match predictions table; **2026 forward simulation** (NOT a backtest — uses live Kalshi prices + MC paths, shows bankroll distribution histogram) |
| Findings | Six-WC Brier/hit-rate summary chart; 5 key findings; per-year table; expected 2026 knockout bracket diagram |

### Discord bot (fully built and deployed)

Two deployment targets, both live as of June 8:

**GitHub Actions** — three scheduled jobs:

| Job | Schedule | What it does |
|-----|----------|--------------|
| `jobs/daily_digest.py` | 07:00 UTC daily | Posts all fixtures today with model W/D/L vs market probs and top 5 edges |
| `jobs/pre_match.py` | Every 30 min | Posts briefing when kickoff is 45–90 min away; places paper bets if edge > threshold |
| `jobs/post_match.py` | Every 30 min | Posts scorecard 10–90 min after full time; settles paper bets; posts P&L |

**Railway** — always-on asyncio worker (`src/bot/live_poller.py`):
- Polls Polymarket + Kalshi every 75s during match windows
- Fires edge alerts when model vs market divergence > 6%
- Fires spread alerts when Polymarket vs Kalshi disagree > 4%

**Bot modules** (`src/bot/`):

| Module | Job |
|--------|-----|
| `storage.py` | SQLite layer; predictions, results, calibration, paper ledger, dedup keys |
| `fixtures.py` | Schedule download + cache; 104 WC 2026 fixtures from fixturedownload.com |
| `notify.py` | All Discord embeds; per-channel webhook routing |
| `market_discovery.py` | Runtime Polymarket + Kalshi market discovery; 30-min cache |
| `results.py` | Multi-source results: Polymarket resolution → API-Football → martj42 CSV |
| `paper_trader.py` | Fractional Kelly, $1K bankroll, 25% cap, correlated-bet guard |
| `live_poller.py` | Asyncio polling worker for Railway |

**Per-channel Discord webhook routing** (as of June 8):

| Env var | Channel | Notifications |
|---------|---------|---------------|
| `DISCORD_WEBHOOK_DIGEST` | `#daily-digest` | Morning digest |
| `DISCORD_WEBHOOK_PRE_MATCH` | `#pre-match` | Pre-match briefing |
| `DISCORD_WEBHOOK_POST_MATCH` | `#post-match` | Post-match scorecard |
| `DISCORD_WEBHOOK_PAPER` | `#paper-trading` | Bet placed, P&L report |
| `DISCORD_WEBHOOK_LIVE` | `#live-alerts` | Live edge alerts, spread alerts |

All channel vars fall back to `DISCORD_WEBHOOK_URL` if unset.

**State persistence:**
- GitHub Actions: commits `wcq_bot.db` back to repo after each run (`[skip ci]`)
- Railway: persistent volume at `/data`; `WCQ_DB_PATH=/data/wcq_bot.db`
- Schedule copied to volume on startup via `railway.json` startCommand

**Secrets required:**

| Location | Secret | Purpose |
|----------|--------|---------|
| GitHub + Railway | `DISCORD_WEBHOOK_URL` | Fallback webhook |
| GitHub + Railway | All `DISCORD_WEBHOOK_*` vars | Per-channel routing |
| GitHub | `API_FOOTBALL_KEY` | Post-match result fetching |
| Railway | `API_FOOTBALL_KEY` | Results fallback in live poller |

## Key model details

- **Elo**: trained on all matches from 1872 → present; 5 % annual mean-reversion;
  5-tier tournament K-weighting (WC finals K=60, qualifiers K=40, friendlies K=20)
- **Draw model**: `P(draw|Δelo) = draw_base × exp(-|Δelo|/scale)`, MLE-fitted:
  draw_base=0.3131, scale=318.2; params stored in `data/draw_params.json`
- **Monte Carlo**: 20 000 simulations of 48-team 2026 bracket (configurable in sidebar)
- **Calibrated draw params** live in `data/draw_params.json`; if missing, sidebar shows
  a calibration button that re-fits and writes the file (~5 s)

## Kalshi round label mapping (critical — gets it wrong if missed)

Kalshi asks "Will X **qualify for** [Round]?" which means the team **reaches** that
round. Our model labels refer to the round a team **wins**. The off-by-one:

| Kalshi market | Model key (mc_survival) | Sim dict key |
|---------------|------------------------|-------------|
| R16           | R32                    | R32         |
| QF            | R16                    | R16         |
| SF            | QF                     | QF          |
| final         | final (= SF value)     | SF          |

This mapping is defined as `_KALSHI_TO_MODEL_ROUND` in the edge tab and
`_FWDMAP_MC` / `_FWDMAP_SIM` in the backtest forward-simulation section.

## Forward simulation design

`run_forward_simulation()` in `streamlit_app.py` (cached) replicates the full
tournament.py simulation loop but returns per-sim outcomes:
  `list[{"R32": set[str], "R16": set[str], "QF": set[str], "SF": set[str], "champion": str}]`

The UI section in the Backtest tab then:
1. Fetches live Kalshi prices, applies the round-label remapping
2. Filters bets above the edge threshold
3. For each simulation path, applies Kelly staking round-by-round (all bets within
   a round sized from the start-of-round bankroll simultaneously)
4. Shows a histogram of 1K–20K final bankrolls + median / percentile / P(profit) metrics

## SVI framing — say this clearly in comments and docstrings

SVI (Stochastic Volatility Inspired) is an options implied-vol interpolation
method. What transfers to this project is the *methodology*: smooth,
low-parameter, no-arbitrage-constrained, market-calibrated curve fitting.
What does NOT transfer is the literal SVI hyperbola — that is tuned to vol smiles.

The mapping:
  - Options maturity axis      → tournament round depth (group → champion)
  - Implied-vol surface value  → P(team survives to that round)
  - Calendar no-arbitrage      → survival monotone non-increasing in round depth
  - Butterfly no-arbitrage     → per-round probs form a valid distribution

## Code style

- Python 3.10+, type hints on public functions
- Every module has an `if __name__ == "__main__":` block that demos its output
- No file reaches "up" to import from something that imports it (one-directional flow)
- Graceful fallback when network calls fail (markets.py already does this)
- Keep modules focused: one clear job each
- Docstrings on all public functions. Inline comments for non-obvious math.

## Libraries

pandas, numpy, scipy (optimization), plotly (all charts), streamlit (dashboard),
requests (API calls), aiohttp (live poller async HTTP). scipy.optimize.minimize is
used for SVI calibration. Do NOT add new dependencies without checking first.

## Key commands

```bash
# From the project root (wcq/):
python src/data/historical.py          # download + verify real data
python tests/test_smoke.py             # run smoke tests
streamlit run app/streamlit_app.py     # launch dashboard

# Bot modules (run from wcq/):
python src/bot/storage.py              # init wcq_bot.db
python src/bot/fixtures.py --download  # refresh schedule cache
python src/bot/live_poller.py          # start live poller locally

# Data lives at data/raw/results.csv (gitignored after first download)
# API endpoints (public, no auth):
#   Polymarket: https://gamma-api.polymarket.com/markets
#   Kalshi:     https://external-api.kalshi.com/trade-api/v2/markets
```

## Known gotchas

- `sys.path.insert(0, parents[2])` in all `src/bot/` files points to `wcq/` (not repo root).
  `parents[3]` would be wrong and cause `No module named 'src'`.
- API-Football league ID for WC 2026 is `_WC_LEAGUE_ID = 1` in `results.py` — confirm
  at tournament start if results aren't coming through.
- Polymarket outcome label format in `_parse_poly_outcome_prices()` (live_poller.py) may
  need adjustment once per-match markets go live (days before June 11).
- GitHub Actions default token permissions may be read-only — all three workflows now
  explicitly set `permissions: contents: write` to allow DB commits.
- Railway shell is ephemeral (no volume mount) — use the `startCommand` cp in
  `railway.json` to seed files onto the volume, not the shell.

## What to build next (priority order)

1. **Forward simulation UX improvements** — histogram can show extreme right-tail
   outliers (Kelly compounding artifact). Consider log-scale x-axis toggle or
   capping at 99th percentile with a note.
2. **Group-stage betting** — Kalshi only covers knockout rounds; Polymarket may
   have group-stage markets. Could add group-winner / top-2 markets to the forward sim.
3. **Per-bet attribution** — in the forward sim, show which individual bets
   contributed most to the P&L distribution.
4. **Recent-form time-decay weighting** in `elo.py` — matches from the last
   12 months get a higher weight multiplier.
5. **Smoke tests** — `tests/test_smoke.py` may not cover the forward simulation
   code path. Add a fast test (100 sims) checking return shape and round-set sizes.
