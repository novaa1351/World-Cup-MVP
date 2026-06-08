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
```

Supporting files: `config.py` (shared settings/paths, no dependencies),
`backtest/engine.py` (staking sim), `viz/charts.py` (plotly charts).

## Current state (as of end of June 7 2026 session)

Everything in the architecture above is **built and wired**. The dashboard has
six tabs and is fully functional:

| Tab | Contents |
|-----|----------|
| Live markets | Raw Polymarket + Kalshi prices; pivot table of Kalshi round survival prices |
| Model forecast | Top-20 Elo ratings; MC survival probability table; group tables |
| Edge detection | Polymarket champion edges (bar chart, scatter); Kalshi round-survival edges with correct round-label remapping |
| Survival surface | SVI-style survival curves per team |
| Backtest | Six historical WC backtests (2002–2022); per-match predictions table; **2026 forward simulation** (NOT a backtest — uses live Kalshi prices + MC paths, shows bankroll distribution histogram) |
| Findings | Six-WC Brier/hit-rate summary chart; 5 key findings; per-year table; expected 2026 knockout bracket diagram |

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

## Forward simulation design (new as of Jun 7)

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
requests (API calls). scipy.optimize.minimize is used for SVI calibration.
Do NOT add new dependencies without checking first.

## Key commands

```bash
# From the project root (wcq/):
python src/data/historical.py      # download + verify real data
python tests/test_smoke.py         # run smoke tests
streamlit run app/streamlit_app.py # launch dashboard

# Data lives at data/raw/results.csv (gitignored after first download)
# API endpoints (public, no auth):
#   Polymarket: https://gamma-api.polymarket.com/markets
#   Kalshi:     https://external-api.kalshi.com/trade-api/v2/markets
```

## What to build next (priority order)

1. **Forward simulation UX improvements** — currently the histogram can show
   extreme right-tail outliers (Kelly compounding artifact) that compress the
   useful range. Consider log-scale x-axis toggle, or capping display at 99th
   percentile with a note.
2. **Group-stage betting** — Kalshi only covers knockout rounds; Polymarket may
   have group-stage markets. Could add group-winner / top-2 markets to the
   forward sim if data is available.
3. **Per-bet attribution** — in the forward sim, show which individual bets
   contributed most to the P&L distribution (useful for understanding model
   conviction vs noise).
4. **Recent-form time-decay weighting** in `elo.py` — matches from the last
   12 months get a higher weight multiplier. Would improve accuracy for teams
   that have changed dramatically in form.
5. **Smoke tests** — `tests/test_smoke.py` exists but may not cover the new
   forward simulation code path. Add a fast test (100 sims) that checks return
   shape and round-set sizes.
