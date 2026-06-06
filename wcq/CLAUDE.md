# World Cup Quant Dashboard

Python quant/data project (CS freshman portfolio piece). Be practical and modular.
Explain tradeoffs briefly when making design choices.

## What this project does

Pulls live prediction-market odds (Polymarket + Kalshi) for the 2026 World Cup,
generates independent model probabilities from historical match data, and surfaces
where the two disagree (value edges, arbitrage). Educational/analytical only —
no actual trading.

## Architecture: two tracks converging

```
results.csv (50k matches)          Polymarket / Kalshi (live odds)
     ↓                                      ↓
historical.py (load/clean)           markets.py (fetch prices)
     ↓                                      ↓
elo.py (team ratings)              implied.py (de-vig → clean probs)
     ↓                                      ↓
match_model.py (win/draw/loss)             |
     ↓                                      |
[tournament.py -- NOT BUILT YET]           |
     ↓                                      |
svi_surface.py (survival curves)           |
     ↓_____________________________________|
                   edges.py (model vs market)
                         ↓
               streamlit_app.py (dashboard)
```

Supporting files: `config.py` (shared settings/paths, no dependencies),
`backtest/engine.py` (staking sim), `viz/charts.py` (plotly charts).

## Key gap right now

`tournament.py` does not exist yet. It is the missing link: takes match
probabilities from match_model.py, simulates the 48-team 2026 bracket via
Monte Carlo, and outputs per-team survival probabilities per round. Those
probabilities become the anchors fed to `svi_surface.py`.

Until tournament.py is built, the edge tab in the dashboard uses placeholder
probabilities (model_p == market_p), so it shows zero edge.

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

Be honest about this distinction in any comments or docs you write.

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

1. `src/models/tournament.py` — Monte Carlo bracket sim, 48 teams, 12 groups of 4,
   2026 format (top 2 + 8 best thirds → R32). Output: dict[team → dict[round → float]].
2. Wire tournament output → SurvivalSurface anchors → edge tab real probabilities.
3. Dixon-Coles bivariate-Poisson upgrade for match_model.py (better draw rate).
4. Calibration/reliability diagram in backtest/ using resolved 2018/2022 data.
5. Recent-form time-decay weighting in elo.py.

## 2026 World Cup format

48 teams, 12 groups (A–L) of 4 teams each. Top 2 from each group advance directly.
8 best third-placed teams also advance. Total of 32 teams in Round of 32.
Standard knockout (R32 → R16 → QF → SF → Final → Champion) from there.
`config.py` defines: ROUNDS = ["group", "R32", "R16", "QF", "SF", "final", "champion"]
