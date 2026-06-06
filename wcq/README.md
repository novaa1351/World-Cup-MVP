# ⚽ World Cup Quant Dashboard — Model vs. Prediction Markets

A portfolio project that pulls public prediction-market odds (Polymarket + Kalshi)
for the 2026 World Cup, generates independent model probabilities, and surfaces
where the two disagree — value edges and theoretical cross-platform arbitrage —
with backtesting and an interactive Streamlit dashboard.

> **Educational tool. It does not place trades and is not financial or betting
> advice.** It reads *public* market data only.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/data/historical.py        
python tests/test_smoke.py           
streamlit run app/streamlit_app.py   
```

## How the SVI idea actually transfers (read before interviews)
Options SVI fits a smooth, low-parameter, **no-arbitrage** curve to sparse market
quotes. We reuse the *methodology*, not the literal equation:

| Options world | This project |
|---|---|
| strike / log-moneyness axis | team latent-strength axis |
| maturity axis | tournament **round depth** (group → champion) |
| implied-vol surface value | P(team survives to that round) |
| butterfly no-arb (valid density) | per-round survival probs form a valid distribution |
| calendar no-arb (monotone variance) | survival **monotone non-increasing** in round depth |

**Be honest about the limit:** the SVI hyperbola is tuned to vol smiles, so we
borrow the *approach* (smooth + no-arb-constrained + market-calibrated fit), not
the formula. The statistically grounded engine for the raw numbers is a
Monte-Carlo bracket sim driven by Elo / match probabilities; the SVI-style
surface is the calibration layer that reconciles the model with sparse market
anchors while preserving no-arbitrage structure. See `src/models/svi_surface.py`.

## Architecture
```
config.py                 paths + public API endpoints + knobs
data/                     cached datasets + fixtures seed
src/data/historical.py    ~50k internationals (1872->), cleaned
src/data/markets.py       Polymarket + Kalshi fetchers (public, graceful fallback)
src/models/elo.py         Elo ratings replayed over history
src/models/match_model.py Elo -> win/draw/loss (upgrade path: Dixon-Coles)
src/models/svi_surface.py SVI-style no-arb survival surface  <-- centerpiece
src/markets/implied.py    price -> implied prob, de-vig, overround
src/markets/edges.py      model-vs-market edges, Kelly sizing, arb flags
src/backtest/engine.py    staking sim, ROI, hit rate, Brier calibration
src/viz/charts.py         plotly (3D surface, edge bars, calibration scatter)
app/streamlit_app.py      dashboard (5 tabs)
tests/test_smoke.py       fast invariants
```

## Roadmap (incremental — each is a self-contained session)
1. **Real model probs in the edge tab.** Wire Monte-Carlo bracket sim
   (`src/models/tournament.py`) → survival anchors → SVI surface → edge table.
2. **Dixon-Coles** goal model for exact scorelines + correct draw rate.
3. **Recent-form / time-decay weighting** in Elo (down-weight 1990s friendlies).
4. **Resolved-market backtest** on 2018/2022 archives or live group-stage results.
5. **Calibration report** (reliability diagram) — the most interview-impressive piece.

## Data & legal notes
- Historical data: martj42 international-results dataset (GitHub mirror, no auth).
- Markets: Polymarket Gamma + Kalshi public endpoints, **read-only, no API key.**
- US users: Polymarket restricts US access and Kalshi is CFTC-regulated; this
  project only *reads public prices and simulates* — keep it that way.
