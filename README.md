# World Cup Quant Dashboard

[![Tests](https://github.com/novaa1351/World-Cup-MVP/actions/workflows/tests.yml/badge.svg)](https://github.com/novaa1351/World-Cup-MVP/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An Elo-based forecasting model for the 2026 FIFA World Cup, with a Monte Carlo tournament simulator, a six-tournament walk-forward backtest, a live Discord bot, and a Streamlit dashboard.

The tournament is over, so this repo reports how the model actually did rather than what it hoped to do.

> **Educational and analytical only. Places no trades and is not financial or betting advice.** Reads public market data.

---

## Results

The question worth asking is not "does this beat a coin flip" but "do the corrections in this project beat plain Elo." Scored on the same 72 real 2026 matches, every variant trained only on pre-tournament data:

| Model, same 72 matches | Brier | Hit rate |
|---|---|---|
| Uniform 1/3 (not a serious baseline, listed for scale) | 0.2222 | |
| **Plain Elo** (the real comparator) | 0.1920 | 56.9% |
| **Plain Elo + confederation offsets + goal-difference form** | **0.1595** | 65.3% |

The corrections improve Brier by **0.0326**, 95% bootstrap CI **[+0.0234, +0.0418]** over 10,000 resamples of matches, excluding zero. That gap, against a real baseline rather than a strawman, is the actual result of this project. The corrections and their validation are described in the [detailed write-up](https://github.com/yousae/World-Cup-MVP/blob/main/WRITEUP.md) sections 4 to 7.

Two honest caveats on that table, both of which cost the project something:

- **The deployed bot was running plain Elo, not the corrected model.** `jobs/pre_match.py` called `compute_elo()` with default arguments, so `confed_offset` and `goal_diff_form` were both off. Its 72 logged live predictions scored Brier **0.1811**, better than pre-tournament plain Elo only because it retrained daily on in-tournament results, and worse than the corrected model would have managed. The 0.1595 row is a walk-forward evaluation computed after the fact, not a real-time track record.
- **No market baseline exists.** Live market-odds capture failed silently for the whole tournament, so every edge and ROI figure in the dashboard is measured against a flat 1/3 baseline rather than real prices. That is a null result, documented rather than papered over.

**Knockout bracket:** the model's own picks propagated through the real bracket tree, with no mid-tournament correction, went 25 of 31 with the champion (Spain) called correctly. That is 81%, but on 31 matches the exact 95% interval is **[62.5%, 92.5%]**, so treat it as an illustration rather than a measurement. It also shifts by a match or two between refits, for reasons explained in the [write-up](https://github.com/yousae/World-Cup-MVP/blob/main/WRITEUP.md) section 10. The bracket is the readable story; the Brier comparison above is the defensible number.

![2026 knockout bracket: model picks vs actual results](docs/img/bracket.png)

📄 **[Read the detailed write-up](https://github.com/yousae/World-Cup-MVP/blob/main/WRITEUP.md)** for the methodology, the ideas that failed, the bugs found in a post-tournament audit, and the limitations. It lives in Yousif's fork.

## Who built what

A two-person project. One of us built the original architecture, the core Elo, match-model and Monte Carlo pipeline, the Discord bot, and the first version of the dashboard. The other contributed the confederation-offset and goal-difference corrections measured above, the real-bracket prediction feature, the no-lookahead test suite, and the post-tournament audit. That second half is documented in detail in a separate [write-up](https://github.com/yousae/World-Cup-MVP/blob/main/WRITEUP.md), which also notes where AI assistance was used.

---

## Quickstart

```bash
cd wcq
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/data/historical.py        # download ~50k historical matches
python tests/test_smoke.py           # fast sanity checks
streamlit run app/streamlit_app.py   # launch the dashboard
```

---

## How it works

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
              src/bot/ (Discord notification layer)
```

**Model details**

- **Elo:** replayed over every international match from 1872 to present, 5% annual mean-reversion, 5-tier tournament K-weighting (World Cup finals K=60 down to friendlies K=20).
- **Draw model:** `P(draw | Δelo) = draw_base × exp(-|Δelo| / scale)`. Ships with `draw_base=0.28, scale=400`, and every number reported above was produced with those values, so a clean clone reproduces them exactly. The parameters can also be MLE-calibrated (`python src/models/match_model.py`, or the dashboard's sidebar button), which fits `draw_base≈0.313, scale≈319` on ~21k competitive matches since 1990. That fitted file is deliberately **not** committed: it is fit on the full match history, so loading it by default would mean the backtests and the 2026 bracket ran on parameters estimated partly from the results they are scored against. The calibration is a reported finding, not the configuration behind the headline numbers.
- **Confederation offsets:** per-confederation Elo corrections fit by MLE on cross-confederation results, correcting a real bias where CONCACAF teams were overrated against UEFA and CONMEBOL opposition. See the [write-up](https://github.com/yousae/World-Cup-MVP/blob/main/WRITEUP.md) sections 1 to 4.
- **Monte Carlo:** 20,000 simulations of the 48-team bracket for round-by-round survival probabilities.

**Methodology guarantees**

- Elo for any backtested tournament is trained strictly on matches before that tournament's start date.
- Fitted parameters (confederation offsets, goal-difference weight) come only from pre-cutoff data.
- Both guarantees are enforced by tests in [`wcq/tests/test_no_lookahead.py`](wcq/tests/test_no_lookahead.py), including a regression test for a lookahead bug found and fixed in a post-tournament audit.

---

## Dashboard

| Tab | Contents |
|-----|----------|
| Live markets | Polymarket + Kalshi prices, Kalshi round-survival pivot |
| Model forecast | Top-20 Elo ratings, Monte Carlo survival probabilities, group tables |
| Edge detection | Model vs market edges, with Kalshi round-label remapping |
| Survival surface | SVI-style survival curves per team across rounds |
| Backtest | Six historical World Cup backtests (2002 to 2022), 2026 forward simulation |
| Findings | Brier/hit-rate summary, key findings, real 2026 bracket vs actual results |

## Discord bot

Posts automated alerts to a private server. Deployed on GitHub Actions (scheduled jobs) plus Railway (always-on live poller). See [DEPLOY.md](wcq/DEPLOY.md).

| Notification | When |
|---|---|
| Daily digest | 07:00 UTC |
| Pre-match briefing | ~1 hour before kickoff |
| Post-match scorecard | 10 to 90 min after full time |
| Paper bet placed / P&L | Alongside pre/post-match |
| Live edge alert | During match, edge > 6% |
| Cross-platform spread | During match, Polymarket vs Kalshi > 4% |

---

## Repo layout

| Path | Job |
|------|-----|
| `wcq/config.py` | Shared paths, API endpoints, model knobs |
| `wcq/src/data/` | Historical results loader, market price fetchers |
| `wcq/src/models/` | Elo, match model, tournament MC, confederations, SVI surface |
| `wcq/src/markets/` | Implied probabilities, de-vig, edges, Kelly sizing |
| `wcq/src/backtest/engine.py` | Walk-forward backtest, Brier scoring, staking sim |
| `wcq/src/bot/` | Discord bot (storage, notify, poller, paper trader, results) |
| `wcq/jobs/` | Scheduled job scripts (digest, pre-match, post-match, backfill) |
| `wcq/app/streamlit_app.py` | Dashboard |
| `wcq/tests/` | Smoke tests and no-lookahead guarantees |
| `wcq/run_live_baseline_comparison.py` | Reproduces the headline table above (corrections vs plain Elo on the 72 live matches) |
| `wcq/run_significance_test.py` | Bootstrap significance test for the model corrections |

---

## How the SVI framing transfers

Options SVI fits a smooth, low-parameter, no-arbitrage curve to sparse market quotes. This project reuses the *methodology*, not the equation:

| Options world | This project |
|---|---|
| Maturity axis | Tournament round depth (group → champion) |
| Implied-vol surface | P(team survives to that round) |
| Butterfly no-arb | Per-round survival probabilities form a valid distribution |
| Calendar no-arb | Survival monotone non-increasing in round depth |

The SVI hyperbola is tuned to volatility smiles, so the approach transfers (smooth, no-arb-constrained, market-calibrated) but the formula does not. See `wcq/src/models/svi_surface.py`.

---

## Data and legal notes

- Historical data: [martj42 international-results dataset](https://github.com/martj42/international_results) (public, no auth)
- Markets: Polymarket Gamma API and Kalshi public endpoints, read-only, no API key
- Polymarket restricts US access and Kalshi is CFTC-regulated. This project only reads public prices and simulates.

## License

[MIT](LICENSE)
