"""World Cup Quant Dashboard: Model vs Prediction Markets.

Run from the project root:
    streamlit run app/streamlit_app.py

EDUCATIONAL TOOL. It surfaces where a toy model disagrees with public markets.
It does not place trades and is not financial or betting advice.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.historical import download_results, load_results
from src.data.markets import get_all, winner_probs, kalshi_survival_probs
from src.models.elo import compute_elo, top_n
from src.models.match_model import dc_params_available, fit_dc_params
from src.models.tournament import simulate_tournament, survival_table, GROUPS_2026, MARKET_TO_FIFA
from src.models.svi_surface import SurvivalSurface
from src.markets.implied import implied_from_book
from src.markets.edges import edge_table, flag_value
from src.backtest.engine import run_wc_backtest, WC_CUTOFFS
from src.viz import charts

st.set_page_config(page_title="World Cup Quant Dashboard", layout="wide")


@st.cache_data(show_spinner="Loading match history + computing Elo...")
def load_elo() -> dict[str, float]:
    matches = download_results()
    return compute_elo(matches)


@st.cache_data(show_spinner="Fetching prediction markets...")
def load_markets() -> pd.DataFrame:
    return get_all()


@st.cache_data(show_spinner=f"Running {config.MC_SIMS:,} Monte Carlo simulations...")
def load_mc(_elo_frozen: tuple) -> dict[str, dict[str, float]]:
    # _elo_frozen is a tuple of (team, rating) pairs so st.cache_data can hash it.
    return simulate_tournament(dict(_elo_frozen), seed=42)


@st.cache_data(show_spinner="Running historical backtest (rebuilding pre-WC Elo)...")
def load_backtest(year: int) -> dict:
    # load_results() reads from the cached local CSV — no network call.
    return run_wc_backtest(year, load_results(), edge_threshold=0.05)


st.title("⚽ World Cup Quant Dashboard")
st.caption("Model vs. prediction markets · educational tool, not betting advice")

# Sidebar: model status
with st.sidebar:
    st.header("Model info")
    if dc_params_available():
        import json
        with open(config.DC_PARAMS_PATH) as _f:
            _p = json.load(_f)
        st.success("Dixon-Coles (DC) model active")
        st.caption(
            f"μ₀={_p['mu_0']:.3f} · γ={_p['gamma']:.5f} · ρ={_p['rho']:.3f}"
            f"\nFit on {_p.get('n_matches', '?'):,} matches"
        )
    else:
        st.warning("DC params not fitted — using fallback model")
        st.caption("Run once to fit:")
        st.code("python src/models/match_model.py")
        if st.button("Fit DC parameters now (≈30 s)"):
            with st.spinner("Fitting Dixon-Coles parameters…"):
                _matches = load_results()
                fit_dc_params(_matches)
            st.success("Done! Reload the page to use the DC model.")
    st.divider()
    st.caption(
        "Elo: tournament K-weighted (eloratings.net) "
        "+ 20 % annual mean-reversion (FiveThirtyEight)."
    )

elo = load_elo()
markets = load_markets()
# Convert elo dict to a hashable form for cache key
mc_survival = load_mc(tuple(sorted(elo.items())))

tab_mkt, tab_model, tab_edge, tab_surf, tab_bt = st.tabs(
    ["Live markets", "Model forecast", "Edge detection",
     "Survival surface", "Backtest"]
)

# --- Live markets -----------------------------------------------------------
with tab_mkt:
    st.subheader("Current market-implied prices")
    if markets.empty:
        st.warning("No markets returned (API may be unavailable).")
    else:
        pm_sub = markets[markets["platform"] == "polymarket"]
        ks_sub = markets[markets["platform"] == "kalshi"]

        st.markdown("#### Polymarket — champion winner markets")
        if pm_sub.empty:
            st.info("No Polymarket data.")
        else:
            raw_yes = winner_probs(pm_sub)
            if raw_yes:
                pm_display = (
                    pd.DataFrame.from_dict(raw_yes, orient="index", columns=["raw YES price"])
                    .join(
                        pd.DataFrame.from_dict(implied_from_book(raw_yes), orient="index", columns=["de-vigged prob"])
                    )
                    .sort_values("de-vigged prob", ascending=False)
                )
                st.dataframe(pm_display.style.format("{:.3f}"), use_container_width=True)
                st.caption(f"Overround: {(sum(raw_yes.values()) - 1)*100:.1f} pp")
            else:
                st.dataframe(pm_sub[["market", "outcome", "price"]], use_container_width=True)

        st.markdown("#### Kalshi — round survival markets (KXWCROUND)")
        if ks_sub.empty:
            st.info("No Kalshi data.")
        else:
            ks_pivot = (
                ks_sub[["team", "round", "price"]]
                .dropna(subset=["team", "round"])
                .pivot_table(index="team", columns="round", values="price", aggfunc="mean")
            )
            # Order columns by round depth
            col_order = [c for c in ["R16", "QF", "SF", "final"] if c in ks_pivot.columns]
            ks_pivot = ks_pivot[col_order].sort_values(col_order[-1] if col_order else "R16", ascending=False)
            st.dataframe(ks_pivot.style.format("{:.3f}"), use_container_width=True)
            st.caption(f"{len(ks_sub['team'].unique())} teams, {len(col_order)} rounds from Kalshi KXWCROUND series")

# --- Model forecast ---------------------------------------------------------
with tab_model:
    st.subheader("Elo ratings (1872 → present internationals)")
    st.dataframe(top_n(elo, 20), use_container_width=True)

    st.divider()
    st.subheader(f"Monte Carlo forecast — {config.MC_SIMS:,} simulations")
    st.caption(
        "Survival probabilities per round. "
        "Teams absent from historical data are assigned the ELO_BASE rating."
    )

    col_sort, col_n = st.columns([2, 1])
    with col_sort:
        sort_round = st.selectbox("Sort by round", config.ROUNDS, index=config.ROUNDS.index("champion"))
    with col_n:
        show_n = st.number_input("Show top N teams", min_value=5, max_value=48, value=16, step=1)

    mc_df = survival_table(mc_survival, sort_by=sort_round, top_n=int(show_n))
    st.dataframe(mc_df.style.format("{:.1%}"), use_container_width=True)

    st.subheader("Group table")
    group_cols = st.columns(4)
    for i, (gid, teams) in enumerate(GROUPS_2026.items()):
        with group_cols[i % 4]:
            champ_odds = {t: mc_survival[t]["champion"] for t in teams}
            sorted_teams = sorted(champ_odds, key=champ_odds.get, reverse=True)  # type: ignore[arg-type]
            rows = [{"team": t, "champion %": f"{champ_odds[t]:.1%}",
                     "group adv %": f"{mc_survival[t]['group']:.1%}"} for t in sorted_teams]
            st.write(f"**Group {gid}**")
            st.dataframe(pd.DataFrame(rows).set_index("team"), use_container_width=True)

# --- Edge detection ---------------------------------------------------------
with tab_edge:
    st.subheader("Model vs market edge")
    st.caption(
        "**Green** = model thinks the market underprices the outcome (potential value). "
        "**Red** = market overprices relative to our Elo+MC model."
    )

    if markets.empty:
        st.info("No market data available — edge detection requires live market prices.")
    else:
        pm_sub = markets[markets["platform"] == "polymarket"]
        ks_sub = markets[markets["platform"] == "kalshi"]

        # ── Section 1: Polymarket champion odds ──────────────────────────────
        st.markdown("### Polymarket — champion (winner) edges")
        if pm_sub.empty:
            st.info("No Polymarket data available.")
        else:
            raw_yes: dict[str, float] = winner_probs(pm_sub)
            if not raw_yes:
                st.warning("Could not parse winner markets from Polymarket titles.")
            else:
                market_p: dict[str, float] = implied_from_book(raw_yes)
                model_p: dict[str, float] = {}
                unmatched_pm: list[str] = []
                for mkt_team, mkt_prob in market_p.items():
                    fifa_name = MARKET_TO_FIFA.get(mkt_team, mkt_team)
                    if fifa_name in mc_survival:
                        model_p[mkt_team] = mc_survival[fifa_name]["champion"]
                    else:
                        unmatched_pm.append(mkt_team)
                if unmatched_pm:
                    st.caption(f"Skipped (not in MC bracket): {', '.join(unmatched_pm)}")
                if model_p:
                    et = edge_table(model_p, market_p)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Teams compared", len(et))
                    c2.metric("Overround", f"{(sum(raw_yes.values())-1)*100:.1f} pp")
                    c3.metric("Largest edge", f"{et['edge_pct'].abs().max():.1f} pp")
                    st.plotly_chart(charts.edge_bars(et), use_container_width=True)
                    st.plotly_chart(charts.model_vs_market_scatter(et), use_container_width=True)
                    flagged = flag_value(et)
                    if not flagged.empty:
                        st.markdown("**Value flags (|edge| > 5 pp)**")
                        st.dataframe(
                            flagged[["outcome", "model_prob", "market_prob", "edge_pct"]]
                            .rename(columns={"edge_pct": "edge (pp)"})
                            .style.format({"model_prob": "{:.1%}", "market_prob": "{:.1%}",
                                           "edge (pp)": "{:+.1f}"}),
                            use_container_width=True,
                        )
                    else:
                        st.info("No champion edges above 5 pp threshold.")
                    with st.expander("Full champion edge table"):
                        st.dataframe(
                            et[["outcome", "model_prob", "market_prob", "edge_pct",
                                "fair_odds", "market_odds"]]
                            .rename(columns={"edge_pct": "edge (pp)"})
                            .style.format({"model_prob": "{:.1%}", "market_prob": "{:.1%}",
                                           "edge (pp)": "{:+.1f}", "fair_odds": "{:.2f}",
                                           "market_odds": "{:.2f}"}),
                            use_container_width=True,
                        )

        st.divider()

        # ── Section 2: Kalshi round-survival edges ────────────────────────────
        st.markdown("### Kalshi — round survival edges (KXWCROUND)")
        st.caption(
            "Kalshi has per-team, per-round binary markets: "
            "'Will X qualify for the Round of 16 / QF / SF / Final?' "
            "The YES mid-price is the market-implied survival probability at that round. "
            "We compare those against our MC model's survival probabilities at the same rounds."
        )
        if ks_sub.empty:
            st.info("No Kalshi data available.")
        else:
            ks_survival = kalshi_survival_probs(ks_sub)
            kalshi_rounds = ["R16", "QF", "SF", "final"]
            round_choice = st.selectbox(
                "Round to analyse",
                [r for r in kalshi_rounds if any(r in probs for probs in ks_survival.values())],
                key="kalshi_round_select",
            )

            ks_model_p: dict[str, float] = {}
            ks_market_p: dict[str, float] = {}
            for mkt_team, round_probs in ks_survival.items():
                if round_choice not in round_probs:
                    continue
                fifa_name = MARKET_TO_FIFA.get(mkt_team, mkt_team)
                if fifa_name not in mc_survival:
                    continue
                ks_model_p[mkt_team] = mc_survival[fifa_name][round_choice]
                ks_market_p[mkt_team] = round_probs[round_choice]

            if ks_model_p:
                ks_et = edge_table(ks_model_p, ks_market_p)
                c1, c2, c3 = st.columns(3)
                c1.metric("Teams compared", len(ks_et))
                c2.metric("Largest edge", f"{ks_et['edge_pct'].abs().max():.1f} pp")
                avg_mkt = ks_market_p  # raw, not de-vigged (each market is independent binary)
                c3.metric("Market avg price", f"{sum(avg_mkt.values())/len(avg_mkt):.3f}",
                          help="Mean YES mid-price across all teams for this round")
                st.plotly_chart(charts.edge_bars(ks_et), use_container_width=True)
                st.plotly_chart(charts.model_vs_market_scatter(ks_et), use_container_width=True)
                ks_flagged = flag_value(ks_et)
                if not ks_flagged.empty:
                    st.markdown(f"**Value flags at {round_choice} (|edge| > 5 pp)**")
                    st.dataframe(
                        ks_flagged[["outcome", "model_prob", "market_prob", "edge_pct"]]
                        .rename(columns={"edge_pct": "edge (pp)"})
                        .style.format({"model_prob": "{:.1%}", "market_prob": "{:.1%}",
                                       "edge (pp)": "{:+.1f}"}),
                        use_container_width=True,
                    )
                else:
                    st.info(f"No edges above 5 pp at {round_choice}.")
                with st.expander(f"Full {round_choice} edge table"):
                    st.dataframe(
                        ks_et[["outcome", "model_prob", "market_prob", "edge_pct",
                               "fair_odds", "market_odds"]]
                        .rename(columns={"edge_pct": "edge (pp)"})
                        .style.format({"model_prob": "{:.1%}", "market_prob": "{:.1%}",
                                       "edge (pp)": "{:+.1f}", "fair_odds": "{:.2f}",
                                       "market_odds": "{:.2f}"}),
                        use_container_width=True,
                    )
            else:
                st.warning("Could not match Kalshi teams to MC survival dictionary.")

# --- Survival surface -------------------------------------------------------
with tab_surf:
    st.subheader("SVI-style survival surface")
    st.caption(
        "Smooth, monotone-non-increasing survival curves calibrated from Monte Carlo anchors. "
        "Methodology borrowed from options SVI: low-parameter sigmoid fit with "
        "no-arbitrage (calendar-monotone) constraint enforced. "
        "The literal SVI hyperbola is tuned to vol smiles; the shape here is a sigmoid."
    )

    all_mc_teams = sorted(mc_survival, key=lambda t: mc_survival[t]["champion"], reverse=True)
    default_teams = all_mc_teams[:8]
    selected_teams = st.multiselect(
        "Teams to display", options=all_mc_teams, default=default_teams
    )

    if selected_teams:
        # Calibrate SurvivalSurface for each selected team using all 7 MC anchors
        surv: dict[str, dict[str, float]] = {}
        for team in selected_teams:
            mc_anchors = mc_survival[team]
            surface = SurvivalSurface(team).fit(mc_anchors)
            surv[team] = surface.survival()

        st.plotly_chart(charts.survival_surface_3d(surv), use_container_width=True)
        st.plotly_chart(charts.survival_curves(surv), use_container_width=True)

        st.subheader("Raw Monte Carlo anchors (before SVI calibration)")
        raw_df = pd.DataFrame(
            {t: mc_survival[t] for t in selected_teams}
        ).T[config.ROUNDS]
        st.dataframe(raw_df.style.format("{:.3f}"), use_container_width=True)
    else:
        st.info("Select at least one team above.")

# --- Backtest ---------------------------------------------------------------
with tab_bt:
    st.subheader("Historical World Cup backtest")
    st.caption(
        "Elo ratings are rebuilt from scratch using only matches played **before** "
        "the selected World Cup started — zero lookahead bias. "
        "The model's match probabilities are then compared against the 64 actual "
        "results. Market baseline = 1/3 flat (equal-odds for each 3-way outcome), "
        "since archived betting odds are unavailable. "
        "Kelly staking bets on any outcome where our model exceeds 1/3 by > 5 pp."
    )

    year_choice = st.radio("Select World Cup", list(WC_CUTOFFS.keys()),
                           format_func=lambda y: f"{y} (Elo cutoff: {WC_CUTOFFS[y][0]})",
                           horizontal=True)

    bt = load_backtest(year_choice)
    data = bt["data"]

    # --- Summary metrics -------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Matches evaluated", len(data) // 3,
                help="64 WC matches × 3 outcomes = 192 rows; shown per match here")
    col2.metric("Brier — our model", f"{bt.get('brier_model', 0):.4f}",
                delta=f"{bt.get('brier_baseline', 0) - bt.get('brier_model', 0):+.4f} vs baseline",
                delta_color="normal",
                help="Lower Brier score = better calibrated. Delta = how much we beat the naive 1/3-equal baseline.")
    col3.metric("Bets placed", bt.get("n_bets", 0))
    col4.metric("Hit rate", f"{bt.get('hit_rate', 0):.1%}",
                help="Fraction of bets that won (edge ≥ 5 pp above 1/3 baseline)")
    col5.metric("Final bankroll", f"${bt.get('final_bankroll', 1000):,.0f}",
                delta=f"{bt.get('roi', 0):+.1%} ROI",
                help="Starting bankroll $1,000, fractional-Kelly staking vs 1/3 flat market")

    st.divider()

    # --- Charts ---------------------------------------------------------------
    left, right = st.columns(2)

    with left:
        # Calibration plot for home-win predictions (most informative outcome)
        home_rows = data[data["outcome_label"] == "home_win"]
        st.plotly_chart(
            charts.calibration_plot(
                home_rows["model_prob"].to_numpy(),
                home_rows["outcome"].to_numpy(),
                title=f"{year_choice} WC — calibration (home-win predictions)",
            ),
            use_container_width=True,
        )

    with right:
        if bt.get("n_bets", 0) > 0:
            st.plotly_chart(
                charts.bankroll_curve(bt["ledger"]),
                use_container_width=True,
            )
        else:
            st.info("No bets placed at the current edge threshold.")

    st.divider()

    # --- Per-match predictions ------------------------------------------------
    st.subheader("Match-level predictions vs outcomes")

    # Pivot back to one row per match for display
    home_rows = data[data["outcome_label"] == "home_win"].copy()
    draw_rows = data[data["outcome_label"] == "draw"].set_index(["home", "away", "date"])
    away_rows = data[data["outcome_label"] == "away_win"].set_index(["home", "away", "date"])

    display = home_rows[["date", "home", "away", "home_score", "away_score",
                          "elo_home", "elo_away", "model_prob"]].copy()
    display = display.rename(columns={"model_prob": "P(home win)"})
    display["P(draw)"] = draw_rows["model_prob"].values
    display["P(away win)"] = away_rows["model_prob"].values
    display["result"] = display.apply(
        lambda r: (f"{int(r.home_score)}–{int(r.away_score)} "
                   + ("✓ home" if r.home_score > r.away_score
                      else ("draw" if r.home_score == r.away_score else "✓ away"))),
        axis=1,
    )
    display["model called"] = display.apply(
        lambda r: ("home" if r["P(home win)"] >= r[["P(draw)", "P(away win)"]].max()
                   and r["P(home win)"] >= r["P(draw)"]
                   else ("draw" if r["P(draw)"] >= r["P(away win)"] else "away")),
        axis=1,
    )

    st.dataframe(
        display[["date", "home", "away", "elo_home", "elo_away",
                 "P(home win)", "P(draw)", "P(away win)", "result", "model called"]]
        .sort_values("date")
        .style.format({
            "elo_home": "{:.0f}", "elo_away": "{:.0f}",
            "P(home win)": "{:.1%}", "P(draw)": "{:.1%}", "P(away win)": "{:.1%}",
        }),
        use_container_width=True,
    )
