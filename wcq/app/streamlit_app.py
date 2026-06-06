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
from src.data.historical import download_results
from src.data.markets import get_all
from src.models.elo import compute_elo, top_n
from src.models.svi_surface import SurvivalSurface
from src.markets.implied import implied_from_book
from src.markets.edges import edge_table, flag_value, kelly_fraction
from src.viz import charts

st.set_page_config(page_title="World Cup Quant Dashboard", layout="wide")


@st.cache_data(show_spinner="Loading match history + Elo...")
def load_elo():
    matches = download_results()
    return compute_elo(matches)


@st.cache_data(show_spinner="Fetching markets...")
def load_markets():
    return get_all()


st.title("⚽ World Cup Quant Dashboard")
st.caption("Model vs. prediction markets · educational tool, not betting advice")

elo = load_elo()
markets = load_markets()

tab_mkt, tab_model, tab_edge, tab_surf, tab_bt = st.tabs(
    ["Live markets", "Model forecast", "Edge detection",
     "Survival surface", "Backtest"]
)

# --- Live markets -----------------------------------------------------------
with tab_mkt:
    st.subheader("Current market-implied prices")
    if markets.empty:
        st.warning("No markets returned.")
    else:
        st.dataframe(markets, use_container_width=True)
        for plat in markets["platform"].unique():
            sub = markets[markets["platform"] == plat]
            book = dict(zip(sub["outcome"], sub["price"]))
            st.write(f"**{plat}** de-vigged:", implied_from_book(book))

# --- Model forecast ---------------------------------------------------------
with tab_model:
    st.subheader("Elo ratings (from 1872->present internationals)")
    st.dataframe(top_n(elo, 20), use_container_width=True)

# --- Edge detection ---------------------------------------------------------
with tab_edge:
    st.subheader("Model vs market edge")
    st.info("Demo wiring: replace `model` with your calibrated probabilities.")
    if not markets.empty:
        plat = st.selectbox("platform", markets["platform"].unique())
        sub = markets[markets["platform"] == plat]
        market_p = implied_from_book(dict(zip(sub["outcome"], sub["price"])))
        # placeholder model: nudge market by Elo rank as a stand-in
        model_p = {k: v for k, v in market_p.items()}  # TODO real model probs
        et = edge_table(model_p, market_p)
        st.plotly_chart(charts.edge_bars(et), use_container_width=True)
        st.plotly_chart(charts.model_vs_market_scatter(et), use_container_width=True)
        st.dataframe(flag_value(et), use_container_width=True)

# --- Survival surface -------------------------------------------------------
with tab_surf:
    st.subheader("SVI-style survival surface")
    demo_anchors = {
        "Brazil": {"group": 0.88, "QF": 0.34, "champion": 0.12},
        "France": {"group": 0.85, "QF": 0.30, "champion": 0.10},
        "Argentina": {"group": 0.86, "QF": 0.31, "champion": 0.11},
        "England": {"group": 0.82, "QF": 0.24, "champion": 0.07},
    }
    surv = {t: SurvivalSurface(t).fit(a).survival() for t, a in demo_anchors.items()}
    st.plotly_chart(charts.survival_surface_3d(surv), use_container_width=True)
    st.plotly_chart(charts.survival_curves(surv), use_container_width=True)

# --- Backtest ---------------------------------------------------------------
with tab_bt:
    st.subheader("Backtest")
    st.info("Plug resolved historical markets into src/backtest/engine.py.")
    st.write("Example Kelly stake (model 22% vs market 18%):",
             round(kelly_fraction(0.22, 0.18), 4))
