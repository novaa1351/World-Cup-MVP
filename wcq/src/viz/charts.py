"""Plotly chart builders used by the Streamlit app."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config


def survival_surface_3d(team_survivals: dict[str, dict[str, float]]) -> go.Figure:
    """The SVI-inspired 'surface': teams x rounds x survival probability."""
    teams = list(team_survivals)
    rounds = config.ROUNDS
    z = np.array([[team_survivals[t][r] for r in rounds] for t in teams])
    fig = go.Figure(go.Surface(z=z, x=rounds, y=teams, colorscale="Viridis"))
    fig.update_layout(
        title="Tournament Survival Surface (SVI-style)",
        scene=dict(xaxis_title="round depth", yaxis_title="team",
                   zaxis_title="survival probability"),
        height=600, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def survival_curves(team_survivals: dict[str, dict[str, float]]) -> go.Figure:
    """2D version: one monotone line per team across rounds."""
    fig = go.Figure()
    for t, surv in team_survivals.items():
        fig.add_trace(go.Scatter(x=config.ROUNDS, y=[surv[r] for r in config.ROUNDS],
                                 mode="lines+markers", name=t))
    fig.update_layout(title="Survival curves (no-arb monotone)",
                      yaxis_title="P(survive)", height=450)
    return fig


def edge_bars(edge_df: pd.DataFrame) -> go.Figure:
    colors = ["#2ca02c" if e >= 0 else "#d62728" for e in edge_df["edge"]]
    fig = go.Figure(go.Bar(x=edge_df["outcome"], y=edge_df["edge_pct"],
                           marker_color=colors))
    fig.update_layout(title="Model edge vs market (percentage points)",
                      yaxis_title="edge (pp)", height=400)
    return fig


def model_vs_market_scatter(edge_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(edge_df, x="market_prob", y="model_prob", text="outcome",
                     title="Model vs market-implied probability")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray"))
    fig.update_traces(textposition="top center")
    fig.update_layout(height=450, xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig
