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


def bankroll_curve(ledger: pd.DataFrame) -> go.Figure:
    """Bankroll over time for one backtest run."""
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(ledger) + 1)),
        y=ledger["bankroll"],
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        hovertemplate="Bet %{x}<br>Bankroll: $%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=ledger["bankroll"].iloc[0] if len(ledger) > 0 else 1000,
                  line_dash="dash", line_color="gray",
                  annotation_text="starting bankroll")
    fig.update_layout(title="Bankroll over time (Kelly staking)",
                      xaxis_title="bet number", yaxis_title="bankroll ($)",
                      height=380)
    return fig


def calibration_plot(
    model_probs: "np.ndarray",
    outcomes: "np.ndarray",
    n_bins: int = 10,
    title: str = "Calibration (reliability diagram)",
) -> go.Figure:
    """Reliability diagram: predicted probability bucket vs actual win rate.

    Perfect calibration sits on the diagonal. Points above the line mean the
    model is underconfident (actual rate > predicted); below = overconfident.
    """
    import numpy as np
    bins = np.linspace(0, 1, n_bins + 1)
    centers, actuals, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (model_probs >= lo) & (model_probs < hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        actuals.append(float(outcomes[mask].mean()))
        counts.append(int(mask.sum()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"),
                             name="perfect calibration"))
    fig.add_trace(go.Scatter(
        x=centers, y=actuals, mode="lines+markers",
        marker=dict(size=[max(6, c // 2) for c in counts], color="#2ca02c"),
        text=[f"n={c}" for c in counts],
        hovertemplate="predicted: %{x:.2f}<br>actual: %{y:.2f}<br>%{text}<extra></extra>",
        name="model",
    ))
    fig.update_layout(title=title, xaxis_title="predicted probability",
                      yaxis_title="actual win rate",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                      height=420)
    return fig


def model_vs_market_scatter(edge_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(edge_df, x="market_prob", y="model_prob", text="outcome",
                     title="Model vs market-implied probability")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray"))
    fig.update_traces(textposition="top center")
    fig.update_layout(height=450, xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig
