"""Load and clean historical international match results."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import requests

import config


def download_results(force: bool = False) -> pd.DataFrame:
    """Pull the historical results CSV (cached locally)."""
    path = config.RAW / "results.csv"
    if force or not path.exists():
        r = requests.get(config.HISTORICAL_RESULTS_URL, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    return load_results()


def load_results() -> pd.DataFrame:
    """Return cleaned, chronologically sorted match history."""
    path = config.RAW / "results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    # 1 = home win, 0 = draw, -1 = away win  (useful target column)
    df["result"] = (df["home_score"] - df["away_score"]).clip(-1, 1)
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    d = download_results()
    print(f"{len(d):,} matches, {d['date'].min().date()} -> {d['date'].max().date()}")
    print(d.tail(3)[["date", "home_team", "away_team", "home_score", "away_score"]])
