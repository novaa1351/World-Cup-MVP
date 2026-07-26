"""Bootstrap significance test: is the full model's Brier improvement real,
or could it be explained by chance given the small per-tournament sample size?

Usage: python run_significance_test.py
"""
import numpy as np
from src.data.historical import load_results
from src.backtest.engine import build_wc_backtest, build_wc_backtest_full, WC_CUTOFFS
from src.models.match_model import _three_way_probs  # unused here, safe to ignore if missing


def brier_per_match(df):
    """Brier score contribution per row (one per outcome-label)."""
    return (df["model_prob"] - df["outcome"]) ** 2


def bootstrap_test(df_plain, df_full, n_boot=5000, seed=42):
    """Resample MATCHES (not rows) with replacement, compare mean Brier each time."""
    rng = np.random.default_rng(seed)

    # Group rows by match (every 3 rows = one match: home_win/draw/away_win)
    n_matches = len(df_plain) // 3
    diffs = []

    brier_plain = brier_per_match(df_plain).to_numpy().reshape(n_matches, 3)
    brier_full = brier_per_match(df_full).to_numpy().reshape(n_matches, 3)

    for _ in range(n_boot):
        idx = rng.integers(0, n_matches, size=n_matches)
        mean_plain = brier_plain[idx].mean()
        mean_full = brier_full[idx].mean()
        diffs.append(mean_plain - mean_full)  # positive = full model better

    diffs = np.array(diffs)
    win_rate = (diffs > 0).mean()
    return diffs.mean(), win_rate


def main():
    matches = load_results()
    print(f"{'Year':<6} {'Mean Brier improvement':<26} {'% of resamples favoring full model'}")
    print("-" * 70)

    all_plain, all_full = [], []
    for year in sorted(WC_CUTOFFS.keys()):
        df_plain = build_wc_backtest(year, matches)
        df_full = build_wc_backtest_full(year, matches)
        mean_diff, win_rate = bootstrap_test(df_plain, df_full)
        print(f"{year:<6} {mean_diff:<26.5f} {win_rate:.1%}")
        all_plain.append(df_plain)
        all_full.append(df_full)

    # Pooled across all 6 tournaments combined (bigger sample = more power)
    import pandas as pd
    pooled_plain = pd.concat(all_plain, ignore_index=True)
    pooled_full = pd.concat(all_full, ignore_index=True)
    mean_diff, win_rate = bootstrap_test(pooled_plain, pooled_full, n_boot=10000)
    print("-" * 70)
    print(f"{'POOLED':<6} {mean_diff:<26.5f} {win_rate:.1%}")


if __name__ == "__main__":
    main()