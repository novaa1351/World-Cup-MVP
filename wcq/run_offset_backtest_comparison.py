"""Compare backtest accuracy: plain Elo vs. confederation-offset-adjusted Elo,
across all six historical World Cups.

Usage: python run_offset_backtest_comparison.py
"""
from src.data.historical import load_results
from src.backtest.engine import (
    build_wc_backtest, build_wc_backtest_with_offset, run_backtest, WC_CUTOFFS,
)


def main():
    matches = load_results()

    print(f"{'Year':<6} {'Brier (plain)':<16} {'Brier (offset)':<16} "
          f"{'Hit rate (plain)':<18} {'Hit rate (offset)':<18} {'Better?'}")
    print("-" * 90)

    for year in sorted(WC_CUTOFFS.keys()):
        try:
            data_plain = build_wc_backtest(year, matches)
            result_plain = run_backtest(data_plain)

            data_offset = build_wc_backtest_with_offset(year, matches)
            result_offset = run_backtest(data_offset)

            brier_plain = result_plain["brier_model"]
            brier_offset = result_offset["brier_model"]
            hit_plain = result_plain["hit_rate"]
            hit_offset = result_offset["hit_rate"]

            better = "offset" if brier_offset < brier_plain else "plain"

            print(f"{year:<6} {brier_plain:<16} {brier_offset:<16} "
                  f"{hit_plain:<18} {hit_offset:<18} {better}")
        except Exception as e:
            print(f"{year:<6} ERROR: {e}")


if __name__ == "__main__":
    main()