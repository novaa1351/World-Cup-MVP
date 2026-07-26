"""Compare backtest accuracy: plain Elo vs. offset-only vs. offset+goal-diff form,
across all six historical World Cups.

Usage: python run_offset_backtest_comparison.py
"""
from src.data.historical import load_results
from src.backtest.engine import (
    build_wc_backtest, build_wc_backtest_with_offset, build_wc_backtest_full,
    run_backtest, WC_CUTOFFS,
)


def main():
    matches = load_results()

    print(f"{'Year':<6} {'Brier plain':<13} {'Brier offset':<14} {'Brier full':<12} "
          f"{'Hit plain':<11} {'Hit offset':<12} {'Hit full':<10}")
    print("-" * 90)

    for year in sorted(WC_CUTOFFS.keys()):
        try:
            r_plain = run_backtest(build_wc_backtest(year, matches))
            r_offset = run_backtest(build_wc_backtest_with_offset(year, matches))
            r_full = run_backtest(build_wc_backtest_full(year, matches))

            print(f"{year:<6} {r_plain['brier_model']:<13} {r_offset['brier_model']:<14} "
                  f"{r_full['brier_model']:<12} {r_plain['hit_rate']:<11} "
                  f"{r_offset['hit_rate']:<12} {r_full['hit_rate']:<10}")
        except Exception as e:
            print(f"{year:<6} ERROR: {e}")


if __name__ == "__main__":
    main()