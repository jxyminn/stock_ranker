"""
Small grid search over LightGBM hyperparameters, scored by the same
walk-forward Information Coefficient (IC) backtest.py uses -- so "better"
means better out-of-sample ranking skill, not a lower loss on one holdout
split.

Reuses backtest.py's panel-building and per-fold training/scoring so the
tuning objective is identical to what backtest.py reports; only the LightGBM
params vary between candidates.

Usage:
    python tune.py
    python tune.py --horizons 10 30 --folds 4
    python tune.py --universe AAPL MSFT NVDA AMD PLTR --folds 3
"""
import argparse
import itertools
import numpy as np
import pandas as pd

from config import HORIZONS, TRAINING_LOOKBACK_DAYS, DEFAULT_TRAIN_UNIVERSE, LGBM_PARAMS
from data_fetcher import fetch_history_batch
from backtest import build_panel, run_fold
from train import fetch_macro_context

# Kept intentionally small -- each combo costs (horizons x folds) LightGBM
# fits, and this already runs 12 combos x 4 horizons x 4 folds = 192 fits by
# default. Widen it later once you know which direction helps.
GRID = {
    "num_leaves": [15, 31, 63],
    "max_depth": [4, 6],
    "learning_rate": [0.02, 0.05],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", nargs="+", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--lookback-days", type=int, default=TRAINING_LOOKBACK_DAYS)
    parser.add_argument("--folds", type=int, default=4, help="Walk-forward folds per horizon (fewer = faster)")
    parser.add_argument("--step-days", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    universe = args.universe or DEFAULT_TRAIN_UNIVERSE
    horizons = args.horizons or HORIZONS

    print(f"Fetching {len(universe)} tickers, {args.lookback_days} days of history...")
    price_data = fetch_history_batch(universe, args.lookback_days)
    print(f"Usable tickers: {len(price_data)}/{len(universe)}")
    if len(price_data) < 10:
        raise SystemExit("Too few tickers with usable data to tune meaningfully.")

    macro_df = fetch_macro_context(args.lookback_days)

    print("Building feature panels per horizon (reused across all hyperparam combos)...")
    panels = {h: build_panel(price_data, h, macro_df) for h in horizons}

    keys = list(GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
    n_fits = len(combos) * len(horizons) * args.folds
    print(f"\nSweeping {len(combos)} hyperparam combos x {len(horizons)} horizons x "
          f"{args.folds} folds ({n_fits} model fits total)...\n")

    results = []
    for params in combos:
        lgbm_params = {k: v for k, v in LGBM_PARAMS.items() if k not in params} | params

        horizon_ics = {}
        for h in horizons:
            full = panels[h]
            dates = full.index.unique().sort_values()
            n_test_dates = args.folds * args.step_days
            if len(dates) <= n_test_dates + 100:
                continue

            fold_start_idx = len(dates) - n_test_dates
            ics = []
            for f in range(args.folds):
                test_start = dates[fold_start_idx + f * args.step_days]
                end_pos = fold_start_idx + (f + 1) * args.step_days
                test_end = dates[end_pos] if end_pos < len(dates) else dates[-1] + pd.Timedelta(days=1)
                res = run_fold(full, h, test_start, test_end, args.top_n, lgbm_params=lgbm_params)
                if res is not None:
                    ics.append(res["mean_ic"])
            horizon_ics[h] = np.nanmean(ics) if ics else np.nan

        overall_ic = np.nanmean(list(horizon_ics.values())) if horizon_ics else np.nan
        results.append({"params": params, "overall_ic": overall_ic, "per_horizon": horizon_ics})
        per_h_str = "  ".join(f"{h}d={v:+.3f}" for h, v in horizon_ics.items())
        print(f"  {params}  ->  mean IC={overall_ic:+.4f}   ({per_h_str})")

    results.sort(key=lambda r: r["overall_ic"] if not np.isnan(r["overall_ic"]) else -999, reverse=True)

    print("\n=== Top 5 configs by mean IC (averaged across horizons/folds) ===")
    for r in results[:5]:
        per_h_str = "  ".join(f"{h}d={v:+.3f}" for h, v in r["per_horizon"].items())
        print(f"  {r['params']}  overall_IC={r['overall_ic']:+.4f}   ({per_h_str})")

    best = results[0]
    print(f"\nBest: {best['params']}  (overall mean IC={best['overall_ic']:+.4f})")
    print("This is scored on a limited fold/horizon set for speed -- worth confirming with the")
    print("full backtest.py (default 6 folds, all horizons) before committing to it.")
    print("To apply: update LGBM_PARAMS in config.py with these values, then re-run train.py.")


if __name__ == "__main__":
    main()
