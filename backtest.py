"""
Walk-forward backtest.

train.py's holdout metrics (MAE/AUC) come from a single train/test split, so
they're one noisy sample of model quality and say nothing about whether the
*ranking* across stocks on a given day is useful. This script instead:

  1. Walks forward through time in a series of test windows.
  2. For each window, retrains on all data strictly before it (expanding
     window -- no lookahead), then scores every ticker's cross-section on
     each date in the window using the exact same composite-score formula
     `predict.py` uses live (imported from there, not re-implemented).
  3. Compares that ranking to what actually happened, per day, via:
       - Information Coefficient (IC): Spearman correlation between a day's
         composite scores and each stock's realized forward return. This is
         the standard way to ask "did the ranking order match reality?"
         independent of whether the raw return predictions were well-scaled.
       - top-vs-bottom spread: mean realized return of the top-N ranked
         stocks minus the bottom-N, per day -- a more intuitive "would this
         ranking have made money, directionally" check.

A positive mean IC that's consistently positive across folds (see
%days_IC>0) is the signal that the ranking has real, non-accidental skill.
An IC near zero, or one that's only positive in some folds, means don't
trust the live rankings.

Usage:
    python backtest.py
    python backtest.py --universe AAPL MSFT NVDA AMD PLTR --horizons 5 10
    python backtest.py --folds 8 --step-days 30 --top-n 3
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_absolute_error

from config import HORIZONS, TRAINING_LOOKBACK_DAYS, DEFAULT_TRAIN_UNIVERSE, RANDOM_STATE, LGBM_PARAMS
from data_fetcher import fetch_history_batch, fetch_earnings
from features import (
    build_features, add_earnings_features, add_macro_features,
    FEATURE_COLUMNS, TECHNICAL_FEATURE_COLUMNS,
)
from labeling import build_labels
from predict import composite_score
from train import fetch_macro_context
from reliability import save_backtest_results


def build_panel(price_data: dict[str, pd.DataFrame], horizon: int,
                 macro_df: pd.DataFrame | None) -> pd.DataFrame:
    """Per-ticker feature+label rows, stacked, date index kept for cross-sectional grouping."""
    frames = []
    for ticker, df in price_data.items():
        feats = build_features(df)
        feats = add_earnings_features(feats, fetch_earnings(ticker))
        feats = add_macro_features(feats, macro_df)
        labels = build_labels(df["close"], horizon)
        combined = pd.concat([feats, labels], axis=1)
        combined["ticker"] = ticker
        frames.append(combined)
    full = pd.concat(frames)
    full = full.dropna(subset=TECHNICAL_FEATURE_COLUMNS + [f"fwd_return_{horizon}", f"up_{horizon}"])
    return full.sort_index()


def evaluate_day(day: pd.DataFrame, top_n: int) -> dict | None:
    """
    Score one day's cross-section: rank by composite score, then compare to
    what actually happened via Spearman IC and a top-N-vs-bottom-N realized-
    return spread.

    `day` must already have a unique positional index (0..len-1) -- callers
    pull this from a panel indexed by date, and that index is duplicated
    across tickers (every ticker has a row for the same date). Passing a
    duplicate-labeled `day` through un-reset previously caused `.loc` on a
    subset of ranked labels to silently expand back out to *all* rows sharing
    that label, so top-N and bottom-N ended up as the same (whole) group and
    the spread came out as exactly zero on every fold.
    """
    if len(day) < 5:  # too few names that day to rank meaningfully
        return None
    score = composite_score(day["pred_return"], day["prob_up"], day["vol_20d"])
    if score.std() < 1e-9:
        return None

    result = {"ic": None, "spread": None}
    ic = score.corr(day["actual"], method="spearman")
    if pd.notna(ic):
        result["ic"] = ic

    n = min(top_n, len(day) // 2)
    if n >= 1:
        ranked = score.sort_values(ascending=False)
        top_ret = day["actual"].loc[ranked.index[:n]].mean()
        bottom_ret = day["actual"].loc[ranked.index[-n:]].mean()
        result["spread"] = top_ret - bottom_ret

    return result


def run_fold(full: pd.DataFrame, horizon: int, test_start, test_end, top_n: int,
             lgbm_params: dict | None = None) -> dict | None:
    """Train on rows strictly before test_start, evaluate on [test_start, test_end)."""
    train_rows = full[full.index < test_start]
    test_rows = full[(full.index >= test_start) & (full.index < test_end)]
    if len(train_rows) < 500 or test_rows.empty:
        return None

    params = lgbm_params if lgbm_params is not None else LGBM_PARAMS
    reg = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1)
    reg.fit(train_rows[FEATURE_COLUMNS], train_rows[f"fwd_return_{horizon}"])

    clf = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1)
    clf.fit(train_rows[FEATURE_COLUMNS], train_rows[f"up_{horizon}"])

    X_test = test_rows[FEATURE_COLUMNS]
    eval_df = pd.DataFrame({
        "pred_return": reg.predict(X_test),
        "prob_up": clf.predict_proba(X_test)[:, 1],
        "vol_20d": test_rows["vol_20d"].values,
        "actual": test_rows[f"fwd_return_{horizon}"].values,
    }, index=test_rows.index)

    day_ics, day_spreads = [], []
    # eval_df's date index is duplicated across tickers -- see evaluate_day's
    # docstring for why each group must be reset to a unique positional index
    # before scoring.
    for _, day in eval_df.groupby(level=0):
        day = day.reset_index(drop=True)
        res = evaluate_day(day, top_n)
        if res is None:
            continue
        if res["ic"] is not None:
            day_ics.append(res["ic"])
        if res["spread"] is not None:
            day_spreads.append(res["spread"])

    return {
        "n_train": len(train_rows),
        "n_days": len(day_ics),
        "mean_ic": np.mean(day_ics) if day_ics else np.nan,
        "pct_days_positive_ic": np.mean([ic > 0 for ic in day_ics]) if day_ics else np.nan,
        "mean_top_bottom_spread_pct": np.mean(day_spreads) * 100 if day_spreads else np.nan,
        "auc": roc_auc_score(test_rows[f"up_{horizon}"], eval_df["prob_up"]),
        "mae": mean_absolute_error(test_rows[f"fwd_return_{horizon}"], eval_df["pred_return"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", nargs="+", default=None,
                         help="Tickers to backtest on. Defaults to the built-in training universe.")
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                         help="Horizons to backtest. Defaults to config.HORIZONS.")
    parser.add_argument("--lookback-days", type=int, default=TRAINING_LOOKBACK_DAYS)
    parser.add_argument("--folds", type=int, default=6, help="Number of walk-forward test windows")
    parser.add_argument("--step-days", type=int, default=40, help="Trading days per test window")
    parser.add_argument("--top-n", type=int, default=5, help="Top/bottom N stocks per day for the spread metric")
    args = parser.parse_args()

    universe = args.universe or DEFAULT_TRAIN_UNIVERSE
    horizons = args.horizons or HORIZONS

    print(f"Fetching {len(universe)} tickers, {args.lookback_days} days of history...")
    price_data = fetch_history_batch(universe, args.lookback_days)
    print(f"Usable tickers: {len(price_data)}/{len(universe)}\n")
    if len(price_data) < 10:
        raise SystemExit("Too few tickers with usable data to backtest meaningfully.")

    macro_df = fetch_macro_context(args.lookback_days)

    horizon_summaries = {}
    for horizon in horizons:
        print(f"=== Horizon {horizon}d ===")
        full = build_panel(price_data, horizon, macro_df)
        dates = full.index.unique().sort_values()

        n_test_dates = args.folds * args.step_days
        if len(dates) <= n_test_dates + 100:
            print(f"  not enough history for {args.folds} folds of {args.step_days} days each; skipping\n")
            continue

        fold_start_idx = len(dates) - n_test_dates
        results = []
        for f in range(args.folds):
            test_start = dates[fold_start_idx + f * args.step_days]
            end_pos = fold_start_idx + (f + 1) * args.step_days
            test_end = dates[end_pos] if end_pos < len(dates) else dates[-1] + pd.Timedelta(days=1)

            res = run_fold(full, horizon, test_start, test_end, args.top_n)
            if res is None:
                continue
            results.append(res)
            print(f"  fold {f + 1}: test_start={test_start.date()}  n_train={res['n_train']:>6}  "
                  f"n_test_days={res['n_days']:>3}  mean_IC={res['mean_ic']:+.3f}  "
                  f"%days_IC>0={res['pct_days_positive_ic'] * 100:4.0f}%  "
                  f"top-bottom_spread={res['mean_top_bottom_spread_pct']:+.2f}%  "
                  f"AUC={res['auc']:.3f}  MAE={res['mae']:.4f}")

        if results:
            mean_ic = np.nanmean([r["mean_ic"] for r in results])
            pct_pos = np.nanmean([r["pct_days_positive_ic"] for r in results])
            mean_spread = np.nanmean([r["mean_top_bottom_spread_pct"] for r in results])
            mean_auc = np.nanmean([r["auc"] for r in results])
            print(f"  -- summary across {len(results)} folds: mean IC={mean_ic:+.3f}  "
                  f"%days_IC>0={pct_pos * 100:.0f}%  mean top-bottom spread={mean_spread:+.2f}%  "
                  f"mean AUC={mean_auc:.3f}")
            horizon_summaries[horizon] = {
                "mean_ic": float(mean_ic),
                "pct_days_positive_ic": float(pct_pos),
                "mean_top_bottom_spread_pct": float(mean_spread),
                "mean_auc": float(mean_auc),
                "n_folds": len(results),
            }
        print()

    if horizon_summaries:
        save_backtest_results(horizon_summaries, len(price_data), args.folds, args.step_days)
        print(f"Saved reliability summary for {len(horizon_summaries)} horizon(s) -- "
              f"predict.py/analyze.py will now show a horizon reliability ranking based on this run.")


if __name__ == "__main__":
    main()
