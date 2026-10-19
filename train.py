"""
Train one regressor + one classifier per horizon on a broad universe of stocks.

Usage:
    python train.py
    python train.py --universe AAPL MSFT NVDA ...
    python train.py --lookback-days 1800
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error

from config import HORIZONS, TRAINING_LOOKBACK_DAYS, TRAINING_UNIVERSE, MODELS_DIR, RANDOM_STATE, LGBM_PARAMS
from data_fetcher import fetch_history_batch, fetch_earnings, fetch_history
from features import (
    build_features, add_earnings_features, build_macro_context, add_macro_features,
    FEATURE_COLUMNS, TECHNICAL_FEATURE_COLUMNS,
)
from labeling import build_labels


def fetch_macro_context(lookback_days: int) -> pd.DataFrame | None:
    """Fetch SPY + VIX once and build the shared market-regime feature frame."""
    spy_df = fetch_history("SPY", lookback_days)
    vix_df = fetch_history("^VIX", lookback_days)
    if spy_df.empty or vix_df.empty:
        print("Warning: couldn't fetch SPY/VIX history; training without macro features.")
        return None
    return build_macro_context(spy_df, vix_df)


def build_training_frame(price_data: dict[str, pd.DataFrame], horizon: int,
                          macro_df: pd.DataFrame | None) -> pd.DataFrame:
    """Stack per-ticker feature+label rows into one training dataframe for a given horizon."""
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
    # Only require technical features to be present -- earnings features are
    # legitimately NaN when a ticker has no earnings history on record, and
    # LightGBM handles NaN splits natively, so we don't want to drop those rows.
    full = full.dropna(subset=TECHNICAL_FEATURE_COLUMNS + [f"fwd_return_{horizon}", f"up_{horizon}"])
    return full


def train_horizon(full: pd.DataFrame, horizon: int) -> dict:
    """Train regressor + classifier for one horizon using a time-respecting split."""
    full = full.sort_index()
    X = full[FEATURE_COLUMNS]
    y_reg = full[f"fwd_return_{horizon}"]
    y_clf = full[f"up_{horizon}"]

    tscv = TimeSeriesSplit(n_splits=4)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]  # final split = most recent holdout

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    y_clf_train, y_clf_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

    # Winsorize the regression TARGET at the 1st/99th percentile of the
    # training split only (never touches y_reg_test, so the reported MAE
    # still reflects real outcomes). A tree regressor's leaf prediction is
    # just the mean label of the training rows that land in it -- a handful
    # of extreme historical forward returns (meme-stock spikes, post-crash
    # rebounds) can otherwise pull that mean to an implausible level for
    # every future ticker that lands in the same leaf, especially at longer
    # horizons where returns compound.
    lo, hi = y_reg_train.quantile([0.01, 0.99])
    y_reg_train = y_reg_train.clip(lo, hi)

    reg = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=RANDOM_STATE, verbose=-1)
    reg.fit(X_train, y_reg_train)
    reg_mae = mean_absolute_error(y_reg_test, reg.predict(X_test))

    clf = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=RANDOM_STATE, verbose=-1)
    clf.fit(X_train, y_clf_train)
    clf_auc = roc_auc_score(y_clf_test, clf.predict_proba(X_test)[:, 1])

    print(f"  [horizon={horizon}d] holdout MAE={reg_mae:.4f}  AUC={clf_auc:.3f}  "
          f"(AUC 0.5 = no better than chance)")

    return {
        "regressor": reg, "classifier": clf, "holdout_mae": reg_mae, "holdout_auc": clf_auc,
        # Recorded so predict.load_models() can detect a stale model -- e.g.
        # after features.py adds/removes a column -- and fail with a clear
        # "retrain" message instead of a cryptic LightGBM shape-mismatch error.
        "feature_columns": list(FEATURE_COLUMNS),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", nargs="+", default=None,
                         help="Tickers to train on. Defaults to config.TRAINING_UNIVERSE "
                              "(50 popular large-caps + every watchlist ticker).")
    parser.add_argument("--lookback-days", type=int, default=TRAINING_LOOKBACK_DAYS)
    args = parser.parse_args()

    universe = args.universe or TRAINING_UNIVERSE
    print(f"Training on {len(universe)} tickers, {args.lookback_days} days of history...")
    price_data = fetch_history_batch(universe, args.lookback_days)
    print(f"Usable tickers: {len(price_data)}/{len(universe)}")

    if len(price_data) < 5:
        raise SystemExit("Too few tickers with usable data to train. Check network/tickers.")

    macro_df = fetch_macro_context(args.lookback_days)

    for horizon in HORIZONS:
        print(f"\nBuilding training set for {horizon}-day horizon...")
        full = build_training_frame(price_data, horizon, macro_df)
        print(f"  {len(full)} training rows across {full['ticker'].nunique()} tickers")
        result = train_horizon(full, horizon)
        joblib.dump(result, MODELS_DIR / f"horizon_{horizon}.joblib")
        print(f"  saved -> {MODELS_DIR / f'horizon_{horizon}.joblib'}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
