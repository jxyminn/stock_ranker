"""
Score and rank a list of tickers, using the latest available (live) data,
across all trained horizons.

Usage:
    python predict.py --tickers AAPL MSFT NVDA
    python predict.py --tickers-file my_list.txt
"""
import argparse
import sys
import joblib
import numpy as np
import pandas as pd

from config import HORIZONS, LIVE_LOOKBACK_DAYS, MODELS_DIR, SCORE_WEIGHTS
from data_fetcher import fetch_history, fetch_earnings, validate_tickers
from features import (
    build_features, add_earnings_features, build_macro_context, add_macro_features,
    FEATURE_COLUMNS, TECHNICAL_FEATURE_COLUMNS,
)
from reliability import load_backtest_results, format_ranking_lines


def load_models() -> dict:
    models = {}
    for h in HORIZONS:
        path = MODELS_DIR / f"horizon_{h}.joblib"
        if not path.exists():
            raise SystemExit(f"No trained model for horizon {h}d. Run train.py first.")
        model = joblib.load(path)

        trained_cols = model.get("feature_columns")
        if trained_cols is None:
            print(f"[predict] warning: horizon {h}d model has no recorded feature schema "
                  f"(trained before this check existed) -- assuming it matches. If predictions "
                  f"error out or look wrong, run `python train.py` to retrain.")
        elif trained_cols != FEATURE_COLUMNS:
            missing = [c for c in FEATURE_COLUMNS if c not in trained_cols]
            extra = [c for c in trained_cols if c not in FEATURE_COLUMNS]
            detail = []
            if missing:
                detail.append(f"code now expects but this model doesn't have: {missing}")
            if extra:
                detail.append(f"model has but code no longer produces: {extra}")
            raise SystemExit(
                f"Model for horizon {h}d ({path}) was trained on a different feature set than "
                f"the current code produces ({'; '.join(detail)}).\n"
                f"Run `python train.py` to retrain. If you're running the Streamlit app, also "
                f"fully restart the server afterward -- it caches loaded models in memory across "
                f"reruns, so a retrain alone won't update an already-running session."
            )

        models[h] = model
    return models


def fetch_macro_context(lookback_days: int) -> pd.DataFrame | None:
    """
    Fetch SPY + VIX and build the shared market-regime feature frame. Uses
    fetch_history's normal disk cache (1hr freshness) rather than always-live,
    since it's called once per ticker scored and SPY/VIX don't need to be
    fetched fresh every single time within one scoring run.
    """
    spy_df = fetch_history("SPY", lookback_days)
    vix_df = fetch_history("^VIX", lookback_days)
    if spy_df.empty or vix_df.empty:
        return None
    return build_macro_context(spy_df, vix_df)


def latest_features_for_ticker(ticker: str) -> pd.Series | None:
    """Fetch live history and compute the most recent feature row for a ticker."""
    df = fetch_history(ticker, LIVE_LOOKBACK_DAYS, use_cache=False)  # no cache: always live
    if df is None or len(df) < 60:
        print(f"[predict] skipping {ticker}: insufficient live history")
        return None
    feats = build_features(df)
    feats = add_earnings_features(feats, fetch_earnings(ticker))
    feats = add_macro_features(feats, fetch_macro_context(LIVE_LOOKBACK_DAYS))
    latest = feats.iloc[-1]
    if latest[TECHNICAL_FEATURE_COLUMNS].isna().any():
        print(f"[predict] skipping {ticker}: incomplete latest features")
        return None
    latest["_vol_20d"] = feats.iloc[-1]["vol_20d"]
    latest["_asof_date"] = df.index[-1]
    latest["_last_close"] = df["close"].iloc[-1]
    return latest


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / std if std > 1e-9 else s * 0


def composite_score(pred_return: pd.Series, prob_up: pd.Series, vol: pd.Series) -> pd.Series:
    """
    Combine model outputs into the ranking score used everywhere -- live
    prediction and the backtest both call this, so a backtest result reflects
    the same formula that would run live, not a re-implementation of it.
    """
    risk_adjusted = pred_return / vol.clip(lower=1e-4)
    return (
        SCORE_WEIGHTS["predicted_return"] * zscore(pred_return)
        + SCORE_WEIGHTS["prob_up"] * zscore(prob_up)
        + SCORE_WEIGHTS["risk_adjusted"] * zscore(risk_adjusted)
    )


def score_universe(tickers: list[str], models: dict) -> dict[int, pd.DataFrame]:
    """Returns {horizon: ranked_dataframe} for every horizon."""
    rows = {}
    for t in tickers:
        feat_row = latest_features_for_ticker(t)
        if feat_row is not None:
            rows[t] = feat_row
    if not rows:
        raise SystemExit("No tickers produced usable live features.")

    feat_df = pd.DataFrame(rows).T
    X = feat_df[FEATURE_COLUMNS].astype(float)

    results = {}
    for h in HORIZONS:
        reg = models[h]["regressor"]
        clf = models[h]["classifier"]

        pred_return = pd.Series(reg.predict(X), index=X.index)
        prob_up = pd.Series(clf.predict_proba(X)[:, 1], index=X.index)
        vol = feat_df["_vol_20d"].astype(float).clip(lower=1e-4)
        risk_adjusted = pred_return / vol
        composite = composite_score(pred_return, prob_up, vol)

        out = pd.DataFrame({
            "ticker": X.index,
            "last_close": feat_df["_last_close"].values,
            "as_of": feat_df["_asof_date"].values,
            f"pred_return_{h}d_pct": (pred_return.values * 100).round(2),
            "prob_up": prob_up.values.round(3),
            "risk_adjusted": risk_adjusted.values.round(3),
            "composite_score": composite.values.round(3),
        }).sort_values("composite_score", ascending=False).reset_index(drop=True)
        out.index = out.index + 1  # 1-based rank
        results[h] = out

    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", nargs="+", help="Space-separated tickers, e.g. AAPL MSFT NVDA")
    group.add_argument("--tickers-file", help="Path to a text file, one ticker per line")
    parser.add_argument("--top", type=int, default=10, help="How many results to print per horizon")
    args = parser.parse_args()

    if args.tickers:
        raw_tickers = args.tickers
    else:
        with open(args.tickers_file) as f:
            raw_tickers = [line.strip() for line in f if line.strip()]

    print(f"Validating {len(raw_tickers)} tickers...")
    tickers = validate_tickers(raw_tickers)
    if not tickers:
        sys.exit("None of the provided tickers returned data.")
    print(f"Valid tickers: {tickers}\n")

    models = load_models()
    results = score_universe(tickers, models)

    for h in HORIZONS:
        print(f"\n=== Top picks for {h}-day horizon ===")
        print(results[h].head(args.top).to_string(index=True))

    print()
    for line in format_ranking_lines(load_backtest_results()):
        print(line)


if __name__ == "__main__":
    main()
