"""
Model explainability via SHAP (SHapley Additive exPlanations): shows which
specific features actually pushed a given prediction up or down, grounded
in the model's own learned splits -- unlike insights.py's hand-written
"technical read" bullets, which describe the general technical picture in
plain English but aren't literally derived from what this model weighted
for this specific prediction.

Computed on demand only (Stock Research page, one ticker at a time) -- not
run for every ticker in the nightly briefing or My Watchlist's bulk load,
since it's extra per-request compute that would meaningfully slow a
200+-ticker batch for comparatively little benefit there.
"""
import numpy as np
import pandas as pd
import shap

FEATURE_LABELS = {
    "ret_1d": "1-day return",
    "ret_5d": "5-day return",
    "ret_10d": "10-day return",
    "ret_20d": "20-day return",
    "sma_10_ratio": "distance from 10-day average",
    "sma_20_ratio": "distance from 20-day average",
    "sma_50_ratio": "distance from 50-day average",
    "ema_12_ratio": "distance from 12-day EMA",
    "ema_26_ratio": "distance from 26-day EMA",
    "rsi_14": "RSI (momentum)",
    "macd": "MACD line",
    "macd_signal": "MACD signal line",
    "macd_hist": "MACD histogram",
    "bb_pct_b": "Bollinger Band position",
    "bb_width": "Bollinger Band width (volatility)",
    "vol_10d": "10-day volatility",
    "vol_20d": "20-day volatility",
    "vol_ratio": "short- vs. long-term volatility ratio",
    "volume_ratio_10d": "volume vs. 10-day average",
    "volume_ratio_20d": "volume vs. 20-day average",
    "high_low_range": "daily trading range",
    "dist_from_52w_high": "distance from 52-week high",
    "dist_from_52w_low": "distance from 52-week low",
    "days_since_earnings": "days since last earnings",
    "days_until_earnings": "days until next earnings",
    "last_earnings_surprise_pct": "last earnings surprise",
    "mkt_ret_5d": "S&P 500 5-day return",
    "mkt_ret_20d": "S&P 500 20-day return",
    "mkt_sma20_ratio": "S&P 500 vs. its 20-day average",
    "vix_level": "VIX (market fear gauge)",
    "vix_change_5d": "VIX 5-day change",
}


def explain_prediction(regressor, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    regressor: a trained LightGBM regressor (models[h]["regressor"]).
    X: a single-row feature DataFrame with the same columns it was trained on.

    Returns the top_n features by |SHAP value|, each as:
      {feature, label, value, contribution_pp, direction}
    contribution_pp is in prediction-percentage-points -- the model predicts
    a fractional return, so SHAP values (in that same fractional unit) are
    scaled by 100 to match the predicted-return-% shown everywhere else in
    the UI. base_value + sum(all SHAP values) == the model's raw prediction.
    """
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X)[0]

    order = np.argsort(-np.abs(shap_values))[:top_n]
    out = []
    for i in order:
        col = X.columns[i]
        contribution_pp = float(shap_values[i]) * 100
        out.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "value": float(X.iloc[0, i]),
            "contribution_pp": round(contribution_pp, 3),
            "direction": "up" if contribution_pp > 0 else "down",
        })
    return out
