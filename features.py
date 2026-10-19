"""
Feature engineering. The same function is used at training time and at live
prediction time, so features must be computable from OHLCV data alone with no
lookahead (every feature at row t only uses data up to and including t).
"""
import numpy as np
import pandas as pd

TECHNICAL_FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "ema_12_ratio", "ema_26_ratio",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_pct_b", "bb_width",
    "vol_10d", "vol_20d", "vol_ratio",
    "volume_ratio_10d", "volume_ratio_20d",
    "high_low_range", "dist_from_52w_high", "dist_from_52w_low",
]

# Earnings-calendar features (see add_earnings_features). Left as NaN when no
# earnings data is available for a ticker/date -- LightGBM handles NaN splits
# natively, so these are intentionally excluded from any dropna() on rows.
EARNINGS_FEATURE_COLUMNS = [
    "days_since_earnings", "days_until_earnings", "last_earnings_surprise_pct",
]

# Market-regime features (see build_macro_context / add_macro_features). Same
# value broadcast across every ticker on a given date, since these describe
# the market as a whole rather than any one stock. NaN-tolerant like earnings.
MACRO_FEATURE_COLUMNS = [
    "mkt_ret_5d", "mkt_ret_20d", "mkt_sma20_ratio", "vix_level", "vix_change_5d",
]

FEATURE_COLUMNS = TECHNICAL_FEATURE_COLUMNS + EARNINGS_FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss==0 divides out to NaN above, but a window with zero losses is
    # not undefined -- it's the maximally overbought case (RSI=100), or, if
    # gains are also zero (flat price), the neutral case (RSI=50). A strong,
    # low-volatility uptrend can realistically hit zero losses over 14 days,
    # so leaving this NaN would silently drop that row from training/live
    # features via the no-NaN requirement on TECHNICAL_FEATURE_COLUMNS.
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: raw OHLCV dataframe with columns [open, high, low, close, volume], indexed by date.
    Output: dataframe of engineered features, same index, with NaN rows (from
    warm-up windows) left in -- caller decides whether to drop them.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Returns over multiple windows
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)

    # Trend: price relative to moving averages
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    out["sma_10_ratio"] = close / sma_10 - 1
    out["sma_20_ratio"] = close / sma_20 - 1
    out["sma_50_ratio"] = close / sma_50 - 1

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["ema_12_ratio"] = close / ema_12 - 1
    out["ema_26_ratio"] = close / ema_26 - 1

    # Momentum
    out["rsi_14"] = _rsi(close, 14)

    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # Volatility / bands
    bb_mid = sma_20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    daily_ret = close.pct_change()
    out["vol_10d"] = daily_ret.rolling(10).std()
    out["vol_20d"] = daily_ret.rolling(20).std()
    out["vol_ratio"] = out["vol_10d"] / out["vol_20d"].replace(0, np.nan)

    # Volume behavior
    vol_sma_10 = volume.rolling(10).mean()
    vol_sma_20 = volume.rolling(20).mean()
    out["volume_ratio_10d"] = volume / vol_sma_10.replace(0, np.nan)
    out["volume_ratio_20d"] = volume / vol_sma_20.replace(0, np.nan)

    # Range / position within recent trading range
    out["high_low_range"] = (high - low) / close.replace(0, np.nan)
    roll_high_252 = high.rolling(252, min_periods=60).max()
    roll_low_252 = low.rolling(252, min_periods=60).min()
    out["dist_from_52w_high"] = close / roll_high_252 - 1
    out["dist_from_52w_low"] = close / roll_low_252 - 1

    return out[TECHNICAL_FEATURE_COLUMNS]


def add_earnings_features(feat_df: pd.DataFrame, earnings: pd.DataFrame | None) -> pd.DataFrame:
    """
    Attach earnings-calendar features to a feature frame indexed by date.

    `earnings` must be indexed by report date with a 'surprise_pct' column
    (as returned by data_fetcher.fetch_earnings) and may include both past
    (actual) and known upcoming (estimated) report dates.

    For each row date t:
      - days_since_earnings / last_earnings_surprise_pct use only report
        dates <= t (the most recent report as of that date -- no lookahead).
      - days_until_earnings uses the next report date > t. This is calendar
        information (companies pre-announce report dates), not future price
        data, so it's safe to use as of date t.
    """
    out = feat_df.copy()
    if earnings is None or earnings.empty:
        for col in EARNINGS_FEATURE_COLUMNS:
            out[col] = np.nan
        return out

    report_dates = earnings.index
    if report_dates.tz is not None:
        report_dates = report_dates.tz_localize(None)
    order = report_dates.argsort()
    report_dates = report_dates.to_numpy()[order]
    surprise_vals = earnings["surprise_pct"].to_numpy()[order]

    row_dates = out.index
    if getattr(row_dates, "tz", None) is not None:
        row_dates = row_dates.tz_localize(None)

    since, until, surprise = [], [], []
    for t in row_dates:
        past = np.where(report_dates <= t)[0]
        future = np.where(report_dates > t)[0]
        if len(past):
            since.append((t - report_dates[past[-1]]) / np.timedelta64(1, "D"))
            surprise.append(surprise_vals[past[-1]])
        else:
            since.append(np.nan)
            surprise.append(np.nan)
        until.append((report_dates[future[0]] - t) / np.timedelta64(1, "D") if len(future) else np.nan)

    out["days_since_earnings"] = since
    out["days_until_earnings"] = until
    out["last_earnings_surprise_pct"] = surprise
    return out


def build_macro_context(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build market-regime features from SPY (broad market) and VIX (fear gauge)
    daily history. Indexed by date; the same row applies to every ticker on
    that date, capturing market-wide context no single stock's own price/
    volume history can (e.g. "the whole market was selling off").
    """
    # yfinance returns ^VIX tz-localized to America/Chicago vs. America/New_York
    # for equities -- both indices are midnight timestamps, so left tz-aware
    # they'd represent different UTC instants and never align on reindex.
    spy_index = spy_df.index.tz_localize(None) if spy_df.index.tz is not None else spy_df.index
    vix_index = vix_df.index.tz_localize(None) if vix_df.index.tz is not None else vix_df.index

    out = pd.DataFrame(index=spy_index)
    spy_close = pd.Series(spy_df["close"].to_numpy(), index=spy_index)
    out["mkt_ret_5d"] = spy_close.pct_change(5)
    out["mkt_ret_20d"] = spy_close.pct_change(20)
    out["mkt_sma20_ratio"] = spy_close / spy_close.rolling(20).mean() - 1

    vix_close = pd.Series(vix_df["close"].to_numpy(), index=vix_index).reindex(out.index).ffill()
    out["vix_level"] = vix_close
    out["vix_change_5d"] = vix_close.pct_change(5)
    return out


def add_macro_features(feat_df: pd.DataFrame, macro_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Attach market-regime features to a per-ticker feature frame, aligned by
    date (left join + forward-fill for any date gaps between the ticker's
    calendar and SPY/VIX's).
    """
    out = feat_df.copy()
    if macro_df is None or macro_df.empty:
        for col in MACRO_FEATURE_COLUMNS:
            out[col] = np.nan
        return out

    idx = out.index.tz_localize(None) if getattr(out.index, "tz", None) is not None else out.index
    macro_idx = macro_df.index.tz_localize(None) if getattr(macro_df.index, "tz", None) is not None else macro_df.index
    macro = macro_df.copy()
    macro.index = macro_idx
    aligned = macro.reindex(idx).ffill()

    for col in MACRO_FEATURE_COLUMNS:
        out[col] = aligned[col].to_numpy()
    return out
