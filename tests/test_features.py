import numpy as np
import pandas as pd
import pytest

from features import (
    build_features, add_earnings_features, build_macro_context, add_macro_features,
    TECHNICAL_FEATURE_COLUMNS, EARNINGS_FEATURE_COLUMNS, MACRO_FEATURE_COLUMNS,
)
from tests.conftest import make_ohlcv


def test_build_features_returns_expected_columns(ohlcv):
    feats = build_features(ohlcv)
    assert list(feats.columns) == TECHNICAL_FEATURE_COLUMNS


def test_rsi_is_bounded(ohlcv):
    feats = build_features(ohlcv)
    rsi = feats["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_rsi_is_100_when_every_day_in_window_is_a_gain():
    # A strictly increasing price has zero losses in the trailing 14-day
    # window -- RSI is mathematically 100 (maximally overbought) here, not
    # undefined; regression test for the avg_loss==0 -> NaN bug.
    dates = pd.bdate_range("2024-01-02", periods=60, tz="America/New_York")
    close = pd.Series(np.linspace(100, 200, 60), index=dates)
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(60, 1_000_000),
    }, index=dates)
    feats = build_features(df)
    assert feats["rsi_14"].iloc[-1] == 100.0


def test_rsi_is_50_for_perfectly_flat_price():
    dates = pd.bdate_range("2024-01-02", periods=60, tz="America/New_York")
    close = pd.Series(np.full(60, 100.0), index=dates)
    df = pd.DataFrame({
        "open": close, "high": close, "low": close,
        "close": close, "volume": np.full(60, 1_000_000),
    }, index=dates)
    feats = build_features(df)
    assert feats["rsi_14"].iloc[-1] == 50.0


def test_no_lookahead_truncated_features_match_full_features(ohlcv):
    """
    Every feature is documented to use only data up to and including row t.
    Truncating the input to N rows must not change any of the first N-derived
    feature values versus computing on the full history -- if it did, some
    feature would be peeking at data past its own row.
    """
    n = 250
    full_feats = build_features(ohlcv)
    truncated_feats = build_features(ohlcv.iloc[:n])

    pd.testing.assert_frame_equal(
        full_feats.iloc[:n].reset_index(drop=True),
        truncated_feats.reset_index(drop=True),
    )


def test_add_earnings_features_none_gives_nan_columns(ohlcv):
    feats = build_features(ohlcv)
    out = add_earnings_features(feats, None)
    for col in EARNINGS_FEATURE_COLUMNS:
        assert col in out.columns
        assert out[col].isna().all()


def test_add_earnings_features_since_until_surprise():
    tz = "America/New_York"
    earnings = pd.DataFrame(
        {"surprise_pct": [5.0, -2.0, 3.0]},
        index=pd.DatetimeIndex(["2024-02-01", "2024-05-01", "2024-08-01"], tz=tz),
    )
    probe_dates = pd.DatetimeIndex(
        ["2024-01-15", "2024-02-01", "2024-02-15", "2024-04-30", "2024-08-01", "2024-09-10"], tz=tz,
    )
    feat_df = pd.DataFrame(index=probe_dates)
    out = add_earnings_features(feat_df, earnings)

    def days_between(a, b):
        # Match the production code: it strips tz before diffing so calendar-day
        # counts aren't off by an hour across a DST transition (e.g. mid-March).
        return (pd.Timestamp(b).tz_localize(None) - pd.Timestamp(a).tz_localize(None)) / pd.Timedelta(days=1)

    # Before any report: no past report to reference, but the next (future)
    # report date is known -- that's calendar info, not a price-lookahead.
    row = out.loc["2024-01-15"]
    assert pd.isna(row["days_since_earnings"]) and pd.isna(row["last_earnings_surprise_pct"])
    assert row["days_until_earnings"] == days_between("2024-01-15", "2024-02-01")

    # Exactly on a report date: that report counts as "past" (t <= t).
    row = out.loc["2024-02-01"]
    assert row["days_since_earnings"] == 0
    assert row["last_earnings_surprise_pct"] == 5.0
    assert row["days_until_earnings"] == days_between("2024-02-01", "2024-05-01")

    # Between two reports: must reference the most recent PAST one only --
    # the no-lookahead check. Must never see the 2024-05-01 surprise (-2.0)
    # before that date has actually happened.
    row = out.loc["2024-04-30"]
    assert row["days_since_earnings"] == days_between("2024-02-01", "2024-04-30")
    assert row["last_earnings_surprise_pct"] == 5.0
    assert row["days_until_earnings"] == days_between("2024-04-30", "2024-05-01")

    # After the last known report: no future report on record.
    row = out.loc["2024-09-10"]
    assert row["days_since_earnings"] == days_between("2024-08-01", "2024-09-10")
    assert row["last_earnings_surprise_pct"] == 3.0
    assert pd.isna(row["days_until_earnings"])


def test_build_macro_context_handles_mismatched_timezones():
    """
    Regression test: yfinance returns ^VIX tz-localized to America/Chicago
    while equities/SPY are America/New_York. Both are midnight timestamps, so
    left tz-aware they represent different UTC instants and silently fail to
    align on reindex -- this previously left vix_level all-NaN with no error.
    """
    dates_ny = pd.bdate_range("2024-01-02", periods=60, tz="America/New_York")
    # Independently localized at midnight in Chicago's own timezone -- this is
    # what real yfinance data looks like (both indices are midnight-local,
    # each in their own market's tz), NOT a tz_convert() of the NY index
    # (which would preserve the UTC instant and shift the wall-clock date).
    dates_chi = pd.bdate_range("2024-01-02", periods=60, tz="America/Chicago")

    spy_close = pd.Series(np.linspace(400, 420, 60), index=dates_ny)
    spy_df = pd.DataFrame({
        "open": spy_close, "high": spy_close, "low": spy_close,
        "close": spy_close, "volume": np.full(60, 1_000_000),
    }, index=dates_ny)

    vix_close = pd.Series(np.linspace(15, 20, 60), index=dates_chi)
    vix_df = pd.DataFrame({
        "open": vix_close, "high": vix_close, "low": vix_close,
        "close": vix_close, "volume": np.zeros(60),
    }, index=dates_chi)

    macro = build_macro_context(spy_df, vix_df)

    assert list(macro.columns) == MACRO_FEATURE_COLUMNS
    # the whole point of the regression test: vix_level must actually populate
    assert macro["vix_level"].notna().sum() > 0
    assert macro["vix_level"].iloc[-1] == pytest.approx(20.0)


def test_add_macro_features_broadcasts_and_ffills(ohlcv):
    feats = build_features(ohlcv)
    macro_dates = ohlcv.index[::2]  # sparse: every other trading day has macro data
    macro = pd.DataFrame(
        {col: np.arange(len(macro_dates), dtype=float) for col in MACRO_FEATURE_COLUMNS},
        index=macro_dates,
    )

    out = add_macro_features(feats, macro)
    for col in MACRO_FEATURE_COLUMNS:
        assert col in out.columns
    # gap days (no exact macro row) must be forward-filled, not NaN, once past
    # the first available macro date
    assert out[MACRO_FEATURE_COLUMNS[0]].iloc[5:].notna().all()


def test_add_macro_features_none_gives_nan_columns(ohlcv):
    feats = build_features(ohlcv)
    out = add_macro_features(feats, None)
    for col in MACRO_FEATURE_COLUMNS:
        assert out[col].isna().all()
