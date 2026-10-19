import numpy as np
import pandas as pd
import pytest

from backtest import evaluate_day, run_fold, build_panel
from features import FEATURE_COLUMNS


def test_evaluate_day_spread_matches_hand_calc():
    # prob_up/vol_20d held constant so composite_score's ranking is driven
    # purely by pred_return -- makes the expected top/bottom split unambiguous.
    day = pd.DataFrame({
        "pred_return": [0.05, 0.04, 0.03, 0.02, 0.01, 0.00],
        "prob_up": [0.5] * 6,
        "vol_20d": [0.02] * 6,
        "actual": [0.10, 0.08, 0.00, -0.02, -0.05, -0.10],
    })
    result = evaluate_day(day, top_n=2)

    # top 2 by pred_return: rows 0,1 (actual 0.10, 0.08) -> mean 0.09
    # bottom 2 by pred_return: rows 4,5 (actual -0.05, -0.10) -> mean -0.075
    assert result["spread"] == pytest.approx(0.09 - (-0.075))
    assert result["ic"] == pytest.approx(1.0)  # pred_return and actual are perfectly rank-aligned


def test_evaluate_day_returns_none_for_too_few_names():
    day = pd.DataFrame({
        "pred_return": [0.01, 0.02, 0.03],
        "prob_up": [0.5, 0.5, 0.5],
        "vol_20d": [0.02, 0.02, 0.02],
        "actual": [0.01, -0.01, 0.02],
    })
    assert evaluate_day(day, top_n=1) is None


def test_evaluate_day_returns_none_for_degenerate_score():
    # every row identical -> composite_score has zero variance -> no ranking is possible
    day = pd.DataFrame({
        "pred_return": [0.01] * 6,
        "prob_up": [0.5] * 6,
        "vol_20d": [0.02] * 6,
        "actual": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
    })
    assert evaluate_day(day, top_n=2) is None


def _synthetic_panel(horizon: int, n_tickers: int = 8, n_dates: int = 100, seed: int = 0) -> pd.DataFrame:
    """
    A tiny multi-ticker panel with duplicate date labels across tickers --
    the exact shape run_fold operates on in production. Each "ticker" has a
    fixed quality baked into ret_5d that also drives its forward return, so a
    model trained on it should learn genuine, learnable rank separation.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates, tz="America/New_York")

    rows = []
    for d in dates:
        for t in range(n_tickers):
            quality = (t - n_tickers / 2) / n_tickers
            row = {col: rng.normal(0, 0.01) for col in FEATURE_COLUMNS}
            row["ret_5d"] = quality + rng.normal(0, 0.001)
            row[f"fwd_return_{horizon}"] = quality + rng.normal(0, 0.001)
            row[f"up_{horizon}"] = float(row[f"fwd_return_{horizon}"] > 0)
            rows.append((d, row))

    full = pd.DataFrame([r for _, r in rows], index=pd.DatetimeIndex([d for d, _ in rows]))
    return full.sort_index()


def test_run_fold_spread_is_not_degenerately_zero():
    """
    Regression test for the duplicate-date-index bug: run_fold's eval_df is
    indexed by date, and that index is duplicated across tickers. Before the
    fix, .loc on a subset of ranked labels silently expanded back out to
    *all* rows sharing that date, so top-N and bottom-N were the same (whole)
    group and the reported spread came out as exactly 0.00% on every fold,
    regardless of how much real dispersion existed in the data.

    This panel bakes in a strong, learnable per-ticker return differential,
    so a correctly-computed spread must be clearly nonzero.
    """
    horizon = 5
    full = _synthetic_panel(horizon)
    dates = full.index.unique().sort_values()
    test_start = dates[70]  # 70 * 8 tickers = 560 training rows, above run_fold's 500-row minimum
    test_end = dates[-1] + pd.Timedelta(days=1)

    fast_params = {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3,
                   "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8}
    result = run_fold(full, horizon, test_start, test_end, top_n=2, lgbm_params=fast_params)

    assert result is not None
    assert result["mean_top_bottom_spread_pct"] != pytest.approx(0.0, abs=1e-9)
    # with a strong baked-in signal, the top-ranked names should realize a
    # meaningfully higher return than the bottom-ranked names, not just "nonzero"
    assert result["mean_top_bottom_spread_pct"] > 1.0


def test_build_panel_columns_present(monkeypatch):
    # build_panel calls fetch_earnings(ticker) per ticker -- mock it out so
    # this test stays offline/fast instead of hitting the network.
    import backtest as backtest_module
    from tests.conftest import make_ohlcv

    monkeypatch.setattr(backtest_module, "fetch_earnings", lambda ticker: pd.DataFrame())

    price_data = {"FAKE": make_ohlcv()}
    full = build_panel(price_data, horizon=5, macro_df=None)
    for col in FEATURE_COLUMNS:
        assert col in full.columns
    assert "fwd_return_5" in full.columns
    assert "up_5" in full.columns
