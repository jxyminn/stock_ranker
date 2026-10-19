import numpy as np
import pandas as pd
import pytest

from config import SCORE_WEIGHTS, HORIZONS
from features import FEATURE_COLUMNS
import predict as predict_module
from predict import zscore, composite_score


def test_zscore_of_constant_series_is_zero_not_nan():
    s = pd.Series([5.0, 5.0, 5.0, 5.0])
    z = zscore(s)
    assert (z == 0).all()
    assert not z.isna().any()


def test_zscore_has_zero_mean_unit_std():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std() == pytest.approx(1.0, abs=1e-9)


def test_composite_score_matches_manual_weighted_sum():
    pred_return = pd.Series([0.01, 0.02, -0.01])
    prob_up = pd.Series([0.5, 0.6, 0.4])
    vol = pd.Series([0.02, 0.02, 0.02])

    risk_adjusted = pred_return / vol
    expected = (
        SCORE_WEIGHTS["predicted_return"] * zscore(pred_return)
        + SCORE_WEIGHTS["prob_up"] * zscore(prob_up)
        + SCORE_WEIGHTS["risk_adjusted"] * zscore(risk_adjusted)
    )

    actual = composite_score(pred_return, prob_up, vol)
    pd.testing.assert_series_equal(actual, expected)


def test_composite_score_ranks_strictly_better_stock_higher():
    # stock B beats stock A on every dimension: higher predicted return,
    # higher probability of a gain, lower volatility (better risk-adjusted) --
    # it must end up with a strictly higher composite score.
    pred_return = pd.Series({"A": 0.01, "B": 0.03, "C": -0.02})
    prob_up = pd.Series({"A": 0.5, "B": 0.65, "C": 0.4})
    vol = pd.Series({"A": 0.03, "B": 0.02, "C": 0.03})

    scores = composite_score(pred_return, prob_up, vol)
    assert scores["B"] > scores["A"] > scores["C"]


def test_composite_score_zero_volatility_does_not_blow_up():
    # vol.clip(lower=1e-4) inside composite_score must prevent a division by
    # zero / inf risk-adjusted score for a (degenerate) zero-volatility row.
    pred_return = pd.Series([0.01, 0.02])
    prob_up = pd.Series([0.5, 0.5])
    vol = pd.Series([0.0, 0.02])

    scores = composite_score(pred_return, prob_up, vol)
    assert np.isfinite(scores).all()


def _make_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "MODELS_DIR", tmp_path)
    for h in HORIZONS:
        (tmp_path / f"horizon_{h}.joblib").touch()


def test_load_models_raises_clear_error_on_schema_mismatch(monkeypatch, tmp_path):
    """
    Regression test for the exact failure this feature exists to prevent:
    previously, a model trained on an older FEATURE_COLUMNS set (e.g. before
    macro features were added) would fail deep inside LightGBM with a cryptic
    "number of features in data (23) is not the same as training data (26)"
    error. load_models() must now catch this at load time with a clear,
    actionable message instead.
    """
    _make_model_files(tmp_path, monkeypatch)
    stale_columns = FEATURE_COLUMNS[:-1]  # simulates a model trained before a feature was added

    monkeypatch.setattr(predict_module.joblib, "load", lambda path: {
        "regressor": None, "classifier": None, "holdout_mae": 0.0, "holdout_auc": 0.5,
        "feature_columns": stale_columns,
    })

    with pytest.raises(SystemExit, match="different feature set"):
        predict_module.load_models()


def test_load_models_accepts_matching_schema(monkeypatch, tmp_path):
    _make_model_files(tmp_path, monkeypatch)

    monkeypatch.setattr(predict_module.joblib, "load", lambda path: {
        "regressor": "reg", "classifier": "clf", "holdout_mae": 0.0, "holdout_auc": 0.5,
        "feature_columns": list(FEATURE_COLUMNS),
    })

    models = predict_module.load_models()
    assert len(models) == len(HORIZONS)


def test_load_models_warns_but_accepts_legacy_model_without_schema(monkeypatch, tmp_path, capsys):
    """Models saved before this check existed have no 'feature_columns' key -- must warn, not crash."""
    _make_model_files(tmp_path, monkeypatch)

    monkeypatch.setattr(predict_module.joblib, "load", lambda path: {
        "regressor": "reg", "classifier": "clf", "holdout_mae": 0.0, "holdout_auc": 0.5,
    })

    models = predict_module.load_models()
    assert len(models) == len(HORIZONS)
    assert "no recorded feature schema" in capsys.readouterr().out
