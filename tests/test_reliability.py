import reliability as reliability_module
from reliability import save_backtest_results, load_backtest_results, reliability_ranking, format_ranking_lines


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(reliability_module, "BACKTEST_RESULTS_PATH", tmp_path / "backtest_results.json")

    summaries = {
        5: {"mean_ic": 0.02, "pct_days_positive_ic": 0.5, "mean_top_bottom_spread_pct": 0.3, "mean_auc": 0.51, "n_folds": 6},
        30: {"mean_ic": 0.06, "pct_days_positive_ic": 0.6, "mean_top_bottom_spread_pct": 4.5, "mean_auc": 0.54, "n_folds": 6},
    }
    save_backtest_results(summaries, universe_size=50, folds=6, step_days=40)

    loaded = load_backtest_results()
    assert loaded is not None
    assert loaded["universe_size"] == 50
    assert loaded["folds"] == 6
    assert loaded["horizons"]["5"]["mean_ic"] == 0.02
    assert loaded["horizons"]["30"]["mean_ic"] == 0.06


def test_load_backtest_results_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(reliability_module, "BACKTEST_RESULTS_PATH", tmp_path / "does_not_exist.json")
    assert load_backtest_results() is None


def test_reliability_ranking_sorted_descending_by_mean_ic():
    results = {
        "horizons": {
            "5": {"mean_ic": 0.02, "mean_auc": 0.51},
            "10": {"mean_ic": -0.01, "mean_auc": 0.49},
            "30": {"mean_ic": 0.06, "mean_auc": 0.54},
        }
    }
    ranked = reliability_ranking(results)
    assert [r["horizon"] for r in ranked] == [30, 5, 10]


def test_reliability_ranking_none_when_no_results():
    assert reliability_ranking(None) is None
    assert reliability_ranking({"horizons": {}}) is None


def test_format_ranking_lines_fallback_message_when_unavailable():
    lines = format_ranking_lines(None)
    assert len(lines) == 1
    assert "run `python backtest.py`" in lines[0]


def test_format_ranking_lines_marks_most_and_least_reliable():
    results = {
        "run_at": "2026-07-07T14:22:00+00:00",
        "folds": 6,
        "universe_size": 50,
        "horizons": {
            "5": {"mean_ic": 0.02, "mean_auc": 0.51},
            "30": {"mean_ic": 0.06, "mean_auc": 0.54},
        },
    }
    lines = format_ranking_lines(results)
    assert "most reliable" in lines[1]
    assert "30d" in lines[1]
    assert "least reliable" in lines[2]
    assert "5d" in lines[2]
