"""
Persists and reads backtest.py's walk-forward reliability summary, so
predict.py, analyze.py, and the Streamlit app can show which horizon to
trust most without re-running the (multi-minute) backtest on every request.

Deliberately has no dependency on backtest.py or predict.py -- both of those
need to read/write this data, and backtest.py already imports from predict.py,
so this lives standalone to avoid a circular import.
"""
import json
from datetime import datetime, timezone

from config import BACKTEST_RESULTS_PATH


def save_backtest_results(horizon_summaries: dict, universe_size: int, folds: int, step_days: int) -> None:
    """horizon_summaries: {horizon: {mean_ic, pct_days_positive_ic, mean_top_bottom_spread_pct, mean_auc, n_folds}}"""
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": universe_size,
        "folds": folds,
        "step_days": step_days,
        "horizons": {str(h): s for h, s in horizon_summaries.items()},
    }
    BACKTEST_RESULTS_PATH.write_text(json.dumps(payload, indent=2))


def load_backtest_results() -> dict | None:
    if not BACKTEST_RESULTS_PATH.exists():
        return None
    try:
        return json.loads(BACKTEST_RESULTS_PATH.read_text())
    except Exception:
        return None


def reliability_ranking(results: dict | None) -> list[dict] | None:
    """Horizons sorted by mean_ic descending (most to least reliable), or None if unavailable."""
    if not results or not results.get("horizons"):
        return None
    ranked = sorted(
        ((int(h), summary) for h, summary in results["horizons"].items()),
        key=lambda item: item[1]["mean_ic"],
        reverse=True,
    )
    return [{"horizon": h, **summary} for h, summary in ranked]


def format_ranking_lines(results: dict | None) -> list[str]:
    """Plain-text lines for CLI output (predict.py, analyze.py) -- shared so both stay in sync."""
    ranked = reliability_ranking(results)
    if not ranked:
        return ["No backtest results found yet -- run `python backtest.py` to rank horizons by "
                "actual walk-forward reliability (this single-holdout MAE/AUC above is noisy)."]

    run_at = results["run_at"][:16].replace("T", " ")
    lines = [
        f"Horizon reliability ranking (from backtest.py, run {run_at} UTC, "
        f"{results['folds']} folds, {results['universe_size']} tickers):",
    ]
    for i, r in enumerate(ranked, start=1):
        note = ""
        if i == 1:
            note = "  (most reliable)"
        elif i == len(ranked):
            note = "  (least reliable)"
        lines.append(f"  {i}. {r['horizon']}d  -- mean IC {r['mean_ic']:+.3f}, "
                     f"AUC {r['mean_auc']:.3f}{note}")
    return lines
