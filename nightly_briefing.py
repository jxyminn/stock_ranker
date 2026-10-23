"""
Generate the nightly watchlist briefing: pull fresh data for every ticker in
config.WATCHLIST_SECTIONS, run the already-trained models for a forecast,
and compare against last night's snapshot to flag what actually changed --
big forecast swings, earnings coming up this week, a moving-average
crossover that just happened, a new high, or notably one-sided news.

This does NOT retrain the models -- that's a separate, much slower job
(.github/workflows/weekly-retrain.yml / manual `python train.py`). This
script only refreshes live data and re-runs inference with whatever models
are already saved in models/.

Also maintains briefing/forecast_log.json: every time a forecast move gets
flagged as notable, that prediction is logged and checked again once its
horizon has elapsed, to build an honest running track record of whether
"notable" forecast flags actually panned out -- see the "Track Record"
section on the Daily Briefing page.

Meant to run once a day via the GitHub Actions workflow in
.github/workflows/nightly.yml, which commits briefing/*.json back to the
repo so the deployed Streamlit app (which has no database of its own) can
read it as a static file.

Usage:
    python nightly_briefing.py
"""
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import (
    BRIEFING_DIR, BRIEFING_EARNINGS_WITHIN_DAYS, BRIEFING_FORECAST_MOVE_PP,
    BRIEFING_NEAR_ATH_PCT, BRIEFING_PROB_UP_MOVE_PP, BRIEFING_SENTIMENT_EXTREME,
    FORECAST_LOG_MAX_RESOLVED, FORECAST_LOG_STALE_AFTER_DAYS, HORIZONS, MODELS_DIR,
)
from data_fetcher import fetch_fundamentals, fetch_news, fetch_price_stats
from features import FEATURE_COLUMNS
from predict import latest_features_for_ticker, load_models
from sentiment import summarize_articles
from watchlist_store import get_effective_sections

SNAPSHOT_PATH = BRIEFING_DIR / "snapshot.json"
LATEST_PATH = BRIEFING_DIR / "latest.json"
FORECAST_LOG_PATH = BRIEFING_DIR / "forecast_log.json"


def trading_days_to_calendar_days(trading_days: int) -> int:
    """Rough trading-day -> calendar-day conversion (5-day weeks, +2 day
    cushion for holidays). Good enough for "has this forecast's horizon
    elapsed yet" -- not a precise trading-calendar count, which would need
    extra data just for this bookkeeping. Errs toward resolving slightly
    late rather than early."""
    return round(trading_days * 7 / 5) + 2


def all_watchlist_tickers() -> list[str]:
    seen: dict[str, None] = {}
    for tickers in get_effective_sections().values():
        for t in tickers:
            seen.setdefault(t, None)
    return list(seen.keys())


def ticker_sections_map() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for section, tickers in get_effective_sections().items():
        for t in tickers:
            out.setdefault(t, []).append(section)
    return out


def build_ticker_snapshot(ticker: str, models: dict | None) -> dict | None:
    """Everything needed for tonight's briefing for one ticker, in a
    JSON-serializable shape."""
    feat = latest_features_for_ticker(ticker)
    if feat is None:
        return None

    fund = fetch_fundamentals(ticker)
    stats = fetch_price_stats(ticker)
    news = summarize_articles(fetch_news(ticker, max_articles=5))

    predictions = {}
    if models is not None:
        X = feat[FEATURE_COLUMNS].astype(float).to_frame().T
        for h in HORIZONS:
            reg = models[h]["regressor"]
            clf = models[h]["classifier"]
            predictions[str(h)] = {
                "pred_return_pct": round(float(reg.predict(X)[0]) * 100, 3),
                "prob_up": round(float(clf.predict_proba(X)[0, 1]), 4),
            }

    sma_50, sma_200 = stats.get("sma_50"), stats.get("sma_200")
    golden_cross = bool(sma_50 > sma_200) if sma_50 is not None and sma_200 is not None else None

    days_until = feat.get("days_until_earnings")

    return {
        "name": fund.get("long_name") or ticker,
        "price": round(float(feat["_last_close"]), 4),
        "day_chg_pct": round(float(feat["ret_1d"]) * 100, 3) if pd.notna(feat.get("ret_1d")) else None,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "golden_cross": golden_cross,
        "all_time_high": stats.get("all_time_high"),
        "days_until_earnings": float(days_until) if pd.notna(days_until) else None,
        "news_sentiment": news.get("mean_score"),
        "predictions": predictions,
    }


def pct_from_ath(price: float, ath: float | None) -> float | None:
    if not ath:
        return None
    return (1 - price / ath) * 100


def diff_ticker(ticker: str, name: str, today: dict, yesterday: dict | None) -> dict:
    """Compare today's snapshot to yesterday's (if any) and return the
    notable-event flags for this ticker. (The accuracy tracker logs every
    prediction independently of these flags -- see update_forecast_log --
    so a stock not showing up here can still be contributing to the Track
    Record.)"""
    notes: list[str] = []

    if yesterday and today.get("predictions") and yesterday.get("predictions"):
        for h in HORIZONS:
            t_pred = today["predictions"].get(str(h))
            y_pred = yesterday["predictions"].get(str(h))
            if not t_pred or not y_pred:
                continue
            d_ret = t_pred["pred_return_pct"] - y_pred["pred_return_pct"]
            d_prob = (t_pred["prob_up"] - y_pred["prob_up"]) * 100
            if abs(d_ret) >= BRIEFING_FORECAST_MOVE_PP:
                notes.append(f"{h}d forecast moved {d_ret:+.1f}pp overnight "
                             f"(now {t_pred['pred_return_pct']:+.1f}%)")
            if abs(d_prob) >= BRIEFING_PROB_UP_MOVE_PP:
                notes.append(f"{h}d chance-of-gain moved {d_prob:+.0f}pp overnight "
                             f"(now {t_pred['prob_up'] * 100:.0f}%)")

    if yesterday and today.get("golden_cross") is not None and yesterday.get("golden_cross") is not None:
        if today["golden_cross"] != yesterday["golden_cross"]:
            notes.append("golden cross (bullish)" if today["golden_cross"] else "death cross (bearish)")

    days_until = today.get("days_until_earnings")
    if days_until is not None and days_until <= BRIEFING_EARNINGS_WITHIN_DAYS:
        notes.append(f"earnings in {days_until:.0f} day{'s' if days_until != 1 else ''}")

    dist = pct_from_ath(today["price"], today.get("all_time_high"))
    if dist is not None and 0 <= dist <= BRIEFING_NEAR_ATH_PCT:
        notes.append("within " + f"{dist:.1f}% of its all-time high")

    sentiment = today.get("news_sentiment")
    if sentiment is not None and abs(sentiment) >= BRIEFING_SENTIMENT_EXTREME:
        notes.append(f"notably {'positive' if sentiment > 0 else 'negative'} headlines "
                     f"(score {sentiment:+.2f})")

    diff = {"ticker": ticker, "name": name, "price": today["price"],
            "day_chg_pct": today.get("day_chg_pct"), "notes": notes}
    return diff


def update_forecast_log(today_snapshot: dict[str, dict], today: datetime) -> dict:
    """
    Resolve any pending predictions whose horizon has elapsed (using today's
    price for that ticker), then seed a fresh entry for every (ticker,
    horizon) that doesn't already have one pending.

    Deliberately tracks EVERY prediction, not just ones flagged as a notable
    overnight move -- only logging the "notable" subset would bias the
    sample toward stocks that were already moving a lot, which isn't a fair
    read of the model's actual accuracy. A given (ticker, horizon) only gets
    a new entry once its previous one resolves, so "pending" stays capped at
    roughly one per ticker per horizon rather than growing every night.

    Mutates nothing on disk itself -- caller writes the returned log.
    """
    log = {"pending": [], "resolved": []}
    if FORECAST_LOG_PATH.exists():
        try:
            log = json.loads(FORECAST_LOG_PATH.read_text())
        except Exception as e:
            print(f"Warning: couldn't read forecast log ({e}); starting fresh.")

    still_pending = []
    newly_resolved = []
    in_flight: set[tuple[str, int]] = set()
    for entry in log.get("pending", []):
        logged_at_date = datetime.fromisoformat(entry["logged_at"]).date()
        calendar_days_elapsed = (today.date() - logged_at_date).days
        needed = trading_days_to_calendar_days(entry["horizon"])
        current = today_snapshot.get(entry["ticker"])

        if calendar_days_elapsed >= needed and current is not None:
            actual_return_pct = (current["price"] / entry["price_at_log"] - 1) * 100
            newly_resolved.append({
                **entry,
                "resolved_at": today.date().isoformat(),
                "price_at_resolution": current["price"],
                "actual_return_pct": round(actual_return_pct, 3),
                "direction_correct": (actual_return_pct > 0) == (entry["pred_return_pct"] > 0),
                "classifier_correct": (actual_return_pct > 0) == (entry["prob_up"] > 0.5),
            })
        elif calendar_days_elapsed >= needed + FORECAST_LOG_STALE_AFTER_DAYS:
            pass  # gave up waiting (ticker likely dropped out of the watchlist) -- just drop it
        else:
            still_pending.append(entry)
            in_flight.add((entry["ticker"], entry["horizon"]))

    for ticker, info in today_snapshot.items():
        for h_str, pred in info.get("predictions", {}).items():
            h = int(h_str)
            if (ticker, h) in in_flight:
                continue
            still_pending.append({
                "ticker": ticker, "horizon": h, "price_at_log": info["price"],
                "pred_return_pct": pred["pred_return_pct"], "prob_up": pred["prob_up"],
                "logged_at": today.date().isoformat(),
            })

    resolved = (log.get("resolved", []) + newly_resolved)[-FORECAST_LOG_MAX_RESOLVED:]
    log = {"pending": still_pending, "resolved": resolved}
    FORECAST_LOG_PATH.write_text(json.dumps(log, indent=2))

    scorecard = {}
    for h in HORIZONS:
        rows = [r for r in resolved if r["horizon"] == h]
        if not rows:
            continue
        n = len(rows)
        scorecard[str(h)] = {
            "n_resolved": n,
            "direction_accuracy": round(sum(r["direction_correct"] for r in rows) / n, 3),
            "classifier_accuracy": round(sum(r["classifier_correct"] for r in rows) / n, 3),
            "mean_abs_error_pp": round(sum(abs(r["actual_return_pct"] - r["pred_return_pct"]) for r in rows) / n, 2),
            "mean_predicted_pct": round(sum(r["pred_return_pct"] for r in rows) / n, 2),
            "mean_actual_pct": round(sum(r["actual_return_pct"] for r in rows) / n, 2),
        }

    return {"scorecard": scorecard, "n_pending": len(still_pending),
            "n_newly_resolved": len(newly_resolved)}


def main():
    models, missing_horizons = (None, HORIZONS)
    if all((MODELS_DIR / f"horizon_{h}.joblib").exists() for h in HORIZONS):
        models = load_models()
        missing_horizons = []

    yesterday_snapshot = {}
    if SNAPSHOT_PATH.exists():
        try:
            yesterday_snapshot = json.loads(SNAPSHOT_PATH.read_text())
        except Exception as e:
            print(f"Warning: couldn't read previous snapshot ({e}); treating this as the first run.")

    sections_map = ticker_sections_map()
    tickers = all_watchlist_tickers()
    print(f"Building tonight's snapshot for {len(tickers)} watchlist tickers...")

    today_snapshot: dict[str, dict] = {}
    skipped: list[str] = []
    for i, t in enumerate(tickers, 1):
        info = build_ticker_snapshot(t, models)
        if info is None:
            skipped.append(t)
        else:
            today_snapshot[t] = info
        if i % 25 == 0:
            print(f"  ...{i}/{len(tickers)}")

    print(f"Got data for {len(today_snapshot)}/{len(tickers)} tickers "
          f"({len(skipped)} skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''})")

    notable = []
    for t, info in today_snapshot.items():
        diff = diff_ticker(t, info["name"], info, yesterday_snapshot.get(t))
        if diff["notes"]:
            notable.append(diff)

    now = datetime.now(timezone.utc)
    tracker_result = update_forecast_log(today_snapshot, now)

    briefing = {
        "generated_at": now.isoformat(),
        "tickers_covered": len(today_snapshot),
        "tickers_skipped": skipped,
        "models_available": not missing_horizons,
        "is_first_run": not bool(yesterday_snapshot),
        "notable": sorted(notable, key=lambda d: len(d["notes"]), reverse=True),
        "track_record": tracker_result["scorecard"],
        "track_record_pending": tracker_result["n_pending"],
    }

    BRIEFING_DIR.mkdir(exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(today_snapshot, indent=2, default=str))
    LATEST_PATH.write_text(json.dumps(briefing, indent=2, default=str))

    print(f"\n{len(notable)} tickers have something notable tonight.")
    print(f"Forecast tracker: {tracker_result['n_newly_resolved']} newly resolved, "
          f"{tracker_result['n_pending']} still pending.")
    print(f"Wrote {SNAPSHOT_PATH}, {LATEST_PATH}, and {FORECAST_LOG_PATH}.")


if __name__ == "__main__":
    main()
