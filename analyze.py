"""
Single-stock deep-dive report: everything the ranker knows about one ticker,
organized to help answer "should I invest in this."

Combines:
  - the trained models' raw predictions per horizon (predicted return,
    probability of a positive move, risk-adjusted ratio), plus each model's
    own walk-forward reliability (holdout MAE/AUC from train.py) so
    predictions aren't read with more confidence than they've earned
  - a plain-language read of the technical indicators driving those
    predictions (trend, momentum, volatility, volume)
  - earnings timing (last surprise, days until next report -- flagged if
    it's imminent, since that's a higher-risk window for a price move)
  - market regime context (S&P/VIX) the model also sees
  - company fundamentals (P/E, market cap, sector, analyst targets) --
    informational only. The trained models never see this; it's pulled
    fresh here purely for the human reading the report.
  - recent headlines with sentiment scoring -- also informational only.
    yfinance's news endpoint has no historical archive, so unlike every
    other feature here there's no way to backfill this into training data
    without lookahead; it's shown live for human context, never as a
    model input (see sentiment.py).

This is a decision-support report, not a recommendation engine -- see the
caveats printed at the bottom of every report, and README.md.

Usage:
    python analyze.py AAPL
    python analyze.py "palantir"
"""
import argparse
import sys

import numpy as np
import pandas as pd

from config import HORIZONS, LIVE_LOOKBACK_DAYS
from data_fetcher import fetch_fundamentals, fetch_news, fetch_history, fetch_earnings, search_ticker, validate_tickers
from features import FEATURE_COLUMNS
from predict import load_models, latest_features_for_ticker
from sentiment import summarize_articles
from reliability import load_backtest_results, format_ranking_lines


def resolve_ticker(query: str) -> str | None:
    """Accept an exact ticker or a company name; resolve names via search."""
    candidate = query.strip().upper()
    if validate_tickers([candidate]):
        return candidate
    matches = search_ticker(query)
    equity_matches = [m for m in matches if m.get("type", "").upper() in ("EQUITY", "")] or matches
    if not equity_matches:
        return None
    best = equity_matches[0]
    print(f"Interpreting '{query}' as {best['symbol']} ({best['name']})")
    return best["symbol"]


def build_price_history(ticker: str, lookback_days: int = LIVE_LOOKBACK_DAYS) -> pd.DataFrame | None:
    """
    OHLCV plus a few chart-friendly technical overlays, in raw price-level
    terms (not the ratio-based features the model uses -- those are built
    for a tree model, not for plotting). Used only for the Deep Dive chart.
    """
    df = fetch_history(ticker, lookback_days, use_cache=False)
    if df is None or df.empty:
        return None
    out = df.copy()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    bb_mid = out["close"].rolling(20).mean()
    bb_std = out["close"].rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    return out


def build_report(ticker: str) -> dict:
    """Gather everything needed for a single-stock deep dive."""
    feat_row = latest_features_for_ticker(ticker)
    if feat_row is None:
        return {"ticker": ticker, "error": "insufficient live data for this ticker"}

    models = load_models()
    X = feat_row[FEATURE_COLUMNS].astype(float).to_frame().T

    predictions = {}
    for h in HORIZONS:
        reg = models[h]["regressor"]
        clf = models[h]["classifier"]
        pred_return = float(reg.predict(X)[0])
        prob_up = float(clf.predict_proba(X)[0, 1])
        vol = max(float(feat_row["_vol_20d"]), 1e-4)
        predictions[h] = {
            "pred_return_pct": pred_return * 100,
            "prob_up": prob_up,
            "risk_adjusted": pred_return / vol,
            "holdout_mae": models[h]["holdout_mae"],
            "holdout_auc": models[h]["holdout_auc"],
        }

    return {
        "ticker": ticker,
        "last_close": feat_row["_last_close"],
        "as_of": feat_row["_asof_date"],
        "features": feat_row,
        "predictions": predictions,
        "fundamentals": fetch_fundamentals(ticker),
        "news": summarize_articles(fetch_news(ticker)),
        "backtest": load_backtest_results(),
        "price_history": build_price_history(ticker),
        "earnings_history": fetch_earnings(ticker),
    }


def fmt_money(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if x >= cutoff:
            return f"${x / cutoff:.2f}{suffix}"
    return f"${x:,.0f}"


def fmt(x, digits=2, suffix="") -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}{suffix}"


def _rsi_read(rsi: float) -> str:
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    return "neutral"


def _trend_read(feat: pd.Series) -> str:
    above_20 = feat["sma_20_ratio"] > 0
    above_50 = feat["sma_50_ratio"] > 0
    if above_20 and above_50:
        return "uptrend (price above both 20d and 50d averages)"
    if not above_20 and not above_50:
        return "downtrend (price below both 20d and 50d averages)"
    return "mixed / sideways (20d and 50d averages disagree)"


def _vix_read(vix: float) -> str:
    if np.isnan(vix):
        return "n/a"
    if vix < 15:
        return "low -- market complacent"
    if vix < 25:
        return "normal"
    return "elevated -- market fearful"


def print_report(report: dict) -> None:
    if "error" in report:
        print(f"Could not build a report for {report['ticker']}: {report['error']}")
        return

    ticker = report["ticker"]
    feat = report["features"]
    fund = report["fundamentals"]
    name = fund.get("long_name") or ticker

    print(f"\n{'=' * 70}")
    print(f"{name} ({ticker})  --  as of {pd.Timestamp(report['as_of']).date()}")
    print(f"{'=' * 70}")
    print(f"Last close: ${report['last_close']:.2f}")

    # ---- Model predictions ----------------------------------------------
    print("\n--- Model predictions (technical + earnings-timing + macro only) ---")
    print(f"{'Horizon':<10}{'Pred. return':<15}{'Prob. up':<12}{'Risk-adj.':<12}{'Model reliability (single holdout split)'}")
    for h in HORIZONS:
        p = report["predictions"][h]
        reliability = f"MAE={p['holdout_mae']:.4f}  AUC={p['holdout_auc']:.3f}"
        print(f"{h}d{'':<8}{p['pred_return_pct']:+.2f}%{'':<9}{p['prob_up']:.1%}{'':<6}"
              f"{p['risk_adjusted']:+.3f}{'':<7}{reliability}")
    print("(AUC 0.5 = no better than a coin flip. Run backtest.py for a fuller, multi-fold")
    print(" reliability check -- a single holdout split like this is a noisy estimate.)")
    for line in format_ranking_lines(report.get("backtest")):
        print(f"  {line}")

    # ---- Technical read ----------------------------------------------------
    print("\n--- Technical read ---")
    print(f"  Trend:      {_trend_read(feat)}")
    print(f"  Momentum:   RSI(14)={feat['rsi_14']:.1f} ({_rsi_read(feat['rsi_14'])}); "
          f"MACD histogram {'positive (bullish)' if feat['macd_hist'] > 0 else 'negative (bearish)'}")
    print(f"  Volatility: {feat['vol_20d'] * (252 ** 0.5) * 100:.1f}% annualized (20d); "
          f"Bollinger %B={feat['bb_pct_b']:.2f} "
          f"({'near/above upper band' if feat['bb_pct_b'] > 0.8 else 'near/below lower band' if feat['bb_pct_b'] < 0.2 else 'mid-band'})")
    print(f"  Volume:     {feat['volume_ratio_20d']:.2f}x the 20d average "
          f"({'well above' if feat['volume_ratio_20d'] > 1.5 else 'below' if feat['volume_ratio_20d'] < 0.7 else 'in line with'} normal)")
    print(f"  52-week:    {feat['dist_from_52w_high'] * 100:+.1f}% from 52w high, "
          f"{feat['dist_from_52w_low'] * 100:+.1f}% above 52w low")

    # ---- Earnings ------------------------------------------------------
    print("\n--- Earnings ---")
    days_until = feat.get("days_until_earnings")
    days_since = feat.get("days_since_earnings")
    surprise = feat.get("last_earnings_surprise_pct")
    if pd.notna(days_since):
        print(f"  Last report: {days_since:.0f} days ago, surprise {surprise:+.2f}%" if pd.notna(surprise)
              else f"  Last report: {days_since:.0f} days ago")
    else:
        print("  Last report: no earnings history on record")
    if pd.notna(days_until):
        flag = "  <-- within 7 days: expect higher volatility than usual" if days_until <= 7 else ""
        print(f"  Next report: in {days_until:.0f} days{flag}")
    else:
        print("  Next report: date unknown")

    # ---- Market regime ---------------------------------------------------
    print("\n--- Market regime (same context the model sees) ---")
    vix = feat.get("vix_level", np.nan)
    print(f"  S&P 500: {feat.get('mkt_ret_5d', np.nan) * 100:+.2f}% (5d), "
          f"{feat.get('mkt_ret_20d', np.nan) * 100:+.2f}% (20d)")
    print(f"  VIX: {fmt(vix, 1)} ({_vix_read(vix)})")

    # ---- Fundamentals (not seen by the model) -----------------------------
    print("\n--- Fundamentals (informational only -- not seen by the model) ---")
    if not fund:
        print("  unavailable")
    else:
        print(f"  Sector / industry: {fund.get('sector', 'n/a')} / {fund.get('industry', 'n/a')}")
        print(f"  Market cap: {fmt_money(fund.get('market_cap'))}")
        print(f"  P/E: trailing {fmt(fund.get('trailing_pe'))}, forward {fmt(fund.get('forward_pe'))}; "
              f"P/B {fmt(fund.get('price_to_book'))}; beta {fmt(fund.get('beta'))}")
        div = fund.get("dividend_yield")
        print(f"  Dividend yield: {fmt(div, 2, '%') if div else 'none'}")
        lo, hi = fund.get("fifty_two_week_low"), fund.get("fifty_two_week_high")
        if lo and hi:
            pos = (report["last_close"] - lo) / (hi - lo) * 100
            print(f"  52-week range: ${lo:.2f} - ${hi:.2f}  (current price is {pos:.0f}% up that range)")
        target = fund.get("target_mean_price")
        if target:
            upside = (target / report["last_close"] - 1) * 100
            n_analysts = fund.get("num_analyst_opinions")
            print(f"  Analyst mean target: ${target:.2f} ({upside:+.1f}% vs. current), "
                  f"consensus '{fund.get('recommendation_key', 'n/a')}'"
                  + (f" ({n_analysts} analysts)" if n_analysts else ""))

    # ---- Recent headlines (not seen by the model) -----------------------
    print("\n--- Recent headlines (informational only -- not seen by the model) ---")
    news = report.get("news", {})
    articles = news.get("articles", [])
    if not articles:
        print("  no recent headlines found")
    else:
        mean_score = news["mean_score"]
        print(f"  Overall tone across {len(articles)} headlines: {news['overall_label']} "
              f"(avg score {mean_score:+.2f}, -1=very negative, +1=very positive)")
        print("  Note: yfinance's news feed mixes stock-specific stories with broader market")
        print("  news, and sentiment is scored automatically -- read headlines yourself too.")
        for a in articles[:5]:
            print(f"    [{a['sentiment_label']:<8} {a['sentiment_score']:+.2f}] {a['title']} ({a['publisher']})")

    print(f"\n{'-' * 70}")
    print("Not financial advice. Models see technical/earnings-timing/macro data only --")
    print("no news or fundamentals-depth is part of the prediction, only this report's")
    print("Fundamentals and Recent headlines sections. Stock prices are close to a random")
    print("walk over short horizons; treat this as one input among several, not a verdict.")
    print(f"{'-' * 70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Ticker symbol or company name, e.g. AAPL or 'palantir'")
    args = parser.parse_args()

    ticker = resolve_ticker(args.query)
    if ticker is None:
        sys.exit(f"No matching ticker found for '{args.query}'.")

    report = build_report(ticker)
    print_report(report)


if __name__ == "__main__":
    main()
