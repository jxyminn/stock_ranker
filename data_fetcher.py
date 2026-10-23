"""
Data access layer. All live/historical price data and ticker search goes through here.
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf
from config import CACHE_DIR


def search_ticker(query: str, max_results: int = 8) -> list[dict]:
    """
    Search for a ticker by company name or partial symbol.
    Returns a list of {symbol, name, exchange} dicts.
    """
    try:
        results = yf.Search(query, max_results=max_results)
        out = []
        for quote in results.quotes:
            out.append({
                "symbol": quote.get("symbol"),
                "name": quote.get("shortname") or quote.get("longname") or "",
                "exchange": quote.get("exchange", ""),
                "type": quote.get("quoteType", ""),
            })
        return out
    except Exception as e:
        print(f"[search_ticker] search failed for '{query}': {e}")
        return []


def validate_tickers(tickers: list[str]) -> list[str]:
    """Filter a list of tickers down to ones that actually return data."""
    valid = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="5d")
            if not hist.empty:
                valid.append(t.upper())
            else:
                print(f"[validate_tickers] '{t}' returned no data, skipping.")
        except Exception as e:
            print(f"[validate_tickers] '{t}' failed: {e}, skipping.")
    return valid


def fetch_history(ticker: str, lookback_days: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch daily OHLCV history for a single ticker.
    Cached to disk per-ticker for the session to avoid hammering the API when
    training on a large universe.
    """
    # Cache key includes lookback_days: a cache entry written for a short
    # (e.g. live, 400-day) request must never be served back to a caller
    # asking for a longer (e.g. training, 1825-day) window, or vice versa.
    cache_file = CACHE_DIR / f"{ticker.replace('/', '_')}_{lookback_days}d.parquet"
    if use_cache and cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        if age_seconds < 3600:  # 1 hour freshness for cache during a training run
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

    period_str = f"{lookback_days}d" if lookback_days <= 3650 else "max"
    df = yf.Ticker(ticker).history(period=period_str, interval="1d", auto_adjust=True)
    if df.empty:
        return df

    df = df.rename(columns=str.lower)
    df.index.name = "date"

    try:
        df.to_parquet(cache_file)
    except Exception:
        pass  # caching is best-effort, not critical

    return df


def fetch_earnings(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch a ticker's earnings report history: past actuals plus the next known
    (estimated) report date. Returns a DataFrame indexed by report date with a
    'surprise_pct' column (actual vs. estimated EPS, %; NaN for not-yet-reported
    dates). Cached to disk like price history -- earnings data doesn't change
    intraday, so a 1-hour cache is generous, not stale.
    """
    cache_file = CACHE_DIR / f"{ticker.replace('/', '_')}_earnings.parquet"
    if use_cache and cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        if age_seconds < 3600:
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=40)
    except Exception as e:
        print(f"[fetch_earnings] '{ticker}' failed: {e}, skipping.")
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)
    out.index.name = "report_date"
    out["surprise_pct"] = raw["Surprise(%)"] if "Surprise(%)" in raw.columns else np.nan
    out = out.sort_index()

    try:
        out.to_parquet(cache_file)
    except Exception:
        pass  # caching is best-effort, not critical

    return out


FUNDAMENTALS_FIELDS = {
    "long_name": "longName",
    "quote_type": "quoteType",  # "EQUITY" / "ETF" / etc. -- the reliable signal, vs. inferring
                                # from missing sector/industry (which also happens on a failed fetch)
    "sector": "sector",
    "industry": "industry",
    "market_cap": "marketCap",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "price_to_book": "priceToBook",
    "dividend_yield": "dividendYield",
    "beta": "beta",
    "fifty_two_week_low": "fiftyTwoWeekLow",
    "fifty_two_week_high": "fiftyTwoWeekHigh",
    "target_mean_price": "targetMeanPrice",
    "target_high_price": "targetHighPrice",
    "target_low_price": "targetLowPrice",
    "recommendation_key": "recommendationKey",
    "num_analyst_opinions": "numberOfAnalystOpinions",
}


def fetch_fundamentals(ticker: str, use_cache: bool = True) -> dict:
    """
    Fetch company fundamentals/valuation snapshot (sector, P/E, market cap,
    dividend yield, 52-week range, analyst targets, etc.) via yfinance's
    Ticker.info. This is informational context for a human decision, not a
    model input -- the trained models only ever see technical/earnings-
    timing/macro features (see features.py).
    """
    cache_file = CACHE_DIR / f"{ticker.replace('/', '_')}_fundamentals.parquet"
    if use_cache and cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        if age_seconds < 3600:
            try:
                return pd.read_parquet(cache_file).iloc[0].to_dict()
            except Exception:
                pass

    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"[fetch_fundamentals] '{ticker}' failed: {e}, skipping.")
        return {}

    if not info:
        return {}

    out = {key: info.get(src_key) for key, src_key in FUNDAMENTALS_FIELDS.items()}

    try:
        pd.DataFrame([out]).to_parquet(cache_file)
    except Exception:
        pass  # caching is best-effort, not critical

    return out


def fetch_news(ticker: str, max_articles: int = 8, use_cache: bool = True) -> list[dict]:
    """
    Fetch recent headlines for a ticker via yfinance's Ticker.news. Returns a
    list of {title, publisher, link, published} dicts, most recent first.

    This is report-only data (see sentiment.py) -- yfinance's `.news` only
    returns the current ~10 recent articles with no historical archive, so
    unlike price/earnings/macro data there's no way to backfill this into
    training data without lookahead. It's shown live in analyze.py for human
    context; the trained models never see it.
    """
    cache_file = CACHE_DIR / f"{ticker.replace('/', '_')}_news.parquet"
    if use_cache and cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        if age_seconds < 1800:  # news moves faster than earnings/fundamentals -- 30min cache
            try:
                return pd.read_parquet(cache_file).to_dict("records")
            except Exception:
                pass

    try:
        raw = yf.Ticker(ticker).news
    except Exception as e:
        print(f"[fetch_news] '{ticker}' failed: {e}, skipping.")
        return []

    articles = []
    for item in raw or []:
        content = item.get("content", {})
        if content.get("contentType") != "STORY":  # skip videos/other non-article content
            continue
        articles.append({
            "title": content.get("title", ""),
            "publisher": (content.get("provider") or {}).get("displayName", ""),
            "link": (content.get("canonicalUrl") or {}).get("url", ""),
            "published": content.get("pubDate"),
        })
        if len(articles) >= max_articles:
            break

    try:
        pd.DataFrame(articles).to_parquet(cache_file)
    except Exception:
        pass  # caching is best-effort, not critical

    return articles


def fetch_insider_transactions(ticker: str, max_rows: int = 15, use_cache: bool = True) -> list[dict]:
    """
    Recent insider buy/sell activity (SEC Form 4 filings), via yfinance's
    aggregated insider_transactions table. Informational only -- like
    fundamentals and news, never seen by the trained models: filing timing
    is irregular and there's no clean historical archive suitable for
    leak-free backtesting.
    """
    cache_file = CACHE_DIR / f"{ticker.replace('/', '_')}_insiders.parquet"
    if use_cache and cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        if age_seconds < 21600:  # 6h -- filings trickle in slowly, no need to refetch every call
            try:
                return pd.read_parquet(cache_file).to_dict("records")
            except Exception:
                pass

    try:
        df = yf.Ticker(ticker).insider_transactions
    except Exception as e:
        print(f"[fetch_insider_transactions] '{ticker}' failed: {e}, skipping.")
        return []
    if df is None or df.empty:
        return []

    df = df.head(max_rows).copy()

    def classify(text) -> str:
        t = (text or "").lower()
        if "sale" in t:
            return "Sale"
        if "purchase" in t or "buy" in t:
            return "Purchase"
        return "Other"

    # yfinance's own "Transaction" column is often blank; the actual
    # description (and thus buy/sell/other classification) lives in "Text".
    df["kind"] = df["Text"].map(classify)
    df["Start Date"] = df["Start Date"].astype(str)

    out = df[["Start Date", "Insider", "Position", "kind", "Shares", "Value", "Text"]].to_dict("records")
    try:
        pd.DataFrame(out).to_parquet(cache_file)
    except Exception:
        pass  # caching is best-effort, not critical
    return out


def fetch_price_stats(ticker: str, use_cache: bool = True) -> dict:
    """
    Long-horizon price context that the 400-day live lookback can't give:
    50-day and 200-day simple moving averages (absolute $ levels, for
    display -- not the ratio features the model uses) and the all-time high
    close with how far the current price sits below it. Pulls the full
    available history (yfinance period="max"), so it's cached for 24h rather
    than the 1h used elsewhere -- these move slowly day to day and a "max"
    history pull is the heaviest request this module makes.
    """
    cache_file = CACHE_DIR / f"{ticker.replace('/', '_')}_price_stats.parquet"
    if use_cache and cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        if age_seconds < 86400:
            try:
                return pd.read_parquet(cache_file).iloc[0].to_dict()
            except Exception:
                pass

    try:
        hist = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    except Exception as e:
        print(f"[fetch_price_stats] '{ticker}' failed: {e}, skipping.")
        return {}
    if hist.empty:
        return {}

    close = hist["Close"]
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    ath_date = close.idxmax()

    out = {
        "sma_50": float(sma_50) if sma_50 is not None and pd.notna(sma_50) else None,
        "sma_200": float(sma_200) if sma_200 is not None and pd.notna(sma_200) else None,
        "all_time_high": float(close.loc[ath_date]),
        "all_time_high_date": str(pd.Timestamp(ath_date).date()),
    }

    try:
        pd.DataFrame([out]).to_parquet(cache_file)
    except Exception:
        pass  # caching is best-effort, not critical

    return out


def fetch_history_batch(tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Fetch history for many tickers, skipping ones with no usable data."""
    data = {}
    for i, t in enumerate(tickers):
        df = fetch_history(t, lookback_days)
        if df is not None and len(df) > 60:  # need enough rows for features + labels
            data[t] = df
        else:
            print(f"[fetch_history_batch] skipping {t}: insufficient data")
        if (i + 1) % 10 == 0:
            print(f"  fetched {i + 1}/{len(tickers)} tickers...")
    return data
