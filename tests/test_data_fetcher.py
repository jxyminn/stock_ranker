from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import data_fetcher
from tests.conftest import make_ohlcv


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own empty cache dir -- no cross-test or real-cache pollution."""
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    return tmp_path


def _yf_history_frame(n_rows: int) -> pd.DataFrame:
    df = make_ohlcv(n=n_rows)
    # fetch_history lowercases columns from yfinance's Title-case originals
    return df.rename(columns=str.title)


def test_fetch_history_cache_key_includes_lookback_days():
    """
    Regression test: a cache entry written for a short (e.g. live, 400-day)
    request must never be served back to a caller asking for a longer (e.g.
    training, 1825-day) window. Previously the cache key was ticker-only, so
    a 400-day fetch silently poisoned a subsequent 1825-day training run.
    """
    call_periods = []

    def fake_history(period, interval, auto_adjust):
        call_periods.append(period)
        n_rows = {"400d": 400, "1825d": 1825}[period]
        return _yf_history_frame(n_rows)

    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = fake_history

    with patch("data_fetcher.yf.Ticker", return_value=mock_ticker):
        df_400 = data_fetcher.fetch_history("FAKE", 400)
        df_1825 = data_fetcher.fetch_history("FAKE", 1825)

    assert len(df_400) == 400
    assert len(df_1825) == 1825
    # both lookback windows must have actually hit the network, not shared a cache slot
    assert call_periods == ["400d", "1825d"]


def test_fetch_history_reuses_cache_for_same_lookback():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _yf_history_frame(400)

    with patch("data_fetcher.yf.Ticker", return_value=mock_ticker) as MockTicker:
        data_fetcher.fetch_history("FAKE", 400)
        data_fetcher.fetch_history("FAKE", 400)

    # second call within the freshness window should be served from disk cache
    assert MockTicker.call_count == 1


def test_fetch_earnings_extracts_surprise_pct():
    raw = pd.DataFrame(
        {
            "EPS Estimate": [1.5, 1.6],
            "Reported EPS": [1.6, 1.5],
            "Surprise(%)": [6.67, -6.25],
        },
        index=pd.DatetimeIndex(["2024-05-01", "2024-02-01"], tz="America/New_York", name="Earnings Date"),
    )
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates.return_value = raw

    with patch("data_fetcher.yf.Ticker", return_value=mock_ticker):
        out = data_fetcher.fetch_earnings("FAKE")

    assert list(out.columns) == ["surprise_pct"]
    assert out.index.is_monotonic_increasing  # sorted ascending
    assert out.loc[pd.Timestamp("2024-02-01", tz="America/New_York"), "surprise_pct"] == -6.25
    assert out.loc[pd.Timestamp("2024-05-01", tz="America/New_York"), "surprise_pct"] == 6.67


def test_fetch_fundamentals_maps_fields_without_rescaling_dividend_yield():
    """
    dividendYield from this yfinance version is already a percent NUMBER
    (e.g. 6.73 means 6.73%), not a 0-1 fraction -- verified by hand against
    real VZ/KO/AAPL data. Locking that in here so a yfinance upgrade that
    changes the convention, or a future refactor, doesn't silently reintroduce
    a 100x scaling bug that would otherwise be easy to miss in review.
    """
    info = {
        "longName": "Fake Corp",
        "sector": "Technology",
        "industry": "Software",
        "marketCap": 123_456_789_000,
        "trailingPE": 25.5,
        "forwardPE": 20.1,
        "dividendYield": 6.73,
        "fiftyTwoWeekLow": 10.0,
        "fiftyTwoWeekHigh": 20.0,
        "targetMeanPrice": 18.0,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 12,
    }
    mock_ticker = MagicMock()
    mock_ticker.info = info

    with patch("data_fetcher.yf.Ticker", return_value=mock_ticker):
        out = data_fetcher.fetch_fundamentals("FAKE")

    assert out["dividend_yield"] == 6.73  # passed through as-is, not / 100 or * 100
    assert out["long_name"] == "Fake Corp"
    assert out["market_cap"] == 123_456_789_000
    assert out["trailing_pe"] == 25.5


def test_fetch_fundamentals_missing_info_returns_empty_dict():
    mock_ticker = MagicMock()
    mock_ticker.info = {}

    with patch("data_fetcher.yf.Ticker", return_value=mock_ticker):
        out = data_fetcher.fetch_fundamentals("FAKE")

    assert out == {}


def test_fetch_news_extracts_fields_and_skips_non_story_content():
    raw = [
        {
            "content": {
                "contentType": "VIDEO",
                "title": "Some video that should be skipped",
                "pubDate": "2024-01-01T00:00:00Z",
                "provider": {"displayName": "Yahoo Finance Video"},
                "canonicalUrl": {"url": "https://example.com/video"},
            }
        },
        {
            "content": {
                "contentType": "STORY",
                "title": "Company beats earnings estimates",
                "pubDate": "2024-01-02T12:00:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.com/story"},
            }
        },
    ]
    mock_ticker = MagicMock()
    mock_ticker.news = raw

    with patch("data_fetcher.yf.Ticker", return_value=mock_ticker):
        articles = data_fetcher.fetch_news("FAKE")

    assert len(articles) == 1  # VIDEO entry skipped
    assert articles[0]["title"] == "Company beats earnings estimates"
    assert articles[0]["publisher"] == "Reuters"
    assert articles[0]["link"] == "https://example.com/story"
    assert articles[0]["published"] == "2024-01-02T12:00:00Z"


def test_fetch_news_respects_max_articles():
    raw = [
        {"content": {"contentType": "STORY", "title": f"Story {i}", "pubDate": "2024-01-01T00:00:00Z",
                      "provider": {}, "canonicalUrl": {}}}
        for i in range(20)
    ]
    mock_ticker = MagicMock()
    mock_ticker.news = raw

    with patch("data_fetcher.yf.Ticker", return_value=mock_ticker):
        articles = data_fetcher.fetch_news("FAKE", max_articles=3)

    assert len(articles) == 3
