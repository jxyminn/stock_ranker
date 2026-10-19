from unittest.mock import patch

import numpy as np
import pytest

from analyze import fmt, fmt_money, resolve_ticker


def test_fmt_none_and_nan_are_na():
    assert fmt(None) == "n/a"
    assert fmt(float("nan")) == "n/a"


def test_fmt_digits_and_suffix():
    assert fmt(3.14159) == "3.14"
    assert fmt(3.14159, 1) == "3.1"
    assert fmt(6.73, 2, "%") == "6.73%"


def test_fmt_money_none_and_nan_are_na():
    assert fmt_money(None) == "n/a"
    assert fmt_money(float("nan")) == "n/a"


def test_fmt_money_magnitude_suffixes():
    assert fmt_money(500) == "$500"
    assert fmt_money(5_000_000) == "$5.00M"
    assert fmt_money(331_980_000_000) == "$331.98B"
    assert fmt_money(4_580_000_000_000) == "$4.58T"


def test_resolve_ticker_exact_symbol_skips_search():
    with patch("analyze.validate_tickers", return_value=["AAPL"]) as mock_validate, \
         patch("analyze.search_ticker") as mock_search:
        result = resolve_ticker("aapl")

    assert result == "AAPL"
    mock_validate.assert_called_once()
    mock_search.assert_not_called()


def test_resolve_ticker_falls_back_to_name_search():
    with patch("analyze.validate_tickers", return_value=[]), \
         patch("analyze.search_ticker", return_value=[
             {"symbol": "PLTR", "name": "Palantir Technologies", "exchange": "NASDAQ", "type": "EQUITY"},
         ]):
        result = resolve_ticker("palantir")

    assert result == "PLTR"


def test_resolve_ticker_returns_none_when_nothing_matches():
    with patch("analyze.validate_tickers", return_value=[]), \
         patch("analyze.search_ticker", return_value=[]):
        result = resolve_ticker("not a real company xyz123")

    assert result is None
