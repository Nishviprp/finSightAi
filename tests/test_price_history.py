"""
Tests for src.price_history.

Uses a real live call to Yahoo Finance for AAPL, consistent with how this
project tests other thin yfinance wrappers (test_financial_metrics.py,
test_yahoo_scraper.py's index/crypto tests).
"""
from unittest.mock import patch

import pytest

from src.price_history import get_price_history

AAPL = "AAPL"
INVALID_SYMBOL = "ZZZZZZINVALID"


def _boom(*_args, **_kwargs):
    raise ConnectionError("simulated network down")


class TestGetPriceHistoryRealSymbol:
    def test_returns_list_for_real_symbol(self):
        rows = get_price_history(AAPL)
        assert isinstance(rows, list)

    def test_returns_thirty_rows_by_default(self):
        rows = get_price_history(AAPL)
        assert len(rows) == 30

    def test_row_schema(self):
        rows = get_price_history(AAPL)
        for row in rows:
            assert set(row.keys()) == {"date", "close_price"}
            assert isinstance(row["date"], str)
            assert isinstance(row["close_price"], float)
            assert row["close_price"] > 0

    def test_dates_are_sorted_oldest_first(self):
        rows = get_price_history(AAPL)
        dates = [r["date"] for r in rows]
        assert dates == sorted(dates)

    def test_dates_are_iso_format(self):
        rows = get_price_history(AAPL)
        for row in rows:
            year, month, day = row["date"].split("-")
            assert len(year) == 4 and len(month) == 2 and len(day) == 2

    def test_custom_days_parameter_is_honored(self):
        rows = get_price_history(AAPL, days=7)
        assert len(rows) == 7


class TestGetPriceHistoryErrorHandling:
    def test_invalid_symbol_returns_none(self):
        assert get_price_history(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        get_price_history.clear()
        import src.price_history as ph
        monkeypatch.setattr(ph.yf, "Ticker", _boom)
        assert get_price_history("NETDOWN") is None


class TestGetPriceHistoryCaching:
    def test_two_calls_with_same_symbol_hit_yahoo_once(self):
        get_price_history.clear()
        with patch("src.price_history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = _fake_history_df()
            get_price_history(AAPL)
            get_price_history(AAPL)
        assert mock_ticker.call_count == 1

    def test_different_symbols_each_hit_yahoo(self):
        get_price_history.clear()
        with patch("src.price_history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = _fake_history_df()
            get_price_history("AAPL")
            get_price_history("MSFT")
        assert mock_ticker.call_count == 2

    def test_different_days_value_hits_yahoo_again(self):
        get_price_history.clear()
        with patch("src.price_history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = _fake_history_df()
            get_price_history(AAPL, days=30)
            get_price_history(AAPL, days=7)
        assert mock_ticker.call_count == 2


def _fake_history_df():
    import pandas as pd

    dates = pd.date_range("2026-06-01", periods=5, freq="D")
    return pd.DataFrame({"Close": [100.0, 101.0, 102.0, 101.5, 103.0]}, index=dates)
