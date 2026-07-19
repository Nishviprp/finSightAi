"""
Tests for src.sector_analytics.

Uses real live calls to Yahoo Finance for all 11 sector ETFs plus a real
constituent-screener query, consistent with how this project tests other
yfinance-backed modules. Verifies the corrected XLC ticker (the task's
"XLCO" doesn't resolve on Yahoo Finance — confirmed live) and the sector
name mapping between GICS (this module's public API) and Yahoo's
Morningstar-based screener taxonomy.
"""
import pytest

from src import sector_analytics as sa

EXPECTED_SECTORS = {
    "Technology", "Healthcare", "Financials", "Energy", "Industrials",
    "Consumer Discretionary", "Consumer Staples", "Materials",
    "Real Estate", "Utilities", "Communication Services",
}


def _boom(*_args, **_kwargs):
    raise ConnectionError("simulated network down")


@pytest.fixture(scope="module")
def performance():
    return sa.get_sector_performance()


class TestSectorEtfMapping:
    def test_all_eleven_sectors_present(self):
        assert set(sa.SECTOR_ETFS.keys()) == EXPECTED_SECTORS

    def test_communication_services_uses_xlc_not_xlco(self):
        # the task's spec said XLCO; verified live that XLCO returns no
        # data on Yahoo Finance and XLC is the real SPDR ETF ticker.
        assert sa.SECTOR_ETFS["Communication Services"] == "XLC"

    def test_screener_sector_names_cover_all_eleven_sectors(self):
        assert set(sa._SCREENER_SECTOR_NAMES.keys()) == EXPECTED_SECTORS

    def test_screener_sector_names_are_real_yahoo_taxonomy_values(self):
        # Live-verified against EquityQuery's own valid_values for "sector"
        # before this map was written; guard against silent taxonomy drift.
        from yfinance.screener.query import EquityQuery

        valid = EquityQuery("gt", ["percentchange", 0]).valid_values["sector"]
        for screener_name in sa._SCREENER_SECTOR_NAMES.values():
            assert screener_name in valid


class TestGetSectorPerformanceRealData:
    def test_returns_dict_with_sectors_key(self, performance):
        assert isinstance(performance, dict)
        assert "sectors" in performance

    def test_all_eleven_sectors_resolved(self, performance):
        assert set(performance["sectors"].keys()) == EXPECTED_SECTORS

    def test_each_sector_has_expected_keys(self, performance):
        expected_keys = {
            "symbol", "name", "price", "change_percent",
            "change_30day", "top_stock", "momentum",
        }
        for row in performance["sectors"].values():
            assert expected_keys.issubset(row.keys())

    def test_each_sector_symbol_matches_the_etf_map(self, performance):
        for name, row in performance["sectors"].items():
            assert row["symbol"] == sa.SECTOR_ETFS[name]

    def test_prices_are_positive(self, performance):
        for row in performance["sectors"].values():
            assert row["price"] > 0

    def test_momentum_is_one_of_the_four_valid_values(self, performance):
        for row in performance["sectors"].values():
            assert row["momentum"] in {"strong", "bullish", "neutral", "bearish"}

    def test_momentum_thresholds_match_change_30day(self, performance):
        for row in performance["sectors"].values():
            change = row["change_30day"]
            momentum = row["momentum"]
            if change is None:
                assert momentum == "neutral"
            elif change >= sa.MOMENTUM_STRONG_THRESHOLD:
                assert momentum == "strong"
            elif change <= sa.MOMENTUM_BEARISH_THRESHOLD:
                assert momentum == "bearish"
            elif change > 0:
                assert momentum == "bullish"
            else:
                assert momentum == "neutral"


class TestMomentumClassificationIsDeterministic:
    def test_strong_boundary(self):
        assert sa._classify_momentum(5.0) == "strong"
        assert sa._classify_momentum(20.0) == "strong"

    def test_bearish_boundary(self):
        assert sa._classify_momentum(-5.0) == "bearish"
        assert sa._classify_momentum(-20.0) == "bearish"

    def test_bullish_range(self):
        assert sa._classify_momentum(0.01) == "bullish"
        assert sa._classify_momentum(4.99) == "bullish"

    def test_neutral_range(self):
        assert sa._classify_momentum(0.0) == "neutral"
        assert sa._classify_momentum(-4.99) == "neutral"

    def test_none_is_neutral(self):
        assert sa._classify_momentum(None) == "neutral"


class TestGetSectorTopStocks:
    def test_returns_real_ranked_constituents(self):
        rows = sa.get_sector_top_stocks("Technology", limit=10)
        assert 1 <= len(rows) <= 10
        for row in rows:
            assert set(row.keys()) == {"symbol", "name", "price", "change_30day"}

    def test_sorted_by_30day_change_descending(self):
        rows = sa.get_sector_top_stocks("Technology", limit=10)
        changes = [r["change_30day"] for r in rows]
        assert changes == sorted(changes, reverse=True)

    def test_unknown_sector_returns_empty_list(self):
        assert sa.get_sector_top_stocks("NotASector") == []

    def test_limit_is_respected(self):
        rows = sa.get_sector_top_stocks("Healthcare", limit=3)
        assert len(rows) <= 3


class TestErrorHandling:
    def test_network_down_returns_none_for_performance(self, monkeypatch):
        sa.get_sector_performance.clear()
        monkeypatch.setattr(sa.yf, "Ticker", _boom)
        assert sa.get_sector_performance() is None

    def test_network_down_returns_empty_list_for_top_stocks(self, monkeypatch):
        sa.get_sector_top_stocks.clear()
        monkeypatch.setattr(sa.yf, "screen", _boom)
        assert sa.get_sector_top_stocks("Technology") == []

    def test_partial_failure_still_returns_the_sectors_that_resolved(self, monkeypatch):
        sa.get_sector_performance.clear()
        real_ticker = sa.yf.Ticker

        def flaky_ticker(symbol, *a, **kw):
            if symbol == "XLK":
                raise ConnectionError("simulated transient failure")
            return real_ticker(symbol, *a, **kw)

        monkeypatch.setattr(sa.yf, "Ticker", flaky_ticker)
        result = sa.get_sector_performance()
        assert result is not None
        assert "Technology" not in result["sectors"]
        assert len(result["sectors"]) == len(EXPECTED_SECTORS) - 1
