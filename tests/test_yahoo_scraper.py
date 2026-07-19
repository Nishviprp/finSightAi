"""
Unit tests for scrapers.yahoo_finance_scraper.

No live HTTP calls are made: yfinance's `screen()` and the module's own
`_get` / `_scrape_markets_page` helpers are monkeypatched.
"""
import pytest
from bs4 import BeautifulSoup

from scrapers import yahoo_finance_scraper as mod


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestParseFloat:
    def test_parses_percent_with_sign(self):
        assert mod._parse_float("+28.79%") == pytest.approx(28.79)

    def test_parses_negative_percent(self):
        assert mod._parse_float("-1.33%") == pytest.approx(-1.33)

    def test_parses_plain_price(self):
        assert mod._parse_float("5.95") == pytest.approx(5.95)

    def test_parses_comma_thousands(self):
        assert mod._parse_float("1,093.40") == pytest.approx(1093.40)


class TestParseVolume:
    def test_parses_millions_suffix(self):
        assert mod._parse_volume("55.692M") == 55692000

    def test_parses_billions_suffix(self):
        assert mod._parse_volume("1.2B") == 1200000000

    def test_parses_comma_separated_integer(self):
        assert mod._parse_volume("858,955") == 858955

    def test_empty_string_returns_zero(self):
        assert mod._parse_volume("") == 0


class TestExtractTotal:
    def test_extracts_total_from_of_n_pattern(self):
        html = '<span>1-25 <b>of 403</b> results</span>'
        assert mod._extract_total(html) == 403

    def test_extracts_total_with_thousands_comma(self):
        html = 'Showing 1-25 of 1,234 stocks'
        assert mod._extract_total(html) == 1234

    def test_returns_none_when_pattern_absent(self):
        assert mod._extract_total("<html><body>no pagination here</body></html>") is None


# ---------------------------------------------------------------------------
# _normalize_quote (yfinance quote dict -> row schema)
# ---------------------------------------------------------------------------

class TestNormalizeQuote:
    def test_prefers_long_name(self):
        quote = {
            "symbol": "LCID",
            "longName": "Lucid Group, Inc.",
            "shortName": "Lucid",
            "regularMarketPrice": 5.95,
            "regularMarketChangePercent": 28.7879,
            "regularMarketVolume": 56365785,
        }
        row = mod._normalize_quote(quote)
        assert row["symbol"] == "LCID"
        assert row["name"] == "Lucid Group, Inc."
        assert row["price"] == pytest.approx(5.95)
        assert row["change_percent"] == pytest.approx(28.7879)
        assert row["volume"] == 56365785

    def test_falls_back_to_short_name(self):
        quote = {"symbol": "XYZ", "shortName": "Xyz Corp", "regularMarketPrice": 1.0}
        row = mod._normalize_quote(quote)
        assert row["name"] == "Xyz Corp"

    def test_missing_fields_default_safely(self):
        row = mod._normalize_quote({"symbol": "XYZ"})
        assert row == {"symbol": "XYZ", "name": "XYZ", "price": 0.0, "change_percent": 0.0, "volume": 0}


# ---------------------------------------------------------------------------
# _parse_row (BeautifulSoup <tr> -> row schema), against real captured markup
# ---------------------------------------------------------------------------

ROW_HTML = """
<table><tbody>
<tr class="row" data-testid="data-table-v2-row" data-testid-row="0">
  <td data-testid-cell="ticker"><span class="ticker-wrapper"><a href="/quote/LCID/">
    <div class="name"><span class="symbol">LCID</span></div></a></span></td>
  <td data-testid-cell="companyshortname.raw"><div title="Lucid Group, Inc.">Lucid Group, Inc.</div></td>
  <td data-testid-cell="intradayprice"><span data-testid="change">5.95</span></td>
  <td data-testid-cell="percentchange"><span data-testid="colorChange">+28.79%</span></td>
  <td data-testid-cell="dayvolume"><span data-testid="change">55.692M</span></td>
</tr>
</tbody></table>
"""


class TestParseRow:
    def _row(self):
        soup = BeautifulSoup(ROW_HTML, "lxml")
        return soup.select_one('tr[data-testid="data-table-v2-row"]')

    def test_parses_real_markup(self):
        row = mod._parse_row(self._row())
        assert row == {
            "symbol": "LCID",
            "name": "Lucid Group, Inc.",
            "price": pytest.approx(5.95),
            "change_percent": pytest.approx(28.79),
            "volume": 55692000,
        }

    def test_returns_none_when_symbol_missing(self):
        soup = BeautifulSoup("<table><tbody><tr></tr></tbody></table>", "lxml")
        assert mod._parse_row(soup.select_one("tr")) is None


# ---------------------------------------------------------------------------
# Public fetch functions: yfinance primary, scrape fallback
# ---------------------------------------------------------------------------

SAMPLE_QUOTES = [
    {
        "symbol": "AAA",
        "longName": "Alpha Co",
        "regularMarketPrice": 10.0,
        "regularMarketChangePercent": 5.0,
        "regularMarketVolume": 1000,
    }
]

SAMPLE_SCRAPE_ROW = {"symbol": "BBB", "name": "Beta Inc", "price": 1.0, "change_percent": -1.0, "volume": 5}


class TestFetchFunctionsUseYfinance:
    def test_fetch_top_gainers_returns_normalized_rows(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", lambda query, count=None, **kw: {"quotes": SAMPLE_QUOTES})
        rows = mod.fetch_top_gainers(count=5)
        assert rows == [
            {"symbol": "AAA", "name": "Alpha Co", "price": 10.0, "change_percent": 5.0, "volume": 1000}
        ]

    def test_fetch_top_losers_returns_normalized_rows(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", lambda query, count=None, **kw: {"quotes": SAMPLE_QUOTES})
        rows = mod.fetch_top_losers(count=5)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAA"

    def test_fetch_most_active_returns_normalized_rows(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", lambda query, count=None, **kw: {"quotes": SAMPLE_QUOTES})
        rows = mod.fetch_most_active(count=5)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAA"

    def test_every_call_pages_in_fixed_25row_chunks_regardless_of_offset(self, monkeypatch):
        # Regression test: yfinance silently caps predefined-screen results at
        # 25 rows whenever `offset` is passed at all, even offset=0. Mixing an
        # offset-less first call with offset-based later ones was tried first,
        # but that hits a different underlying pagination baseline and
        # produces duplicate rows at the page boundary (verified live). So
        # every call — including offset=0 — must request exactly 25 with an
        # explicit offset, looping to accumulate `count` rows.
        calls = []

        def fake_screen(query, count=None, offset=None, **kw):
            calls.append({"count": count, "offset": offset})
            batch = [dict(SAMPLE_QUOTES[0], symbol=f"S{offset}-{i}") for i in range(25)]
            return {"quotes": batch, "total": 999}

        monkeypatch.setattr(mod.yf, "screen", fake_screen)

        rows0, total0 = mod._from_yfinance("day_gainers", count=50, offset=0)
        assert calls == [{"count": 25, "offset": 0}, {"count": 25, "offset": 25}]
        assert len(rows0) == 50
        assert total0 == 999

        calls.clear()
        rows1, total1 = mod._from_yfinance("day_gainers", count=50, offset=50)
        assert calls == [{"count": 25, "offset": 50}, {"count": 25, "offset": 75}]
        assert len(rows1) == 50

    def test_total_function_returns_real_total_from_response(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", lambda *a, **kw: {"quotes": SAMPLE_QUOTES, "total": 403})
        assert mod.fetch_top_gainers_total() == 403


class TestFetchFunctionsFallBackToScrape:
    def test_falls_back_when_yfinance_raises(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(mod.yf, "screen", boom)
        monkeypatch.setattr(mod, "_scrape_markets_page", lambda path, count, offset=0: ([SAMPLE_SCRAPE_ROW], 99))
        rows = mod.fetch_top_gainers(count=5)
        assert rows == [SAMPLE_SCRAPE_ROW]

    def test_falls_back_when_yfinance_returns_no_quotes(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", lambda *a, **kw: {"quotes": []})
        monkeypatch.setattr(mod, "_scrape_markets_page", lambda path, count, offset=0: ([SAMPLE_SCRAPE_ROW], 99))
        rows = mod.fetch_top_losers(count=5)
        assert rows == [SAMPLE_SCRAPE_ROW]

    def test_network_down_on_both_paths_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network down")))
        monkeypatch.setattr(mod, "_get", lambda url: None)
        rows = mod.fetch_most_active(count=5)
        assert rows == []


class TestFiftyTwoWeekFunctionsScrapeDirectly:
    def test_fetch_52week_gainers_uses_scrape_path_not_yfinance(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", lambda *a, **kw: pytest.fail("yfinance should not be called"))
        seen = {}

        def fake_scrape(path, count, offset=0):
            seen["path"] = path
            seen["count"] = count
            seen["offset"] = offset
            return [SAMPLE_SCRAPE_ROW], 250

        monkeypatch.setattr(mod, "_scrape_markets_page", fake_scrape)
        rows = mod.fetch_52week_gainers(count=5, offset=10)
        assert seen == {"path": "52-week-gainers", "count": 5, "offset": 10}
        assert rows == [SAMPLE_SCRAPE_ROW]

    def test_fetch_52week_losers_uses_scrape_path_not_yfinance(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", lambda *a, **kw: pytest.fail("yfinance should not be called"))
        seen = {}

        def fake_scrape(path, count, offset=0):
            seen["path"] = path
            return [], None

        monkeypatch.setattr(mod, "_scrape_markets_page", fake_scrape)
        rows = mod.fetch_52week_losers(count=5)
        assert seen == {"path": "52-week-losers"}
        assert rows == []

    def test_52week_gainers_total_reads_extracted_total(self, monkeypatch):
        monkeypatch.setattr(mod, "_scrape_markets_page", lambda path, count, offset=0: ([], 137))
        assert mod.fetch_52week_gainers_total() == 137


# ---------------------------------------------------------------------------
# fetch_all_time_high / fetch_all_time_low
#
# These hit real Yahoo Finance data (yf.screen + a batched yf.download for
# days-from-extreme), unlike the mocked tests above — there is no dedicated
# Yahoo "all-time high/low" page to scrape (finance.yahoo.com/markets/stocks/
# all-time-high/ 302-redirects to /most-active/), so this is built on the
# 52-week-high/low fields from the screener API and verified against live
# results, same as test_financial_metrics.py does for AAPL.
# ---------------------------------------------------------------------------

class TestFetchAllTimeHigh:
    def test_returns_requested_number_of_rows(self):
        rows = mod.fetch_all_time_high(limit=25, offset=0)
        assert rows is not None
        assert len(rows) == 25

    def test_row_schema(self):
        rows = mod.fetch_all_time_high(limit=5, offset=0)
        for row in rows:
            assert set(row.keys()) == {
                "symbol", "name", "price", "change_percent",
                "all_time_high_price", "days_from_high",
            }
            assert isinstance(row["symbol"], str) and row["symbol"]
            assert isinstance(row["price"], float)
            assert isinstance(row["all_time_high_price"], float)
            assert row["days_from_high"] is None or isinstance(row["days_from_high"], int)

    def test_price_is_at_or_below_the_52week_high(self):
        rows = mod.fetch_all_time_high(limit=25, offset=0)
        for row in rows:
            # small slack: price and 52wk-high come from slightly different
            # quote snapshots a few seconds apart
            assert row["price"] <= row["all_time_high_price"] * 1.01

    def test_offset_pagination_returns_next_batch_without_overlap(self):
        page1 = mod.fetch_all_time_high(limit=25, offset=0)
        page2 = mod.fetch_all_time_high(limit=25, offset=25)
        assert len(page2) == 25
        symbols1 = {r["symbol"] for r in page1}
        symbols2 = {r["symbol"] for r in page2}
        assert symbols1.isdisjoint(symbols2)

    def test_invalid_negative_offset_returns_empty_list(self):
        assert mod.fetch_all_time_high(limit=25, offset=-10) == []

    def test_offset_beyond_available_rows_returns_empty_list(self):
        assert mod.fetch_all_time_high(limit=25, offset=100_000) == []

    def test_zero_limit_returns_empty_list(self):
        assert mod.fetch_all_time_high(limit=0, offset=0) == []

    def test_network_down_returns_none(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", _boom)
        assert mod.fetch_all_time_high(limit=25, offset=0) is None


class TestFetchAllTimeLow:
    def test_returns_requested_number_of_rows(self):
        rows = mod.fetch_all_time_low(limit=25, offset=0)
        assert rows is not None
        assert len(rows) == 25

    def test_row_schema(self):
        rows = mod.fetch_all_time_low(limit=5, offset=0)
        for row in rows:
            assert set(row.keys()) == {
                "symbol", "name", "price", "change_percent",
                "all_time_low_price", "days_from_low",
            }
            assert isinstance(row["all_time_low_price"], float)
            assert row["days_from_low"] is None or isinstance(row["days_from_low"], int)

    def test_price_is_at_or_above_the_52week_low(self):
        rows = mod.fetch_all_time_low(limit=25, offset=0)
        for row in rows:
            assert row["price"] >= row["all_time_low_price"] * 0.99

    def test_offset_pagination_returns_next_batch_without_overlap(self):
        page1 = mod.fetch_all_time_low(limit=10, offset=0)
        page2 = mod.fetch_all_time_low(limit=10, offset=10)
        symbols1 = {r["symbol"] for r in page1}
        symbols2 = {r["symbol"] for r in page2}
        assert symbols1.isdisjoint(symbols2)

    def test_invalid_negative_offset_returns_empty_list(self):
        assert mod.fetch_all_time_low(limit=25, offset=-1) == []

    def test_network_down_returns_none(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", _boom)
        assert mod.fetch_all_time_low(limit=25, offset=0) is None

    def test_universe_fetch_failure_returns_empty_list_not_none(self, monkeypatch):
        # yf.screen succeeding but returning no quotes is a legitimate empty
        # result, not a failure -> [] not None.
        monkeypatch.setattr(mod.yf, "screen", lambda *a, **kw: {"quotes": []})
        assert mod.fetch_all_time_low(limit=25, offset=0) == []


class TestAllTimeHighLowTotals:
    def test_high_total_is_positive_int(self):
        total = mod.fetch_all_time_high_total()
        assert isinstance(total, int)
        assert total > 0

    def test_low_total_is_positive_int(self):
        total = mod.fetch_all_time_low_total()
        assert isinstance(total, int)
        assert total > 0

    def test_high_total_matches_actual_candidate_pool_size(self):
        total = mod.fetch_all_time_high_total()
        # paging with a limit larger than the pool should return exactly `total` rows
        all_rows = mod.fetch_all_time_high(limit=total, offset=0)
        assert len(all_rows) == total

    def test_network_down_returns_none(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "screen", _boom)
        assert mod.fetch_all_time_high_total() is None
        assert mod.fetch_all_time_low_total() is None


# ---------------------------------------------------------------------------
# Default row count (50) and offset pagination for the original 5 screeners
#
# Live calls, same rationale as the all-time-high/low tests above: this is
# exercising real pagination behavior of yfinance's screen() (count+offset)
# and the scraped markets-page fallback (?count=&start=), not just the
# module's own logic.
# ---------------------------------------------------------------------------

class TestDefaultRowCountIsFifty:
    def test_top_gainers_default_returns_fifty_rows(self):
        assert len(mod.fetch_top_gainers()) == 50

    def test_most_active_default_returns_fifty_rows(self):
        assert len(mod.fetch_most_active()) == 50

    def test_52week_gainers_default_returns_fifty_rows(self):
        assert len(mod.fetch_52week_gainers()) == 50

    def test_all_time_high_default_returns_fifty_rows(self):
        assert len(mod.fetch_all_time_high()) == 50


class TestOffsetPaginationOnOriginalScreeners:
    def test_top_gainers_offset_returns_next_batch_without_overlap(self):
        page1 = mod.fetch_top_gainers(count=25, offset=0)
        page2 = mod.fetch_top_gainers(count=25, offset=25)
        symbols1 = {r["symbol"] for r in page1}
        symbols2 = {r["symbol"] for r in page2}
        assert len(page2) == 25
        assert symbols1.isdisjoint(symbols2)

    def test_52week_gainers_offset_returns_next_batch_without_overlap(self):
        page1 = mod.fetch_52week_gainers(count=25, offset=0)
        page2 = mod.fetch_52week_gainers(count=25, offset=25)
        symbols1 = {r["symbol"] for r in page1}
        symbols2 = {r["symbol"] for r in page2}
        assert symbols1.isdisjoint(symbols2)

    def test_top_gainers_50row_load_more_matches_real_ui_usage_without_overlap(self):
        # Regression test for a real bug: fetching page 1 via an offset=0
        # "fast path" that omitted the `offset` kwarg, then page 2 via an
        # offset=50 chunked path, hit two different Yahoo pagination
        # baselines and produced 2 duplicate rows at the boundary. This
        # mirrors the exact "Load More" call shape (count=50 both times).
        page1 = mod.fetch_top_gainers(count=50, offset=0)
        page2 = mod.fetch_top_gainers(count=50, offset=50)
        symbols1 = {r["symbol"] for r in page1}
        symbols2 = {r["symbol"] for r in page2}
        assert len(page1) == 50
        assert len(page2) == 50
        assert symbols1.isdisjoint(symbols2)


class TestOriginalScreenerTotals:
    def test_top_gainers_total_is_a_real_positive_int(self):
        total = mod.fetch_top_gainers_total()
        assert isinstance(total, int)
        assert total > 0

    def test_most_active_total_is_a_real_positive_int(self):
        total = mod.fetch_most_active_total()
        assert isinstance(total, int)
        assert total > 0

    def test_52week_losers_total_is_a_real_positive_int_or_none(self):
        # scrape-path total depends on Yahoo rendering the "of N" pager text;
        # tolerate None but never a fabricated/negative number.
        total = mod.fetch_52week_losers_total()
        assert total is None or (isinstance(total, int) and total > 0)


# ---------------------------------------------------------------------------
# fetch_us_indices — live yfinance Ticker().info calls, same rationale as the
# other live-tested screeners: this is a thin wrapper whose correctness
# genuinely depends on real field names and real data resolving.
# ---------------------------------------------------------------------------

class TestFetchUsIndices:
    def test_returns_all_four_indices(self):
        rows = mod.fetch_us_indices()
        assert rows is not None
        assert {r["symbol"] for r in rows} == {"^GSPC", "^DJI", "^IXIC", "^RUT"}

    def test_row_schema(self):
        rows = mod.fetch_us_indices()
        for row in rows:
            assert set(row.keys()) == {"symbol", "name", "price", "change_percent", "change_dollar"}
            assert isinstance(row["symbol"], str) and row["symbol"]
            assert isinstance(row["name"], str) and row["name"]
            assert isinstance(row["price"], float)
            assert isinstance(row["change_percent"], float)
            assert isinstance(row["change_dollar"], float)

    def test_prices_are_realistic_positive_values(self):
        # sanity bound only — these indices trade in the thousands, never
        # near zero or negative.
        rows = mod.fetch_us_indices()
        for row in rows:
            assert row["price"] > 100

    def test_names_are_recognizable(self):
        rows = mod.fetch_us_indices()
        names_by_symbol = {r["symbol"]: r["name"] for r in rows}
        assert "S&P 500" in names_by_symbol["^GSPC"]
        assert "Dow" in names_by_symbol["^DJI"]
        assert "NASDAQ" in names_by_symbol["^IXIC"] or "Nasdaq" in names_by_symbol["^IXIC"]
        assert "Russell 2000" in names_by_symbol["^RUT"]

    def test_partial_failure_still_returns_the_symbols_that_resolved(self, monkeypatch):
        real_ticker = mod.yf.Ticker

        def flaky_ticker(symbol, *a, **kw):
            if symbol == "^RUT":
                raise ConnectionError("simulated transient failure")
            return real_ticker(symbol, *a, **kw)

        monkeypatch.setattr(mod.yf, "Ticker", flaky_ticker)
        rows = mod.fetch_us_indices()
        assert rows is not None
        symbols = {r["symbol"] for r in rows}
        assert symbols == {"^GSPC", "^DJI", "^IXIC"}

    def test_network_down_returns_none(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "Ticker", _boom)
        assert mod.fetch_us_indices() is None


# ---------------------------------------------------------------------------
# fetch_world_indices — live yfinance Ticker().info calls, same rationale as
# fetch_us_indices above.
# ---------------------------------------------------------------------------

_WORLD_INDEX_SYMBOLS = {"^N225", "^GDAXI", "^FTSE", "000001.SS", "^HSI", "^BSESN"}


class TestFetchWorldIndices:
    def test_returns_all_six_indices(self):
        rows = mod.fetch_world_indices()
        assert rows is not None
        assert {r["symbol"] for r in rows} == _WORLD_INDEX_SYMBOLS

    def test_row_schema(self):
        rows = mod.fetch_world_indices()
        for row in rows:
            assert set(row.keys()) == {"symbol", "name", "price", "change_percent", "change_dollar"}
            assert isinstance(row["symbol"], str) and row["symbol"]
            assert isinstance(row["name"], str) and row["name"]
            assert isinstance(row["price"], float)
            assert isinstance(row["change_percent"], float)
            assert isinstance(row["change_dollar"], float)

    def test_prices_are_realistic_positive_values(self):
        rows = mod.fetch_world_indices()
        for row in rows:
            assert row["price"] > 100

    def test_shanghai_composite_gets_a_real_name_despite_yahoo_returning_none(self):
        # Regression test: Yahoo's own longName/shortName for 000001.SS are
        # both None even though real price/change data is present (verified
        # live) — the old validity check gated on name presence and would
        # have silently dropped this index entirely. Confirm it's present
        # with a non-empty name (either Yahoo's own, when it has one that
        # call, or the documented fallback).
        rows = mod.fetch_world_indices()
        shanghai = next(r for r in rows if r["symbol"] == "000001.SS")
        assert shanghai["name"]

    def test_fallback_name_used_when_yahoo_returns_no_name(self, monkeypatch):
        monkeypatch.setattr(
            mod.yf, "Ticker",
            lambda symbol, *a, **kw: type(
                "FakeTicker", (), {"info": {
                    "regularMarketPrice": 3764.15,
                    "regularMarketChange": -118.26,
                    "regularMarketChangePercent": -3.05,
                    "longName": None, "shortName": None,
                }}
            )(),
        )
        rows = mod.fetch_world_indices()
        assert rows is not None
        assert all(r["name"] == mod.WORLD_INDEX_SYMBOLS[r["symbol"]] for r in rows)

    def test_partial_failure_still_returns_the_symbols_that_resolved(self, monkeypatch):
        real_ticker = mod.yf.Ticker

        def flaky_ticker(symbol, *a, **kw):
            if symbol == "^HSI":
                raise ConnectionError("simulated transient failure")
            return real_ticker(symbol, *a, **kw)

        monkeypatch.setattr(mod.yf, "Ticker", flaky_ticker)
        rows = mod.fetch_world_indices()
        assert rows is not None
        symbols = {r["symbol"] for r in rows}
        assert symbols == _WORLD_INDEX_SYMBOLS - {"^HSI"}

    def test_delisted_or_invalid_symbol_excluded_not_crashed(self, monkeypatch):
        real_ticker = mod.yf.Ticker

        def ticker_with_one_delisted(symbol, *a, **kw):
            if symbol == "^N225":
                return type("FakeTicker", (), {"info": {}})()  # empty info: delisted/invalid
            return real_ticker(symbol, *a, **kw)

        monkeypatch.setattr(mod.yf, "Ticker", ticker_with_one_delisted)
        rows = mod.fetch_world_indices()
        assert rows is not None
        symbols = {r["symbol"] for r in rows}
        assert "^N225" not in symbols
        assert len(symbols) == 5

    def test_network_down_returns_none(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "Ticker", _boom)
        assert mod.fetch_world_indices() is None


# ---------------------------------------------------------------------------
# fetch_top_10_crypto — live yfinance Ticker().info calls, same rationale as
# fetch_us_indices / fetch_world_indices above.
# ---------------------------------------------------------------------------

_CRYPTO_YFINANCE_SYMBOLS = {
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
    "XRP-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD", "DOT-USD",
}
_CRYPTO_CLEAN_SYMBOLS = {s.removesuffix("-USD") for s in _CRYPTO_YFINANCE_SYMBOLS}


class TestFetchTop10Crypto:
    def test_returns_all_ten_cryptos(self):
        rows = mod.fetch_top_10_crypto()
        assert rows is not None
        assert {r["symbol"] for r in rows} == _CRYPTO_CLEAN_SYMBOLS

    def test_symbols_are_clean_without_usd_suffix(self):
        rows = mod.fetch_top_10_crypto()
        for row in rows:
            assert "-USD" not in row["symbol"]
        symbols = {r["symbol"] for r in rows}
        assert "BTC" in symbols
        assert "BTC-USD" not in symbols

    def test_row_schema(self):
        rows = mod.fetch_top_10_crypto()
        for row in rows:
            assert set(row.keys()) == {
                "symbol", "name", "price", "change_percent", "change_dollar", "market_cap",
            }
            assert isinstance(row["symbol"], str) and row["symbol"]
            assert isinstance(row["name"], str) and row["name"]
            assert isinstance(row["price"], float)
            assert isinstance(row["change_percent"], float)
            assert isinstance(row["change_dollar"], float)

    def test_market_cap_present_and_positive_for_every_coin(self):
        rows = mod.fetch_top_10_crypto()
        for row in rows:
            assert row["market_cap"] is not None
            assert isinstance(row["market_cap"], float)
            assert row["market_cap"] > 0

    def test_bitcoin_has_the_largest_market_cap(self):
        # sanity check on real live data, not a hardcoded figure
        rows = mod.fetch_top_10_crypto()
        by_cap = sorted(rows, key=lambda r: r["market_cap"], reverse=True)
        assert by_cap[0]["symbol"] == "BTC"

    def test_partial_failure_still_returns_the_symbols_that_resolved(self, monkeypatch):
        real_ticker = mod.yf.Ticker

        def flaky_ticker(symbol, *a, **kw):
            if symbol == "DOT-USD":
                raise ConnectionError("simulated transient failure")
            return real_ticker(symbol, *a, **kw)

        monkeypatch.setattr(mod.yf, "Ticker", flaky_ticker)
        rows = mod.fetch_top_10_crypto()
        assert rows is not None
        symbols = {r["symbol"] for r in rows}
        assert symbols == _CRYPTO_CLEAN_SYMBOLS - {"DOT"}

    def test_delisted_or_invalid_symbol_excluded_not_crashed(self, monkeypatch):
        real_ticker = mod.yf.Ticker

        def ticker_with_one_delisted(symbol, *a, **kw):
            if symbol == "SHIB-USD":
                return type("FakeTicker", (), {"info": {}})()  # empty info: delisted/invalid
            return real_ticker(symbol, *a, **kw)

        monkeypatch.setattr(mod.yf, "Ticker", ticker_with_one_delisted)
        rows = mod.fetch_top_10_crypto()
        assert rows is not None
        symbols = {r["symbol"] for r in rows}
        assert "SHIB" not in symbols
        assert len(symbols) == 9

    def test_network_down_returns_none(self, monkeypatch):
        monkeypatch.setattr(mod.yf, "Ticker", _boom)
        assert mod.fetch_top_10_crypto() is None


def _boom(*_args, **_kwargs):
    raise ConnectionError("simulated network down")
