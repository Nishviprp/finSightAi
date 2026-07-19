"""
Tests for src.financial_metrics.

Uses real live calls to Yahoo Finance for AAPL (per task requirements —
this module is a thin wrapper around yfinance, so its correctness genuinely
depends on live field names). Error-handling tests monkeypatch yfinance.Ticker
to simulate an unreachable network and a distinct symbol per test to avoid the
@st.cache_data(ttl=3600) cache returning a previously-cached real result.
"""
import math

import pandas as pd
import pytest

from src import financial_metrics as fm

AAPL = "AAPL"
INVALID_SYMBOL = "ZZZZZZINVALID"


# ---------------------------------------------------------------------------
# get_company_profile
# ---------------------------------------------------------------------------

class TestGetCompanyProfile:
    def test_returns_dict_for_real_symbol(self):
        profile = fm.get_company_profile(AAPL)
        assert isinstance(profile, dict)

    def test_has_expected_keys(self):
        profile = fm.get_company_profile(AAPL)
        expected_keys = {
            "symbol", "name", "sector", "industry", "website", "founded",
            "founders", "ceo", "employee_count", "business_summary",
            "city", "state", "country",
        }
        assert expected_keys.issubset(profile.keys())

    def test_populates_real_fields(self):
        profile = fm.get_company_profile(AAPL)
        assert profile["symbol"] == "AAPL"
        assert "Apple" in profile["name"]
        assert profile["sector"]
        assert isinstance(profile["employee_count"], int)
        assert profile["employee_count"] > 0
        assert profile["business_summary"]

    def test_ceo_extracted_from_company_officers(self):
        profile = fm.get_company_profile(AAPL)
        assert profile["ceo"] is not None
        assert "Cook" in profile["ceo"]

    def test_founded_and_founders_are_none_not_fabricated(self):
        profile = fm.get_company_profile(AAPL)
        assert profile["founded"] is None
        assert profile["founders"] is None

    def test_invalid_symbol_returns_none(self):
        assert fm.get_company_profile(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        fm.get_company_profile.clear()
        monkeypatch.setattr(fm.yf, "Ticker", _boom_ticker)
        assert fm.get_company_profile("NETDOWN1") is None


# ---------------------------------------------------------------------------
# get_income_statement
# ---------------------------------------------------------------------------

class TestGetIncomeStatement:
    def test_returns_dataframe_for_real_symbol(self):
        df = fm.get_income_statement(AAPL)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_rows(self):
        df = fm.get_income_statement(AAPL)
        expected_rows = {
            "Revenue", "Operating Income", "Net Income",
            "Gross Profit Margin %", "Operating Margin %", "Net Profit Margin %",
        }
        assert expected_rows.issubset(set(df.index))

    def test_has_up_to_five_annual_columns(self):
        df = fm.get_income_statement(AAPL)
        assert 1 <= len(df.columns) <= 5

    def test_revenue_is_positive_for_most_recent_year(self):
        df = fm.get_income_statement(AAPL)
        most_recent_col = df.columns[0]
        assert df.loc["Revenue", most_recent_col] > 0

    def test_margins_are_consistent_with_raw_figures(self):
        df = fm.get_income_statement(AAPL)
        col = df.columns[0]
        revenue = df.loc["Revenue", col]
        net_income = df.loc["Net Income", col]
        expected_margin = round(net_income / revenue * 100, 2)
        assert df.loc["Net Profit Margin %", col] == pytest.approx(expected_margin)

    def test_invalid_symbol_returns_none(self):
        assert fm.get_income_statement(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        fm.get_income_statement.clear()
        monkeypatch.setattr(fm.yf, "Ticker", _boom_ticker)
        assert fm.get_income_statement("NETDOWN2") is None


# ---------------------------------------------------------------------------
# get_balance_sheet
# ---------------------------------------------------------------------------

class TestGetBalanceSheet:
    def test_returns_dataframe_for_real_symbol(self):
        df = fm.get_balance_sheet(AAPL)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_rows(self):
        df = fm.get_balance_sheet(AAPL)
        expected_rows = {
            "Total Assets", "Total Liabilities", "Shareholders Equity",
            "Current Ratio", "Debt-to-Equity Ratio",
        }
        assert expected_rows.issubset(set(df.index))

    def test_total_assets_positive(self):
        df = fm.get_balance_sheet(AAPL)
        most_recent_col = df.columns[0]
        assert df.loc["Total Assets", most_recent_col] > 0

    def test_invalid_symbol_returns_none(self):
        assert fm.get_balance_sheet(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        fm.get_balance_sheet.clear()
        monkeypatch.setattr(fm.yf, "Ticker", _boom_ticker)
        assert fm.get_balance_sheet("NETDOWN3") is None


# ---------------------------------------------------------------------------
# get_cash_flow
# ---------------------------------------------------------------------------

class TestGetCashFlow:
    def test_returns_dataframe_for_real_symbol(self):
        df = fm.get_cash_flow(AAPL)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_rows(self):
        df = fm.get_cash_flow(AAPL)
        expected_rows = {"Operating Cash Flow", "Free Cash Flow", "Capital Expenditures"}
        assert expected_rows.issubset(set(df.index))

    def test_invalid_symbol_returns_none(self):
        assert fm.get_cash_flow(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        fm.get_cash_flow.clear()
        monkeypatch.setattr(fm.yf, "Ticker", _boom_ticker)
        assert fm.get_cash_flow("NETDOWN4") is None


# ---------------------------------------------------------------------------
# get_earnings_history
# ---------------------------------------------------------------------------

class TestGetEarningsHistory:
    def test_returns_list_for_real_symbol(self):
        history = fm.get_earnings_history(AAPL)
        assert isinstance(history, list)

    def test_at_most_eight_quarters(self):
        history = fm.get_earnings_history(AAPL)
        assert 1 <= len(history) <= 8

    def test_entries_have_expected_keys_and_types(self):
        history = fm.get_earnings_history(AAPL)
        for entry in history:
            assert set(entry.keys()) == {"date", "eps_estimate", "eps_actual", "surprise_percent"}
            assert isinstance(entry["date"], str)
            assert entry["eps_actual"] is None or isinstance(entry["eps_actual"], float)

    def test_excludes_unreported_future_quarters(self):
        history = fm.get_earnings_history(AAPL)
        for entry in history:
            assert entry["eps_actual"] is not None
            assert not (isinstance(entry["eps_actual"], float) and math.isnan(entry["eps_actual"]))

    def test_sorted_most_recent_first(self):
        history = fm.get_earnings_history(AAPL)
        dates = [entry["date"] for entry in history]
        assert dates == sorted(dates, reverse=True)

    def test_invalid_symbol_returns_none(self):
        assert fm.get_earnings_history(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        fm.get_earnings_history.clear()
        monkeypatch.setattr(fm.yf, "Ticker", _boom_ticker)
        assert fm.get_earnings_history("NETDOWN5") is None


# ---------------------------------------------------------------------------
# get_analyst_ratings
# ---------------------------------------------------------------------------

class TestGetAnalystRatings:
    def test_returns_dict_for_real_symbol(self):
        ratings = fm.get_analyst_ratings(AAPL)
        assert isinstance(ratings, dict)

    def test_has_expected_keys(self):
        ratings = fm.get_analyst_ratings(AAPL)
        expected_keys = {
            "average_rating", "rating_key", "number_of_analysts",
            "buy_count", "hold_count", "sell_count",
            "price_target_mean", "price_target_high", "price_target_low",
        }
        assert expected_keys.issubset(ratings.keys())

    def test_average_rating_within_1_to_5_scale(self):
        ratings = fm.get_analyst_ratings(AAPL)
        assert 1.0 <= ratings["average_rating"] <= 5.0

    def test_analyst_counts_are_non_negative_ints(self):
        ratings = fm.get_analyst_ratings(AAPL)
        for key in ("buy_count", "hold_count", "sell_count"):
            assert isinstance(ratings[key], int)
            assert ratings[key] >= 0

    def test_price_targets_ordered_low_to_high(self):
        ratings = fm.get_analyst_ratings(AAPL)
        assert ratings["price_target_low"] <= ratings["price_target_mean"] <= ratings["price_target_high"]

    def test_invalid_symbol_returns_none(self):
        assert fm.get_analyst_ratings(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        fm.get_analyst_ratings.clear()
        monkeypatch.setattr(fm.yf, "Ticker", _boom_ticker)
        assert fm.get_analyst_ratings("NETDOWN6") is None


# ---------------------------------------------------------------------------
# Shared fixture helper
# ---------------------------------------------------------------------------

def _boom_ticker(*_args, **_kwargs):
    raise ConnectionError("simulated network down")
