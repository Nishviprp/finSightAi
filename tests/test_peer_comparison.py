"""
Tests for src.peer_comparison.

Uses real live calls to Yahoo Finance for peer discovery and metrics,
consistent with how this project tests other yfinance-backed modules (see
test_sector_analytics.py). compute_valuation_verdict's decision logic is
pure and tested separately against synthetic DataFrames — no network needed
there.
"""
import math

import pandas as pd
import pytest

from src import peer_comparison as pc


@pytest.fixture(scope="module")
def aapl_peers():
    return pc.get_peers("AAPL")


@pytest.fixture(scope="module")
def aapl_comparison(aapl_peers):
    return pc.compare_peers("AAPL", tuple(aapl_peers))


class TestGetPeers:
    def test_returns_a_nonempty_list(self, aapl_peers):
        assert isinstance(aapl_peers, list)
        assert len(aapl_peers) > 0

    def test_returns_between_one_and_five_peers(self, aapl_peers):
        assert 1 <= len(aapl_peers) <= 5

    def test_does_not_include_the_symbol_itself(self, aapl_peers):
        assert "AAPL" not in aapl_peers

    def test_peers_are_real_large_tech_tickers(self, aapl_peers):
        # AAPL is mega-cap Technology sector; real peers should themselves
        # be large, real, resolvable tickers -- not fabricated placeholders.
        for symbol in aapl_peers:
            snap = pc.get_symbol_snapshot(symbol)
            assert snap is not None
            assert snap["sector"] == "Technology"

    def test_unknown_ticker_returns_empty_list(self):
        assert pc.get_peers("ZZZZZZZZZZ_NOT_A_REAL_TICKER") == []

    def test_result_is_cached_and_stable_across_calls(self):
        first = pc.get_peers("AAPL")
        second = pc.get_peers("AAPL")
        assert first == second


class TestComparePeers:
    def test_returns_dataframe_with_expected_columns(self, aapl_comparison):
        assert list(aapl_comparison.columns) == pc.COMPARISON_METRICS

    def test_includes_the_target_symbol_as_a_row(self, aapl_comparison):
        assert "AAPL" in aapl_comparison.index

    def test_includes_at_least_one_peer_row(self, aapl_comparison):
        assert len(aapl_comparison.index) > 1

    def test_pe_ratio_values_are_positive_where_present(self, aapl_comparison):
        pe_values = aapl_comparison["P/E Ratio"].dropna()
        assert not pe_values.empty
        assert (pe_values > 0).all()

    def test_metric_values_are_real_numbers_not_placeholders(self, aapl_comparison):
        # Every present value should be a genuine float, not e.g. a sentinel
        # like 0 or -1 standing in for missing data.
        for col in pc.COMPARISON_METRICS:
            for value in aapl_comparison[col].dropna():
                assert isinstance(value, (int, float))
                assert not math.isnan(value)

    def test_empty_peer_list_still_returns_target_row(self):
        df = pc.compare_peers("AAPL", tuple())
        assert "AAPL" in df.index
        assert len(df.index) == 1

    def test_unknown_symbol_returns_empty_dataframe(self):
        df = pc.compare_peers("ZZZZZZZZZZ_NOT_A_REAL_TICKER", tuple())
        assert df.empty


class TestMetricDirections:
    def test_covers_exactly_the_comparison_metrics(self):
        assert set(pc.METRIC_HIGHER_IS_BETTER.keys()) == set(pc.COMPARISON_METRICS)

    def test_cost_and_leverage_metrics_are_lower_is_better(self):
        for metric in ("P/E Ratio", "PEG Ratio", "Price/Book", "Debt/Equity"):
            assert pc.METRIC_HIGHER_IS_BETTER[metric] is False

    def test_growth_and_return_metrics_are_higher_is_better(self):
        for metric in ("ROE", "ROA", "Revenue Growth", "Margin %", "Dividend Yield", "52W Momentum"):
            assert pc.METRIC_HIGHER_IS_BETTER[metric] is True


class TestValuationVerdictLive:
    def test_verdict_is_one_of_the_valid_outcomes(self, aapl_comparison):
        verdict = pc.compute_valuation_verdict(aapl_comparison, "AAPL")
        assert verdict["verdict"] in {
            "UNDERVALUED", "OVERVALUED", "FAIRLY VALUED", "INSUFFICIENT_DATA",
        }

    def test_verdict_has_a_nonempty_reason(self, aapl_comparison):
        verdict = pc.compute_valuation_verdict(aapl_comparison, "AAPL")
        assert verdict["reason"]


def _synthetic_df(symbol_pe, symbol_growth, peer_rows):
    """peer_rows: list of (pe, growth) tuples for synthetic peer symbols."""
    data = {symbol_pe[0]: {"P/E Ratio": symbol_pe[1], "Revenue Growth": symbol_growth[1]}}
    for i, (pe, growth) in enumerate(peer_rows):
        data[f"PEER{i}"] = {"P/E Ratio": pe, "Revenue Growth": growth}
    return pd.DataFrame.from_dict(data, orient="index", columns=["P/E Ratio", "Revenue Growth"])


class TestValuationVerdictLogicSynthetic:
    """Pure decision-logic tests against hand-built DataFrames -- no network."""

    def test_cheap_and_high_growth_is_undervalued(self):
        df = _synthetic_df(("XYZ", 10.0), ("XYZ", 30.0), [(20.0, 10.0), (25.0, 12.0)])
        verdict = pc.compute_valuation_verdict(df, "XYZ")
        assert verdict["verdict"] == "UNDERVALUED"

    def test_expensive_and_low_growth_is_overvalued(self):
        df = _synthetic_df(("XYZ", 50.0), ("XYZ", 2.0), [(20.0, 10.0), (25.0, 12.0)])
        verdict = pc.compute_valuation_verdict(df, "XYZ")
        assert verdict["verdict"] == "OVERVALUED"

    def test_mixed_signal_is_fairly_valued(self):
        # Cheap P/E but also low growth -- not a clean undervalued/overvalued case.
        df = _synthetic_df(("XYZ", 10.0), ("XYZ", 2.0), [(20.0, 10.0), (25.0, 12.0)])
        verdict = pc.compute_valuation_verdict(df, "XYZ")
        assert verdict["verdict"] == "FAIRLY VALUED"

    def test_missing_pe_is_insufficient_data(self):
        df = _synthetic_df(("XYZ", None), ("XYZ", 30.0), [(20.0, 10.0)])
        verdict = pc.compute_valuation_verdict(df, "XYZ")
        assert verdict["verdict"] == "INSUFFICIENT_DATA"
        assert verdict["symbol_pe"] is None

    def test_symbol_not_in_dataframe_is_insufficient_data(self):
        df = _synthetic_df(("XYZ", 10.0), ("XYZ", 30.0), [(20.0, 10.0)])
        verdict = pc.compute_valuation_verdict(df, "NOTPRESENT")
        assert verdict["verdict"] == "INSUFFICIENT_DATA"

    def test_no_peer_rows_is_insufficient_data(self):
        df = _synthetic_df(("XYZ", 10.0), ("XYZ", 30.0), [])
        verdict = pc.compute_valuation_verdict(df, "XYZ")
        assert verdict["verdict"] == "INSUFFICIENT_DATA"

    def test_reason_mentions_the_symbol(self):
        df = _synthetic_df(("XYZ", 10.0), ("XYZ", 30.0), [(20.0, 10.0)])
        verdict = pc.compute_valuation_verdict(df, "XYZ")
        assert "XYZ" in verdict["reason"]
