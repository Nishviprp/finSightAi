"""
Tests for src.trading_signals.

Uses real live calls to Yahoo Finance for AAPL and a small real-symbol
batch, consistent with how this project tests other yfinance-backed
modules (test_financial_metrics.py, test_yahoo_scraper.py's index/crypto
tests). The _score_signal tests below are the ones that actually prove the
signal logic is a deterministic rule, not randomness — they feed fixed,
hand-picked indicator values and assert the exact expected output.
"""
import time

import pandas as pd
import pytest

from src import trading_signals as ts

AAPL = "AAPL"
INVALID_SYMBOL = "ZZZZZZINVALID"


# ---------------------------------------------------------------------------
# Manual indicator functions — pure, no network. calculate_rsi is checked
# against Wilder's own textbook 14-period example (from "New Concepts in
# Technical Trading Systems"), which nails down the exact seeding method,
# not just "some smoothed average that looks RSI-shaped".
# ---------------------------------------------------------------------------

_WILDER_TEXTBOOK_PRICES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
    45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
]


class TestCalculateRsi:
    def test_matches_wilders_textbook_example(self):
        rsi = ts.calculate_rsi(pd.Series(_WILDER_TEXTBOOK_PRICES), period=14)
        assert rsi == pytest.approx(70.46, abs=0.01)

    def test_accepts_plain_list_not_just_series(self):
        rsi = ts.calculate_rsi(_WILDER_TEXTBOOK_PRICES, period=14)
        assert rsi == pytest.approx(70.46, abs=0.01)

    def test_all_gains_is_100(self):
        rising = list(range(1, 30))
        assert ts.calculate_rsi(rising, period=14) == 100.0

    def test_all_losses_is_0(self):
        falling = list(range(30, 1, -1))
        assert ts.calculate_rsi(falling, period=14) == 0.0

    def test_stays_in_valid_range_on_real_mixed_data(self):
        mixed = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                 44.84, 43.08, 42.89, 43.03, 44.61, 45.28, 46.28, 45.9,
                 44.3, 43.1, 42.8, 43.9]
        rsi = ts.calculate_rsi(mixed, period=14)
        assert 0.0 <= rsi <= 100.0

    def test_insufficient_history_returns_none(self):
        assert ts.calculate_rsi([1.0, 2.0, 3.0], period=14) is None

    def test_none_returned_for_flat_period_plus_one_boundary(self):
        # exactly period+1 points is the minimum accepted length
        assert ts.calculate_rsi([1.0] * 15, period=14) is not None
        assert ts.calculate_rsi([1.0] * 14, period=14) is None


class TestCalculateMacd:
    def test_constant_series_has_zero_macd_signal_and_histogram(self):
        macd, signal, hist = ts.calculate_macd([50.0] * 60)
        assert macd == pytest.approx(0.0, abs=1e-9)
        assert signal == pytest.approx(0.0, abs=1e-9)
        assert hist == pytest.approx(0.0, abs=1e-9)

    def test_histogram_equals_macd_minus_signal(self):
        prices = [100 + i * 0.3 + (2 if i % 5 == 0 else 0) for i in range(80)]
        macd, signal, hist = ts.calculate_macd(prices)
        assert hist == pytest.approx(macd - signal, abs=1e-9)

    def test_steadily_rising_prices_give_positive_macd(self):
        rising = [100 + i for i in range(80)]
        macd, _signal, _hist = ts.calculate_macd(rising)
        assert macd > 0

    def test_steadily_falling_prices_give_negative_macd(self):
        falling = [200 - i for i in range(80)]
        macd, _signal, _hist = ts.calculate_macd(falling)
        assert macd < 0

    def test_returns_none_for_empty_series(self):
        assert ts.calculate_macd([]) is None


class TestCalculateSma:
    def test_matches_hand_computed_average(self):
        sma = ts.calculate_sma(list(range(1, 11)), period=5)
        assert sma.iloc[-1] == pytest.approx(8.0)  # mean(6..10)

    def test_leading_values_are_nan_before_period_is_reached(self):
        sma = ts.calculate_sma(list(range(1, 11)), period=5)
        assert sma.iloc[:4].isna().all()
        assert not pd.isna(sma.iloc[4])


class TestCalculateEma:
    def test_constant_series_converges_to_that_constant(self):
        ema = ts.calculate_ema([100.0] * 30, period=12)
        assert ema.iloc[-1] == pytest.approx(100.0)

    def test_reacts_faster_than_sma_to_a_recent_jump(self):
        prices = [100.0] * 20 + [200.0] * 5
        ema_last = ts.calculate_ema(prices, period=12).iloc[-1]
        sma_last = ts.calculate_sma(prices, period=12).iloc[-1]
        assert ema_last > sma_last


def _boom_ticker(*_args, **_kwargs):
    raise ConnectionError("simulated network down")


# ---------------------------------------------------------------------------
# calculate_signals — real symbol
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aapl_signal():
    return ts.calculate_signals(AAPL)


class TestCalculateSignalsRealSymbol:
    def test_returns_dict(self, aapl_signal):
        assert isinstance(aapl_signal, dict)

    def test_has_expected_keys(self, aapl_signal):
        expected_keys = {
            "symbol", "name", "price", "signal", "confidence",
            "rsi_30min", "rsi_14d", "macd_trend", "ma_crossover",
            "volume_ratio", "reason",
        }
        assert expected_keys.issubset(aapl_signal.keys())

    def test_symbol_and_name(self, aapl_signal):
        assert aapl_signal["symbol"] == "AAPL"
        assert "Apple" in aapl_signal["name"]

    def test_price_is_positive_float(self, aapl_signal):
        assert isinstance(aapl_signal["price"], float)
        assert aapl_signal["price"] > 0

    def test_signal_is_one_of_the_three_valid_values(self, aapl_signal):
        assert aapl_signal["signal"] in {"BUY", "SELL", "HOLD"}

    def test_confidence_is_in_0_to_100_range(self, aapl_signal):
        assert 0.0 <= aapl_signal["confidence"] <= 100.0

    def test_rsi_14d_is_in_valid_rsi_range(self, aapl_signal):
        assert 0.0 <= aapl_signal["rsi_14d"] <= 100.0

    def test_rsi_30min_is_in_valid_rsi_range_or_none(self, aapl_signal):
        rsi_30min = aapl_signal["rsi_30min"]
        assert rsi_30min is None or 0.0 <= rsi_30min <= 100.0

    def test_macd_trend_is_bullish_or_bearish(self, aapl_signal):
        assert aapl_signal["macd_trend"] in {"bullish", "bearish"}

    def test_ma_crossover_is_golden_or_death_cross(self, aapl_signal):
        assert aapl_signal["ma_crossover"] in {"golden_cross", "death_cross"}

    def test_reason_is_a_nonempty_string_mentioning_the_indicators(self, aapl_signal):
        reason = aapl_signal["reason"]
        assert isinstance(reason, str) and reason
        assert "MACD" in reason

    def test_reason_matches_the_returned_signal_direction(self, aapl_signal):
        # sanity cross-check: if RSI fired, the reason text should say so
        rsi = aapl_signal["rsi_14d"]
        if rsi < ts.RSI_OVERSOLD:
            assert "oversold" in aapl_signal["reason"].lower()
        elif rsi > ts.RSI_OVERBOUGHT:
            assert "overbought" in aapl_signal["reason"].lower()


class TestCalculateSignalsErrorHandling:
    def test_invalid_symbol_returns_none(self):
        assert ts.calculate_signals(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        ts.calculate_signals.clear()
        monkeypatch.setattr(ts.yf, "Ticker", _boom_ticker)
        assert ts.calculate_signals("NETDOWN") is None

    def test_empty_history_returns_none(self, monkeypatch):
        import pandas as pd

        ts.calculate_signals.clear()

        class EmptyHistoryTicker:
            def __init__(self, *_a, **_kw):
                pass

            def history(self, *_a, **_kw):
                return pd.DataFrame()

        monkeypatch.setattr(ts.yf, "Ticker", EmptyHistoryTicker)
        assert ts.calculate_signals("EMPTYHIST") is None


# ---------------------------------------------------------------------------
# _score_signal — deterministic rule, proves the logic isn't random
# ---------------------------------------------------------------------------

class TestScoreSignalIsADeterministicRule:
    def test_textbook_buy_case_hits_high_confidence_buy(self):
        # exact task-spec BUY case: RSI oversold + bullish MACD + golden cross
        signal, confidence, reason = ts._score_signal(25.0, "bullish", "golden_cross", None)
        assert signal == "BUY"
        assert confidence == 100.0
        assert "oversold" in reason

    def test_textbook_sell_case_hits_high_confidence_sell(self):
        signal, confidence, reason = ts._score_signal(75.0, "bearish", "death_cross", None)
        assert signal == "SELL"
        assert confidence == 100.0
        assert "overbought" in reason

    def test_fully_neutral_case_is_hold_with_zero_confidence(self):
        # RSI neutral, MACD bearish (-1), MA golden cross (+1) -> net 0
        signal, confidence, _ = ts._score_signal(50.0, "bearish", "golden_cross", None)
        assert signal == "HOLD"
        assert confidence == 0.0

    def test_mixed_case_partial_agreement_is_hold_not_buy(self):
        # RSI oversold (+2) but MACD bearish (-1) and death cross (-1) -> net 0
        signal, confidence, _ = ts._score_signal(25.0, "bearish", "death_cross", None)
        assert signal == "HOLD"
        assert confidence == 0.0

    def test_same_inputs_always_produce_same_output(self):
        # determinism check: this is a rule, not a random draw
        results = {ts._score_signal(22.0, "bullish", "golden_cross", None) for _ in range(20)}
        assert len(results) == 1

    def test_result_depends_only_on_its_inputs_not_on_call_order(self):
        first = ts._score_signal(80.0, "bearish", "death_cross", None)
        ts._score_signal(20.0, "bullish", "golden_cross", None)  # unrelated call in between
        second = ts._score_signal(80.0, "bearish", "death_cross", None)
        assert first == second

    def test_volume_spike_is_mentioned_but_does_not_change_the_signal(self):
        without_spike = ts._score_signal(25.0, "bullish", "golden_cross", 1.0)
        with_spike = ts._score_signal(25.0, "bullish", "golden_cross", 3.0)
        assert without_spike[0] == with_spike[0]
        assert without_spike[1] == with_spike[1]
        assert "volume" not in without_spike[2].lower()
        assert "volume" in with_spike[2].lower()

    def test_macd_and_ma_agreement_alone_is_enough_to_buy(self):
        # Regression test for a real bug: net is always even (rsi_vote is
        # 0/±2, macd_vote+ma_vote is always -2/0/+2), so a threshold of 3
        # could only ever be crossed by |net|=4 (all three indicators
        # unanimous) -- net=2 (e.g. neutral RSI, MACD+MA agreeing) was
        # silently forced to HOLD even though two of three indicators
        # agreed. This is the exact combination that made real scans come
        # back all-HOLD: most symbols at any given moment have a neutral
        # RSI, so BUY/SELL effectively never fired.
        signal, confidence, _ = ts._score_signal(50.0, "bullish", "golden_cross", None)
        assert signal == "BUY"
        assert confidence == 50.0

    def test_macd_and_ma_agreement_alone_is_enough_to_sell(self):
        signal, confidence, _ = ts._score_signal(50.0, "bearish", "death_cross", None)
        assert signal == "SELL"
        assert confidence == 50.0

    def test_unanimous_agreement_still_scores_full_confidence(self):
        signal, confidence, _ = ts._score_signal(20.0, "bullish", "golden_cross", None)
        assert signal == "BUY"
        assert confidence == 100.0

    def test_neutral_rsi_reason_still_reports_the_real_value(self):
        # Regression test for a real bug: the reason string only mentioned
        # RSI when it was oversold/overbought, so every neutral-RSI symbol
        # (the common case) collapsed onto one of just 4 template strings
        # (bullish/bearish x golden/death cross) regardless of its actual
        # RSI value -- looking identical across dozens of real, different
        # symbols even though the underlying numbers genuinely differed.
        _, _, reason_a = ts._score_signal(45.0, "bullish", "golden_cross", None)
        _, _, reason_b = ts._score_signal(55.0, "bullish", "golden_cross", None)
        assert "45.0" in reason_a
        assert "55.0" in reason_b
        assert reason_a != reason_b


# ---------------------------------------------------------------------------
# scan_signals — real small batch
# ---------------------------------------------------------------------------

_SCAN_SYMBOLS = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA")


class TestScanSignals:
    def test_returns_a_row_per_resolved_symbol(self):
        rows = ts.scan_signals(_SCAN_SYMBOLS)
        assert len(rows) == len(_SCAN_SYMBOLS)
        assert {r["symbol"] for r in rows} == set(_SCAN_SYMBOLS)

    def test_sorted_by_confidence_descending(self):
        rows = ts.scan_signals(_SCAN_SYMBOLS)
        confidences = [r["confidence"] for r in rows]
        assert confidences == sorted(confidences, reverse=True)

    def test_rsi_30min_is_none_in_batch_results(self):
        # documented tradeoff: scan_signals skips the per-symbol intraday
        # fetch entirely to stay within the <10s/50-symbol budget, since
        # rsi_30min never feeds the signal decision anyway.
        rows = ts.scan_signals(_SCAN_SYMBOLS)
        assert all(r["rsi_30min"] is None for r in rows)

    def test_names_are_used_when_provided(self):
        # _names is deliberately excluded from the cache key (see
        # scan_signals' docstring), so a stale cached result from another
        # test using the same symbols tuple would otherwise mask this.
        ts.scan_signals.clear()
        rows = ts.scan_signals(_SCAN_SYMBOLS, _names={"AAPL": "Custom Apple Name"})
        aapl_row = next(r for r in rows if r["symbol"] == "AAPL")
        assert aapl_row["name"] == "Custom Apple Name"

    def test_missing_name_falls_back_to_symbol(self):
        ts.scan_signals.clear()
        rows = ts.scan_signals(_SCAN_SYMBOLS, _names={})
        aapl_row = next(r for r in rows if r["symbol"] == "AAPL")
        assert aapl_row["name"] == "AAPL"

    def test_empty_symbols_returns_empty_list(self):
        assert ts.scan_signals(()) == []

    def test_scans_fifty_symbols_reasonably_fast(self):
        symbols = (
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "TSLA", "AVGO", "LLY",
            "JPM", "V", "XOM", "UNH", "MA", "HD", "PG", "COST", "JNJ", "MRK",
            "ABBV", "WMT", "CRM", "BAC", "KO", "PEP", "ADBE", "NFLX", "CSCO", "TMO",
            "ACN", "ABT", "LIN", "ORCL", "MCD", "DHR", "WFC", "TXN", "NEE", "PM",
            "NKE", "UNP", "RTX", "BMY", "QCOM", "HON", "UPS", "LOW", "SBUX", "IBM",
        )
        assert len(symbols) == 50
        ts.scan_signals.clear()
        start = time.time()
        rows = ts.scan_signals(symbols)
        elapsed = time.time() - start
        assert len(rows) > 0
        # Generous ceiling: live Yahoo Finance latency varies run to run
        # (observed 5-16s across repeated trials during development, driven
        # by their server response time, not this function's own logic —
        # a single batched download is already the fastest shape available).
        assert elapsed < 30, f"scan_signals took {elapsed:.1f}s for 50 symbols"


class TestScanSignalsErrorHandling:
    def test_network_down_returns_empty_list_not_none(self, monkeypatch):
        # Batch failures degrade to "nothing scanned" rather than a hard
        # failure signal, since a partial scan is still useful.
        ts.scan_signals.clear()
        monkeypatch.setattr(ts.yf, "download", _boom_ticker)
        assert ts.scan_signals(("AAPL", "MSFT")) == []


# ---------------------------------------------------------------------------
# Signal history recording + accuracy — real current-price checks against a
# temp SQLite store (never the real one, so tests can't pollute real history)
# ---------------------------------------------------------------------------

def _isolated_store(tmp_path, monkeypatch):
    from src.signal_history import SignalHistoryStore

    store = SignalHistoryStore(db_path=tmp_path / "signals_test.db")
    monkeypatch.setattr(ts, "SignalHistoryStore", lambda: store)
    return store


class TestRecordScan:
    def test_records_only_actionable_rows(self, tmp_path, monkeypatch):
        store = _isolated_store(tmp_path, monkeypatch)
        recorded = ts.record_scan(
            [
                {"symbol": "AAPL", "signal": "BUY", "confidence": 80.0, "price": 300.0},
                {"symbol": "MSFT", "signal": "HOLD", "confidence": 10.0, "price": 400.0},
            ]
        )
        assert recorded == 1
        assert store.count() == 1


class TestComputeSignalAccuracy:
    def test_no_history_returns_none(self, tmp_path, monkeypatch):
        _isolated_store(tmp_path, monkeypatch)
        assert ts.compute_signal_accuracy() is None

    def test_too_recent_history_returns_none(self, tmp_path, monkeypatch):
        store = _isolated_store(tmp_path, monkeypatch)
        store.record([{"symbol": "AAPL", "signal": "BUY", "confidence": 80.0, "price": 300.0}])
        assert ts.compute_signal_accuracy(min_age_days=1) is None

    def test_evaluates_real_current_price_against_recorded_signal(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime, timedelta, timezone

        store = _isolated_store(tmp_path, monkeypatch)
        # Deliberately low price: real current AAPL price is virtually
        # guaranteed to be higher, so this BUY should evaluate as a hit.
        store.record([{"symbol": "AAPL", "signal": "BUY", "confidence": 80.0, "price": 1.0}])
        old_ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        with sqlite3.connect(store.db_path) as conn:
            conn.execute("UPDATE signal_history SET scanned_at = ?", (old_ts,))

        result = ts.compute_signal_accuracy(min_age_days=1)
        assert result is not None
        assert result["total_evaluated"] == 1
        assert result["buy_total"] == 1
        assert result["buy_hits"] == 1
        assert result["buy_hit_rate"] == 100.0
        assert result["hit_rate"] == 100.0

    def test_sell_signal_that_moved_the_wrong_way_is_a_miss(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime, timedelta, timezone

        store = _isolated_store(tmp_path, monkeypatch)
        # A SELL recorded at a price far below any real current AAPL price
        # should evaluate as a miss (price didn't fall below signal price).
        store.record([{"symbol": "AAPL", "signal": "SELL", "confidence": 60.0, "price": 1.0}])
        old_ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        with sqlite3.connect(store.db_path) as conn:
            conn.execute("UPDATE signal_history SET scanned_at = ?", (old_ts,))

        result = ts.compute_signal_accuracy(min_age_days=1)
        assert result["sell_total"] == 1
        assert result["sell_hits"] == 0
        assert result["sell_hit_rate"] == 0.0
