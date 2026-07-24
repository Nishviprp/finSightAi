"""
Historical backtest for the ensemble voting signal system in
src.trading_signals (_compute_ensemble_indicators / _ensemble_signal_from_votes).

Walks real daily OHLCV history day by day over an evaluation window,
computing indicators using only data up to and including that day (every
column in _compute_ensemble_indicators' output is inherently causal —
Bollinger Bands, StochRSI, EMA and the rolling volume average are all
trailing calculations), and grades each BUY/SELL signal by whether the
close price at *any* point in the next LOOKFORWARD_DAYS trading days moved
in the predicted direction versus the signal-day close. HOLD signals are
never graded — there is no direction to check them against, the same
convention src.signal_history.SignalHistoryStore already uses for the
classic signal system's real-world accuracy tracking.

What "accuracy" means here — and its real limits: a "hit" only requires
the closing price to have moved the right way, by any amount, at any of
the next 5 trading days. There's no slippage, spread, fees, position
sizing, or stop-loss modeling, and no requirement that the move persist.
This is a directional backtest of the indicator logic, not a simulation of
a tradable strategy — treat the accuracy numbers here as an optimistic
upper bound, not a promise of real trading performance. Nothing in this
module is investment advice.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from src.trading_signals import _compute_classic_indicators_series, _compute_ensemble_indicators, _ensemble_signal_from_votes, _score_signal

logger = logging.getLogger(__name__)

LOOKFORWARD_DAYS = 5
WARMUP_CALENDAR_DAYS = 130  # ~90 trading days -- comfortably past BB(20)/StochRSI/EMA(26) warmup
GRADING_BUFFER_CALENDAR_DAYS = 12  # comfortably past LOOKFORWARD_DAYS=5 trading days

# The four voting indicators, keyed to the boolean columns
# _compute_ensemble_indicators() produces -- used to build the
# indicator-importance report without duplicating that mapping everywhere.
VOTING_INDICATORS: dict[str, tuple[str, str]] = {
    "RSI": ("buy_rsi", "sell_rsi"),
    "Bollinger Bands": ("buy_bb", "sell_bb"),
    "Stochastic RSI crossover": ("stoch_bull_cross", "stoch_bear_cross"),
    "Volume spike": ("volume_spike", "volume_spike"),
}


def fetch_backtest_history(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV for *symbol* covering the evaluation window
    [*start*, *end*] (each "YYYY-MM-DD") plus enough buffer on both sides
    for indicator warmup and forward-looking grading.

    Returns the raw DataFrame (full buffer range, not sliced to the
    window — backtest_symbol does that after computing indicators over the
    whole thing) or None if nothing came back.
    """
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    buffer_end = (pd.Timestamp(end) + pd.Timedelta(days=GRADING_BUFFER_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker(symbol).history(start=warmup_start, end=buffer_end, interval="1d")
    except Exception as exc:
        logger.warning("Backtest history fetch failed for %s: %s", symbol, exc)
        return None
    if hist is None or hist.empty:
        return None
    return hist.dropna(subset=["Close", "High", "Low", "Volume"])


def backtest_symbol(symbol: str, start: str, end: str, threshold: int) -> Optional[dict]:
    """
    Backtest the ensemble signal for *symbol* over [*start*, *end*] at vote
    *threshold*.

    Returns None if history couldn't be fetched. Otherwise a dict:
      symbol, threshold, start, end,
      total_signals, buy_signals, sell_signals,
      correct, accuracy (None if total_signals == 0),
      buy_correct, buy_accuracy, sell_correct, sell_accuracy,
      indicator_hits: {indicator_name: {"fired": n, "correct": n, "hit_rate": pct}}
        -- among graded (BUY/SELL) signal-days where that indicator's vote
        fired, what fraction were correct. This is standalone per
        indicator (a day can have multiple indicators fire at once), so it
        answers "when this indicator votes, how often is the day's overall
        signal right" -- not an isolated single-indicator strategy.
      ema_standalone: {"bullish_up_rate": pct, "bearish_down_rate": pct, "bullish_days": n, "bearish_days": n}
        -- EMA(12/26) trend isn't part of the vote; evaluated separately:
        of days classified bullish, what fraction saw a higher close within
        LOOKFORWARD_DAYS (and the analogous check for bearish/lower).
      signals: [ {date, signal, buy_votes, sell_votes, correct}, ... ] graded rows only
    """
    daily = fetch_backtest_history(symbol, start, end)
    if daily is None or daily.empty:
        logger.warning("No backtest history for %s", symbol)
        return None

    indicators = _compute_ensemble_indicators(daily)
    if indicators is None or indicators.empty:
        return None

    close = indicators["close"]
    window_mask = (indicators.index >= pd.Timestamp(start, tz=indicators.index.tz)) & (
        indicators.index <= pd.Timestamp(end, tz=indicators.index.tz)
    )
    window_positions = [i for i, in_window in enumerate(window_mask) if in_window]

    graded_signals: list[dict] = []
    indicator_hits = {name: {"fired": 0, "correct": 0} for name in VOTING_INDICATORS}
    ema_stats = {"bullish_days": 0, "bullish_up": 0, "bearish_days": 0, "bearish_down": 0}

    n = len(indicators)
    for i in window_positions:
        if i + LOOKFORWARD_DAYS >= n:
            continue  # not enough real future data in the fetched range to grade this day

        row = indicators.iloc[i]
        if pd.isna(row["rsi"]) or pd.isna(row["stoch_k"]) or pd.isna(row["upper_bb"]) or pd.isna(row["ema_fast"]):
            continue  # still inside warmup for this particular symbol/date

        buy_votes, sell_votes = int(row["buy_votes"]), int(row["sell_votes"])
        signal, _confidence = _ensemble_signal_from_votes(buy_votes, sell_votes, threshold)

        entry_close = close.iloc[i]
        future_closes = close.iloc[i + 1 : i + 1 + LOOKFORWARD_DAYS]

        # EMA standalone check runs on every warmed-up day regardless of
        # whether the vote produced a BUY/SELL/HOLD -- it isn't a voting
        # indicator, so it's graded independently of the signal outcome.
        if bool(row["ema_bullish"]):
            ema_stats["bullish_days"] += 1
            if (future_closes > entry_close).any():
                ema_stats["bullish_up"] += 1
        else:
            ema_stats["bearish_days"] += 1
            if (future_closes < entry_close).any():
                ema_stats["bearish_down"] += 1

        if signal == "HOLD":
            continue

        correct = bool((future_closes > entry_close).any()) if signal == "BUY" else bool((future_closes < entry_close).any())

        for name, (buy_col, sell_col) in VOTING_INDICATORS.items():
            fired = bool(row[buy_col]) if signal == "BUY" else bool(row[sell_col])
            if fired:
                indicator_hits[name]["fired"] += 1
                if correct:
                    indicator_hits[name]["correct"] += 1

        graded_signals.append(
            {
                "date": indicators.index[i].strftime("%Y-%m-%d"),
                "signal": signal,
                "buy_votes": buy_votes,
                "sell_votes": sell_votes,
                "entry_close": float(entry_close),
                "correct": correct,
            }
        )

    total = len(graded_signals)
    buy_rows = [s for s in graded_signals if s["signal"] == "BUY"]
    sell_rows = [s for s in graded_signals if s["signal"] == "SELL"]
    correct_total = sum(1 for s in graded_signals if s["correct"])
    buy_correct = sum(1 for s in buy_rows if s["correct"])
    sell_correct = sum(1 for s in sell_rows if s["correct"])

    indicator_report = {}
    for name, stats in indicator_hits.items():
        fired = stats["fired"]
        indicator_report[name] = {
            "fired": fired,
            "correct": stats["correct"],
            "hit_rate": round(stats["correct"] / fired * 100, 1) if fired else None,
        }

    return {
        "symbol": symbol.upper(),
        "threshold": threshold,
        "start": start,
        "end": end,
        "total_signals": total,
        "buy_signals": len(buy_rows),
        "sell_signals": len(sell_rows),
        "correct": correct_total,
        "accuracy": round(correct_total / total * 100, 1) if total else None,
        "buy_correct": buy_correct,
        "buy_accuracy": round(buy_correct / len(buy_rows) * 100, 1) if buy_rows else None,
        "sell_correct": sell_correct,
        "sell_accuracy": round(sell_correct / len(sell_rows) * 100, 1) if sell_rows else None,
        "indicator_hits": indicator_report,
        "ema_standalone": {
            "bullish_days": ema_stats["bullish_days"],
            "bullish_up_rate": round(ema_stats["bullish_up"] / ema_stats["bullish_days"] * 100, 1) if ema_stats["bullish_days"] else None,
            "bearish_days": ema_stats["bearish_days"],
            "bearish_down_rate": round(ema_stats["bearish_down"] / ema_stats["bearish_days"] * 100, 1) if ema_stats["bearish_days"] else None,
        },
        "signals": graded_signals,
    }


def backtest_classic_symbol(symbol: str, start: str, end: str) -> Optional[dict]:
    """
    Same methodology as backtest_symbol(), but grades the *classic*
    calculate_signals()/_score_signal() weighted-vote system (RSI/MACD/MA,
    fixed internal threshold) instead of the ensemble — for a direct
    before/after comparison using identical data windows and identical
    grading (any close in the next LOOKFORWARD_DAYS trading days moved the
    predicted direction vs the signal-day close).

    Returns the same shape as backtest_symbol() minus indicator_hits/
    ema_standalone (those are ensemble-specific).
    """
    daily = fetch_backtest_history(symbol, start, end)
    if daily is None or daily.empty:
        return None

    indicators = _compute_classic_indicators_series(daily)
    if indicators is None or indicators.empty:
        return None

    close = indicators["close"]
    window_mask = (indicators.index >= pd.Timestamp(start, tz=indicators.index.tz)) & (
        indicators.index <= pd.Timestamp(end, tz=indicators.index.tz)
    )
    window_positions = [i for i, in_window in enumerate(window_mask) if in_window]

    graded_signals: list[dict] = []
    n = len(indicators)
    for i in window_positions:
        if i + LOOKFORWARD_DAYS >= n:
            continue
        row = indicators.iloc[i]
        if not bool(row["_valid"]):
            continue

        signal, _confidence, _reason = _score_signal(
            float(row["rsi_14d"]), row["macd_trend"], row["ma_crossover"],
            float(row["volume_ratio"]) if not pd.isna(row["volume_ratio"]) else None,
        )
        if signal == "HOLD":
            continue

        entry_close = close.iloc[i]
        future_closes = close.iloc[i + 1 : i + 1 + LOOKFORWARD_DAYS]
        correct = bool((future_closes > entry_close).any()) if signal == "BUY" else bool((future_closes < entry_close).any())

        graded_signals.append(
            {"date": indicators.index[i].strftime("%Y-%m-%d"), "signal": signal, "entry_close": float(entry_close), "correct": correct}
        )

    total = len(graded_signals)
    buy_rows = [s for s in graded_signals if s["signal"] == "BUY"]
    sell_rows = [s for s in graded_signals if s["signal"] == "SELL"]
    correct_total = sum(1 for s in graded_signals if s["correct"])
    buy_correct = sum(1 for s in buy_rows if s["correct"])
    sell_correct = sum(1 for s in sell_rows if s["correct"])

    return {
        "symbol": symbol.upper(),
        "start": start,
        "end": end,
        "total_signals": total,
        "buy_signals": len(buy_rows),
        "sell_signals": len(sell_rows),
        "correct": correct_total,
        "accuracy": round(correct_total / total * 100, 1) if total else None,
        "buy_correct": buy_correct,
        "buy_accuracy": round(buy_correct / len(buy_rows) * 100, 1) if buy_rows else None,
        "sell_correct": sell_correct,
        "sell_accuracy": round(sell_correct / len(sell_rows) * 100, 1) if sell_rows else None,
        "signals": graded_signals,
    }


def backtest_classic_many(symbols: tuple[str, ...], start: str, end: str) -> list[dict]:
    results = []
    for symbol in symbols:
        result = backtest_classic_symbol(symbol, start, end)
        if result is not None:
            results.append(result)
    return results


def backtest_many(symbols: tuple[str, ...], start: str, end: str, threshold: int) -> list[dict]:
    """backtest_symbol() for each symbol; symbols that fail to resolve are omitted."""
    results = []
    for symbol in symbols:
        result = backtest_symbol(symbol, start, end, threshold)
        if result is not None:
            results.append(result)
    return results


def aggregate_accuracy(results: list[dict]) -> Optional[dict]:
    """Pool total/correct signal counts across multiple backtest_symbol()
    results into one overall accuracy figure. Returns None if there were no
    graded signals across any symbol."""
    total = sum(r["total_signals"] for r in results)
    correct = sum(r["correct"] for r in results)
    if total == 0:
        return None
    buy_total = sum(r["buy_signals"] for r in results)
    buy_correct = sum(r["buy_correct"] for r in results)
    sell_total = sum(r["sell_signals"] for r in results)
    sell_correct = sum(r["sell_correct"] for r in results)
    return {
        "total_signals": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1),
        "buy_accuracy": round(buy_correct / buy_total * 100, 1) if buy_total else None,
        "sell_accuracy": round(sell_correct / sell_total * 100, 1) if sell_total else None,
    }
