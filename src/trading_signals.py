"""
Rule-based technical-indicator signals (RSI, MACD, moving-average crossover)
for a single ticker or a batch of tickers.

IMPORTANT — what this actually is: despite "AI Trading Signals" branding
elsewhere in this project, there is no machine learning here. This is
classical, decades-old technical analysis — three well-known indicators
combined by a simple, fully-documented weighted-vote rule (see
_score_signal below). `confidence` is that vote's strength as a percentage
of the maximum possible agreement, not a backtested statistical probability
or a prediction of future returns. Technical indicators computed from
public price history are not investment advice, and nothing in this module
should be read as a recommendation to buy or sell any security.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from src.signal_history import SignalHistoryStore

logger = logging.getLogger(__name__)

DAILY_HISTORY_PERIOD = "6mo"
INTRADAY_HISTORY_PERIOD = "5d"
INTRADAY_INTERVAL = "30m"

RSI_LENGTH = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_SHORT, MA_LONG = 20, 50
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_RATIO = 2.0

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Vote weights: RSI carries double weight since the task's own BUY/SELL
# rules treat an RSI extreme as the primary trigger, with MACD/MA as
# confirmation. Max possible |net| = 2 (RSI) + 1 (MACD) + 1 (MA) = 4.
_RSI_WEIGHT = 2
_MACD_WEIGHT = 1
_MA_WEIGHT = 1
_MAX_SCORE = _RSI_WEIGHT + _MACD_WEIGHT + _MA_WEIGHT
_STRONG_THRESHOLD = 3  # |net| >= 3 -> BUY/SELL; otherwise HOLD

MIN_DAILY_HISTORY_POINTS = MA_LONG + MACD_SLOW  # enough for MA50 and MACD(26) to be past warmup


# ---------------------------------------------------------------------------
# Manual technical indicators — pure pandas, no external TA library.
#
# pandas_ta was removed for Streamlit Cloud Python 3.14 compatibility (see
# git history), but the calls into it (`ta.rsi`, `ta.macd`, `ta.sma`) were
# left in place with `ta` no longer imported — every indicator computation
# below was silently failing with a NameError caught by the broad
# `except Exception` in _compute_daily_indicators, so calculate_signals()
# and scan_signals() returned None/empty for everything. These four
# functions replace that dependency entirely.
# ---------------------------------------------------------------------------

def _as_series(prices) -> pd.Series:
    return prices if isinstance(prices, pd.Series) else pd.Series(list(prices), dtype=float)


def calculate_sma(prices, period: int) -> pd.Series:
    """Simple moving average over *period* bars. Returns a Series aligned
    to *prices* (leading `period - 1` entries are NaN)."""
    return _as_series(prices).rolling(window=period).mean()


def calculate_ema(prices, period: int) -> pd.Series:
    """Exponential moving average with span=*period* (the standard
    definition: alpha = 2 / (period + 1)). Returns a Series aligned to
    *prices*."""
    return _as_series(prices).ewm(span=period, adjust=False).mean()


def calculate_rsi(prices, period: int = RSI_LENGTH) -> Optional[float]:
    """
    Wilder's RSI: the original, textbook formula — average gain/loss over
    the first *period* bars seeded as a plain average, then smoothed
    recursively thereafter (avg[t] = (avg[t-1] * (period-1) + value[t]) /
    period). This is deliberately not `ewm(adjust=False)` on its own,
    which bootstraps its recursion from the very first delta instead of an
    initial *period*-bar average — close enough once a series has run for
    many periods, but measurably off Wilder's real definition on shorter
    ones.

    Returns the latest RSI value (0-100), or None if there isn't at least
    *period* + 1 usable price points.
    """
    series = _as_series(prices)
    if len(series) < period + 1:
        return None

    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    for i in range(period, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    latest_gain = avg_gain.iloc[-1]
    latest_loss = avg_loss.iloc[-1]
    if pd.isna(latest_gain) or pd.isna(latest_loss):
        return None
    if latest_loss == 0:
        return 100.0

    rs = latest_gain / latest_loss
    return float(100 - (100 / (1 + rs)))


def calculate_macd(
    prices, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL
) -> Optional[tuple[float, float, float]]:
    """
    Standard MACD: MACD line = EMA(fast) - EMA(slow), signal line =
    EMA(signal) of the MACD line, histogram = MACD - signal.

    Returns the latest (macd_value, signal_value, histogram), or None if
    there isn't enough history for the signal line to be past warmup.
    """
    series = _as_series(prices)
    if series.empty:
        return None

    macd_line = calculate_ema(series, fast) - calculate_ema(series, slow)
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_hist = histogram.iloc[-1]
    if pd.isna(latest_macd) or pd.isna(latest_signal) or pd.isna(latest_hist):
        return None

    return float(latest_macd), float(latest_signal), float(latest_hist)


@st.cache_data(ttl=14400, show_spinner=False)
def calculate_signals(symbol: str) -> Optional[dict]:
    """
    Compute a rule-based technical signal for *symbol*.

    Returns a dict:
        symbol, name, price
        signal          – "BUY" | "SELL" | "HOLD"
        confidence      – float 0-100: |vote total| as a percentage of the
                           maximum possible agreement across RSI/MACD/MA —
                           see the module docstring for what this is (and
                           isn't).
        rsi_30min       – RSI(14) on 30-minute bars (most recent), or None
                           if intraday data wasn't available
        rsi_14d         – RSI(14) on daily closes
        macd_trend      – "bullish" | "bearish"
        ma_crossover    – "golden_cross" | "death_cross"
        volume_ratio    – latest volume / 20-day average volume
        reason          – human-readable summary of which indicators fired

    Returns None on any failure (invalid symbol, network error, insufficient
    history) — never raises.
    """
    try:
        daily = _fetch_daily_ohlcv(symbol)
        if daily is None or len(daily) < MIN_DAILY_HISTORY_POINTS:
            logger.warning("Not enough daily history for %s to compute signals", symbol)
            return None

        indicators = _compute_daily_indicators(daily)
        if indicators is None:
            return None

        rsi_30min = _fetch_intraday_rsi(symbol)

        signal, confidence, reason = _score_signal(
            indicators["rsi_14d"], indicators["macd_trend"],
            indicators["ma_crossover"], indicators["volume_ratio"],
        )

        return {
            "symbol": symbol.upper(),
            "name": _resolve_name(symbol),
            "price": indicators["price"],
            "signal": signal,
            "confidence": confidence,
            "rsi_30min": rsi_30min,
            "rsi_14d": indicators["rsi_14d"],
            "macd_trend": indicators["macd_trend"],
            "ma_crossover": indicators["ma_crossover"],
            "volume_ratio": indicators["volume_ratio"],
            "reason": reason,
        }
    except Exception as exc:
        logger.warning("calculate_signals failed for %s: %s", symbol, exc)
        return None


@st.cache_data(ttl=14400, show_spinner=False)
def scan_signals(symbols: tuple[str, ...], _names: Optional[dict[str, str]] = None) -> list[dict]:
    """
    Compute signals for many symbols at once via one batched daily-OHLCV
    yfinance download rather than one call per symbol.

    Unlike calculate_signals(), this intentionally skips the per-symbol
    intraday fetch and always returns rsi_30min=None. Measured live: a
    second batched download (30-min bars, 50 symbols) added ~5-6s on top of
    the ~5s daily download, pushing a 50-symbol scan to ~11-12s against the
    <10s target — and rsi_30min never feeds the BUY/SELL/HOLD decision in
    the first place (see _score_signal: only rsi_14d, macd_trend, and
    ma_crossover — all daily-timeframe — vote). Trading that purely
    informational field for a real, verified 50-symbol scan under budget
    is a correctness-preserving tradeoff, not a shortcut on the signal
    itself. Call calculate_signals() for a single symbol if you need its
    real intraday RSI.

    *_names* is an optional {symbol: company_name} map — pass the names
    already on hand from whatever screener produced *symbols* (the leading
    underscore excludes it from Streamlit's cache key, since it's just a
    display convenience, not part of what should invalidate the cache).
    Resolving names here instead, one yfinance .info call per symbol, would
    on its own blow the time budget for no computational reason — the
    caller already has this data. Symbols missing from *_names* just
    display as their symbol.

    Returns rows (same schema as calculate_signals()) sorted by confidence,
    highest first. Symbols that fail to resolve are simply omitted, not a
    hard failure of the whole scan — an empty list means nothing scanned
    successfully, not "the market is down."
    """
    if not symbols:
        return []

    names = _names or {}
    daily_batch = _fetch_daily_ohlcv_batch(symbols)

    rows: list[dict] = []
    for symbol in symbols:
        daily = daily_batch.get(symbol)
        if daily is None or len(daily) < MIN_DAILY_HISTORY_POINTS:
            continue

        indicators = _compute_daily_indicators(daily)
        if indicators is None:
            continue

        signal, confidence, reason = _score_signal(
            indicators["rsi_14d"], indicators["macd_trend"],
            indicators["ma_crossover"], indicators["volume_ratio"],
        )

        rows.append(
            {
                "symbol": symbol.upper(),
                "name": names.get(symbol, symbol.upper()),
                "price": indicators["price"],
                "signal": signal,
                "confidence": confidence,
                "rsi_30min": None,
                "rsi_14d": indicators["rsi_14d"],
                "macd_trend": indicators["macd_trend"],
                "ma_crossover": indicators["ma_crossover"],
                "volume_ratio": indicators["volume_ratio"],
                "reason": reason,
            }
        )

    rows.sort(key=lambda r: r["confidence"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def _compute_daily_indicators(daily: pd.DataFrame) -> Optional[dict]:
    try:
        close = daily["Close"]
        volume = daily["Volume"]

        rsi_latest = calculate_rsi(close, RSI_LENGTH)
        macd_result = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        if rsi_latest is None or macd_result is None:
            return None
        macd_line, macd_signal_line, _histogram = macd_result

        ma_short = calculate_sma(close, MA_SHORT)
        ma_long = calculate_sma(close, MA_LONG)
        avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()

        ma_short_latest = ma_short.iloc[-1]
        ma_long_latest = ma_long.iloc[-1]
        latest_volume = volume.iloc[-1]
        avg_volume_latest = avg_volume.iloc[-1]

        if any(pd.isna(v) for v in (ma_short_latest, ma_long_latest)):
            return None

        volume_ratio = (
            float(latest_volume / avg_volume_latest)
            if avg_volume_latest and not pd.isna(avg_volume_latest) and avg_volume_latest > 0
            else None
        )

        return {
            "price": float(close.iloc[-1]),
            "rsi_14d": rsi_latest,
            "macd_trend": "bullish" if macd_line > macd_signal_line else "bearish",
            "ma_crossover": "golden_cross" if ma_short_latest > ma_long_latest else "death_cross",
            "volume_ratio": volume_ratio,
        }
    except Exception as exc:
        logger.warning("Indicator computation failed: %s", exc)
        return None


def _score_signal(
    rsi_14d: float, macd_trend: str, ma_crossover: str, volume_ratio: Optional[float]
) -> tuple[str, float, str]:
    """Weighted-vote rule described in the module docstring. Returns
    (signal, confidence 0-100, human-readable reason).
    """
    reasons: list[str] = []

    if rsi_14d < RSI_OVERSOLD:
        rsi_vote = _RSI_WEIGHT
        reasons.append(f"RSI oversold ({rsi_14d:.1f})")
    elif rsi_14d > RSI_OVERBOUGHT:
        rsi_vote = -_RSI_WEIGHT
        reasons.append(f"RSI overbought ({rsi_14d:.1f})")
    else:
        rsi_vote = 0

    macd_vote = _MACD_WEIGHT if macd_trend == "bullish" else -_MACD_WEIGHT
    reasons.append(f"{macd_trend} MACD crossover")

    ma_vote = _MA_WEIGHT if ma_crossover == "golden_cross" else -_MA_WEIGHT
    reasons.append("golden cross (MA20 > MA50)" if ma_crossover == "golden_cross" else "death cross (MA20 < MA50)")

    if volume_ratio is not None and volume_ratio >= VOLUME_SPIKE_RATIO:
        reasons.append(f"unusual volume ({volume_ratio:.1f}x average)")

    net = rsi_vote + macd_vote + ma_vote
    confidence = round(abs(net) / _MAX_SCORE * 100, 1)

    if net >= _STRONG_THRESHOLD:
        signal = "BUY"
    elif net <= -_STRONG_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, confidence, " + ".join(reasons)


# ---------------------------------------------------------------------------
# Data fetch: single-symbol
# ---------------------------------------------------------------------------

def _fetch_daily_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    hist = yf.Ticker(symbol).history(period=DAILY_HISTORY_PERIOD, interval="1d")
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    return hist.dropna(subset=["Close", "Volume"])


def _fetch_intraday_rsi(symbol: str) -> Optional[float]:
    try:
        hist = yf.Ticker(symbol).history(period=INTRADAY_HISTORY_PERIOD, interval=INTRADAY_INTERVAL)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        return calculate_rsi(hist["Close"].dropna(), RSI_LENGTH)
    except Exception as exc:
        logger.warning("Intraday RSI fetch failed for %s: %s", symbol, exc)
        return None


def _resolve_name(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).info
        return (info.get("longName") or info.get("shortName") or symbol.upper()) if info else symbol.upper()
    except Exception:
        return symbol.upper()


# ---------------------------------------------------------------------------
# Data fetch: batched (used by scan_signals for the <10s/50-symbol target)
# ---------------------------------------------------------------------------

def _fetch_daily_ohlcv_batch(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    try:
        data = yf.download(
            list(symbols), period=DAILY_HISTORY_PERIOD, interval="1d",
            group_by="ticker", progress=False, auto_adjust=True, threads=True,
        )
    except Exception as exc:
        logger.warning("Batched daily OHLCV download failed for %d symbols: %s", len(symbols), exc)
        return {}

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            # group_by="ticker" always nests columns under the ticker,
            # even for a single-symbol request — no special-casing needed.
            df = data[symbol]
            df = df.dropna(subset=["Close", "Volume"])
            if not df.empty:
                result[symbol] = df
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Signal history / accuracy tracking
# ---------------------------------------------------------------------------

MIN_EVALUATION_AGE_DAYS = 1


def record_scan(rows: list[dict]) -> int:
    """Persist BUY/SELL rows from a scan to the local signal-history log
    (HOLD rows are skipped — there's no direction to later check against).
    Returns how many rows were recorded.
    """
    try:
        return SignalHistoryStore().record(rows)
    except Exception as exc:
        logger.warning("Failed to record signal history: %s", exc)
        return 0


def compute_signal_accuracy(min_age_days: int = MIN_EVALUATION_AGE_DAYS) -> Optional[dict]:
    """
    Evaluate past BUY/SELL signals (at least *min_age_days* old) against
    what actually happened to price since.

    A BUY "hits" if the current price is above the price recorded at signal
    time; a SELL "hits" if it's below. This is a directional check only —
    _score_signal has no concept of a specific price target, so "hit" means
    "moved the predicted direction by any amount," not "reached a target."

    Returns None if there's no evaluable history yet — the honest starting
    state for a feature with no track record until the app has actually
    been used for at least *min_age_days* — not an error.
    """
    try:
        evaluable = SignalHistoryStore().get_evaluable(min_age_days=min_age_days)
    except Exception as exc:
        logger.warning("Failed to read signal history: %s", exc)
        return None

    if not evaluable:
        return None

    symbols = tuple(sorted({row["symbol"] for row in evaluable}))
    current_prices = _fetch_current_prices(symbols)

    buy_total = buy_hits = sell_total = sell_hits = 0
    for row in evaluable:
        current = current_prices.get(row["symbol"])
        if current is None:
            continue
        if row["signal"] == "BUY":
            buy_total += 1
            if current > row["price_at_signal"]:
                buy_hits += 1
        elif row["signal"] == "SELL":
            sell_total += 1
            if current < row["price_at_signal"]:
                sell_hits += 1

    total = buy_total + sell_total
    if total == 0:
        return None

    return {
        "total_evaluated": total,
        "hit_rate": round((buy_hits + sell_hits) / total * 100, 1),
        "buy_total": buy_total,
        "buy_hits": buy_hits,
        "buy_hit_rate": round(buy_hits / buy_total * 100, 1) if buy_total else None,
        "sell_total": sell_total,
        "sell_hits": sell_hits,
        "sell_hit_rate": round(sell_hits / sell_total * 100, 1) if sell_total else None,
    }


def _fetch_current_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    """Latest close price per symbol, via one batched download."""
    if not symbols:
        return {}
    try:
        data = yf.download(
            list(symbols), period="5d", interval="1d",
            group_by="ticker", progress=False, auto_adjust=True, threads=True,
        )
    except Exception as exc:
        logger.warning("Current-price batch fetch failed for %d symbols: %s", len(symbols), exc)
        return {}

    result: dict[str, float] = {}
    for symbol in symbols:
        try:
            # group_by="ticker" always nests columns under the ticker,
            # even for a single-symbol request — no special-casing needed.
            df = data[symbol]
            close = df["Close"].dropna()
            if not close.empty:
                result[symbol] = float(close.iloc[-1])
        except Exception:
            continue
    return result
