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

# |net| >= 2 -> BUY/SELL; otherwise HOLD.
#
# Must be even, not odd: rsi_vote is always 0 or ±2 (even), and
# macd_vote + ma_vote is always -2, 0, or +2 (sum of two ±1's is always
# even) — so net = rsi_vote + macd_vote + ma_vote is *always* even,
# landing in {-4, -2, 0, 2, 4}. A threshold of 3 (the original value
# here) sits between the achievable 2 and 4, so in practice it could
# only ever be crossed by |net| = 4 — i.e. it silently demanded all
# three indicators unanimously agree before calling BUY/SELL, not "RSI
# extreme plus one confirming indicator" as the weights above intend.
# That confluence is rare enough that real scans came back all-HOLD.
# Threshold = 2 restores the intended rule: an RSI extreme alone (if
# MACD/MA cancel out), or MACD+MA agreeing even without an RSI extreme,
# is enough to call a direction; full 3-way agreement (|net| = 4) still
# scores the maximum 100% confidence.
_STRONG_THRESHOLD = 2

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


def calculate_rsi_series(prices, period: int = RSI_LENGTH) -> pd.Series:
    """
    Full Wilder's RSI series, aligned index-for-index to *prices* (not just
    the latest value — calculate_rsi() below is now a thin wrapper around
    this). Needed by calculate_stochastic_rsi() (StochRSI is literally the
    Stochastic oscillator applied to a rolling window of RSI *values*, not
    price) and by the backtest engine, which computes every indicator once,
    vectorized, over the whole history rather than recomputing from scratch
    at each historical day.

    Every value at row i depends only on rows <= i (Wilder's recursion is
    inherently causal) — safe to use directly in a no-look-ahead backtest.

    Leading entries are NaN until *period* + 1 real price points have
    accumulated, same warmup as the scalar calculate_rsi().
    """
    series = _as_series(prices)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    # avg_gain/avg_loss are NaN until index `period` (delta's own index 0 is
    # itself NaN, so the first fully-populated `period`-window average
    # lands at index `period`, not `period - 1`).
    for i in range(period + 1, len(series)):
        if pd.isna(avg_gain.iloc[i - 1]):
            continue
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi


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

    latest = calculate_rsi_series(series, period).iloc[-1]
    return None if pd.isna(latest) else float(latest)


def calculate_bollinger_bands(
    prices, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: a *period*-bar SMA (the middle band) plus/minus
    *num_std* rolling standard deviations.

    Returns (upper, middle, lower) as three Series aligned to *prices*.
    Leading `period - 1` entries are NaN.
    """
    series = _as_series(prices)
    middle = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def calculate_stochastic_rsi(
    prices,
    rsi_period: int = RSI_LENGTH,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI: the Stochastic oscillator's %K/%D formula applied to a
    rolling window of RSI *values* (not price) — distinct from, and more
    sensitive than, the plain price-based Stochastic oscillator.

    %K (raw) = (RSI - rolling_min(RSI, stoch_period)) /
               (rolling_max(RSI, stoch_period) - rolling_min(RSI, stoch_period)) * 100,
    then %K = SMA(%K_raw, k_smooth) and %D = SMA(%K, d_smooth) — standard
    smoothing (defaults 3/3) so %K/%D aren't too noisy to cross meaningfully.

    Returns (%K, %D) as two Series aligned to *prices*.
    """
    rsi = calculate_rsi_series(prices, rsi_period)
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    span = (rsi_max - rsi_min).replace(0, float("nan"))  # flat RSI window -> undefined, not divide-by-zero
    stoch_rsi_raw = (rsi - rsi_min) / span * 100

    k = stoch_rsi_raw.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d


def calculate_volume_ratio_series(volume, period: int = VOLUME_LOOKBACK) -> pd.Series:
    """Latest volume / trailing *period*-bar average volume, as a full
    series aligned to *volume* — the series form of the volume_ratio field
    _compute_daily_indicators() already returns as a scalar."""
    series = _as_series(volume)
    avg_volume = series.rolling(window=period).mean()
    return series / avg_volume


def _crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    """True at index i iff *a* was <= *b* at i-1 and is > *b* at i — a
    fresh upward crossing, not just "currently above" (which would also
    flag every day *after* an old crossover, not the crossover event
    itself)."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


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


# ---------------------------------------------------------------------------
# Ensemble voting system — a second, independent signal alongside the
# classic RSI/MACD/MA one above (calculate_signals() / scan_signals() are
# untouched; this is calculate_signals_ensemble() / scan_signals_ensemble()).
#
# Four indicators vote, each contributing at most one point toward BUY and
# one toward SELL for a given day:
#   - RSI(14) < 35 (oversold)                       -> BUY vote
#     RSI(14) > 65 (overbought)                      -> SELL vote
#   - Close below the lower Bollinger Band            -> BUY vote
#     Close above the upper Bollinger Band            -> SELL vote
#   - Stochastic RSI %K crosses above %D              -> BUY vote
#     Stochastic RSI %K crosses below %D              -> SELL vote
#   - Volume spike (>= 2x the 20-day average)          -> BUY *and* SELL vote
#     (a volume spike has no direction of its own — it's a conviction
#     multiplier that confirms whichever move is already happening, exactly
#     as the task spec lists it identically in both the BUY and SELL
#     condition sets)
# BUY fires when buy_votes >= threshold; SELL when sell_votes >= threshold;
# otherwise HOLD. Each indicator's own buy/sell pair is mutually exclusive
# (RSI can't be simultaneously <35 and >65, etc.), but the four indicators
# are computed independently of each other and can genuinely disagree —
# e.g. RSI and Bollinger Bands both read overbought while Stochastic RSI
# flashes a bullish crossover the same day (a real pattern seen in the
# backtest below, not a hypothetical). So buy_votes and sell_votes *can*
# both reach the threshold on the same day; the tie-break here (higher
# vote count wins, BUY wins an exact tie) is a real, reachable code path.
#
# EMA(12) vs EMA(26) crossover is also computed and reported (per the task's
# Step 1 ask to add it) but is deliberately NOT part of the vote tally —
# the task's own Step 2 vote list names only the four indicators above. Its
# standalone predictive value is evaluated separately by the backtest
# engine's indicator-importance report.
# ---------------------------------------------------------------------------

ENSEMBLE_RSI_OVERSOLD = 35
ENSEMBLE_RSI_OVERBOUGHT = 65
BB_PERIOD = 20
BB_NUM_STD = 2.0
STOCH_PERIOD = 14
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3
ENSEMBLE_EMA_FAST, ENSEMBLE_EMA_SLOW = 12, 26
ENSEMBLE_VOTE_THRESHOLD = 3  # default; the backtest also sweeps 2 and 4
ENSEMBLE_MAX_VOTES = 4

ENSEMBLE_MIN_HISTORY_POINTS = max(BB_PERIOD, STOCH_PERIOD * 2 + STOCH_K_SMOOTH + STOCH_D_SMOOTH, ENSEMBLE_EMA_SLOW) + 10


def _compute_ensemble_indicators(daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Every ensemble indicator and vote, computed once, vectorized, over the
    *entire* history in *daily* — aligned index-for-index to it. Every
    column's value at row i depends only on rows <= i (Bollinger Bands,
    StochRSI, EMA and the rolling volume average are all inherently
    causal/trailing), which is what makes this table safe to reuse as-is
    for both the live "latest row" signal (calculate_signals_ensemble) and
    the backtest engine (which walks every historical row and must never
    peek ahead).

    Returns None if *daily* is missing required OHLCV columns.
    """
    try:
        close, high, low, volume = daily["Close"], daily["High"], daily["Low"], daily["Volume"]
    except KeyError as exc:
        logger.warning("Ensemble indicators missing required column: %s", exc)
        return None

    rsi = calculate_rsi_series(close, RSI_LENGTH)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close, BB_PERIOD, BB_NUM_STD)
    stoch_k, stoch_d = calculate_stochastic_rsi(close, RSI_LENGTH, STOCH_PERIOD, STOCH_K_SMOOTH, STOCH_D_SMOOTH)
    volume_ratio = calculate_volume_ratio_series(volume, VOLUME_LOOKBACK)
    ema_fast = calculate_ema(close, ENSEMBLE_EMA_FAST)
    ema_slow = calculate_ema(close, ENSEMBLE_EMA_SLOW)

    buy_rsi = rsi < ENSEMBLE_RSI_OVERSOLD
    sell_rsi = rsi > ENSEMBLE_RSI_OVERBOUGHT
    buy_bb = close < lower_bb
    sell_bb = close > upper_bb
    stoch_bull_cross = _crossed_above(stoch_k, stoch_d)
    stoch_bear_cross = _crossed_below(stoch_k, stoch_d)
    volume_spike = volume_ratio >= VOLUME_SPIKE_RATIO

    buy_votes = buy_rsi.astype(int) + buy_bb.astype(int) + stoch_bull_cross.astype(int) + volume_spike.astype(int)
    sell_votes = sell_rsi.astype(int) + sell_bb.astype(int) + stoch_bear_cross.astype(int) + volume_spike.astype(int)

    return pd.DataFrame(
        {
            "close": close,
            "rsi": rsi,
            "upper_bb": upper_bb,
            "middle_bb": middle_bb,
            "lower_bb": lower_bb,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "volume_ratio": volume_ratio,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_bullish": ema_fast > ema_slow,
            "buy_rsi": buy_rsi,
            "sell_rsi": sell_rsi,
            "buy_bb": buy_bb,
            "sell_bb": sell_bb,
            "stoch_bull_cross": stoch_bull_cross,
            "stoch_bear_cross": stoch_bear_cross,
            "volume_spike": volume_spike,
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
        }
    )


def _ensemble_signal_from_votes(buy_votes: int, sell_votes: int, threshold: int) -> tuple[str, float]:
    """Returns (signal, confidence 0-100). confidence is the winning side's
    vote count as a percentage of ENSEMBLE_MAX_VOTES — analogous to the
    classic system's confidence, not a backtested probability."""
    if buy_votes >= threshold and buy_votes >= sell_votes:
        return "BUY", round(buy_votes / ENSEMBLE_MAX_VOTES * 100, 1)
    if sell_votes >= threshold and sell_votes > buy_votes:
        return "SELL", round(sell_votes / ENSEMBLE_MAX_VOTES * 100, 1)
    return "HOLD", round(max(buy_votes, sell_votes) / ENSEMBLE_MAX_VOTES * 100, 1)


def _build_ensemble_reason(row: pd.Series) -> str:
    parts: list[str] = []
    if row["buy_rsi"]:
        parts.append(f"RSI oversold ({row['rsi']:.1f})")
    elif row["sell_rsi"]:
        parts.append(f"RSI overbought ({row['rsi']:.1f})")
    else:
        parts.append(f"RSI neutral ({row['rsi']:.1f})")

    if row["buy_bb"]:
        parts.append("price below lower Bollinger Band")
    elif row["sell_bb"]:
        parts.append("price above upper Bollinger Band")
    else:
        parts.append("price within Bollinger Bands")

    if row["stoch_bull_cross"]:
        parts.append("StochRSI bullish crossover")
    elif row["stoch_bear_cross"]:
        parts.append("StochRSI bearish crossover")
    else:
        parts.append(f"StochRSI no fresh crossover (K={row['stoch_k']:.1f}, D={row['stoch_d']:.1f})")

    if row["volume_spike"]:
        parts.append(f"volume spike ({row['volume_ratio']:.1f}x average)")

    parts.append(f"EMA12/26 trend: {'bullish' if row['ema_bullish'] else 'bearish'} (not voted)")

    return " + ".join(parts)


@st.cache_data(ttl=14400, show_spinner=False)
def calculate_signals_ensemble(symbol: str, threshold: int = ENSEMBLE_VOTE_THRESHOLD) -> Optional[dict]:
    """
    Compute the ensemble-vote signal for *symbol* — RSI + Bollinger Bands +
    Stochastic RSI + volume spike, see the module-level comment above for
    the exact voting rule. Independent of calculate_signals(); does not
    affect it.

    Returns a dict: symbol, name, price, signal, confidence, buy_votes,
    sell_votes, rsi_14d, bollinger_upper, bollinger_lower, stoch_k, stoch_d,
    volume_ratio, ema_trend ("bullish"|"bearish", informational only, not
    voted), reason.

    Returns None on any failure (invalid symbol, network error, insufficient
    history) — never raises.
    """
    try:
        daily = _fetch_daily_ohlcv(symbol)
        if daily is None or len(daily) < ENSEMBLE_MIN_HISTORY_POINTS:
            logger.warning("Not enough daily history for %s to compute ensemble signal", symbol)
            return None

        indicators = _compute_ensemble_indicators(daily)
        if indicators is None or indicators.empty:
            return None

        last = indicators.iloc[-1]
        if pd.isna(last["rsi"]) or pd.isna(last["stoch_k"]) or pd.isna(last["upper_bb"]):
            return None

        signal, confidence = _ensemble_signal_from_votes(int(last["buy_votes"]), int(last["sell_votes"]), threshold)

        return {
            "symbol": symbol.upper(),
            "name": _resolve_name(symbol),
            "price": float(last["close"]),
            "signal": signal,
            "confidence": confidence,
            "buy_votes": int(last["buy_votes"]),
            "sell_votes": int(last["sell_votes"]),
            "rsi_14d": float(last["rsi"]),
            "bollinger_upper": float(last["upper_bb"]),
            "bollinger_lower": float(last["lower_bb"]),
            "stoch_k": float(last["stoch_k"]),
            "stoch_d": float(last["stoch_d"]),
            "volume_ratio": float(last["volume_ratio"]) if not pd.isna(last["volume_ratio"]) else None,
            "ema_trend": "bullish" if bool(last["ema_bullish"]) else "bearish",
            "reason": _build_ensemble_reason(last),
        }
    except Exception as exc:
        logger.warning("calculate_signals_ensemble failed for %s: %s", symbol, exc)
        return None


@st.cache_data(ttl=14400, show_spinner=False)
def scan_signals_ensemble(
    symbols: tuple[str, ...], threshold: int = ENSEMBLE_VOTE_THRESHOLD, _names: Optional[dict[str, str]] = None
) -> list[dict]:
    """
    Ensemble-vote analog of scan_signals(): one batched daily-OHLCV download
    for all *symbols*, same schema as calculate_signals_ensemble() per row,
    sorted by confidence descending. Symbols that fail to resolve are
    omitted, not a hard failure of the whole scan.
    """
    if not symbols:
        return []

    names = _names or {}
    daily_batch = _fetch_daily_ohlcv_batch(symbols)

    rows: list[dict] = []
    for symbol in symbols:
        daily = daily_batch.get(symbol)
        if daily is None or len(daily) < ENSEMBLE_MIN_HISTORY_POINTS:
            continue

        indicators = _compute_ensemble_indicators(daily)
        if indicators is None or indicators.empty:
            continue

        last = indicators.iloc[-1]
        if pd.isna(last["rsi"]) or pd.isna(last["stoch_k"]) or pd.isna(last["upper_bb"]):
            continue

        signal, confidence = _ensemble_signal_from_votes(int(last["buy_votes"]), int(last["sell_votes"]), threshold)

        rows.append(
            {
                "symbol": symbol.upper(),
                "name": names.get(symbol, symbol.upper()),
                "price": float(last["close"]),
                "signal": signal,
                "confidence": confidence,
                "buy_votes": int(last["buy_votes"]),
                "sell_votes": int(last["sell_votes"]),
                "rsi_14d": float(last["rsi"]),
                "bollinger_upper": float(last["upper_bb"]),
                "bollinger_lower": float(last["lower_bb"]),
                "stoch_k": float(last["stoch_k"]),
                "stoch_d": float(last["stoch_d"]),
                "volume_ratio": float(last["volume_ratio"]) if not pd.isna(last["volume_ratio"]) else None,
                "ema_trend": "bullish" if bool(last["ema_bullish"]) else "bearish",
                "reason": _build_ensemble_reason(last),
            }
        )

    rows.sort(key=lambda r: r["confidence"], reverse=True)
    return rows


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


def _compute_classic_indicators_series(daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Full-history, index-aligned version of _compute_daily_indicators() above
    — same three indicators (RSI, MACD, MA20/50 crossover), same causal
    (no-look-ahead) computation, but every row instead of just the latest
    one. Exists solely so the backtest engine can grade the *classic*
    calculate_signals()/_score_signal() system with the exact same
    day-by-day methodology used for the ensemble system, for an apples-to-
    apples before/after comparison — calculate_signals() itself is
    untouched and doesn't use this.
    """
    try:
        close, volume = daily["Close"], daily["Volume"]
    except KeyError as exc:
        logger.warning("Classic indicator series missing required column: %s", exc)
        return None

    rsi = calculate_rsi_series(close, RSI_LENGTH)
    macd_line = calculate_ema(close, MACD_FAST) - calculate_ema(close, MACD_SLOW)
    macd_signal_line = calculate_ema(macd_line, MACD_SIGNAL)
    ma_short = calculate_sma(close, MA_SHORT)
    ma_long = calculate_sma(close, MA_LONG)
    avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    volume_ratio = volume / avg_volume

    return pd.DataFrame(
        {
            "close": close,
            "rsi_14d": rsi,
            "macd_trend": ("bullish" if v else "bearish" for v in (macd_line > macd_signal_line)),
            "ma_crossover": ("golden_cross" if v else "death_cross" for v in (ma_short > ma_long)),
            "volume_ratio": volume_ratio,
            "_valid": rsi.notna() & macd_line.notna() & macd_signal_line.notna() & ma_short.notna() & ma_long.notna(),
        }
    )


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
        # Still report the real RSI reading even when it's not extreme —
        # otherwise every neutral-RSI symbol collapses onto one of only 4
        # template strings (bullish/bearish x golden/death cross), which
        # looks identical across dozens of genuinely different symbols
        # even though their actual RSI values differ.
        reasons.append(f"RSI neutral ({rsi_14d:.1f})")

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
