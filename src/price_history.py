"""
Recent daily closing-price history for one or many tickers, for sparkline
charts.
"""
from __future__ import annotations

import logging
from typing import Optional

import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(symbol: str, days: int = 30) -> Optional[list[dict]]:
    """
    Fetch the last *days* trading days of daily closing prices for *symbol*.

    Uses yfinance's `period=f"{days}d"` rather than a fixed "1mo" period —
    "1mo" only returns ~20 trading days (weekends/holidays excluded), while
    "Nd" returns exactly N real trading-day rows, which is what "last 30
    days" of price history means here.

    Returns a list of {date, close_price} dicts, oldest first, where date is
    an "YYYY-MM-DD" string. Returns None if the symbol is invalid or the
    request fails (network error, etc.) — never raises.
    """
    try:
        hist = yf.Ticker(symbol).history(period=f"{days}d", interval="1d")
        if hist is None or hist.empty or "Close" not in hist.columns:
            logger.warning("No price history for %s", symbol)
            return None

        close = hist["Close"].dropna()
        if close.empty:
            logger.warning("No price history for %s", symbol)
            return None

        return [
            {"date": index.strftime("%Y-%m-%d"), "close_price": float(price)}
            for index, price in close.items()
        ]
    except Exception as exc:
        logger.warning("get_price_history failed for %s: %s", symbol, exc)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history_batch(symbols: tuple[str, ...], days: int = 30) -> dict[str, list[float]]:
    """
    Fetch the last *days* trading days of closing prices for many symbols at
    once, via a single batched yfinance download — used to populate a
    sparkline column across a whole table without one API call per row.

    Returns {symbol: [close_price, ...]} (oldest first). Symbols that failed
    to resolve (delisted, invalid, no data) are simply omitted from the
    result rather than raising — callers should treat a missing key as "no
    sparkline for this row," not a hard failure of the whole batch.
    """
    if not symbols:
        return {}

    try:
        data = yf.download(
            list(symbols), period=f"{days}d", interval="1d",
            group_by="ticker", progress=False, auto_adjust=True, threads=True,
        )
    except Exception as exc:
        logger.warning("Batch price history download failed for %d symbols: %s", len(symbols), exc)
        return {}

    result: dict[str, list[float]] = {}
    for symbol in symbols:
        try:
            close = data[symbol]["Close"].dropna() if len(symbols) > 1 else data["Close"].dropna()
            if not close.empty:
                result[symbol] = [float(p) for p in close]
        except Exception as exc:
            logger.warning("No batched price history for %s: %s", symbol, exc)
    return result
