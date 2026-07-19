"""
Sector performance snapshot: the 11 GICS sectors via their SPDR sector
ETFs, plus (per sector) which real constituent stocks are actually driving
that performance, sourced from Yahoo's screener API — never hardcoded.

Sector taxonomy note: the SPDR sector ETFs (and the sector names this
module's public API uses) follow GICS naming ("Financials", "Consumer
Discretionary", "Materials"). Yahoo's screener API classifies individual
stocks with a different, Morningstar-based taxonomy ("Financial Services",
"Consumer Cyclical", "Basic Materials") — verified live against the
screener's own valid_values before writing _SCREENER_SECTOR_NAMES, which
maps between the two so per-sector constituent lookups query the right
stocks.

Ticker correction: Communication Services' SPDR ETF is XLC, not XLCO —
XLCO returns no data on Yahoo Finance (verified live; a plausible typo
for XLC, since Consumer Staples is XLP and Real Estate is XLRE).
"""
from __future__ import annotations

import logging
from typing import Optional

import streamlit as st
import yfinance as yf
from yfinance.screener.query import EquityQuery

logger = logging.getLogger(__name__)

SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

_SCREENER_SECTOR_NAMES: dict[str, str] = {
    "Technology": "Technology",
    "Healthcare": "Healthcare",
    "Financials": "Financial Services",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Communication Services": "Communication Services",
}

_CONSTITUENT_POOL_SIZE = 20
_MIN_CONSTITUENT_MARKET_CAP = 2_000_000_000

MOMENTUM_STRONG_THRESHOLD = 5.0
MOMENTUM_BEARISH_THRESHOLD = -5.0


@st.cache_data(ttl=1800, show_spinner=False)
def get_sector_performance() -> Optional[dict]:
    """
    Snapshot of all 11 GICS sectors via their SPDR sector ETFs.

    Returns {"sectors": {sector_name: row, ...}} where each row has:
        symbol, name, price, change_percent (today), change_30day,
        top_stock (today's biggest % mover among liquid, >=$2B market-cap
        sector constituents — a deliberately cheap screener lookup, or None
        if it couldn't be resolved), momentum ("strong" | "bullish" |
        "neutral" | "bearish", classified from change_30day — see
        _classify_momentum).

    Note top_stock here is today's mover, not 30-day-ranked — computing a
    true 30-day-ranked pool for all 11 sectors on every snapshot load was
    measured at ~20s, mostly wasted since most users won't drill into most
    sectors. The real 30-day-ranked top 10 lives in get_sector_top_stocks(),
    called lazily by the drill-down UI for just the one sector clicked.

    A sector that fails to resolve is omitted, not a hard failure — this
    returns None only if none of the 11 resolved (e.g. network down).
    """
    sectors: dict[str, dict] = {}
    for sector_name, etf_symbol in SECTOR_ETFS.items():
        row = _build_sector_row(sector_name, etf_symbol)
        if row:
            sectors[sector_name] = row

    return {"sectors": sectors} if sectors else None


@st.cache_data(ttl=1800, show_spinner=False)
def get_sector_top_stocks(sector_name: str, limit: int = 10) -> list[dict]:
    """
    Real constituent stocks in *sector_name* (GICS naming, e.g. "Financials"
    — see SECTOR_ETFS for the full set), ranked by each stock's own actual
    30-day price change (not the sector ETF's aggregate change).

    Returns [] if the sector name isn't recognized or nothing resolved —
    never raises.
    """
    screener_sector = _SCREENER_SECTOR_NAMES.get(sector_name)
    if not screener_sector:
        logger.warning("Unknown sector for constituent lookup: %s", sector_name)
        return []

    try:
        candidates = _fetch_sector_candidates(screener_sector)
        if not candidates:
            return []

        symbols = tuple(c["symbol"] for c in candidates)
        changes = _fetch_30day_changes_batch(symbols)

        rows = []
        for candidate in candidates:
            change = changes.get(candidate["symbol"])
            if change is None:
                continue
            rows.append({**candidate, "change_30day": change})

        rows.sort(key=lambda r: r["change_30day"], reverse=True)
        return rows[:limit]
    except Exception as exc:
        logger.warning("get_sector_top_stocks failed for %s: %s", sector_name, exc)
        return []


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _build_sector_row(sector_name: str, etf_symbol: str) -> Optional[dict]:
    try:
        etf_quote = _fetch_etf_quote(etf_symbol)
        if etf_quote is None:
            return None

        change_30day = _fetch_30day_change(etf_symbol)
        # Deliberately cheap here: today's biggest mover in the sector via
        # one screener query, not the full 30-day-ranked pool (that's what
        # get_sector_top_stocks() is for — called lazily by the drill-down
        # modal, not on every snapshot load). Verified live: this alone cut
        # the full 11-sector snapshot from ~20s to a few seconds.
        top_stock = _fetch_sector_top_mover_today(sector_name)

        return {
            "symbol": etf_symbol,
            "name": etf_quote["name"],
            "price": etf_quote["price"],
            "change_percent": etf_quote["change_percent"],
            "change_30day": change_30day,
            "top_stock": top_stock,
            "momentum": _classify_momentum(change_30day),
        }
    except Exception as exc:
        logger.warning("Could not build sector row for %s (%s): %s", sector_name, etf_symbol, exc)
        return None


def _fetch_sector_top_mover_today(sector_name: str) -> Optional[str]:
    """Today's biggest % gainer in *sector_name* among liquid (>=$2B
    market cap) constituents — one cheap screener query, no batch history.
    """
    screener_sector = _SCREENER_SECTOR_NAMES.get(sector_name)
    if not screener_sector:
        return None
    try:
        query = EquityQuery(
            "and",
            [
                EquityQuery("eq", ["sector", screener_sector]),
                EquityQuery("eq", ["region", "us"]),
                EquityQuery("gt", ["intradaymarketcap", _MIN_CONSTITUENT_MARKET_CAP]),
            ],
        )
        result = yf.screen(query, size=1, sortField="percentchange", sortAsc=False)
        quotes = result.get("quotes", []) if result else []
        return quotes[0]["symbol"] if quotes else None
    except Exception as exc:
        logger.warning("Could not fetch today's top mover for %s: %s", sector_name, exc)
        return None


def _classify_momentum(change_30day: Optional[float]) -> str:
    if change_30day is None:
        return "neutral"
    if change_30day >= MOMENTUM_STRONG_THRESHOLD:
        return "strong"
    if change_30day <= MOMENTUM_BEARISH_THRESHOLD:
        return "bearish"
    if change_30day > 0:
        return "bullish"
    return "neutral"


def _fetch_etf_quote(symbol: str) -> Optional[dict]:
    try:
        info = yf.Ticker(symbol).info
        price = info.get("regularMarketPrice") if info else None
        if price is None:
            logger.warning("No quote data for sector ETF %s", symbol)
            return None
        name = (info.get("longName") or info.get("shortName") or symbol).strip()
        return {
            "name": name,
            "price": float(price),
            "change_percent": float(info.get("regularMarketChangePercent", 0.0) or 0.0),
        }
    except Exception as exc:
        logger.warning("Could not fetch ETF quote for %s: %s", symbol, exc)
        return None


def _fetch_30day_change(symbol: str) -> Optional[float]:
    try:
        hist = yf.Ticker(symbol).history(period="30d", interval="1d")
        if hist is None or hist.empty:
            return None
        close = hist["Close"].dropna()
        if len(close) < 2:
            return None
        return float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)
    except Exception as exc:
        logger.warning("Could not fetch 30-day change for %s: %s", symbol, exc)
        return None


def _fetch_sector_candidates(screener_sector: str) -> list[dict]:
    query = EquityQuery(
        "and",
        [
            EquityQuery("eq", ["sector", screener_sector]),
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("gt", ["intradaymarketcap", _MIN_CONSTITUENT_MARKET_CAP]),
        ],
    )
    result = yf.screen(query, size=_CONSTITUENT_POOL_SIZE, sortField="intradaymarketcap", sortAsc=False)
    quotes = result.get("quotes", []) if result else []
    return [
        {
            "symbol": q["symbol"],
            "name": q.get("longName") or q.get("shortName") or q["symbol"],
            "price": float(q.get("regularMarketPrice", 0.0) or 0.0),
        }
        for q in quotes
        if q.get("symbol")
    ]


def _fetch_30day_changes_batch(symbols: tuple[str, ...]) -> dict[str, float]:
    if not symbols:
        return {}
    try:
        data = yf.download(
            list(symbols), period="30d", interval="1d",
            group_by="ticker", progress=False, auto_adjust=True, threads=True,
        )
    except Exception as exc:
        logger.warning("Batched 30-day download failed for %d symbols: %s", len(symbols), exc)
        return {}

    result: dict[str, float] = {}
    for symbol in symbols:
        try:
            # group_by="ticker" always nests columns under the ticker, even
            # for a single-symbol request.
            close = data[symbol]["Close"].dropna()
            if len(close) < 2:
                continue
            result[symbol] = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)
        except Exception:
            continue
    return result
