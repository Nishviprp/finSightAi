"""
Peer comparison: identify a company's real sector/market-cap peers and
compare 10 valuation and quality metrics across the group, sourced entirely
from yfinance `.info` (free tier) — no fabricated or hardcoded competitor
lists.

Peer discovery note: Yahoo's `.info["sector"]` field for an individual stock
uses the same taxonomy as the screener's `sector` field (both Yahoo/
Morningstar-based) — verified live, e.g. AAPL's `.info["sector"]` is
"Technology" and querying the screener with `sector == "Technology"` returns
AAPL itself among the results. No GICS<->Morningstar remapping is needed
here (contrast with sector_analytics.py, which maps its own GICS *sector
ETF* names to the screener taxonomy — a different problem).

Every public function is cached with `@st.cache_data(ttl=3600)` and never
raises: on any failure it logs a warning and returns None/[]/an empty
DataFrame instead of propagating the exception.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance.screener.query import EquityQuery

logger = logging.getLogger(__name__)

_PEER_POOL_SIZE = 30
_MIN_PEER_MARKET_CAP = 500_000_000
_DEFAULT_PEER_LIMIT = 5

# Column order doubles as the canonical metric list used everywhere else in
# this module (compare_peers, the UI's styling pass).
COMPARISON_METRICS: list[str] = [
    "P/E Ratio", "PEG Ratio", "Price/Book", "Debt/Equity",
    "ROE", "ROA", "Revenue Growth", "Margin %", "Dividend Yield", "52W Momentum",
]

# Whether a higher value is the "better" one for that metric — drives
# best/worst color-coding in the UI. P/E, PEG, Price/Book and Debt/Equity
# are cost/leverage metrics where lower is cheaper/safer; the rest are
# growth/quality/return metrics where higher is better.
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "P/E Ratio": False,
    "PEG Ratio": False,
    "Price/Book": False,
    "Debt/Equity": False,
    "ROE": True,
    "ROA": True,
    "Revenue Growth": True,
    "Margin %": True,
    "Dividend Yield": True,
    "52W Momentum": True,
}


# ---------------------------------------------------------------------------
# Per-symbol snapshot
# ---------------------------------------------------------------------------

def _pct(value) -> Optional[float]:
    """yfinance reports ROE/ROA/growth/margin/52W-change as fractions
    (0.166 == 16.6%) — convert to a plain percentage number."""
    if value is None:
        return None
    try:
        return float(value) * 100
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_symbol_snapshot(symbol: str) -> Optional[dict]:
    """
    Fetch identity + the 10 comparison metrics for *symbol* from yfinance
    `.info`, in one call.

    Returns a dict: symbol, name, sector, market_cap, price, plus the 10
    metrics keyed exactly as COMPARISON_METRICS. Any individual metric
    missing from Yahoo's data is None — never fabricated.

    Returns None if the symbol doesn't resolve.
    """
    try:
        info = yf.Ticker(symbol).info
        if not info or not (info.get("longName") or info.get("shortName")):
            logger.warning("No data found for %s", symbol)
            return None

        return {
            "symbol": symbol.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "market_cap": info.get("marketCap"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "P/E Ratio": info.get("trailingPE"),
            "PEG Ratio": info.get("pegRatio") or info.get("trailingPegRatio"),
            "Price/Book": info.get("priceToBook"),
            "Debt/Equity": info.get("debtToEquity"),
            "ROE": _pct(info.get("returnOnEquity")),
            "ROA": _pct(info.get("returnOnAssets")),
            "Revenue Growth": _pct(info.get("revenueGrowth")),
            "Margin %": _pct(info.get("profitMargins")),
            # Unlike ROE/ROA/growth/margin, yfinance already reports
            # dividendYield as a plain percentage number (0.32 == 0.32%),
            # not a fraction — verified live. No further scaling.
            "Dividend Yield": info.get("dividendYield"),
            "52W Momentum": _pct(info.get("52WeekChange")),
        }
    except Exception as exc:
        logger.warning("get_symbol_snapshot failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Peer discovery
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_peers(symbol: str, limit: int = _DEFAULT_PEER_LIMIT) -> list[str]:
    """
    Identify *symbol*'s top *limit* real competitors: same Yahoo sector,
    ranked by closeness in market cap (log-scale, since market caps span
    orders of magnitude).

    Returns a list of ticker symbols (may be shorter than *limit* if the
    sector has fewer qualifying constituents). Returns [] if *symbol*
    doesn't resolve or has no sector on file.
    """
    target = get_symbol_snapshot(symbol)
    if target is None or not target.get("sector"):
        logger.warning("Cannot find peers for %s: no sector data", symbol)
        return []

    sector = target["sector"]
    target_cap = target.get("market_cap")

    try:
        candidates = _fetch_sector_candidates(sector)
    except Exception as exc:
        logger.warning("Peer screener query failed for %s (sector=%s): %s", symbol, sector, exc)
        return []

    candidates = [c for c in candidates if c["symbol"].upper() != symbol.upper()]
    if not candidates:
        return []

    def _distance(candidate: dict) -> float:
        cap = candidate.get("market_cap")
        if not cap or not target_cap:
            return float("inf")
        return abs(math.log(cap) - math.log(target_cap))

    candidates.sort(key=_distance)
    return [c["symbol"] for c in candidates[:limit]]


def _fetch_sector_candidates(sector: str) -> list[dict]:
    query = EquityQuery(
        "and",
        [
            EquityQuery("eq", ["sector", sector]),
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("gt", ["intradaymarketcap", _MIN_PEER_MARKET_CAP]),
        ],
    )
    result = yf.screen(query, size=_PEER_POOL_SIZE, sortField="intradaymarketcap", sortAsc=False)
    quotes = result.get("quotes", []) if result else []
    return [
        {"symbol": q["symbol"], "market_cap": q.get("marketCap") or q.get("intradaymarketcap")}
        for q in quotes
        if q.get("symbol")
    ]


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def compare_peers(symbol: str, peers: tuple[str, ...]) -> pd.DataFrame:
    """
    Build a metrics-comparison table for *symbol* and *peers*.

    Returns a DataFrame indexed by ticker symbol (symbol's own row first,
    then peers in the order given) with columns == COMPARISON_METRICS.
    Values are real numbers from yfinance; a metric Yahoo doesn't report
    for a given symbol is NaN, never a fabricated placeholder. A symbol
    that fails to resolve entirely is omitted from the result.

    Returns an empty DataFrame if nothing resolved.
    """
    ordered_symbols = [symbol.upper()]
    for p in peers:
        p = p.upper()
        if p not in ordered_symbols:
            ordered_symbols.append(p)

    rows: dict[str, dict] = {}
    for s in ordered_symbols:
        snap = get_symbol_snapshot(s)
        if snap is None:
            continue
        rows[s] = {metric: snap.get(metric) for metric in COMPARISON_METRICS}

    if not rows:
        return pd.DataFrame(columns=COMPARISON_METRICS)

    df = pd.DataFrame.from_dict(rows, orient="index", columns=COMPARISON_METRICS)
    df.index.name = "Symbol"
    return df


# ---------------------------------------------------------------------------
# Valuation verdict
# ---------------------------------------------------------------------------

def compute_valuation_verdict(df: pd.DataFrame, symbol: str) -> dict:
    """
    Heuristic cheap/fair/expensive verdict for *symbol* against the peer
    rows already in *df* (as returned by compare_peers).

    Compares P/E Ratio and Revenue Growth against the peer group's median:
    cheaper P/E + faster growth than peers -> UNDERVALUED; pricier P/E +
    slower growth -> OVERVALUED; anything else (one favorable, one not) ->
    FAIRLY VALUED. Returns verdict "INSUFFICIENT_DATA" if P/E or growth is
    missing for *symbol* or there's no peer data to compare against — this
    never guesses.

    Returns a dict: verdict, reason, symbol_pe, peer_median_pe,
    symbol_growth, peer_median_growth (the last four are None when verdict
    is INSUFFICIENT_DATA).
    """
    symbol = symbol.upper()
    empty = {
        "verdict": "INSUFFICIENT_DATA", "reason": "",
        "symbol_pe": None, "peer_median_pe": None,
        "symbol_growth": None, "peer_median_growth": None,
    }

    if symbol not in df.index:
        empty["reason"] = f"No metrics data for {symbol}."
        return empty

    peer_df = df.drop(index=symbol)
    if peer_df.empty:
        empty["reason"] = "No peer data available to compare against."
        return empty

    pe = df.loc[symbol, "P/E Ratio"]
    growth = df.loc[symbol, "Revenue Growth"]
    peer_median_pe = peer_df["P/E Ratio"].median()
    peer_median_growth = peer_df["Revenue Growth"].median()

    if any(_is_missing(v) for v in (pe, growth, peer_median_pe, peer_median_growth)):
        empty["reason"] = f"P/E or revenue growth data unavailable for {symbol} or its peers."
        return empty

    cheap = pe < peer_median_pe
    high_growth = growth > peer_median_growth

    if cheap and high_growth:
        verdict = "UNDERVALUED"
    elif not cheap and not high_growth:
        verdict = "OVERVALUED"
    else:
        verdict = "FAIRLY VALUED"

    reason = (
        f"{symbol} trades at {pe:.1f}x P/E vs peer median {peer_median_pe:.1f}x, "
        f"with {growth:.1f}% revenue growth vs peer median {peer_median_growth:.1f}%."
    )

    return {
        "verdict": verdict, "reason": reason,
        "symbol_pe": float(pe), "peer_median_pe": float(peer_median_pe),
        "symbol_growth": float(growth), "peer_median_growth": float(peer_median_growth),
    }


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
