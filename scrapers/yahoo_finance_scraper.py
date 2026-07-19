#!/usr/bin/env python3
"""
Yahoo Finance market-movers scraper.

Fetches top gainers, top losers, most active, and 52-week high/low movers.

Primary path uses the `yfinance` library's predefined screeners
(day_gainers / day_losers / most_actives). yfinance has no predefined
screener for 52-week movers, and if the screener call fails or returns
no rows, every function falls back to scraping the public Yahoo Finance
markets pages (finance.yahoo.com/markets/stocks/...) directly with
BeautifulSoup — those pages are server-rendered, so a plain GET returns
the full data table.

No API key is required; only free, publicly available endpoints are used.

Rate limiting: this module itself performs no caching or throttling — every
public function issues a live request on every call. Rate-limiting is the
caller's responsibility. The Streamlit UI layer (src/ui/stock_screener.py)
wraps every fetch in @st.cache_data(ttl=300s), a cache shared across all
user sessions, so repeated views of the same screener page within 5 minutes
cost Yahoo Finance one request rather than one per view. Any other caller of
this module should apply equivalent caching before hitting these functions
in a loop or on a schedule.
"""
from __future__ import annotations

import argparse
import logging
import re
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from yfinance.screener.query import EquityQuery

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

MARKETS_BASE_URL = "https://finance.yahoo.com/markets/stocks"

DEFAULT_COUNT = 50

US_INDEX_SYMBOLS = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones Industrial Average",
    "^IXIC": "Nasdaq Composite",
    "^RUT": "Russell 2000",
}

WORLD_INDEX_SYMBOLS = {
    "^N225": "Nikkei 225 (Japan)",
    "^GDAXI": "DAX (Germany)",
    "^FTSE": "FTSE 100 (UK)",
    "000001.SS": "Shanghai Composite (China)",
    "^HSI": "Hang Seng (Hong Kong)",
    "^BSESN": "BSE Sensex (India)",
}

CRYPTO_SYMBOLS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "BNB",
    "SOL-USD": "Solana",
    "ADA-USD": "Cardano",
    "XRP-USD": "XRP",
    "DOGE-USD": "Dogecoin",
    "AVAX-USD": "Avalanche",
    "SHIB-USD": "Shiba Inu",
    "DOT-USD": "Polkadot",
}

_session = requests.Session()
_session.headers.update(HEADERS)

_SUFFIX_MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_top_gainers(count: int = DEFAULT_COUNT, offset: int = 0) -> list[dict]:
    """Fetch today's top-gaining US equities by percent change."""
    return _fetch("day_gainers", "gainers", count, offset)[0]


def fetch_top_gainers_total() -> Optional[int]:
    """Total number of stocks matching fetch_top_gainers(), or None on failure."""
    return _fetch("day_gainers", "gainers", 1, 0)[1]


def fetch_top_losers(count: int = DEFAULT_COUNT, offset: int = 0) -> list[dict]:
    """Fetch today's top-losing US equities by percent change."""
    return _fetch("day_losers", "losers", count, offset)[0]


def fetch_top_losers_total() -> Optional[int]:
    """Total number of stocks matching fetch_top_losers(), or None on failure."""
    return _fetch("day_losers", "losers", 1, 0)[1]


def fetch_most_active(count: int = DEFAULT_COUNT, offset: int = 0) -> list[dict]:
    """Fetch today's most actively traded US equities by volume."""
    return _fetch("most_actives", "most-active", count, offset)[0]


def fetch_most_active_total() -> Optional[int]:
    """Total number of stocks matching fetch_most_active(), or None on failure."""
    return _fetch("most_actives", "most-active", 1, 0)[1]


def fetch_52week_gainers(count: int = DEFAULT_COUNT, offset: int = 0) -> list[dict]:
    """Fetch US equities with the largest 52-week gains.

    yfinance has no predefined screener for this category, so it is
    fetched directly from Yahoo's public markets page.
    """
    return _scrape_markets_page("52-week-gainers", count, offset)[0]


def fetch_52week_gainers_total() -> Optional[int]:
    """Total number of stocks matching fetch_52week_gainers(), or None on failure."""
    return _scrape_markets_page("52-week-gainers", 1, 0)[1]


def fetch_52week_losers(count: int = DEFAULT_COUNT, offset: int = 0) -> list[dict]:
    """Fetch US equities with the largest 52-week losses.

    yfinance has no predefined screener for this category, so it is
    fetched directly from Yahoo's public markets page.
    """
    return _scrape_markets_page("52-week-losers", count, offset)[0]


def fetch_52week_losers_total() -> Optional[int]:
    """Total number of stocks matching fetch_52week_losers(), or None on failure."""
    return _scrape_markets_page("52-week-losers", 1, 0)[1]


def fetch_all_time_high(limit: int = DEFAULT_COUNT, offset: int = 0) -> Optional[list[dict]]:
    """Fetch US equities trading at or near their 52-week high, paginated.

    Yahoo has no free "all-time" (since-IPO) high screener or page — both
    finance.yahoo.com/markets/stocks/all-time-high/ and its API redirect to
    /markets/stocks/most-active/. This uses the 52-week high as the closest
    available proxy, which is also the convention most retail platforms use
    for "at highs" lists. Each row additionally reports the 52-week high
    price and how many days ago it was set (from a batched 1y price-history
    lookup, computed only for the requested page).

    Row schema: symbol, name, price, change_percent, all_time_high_price,
    days_from_high.

    Returns a (possibly empty) list on success — an empty list means no more
    rows at that offset, not a failure. Returns None if the underlying
    screener call itself fails (network error, etc.) — never raises.
    """
    return _fetch_high_low_page("high", limit, offset)


def fetch_all_time_low(limit: int = DEFAULT_COUNT, offset: int = 0) -> Optional[list[dict]]:
    """Fetch US equities trading at or near their 52-week low, paginated.

    Same 52-week-as-proxy caveat as fetch_all_time_high() — see its
    docstring for why "all-time" means "52-week" here.

    Row schema: symbol, name, price, change_percent, all_time_low_price,
    days_from_low.

    Returns a (possibly empty) list on success — an empty list means no more
    rows at that offset, not a failure. Returns None if the underlying
    screener call itself fails (network error, etc.) — never raises.
    """
    return _fetch_high_low_page("low", limit, offset)


def fetch_us_indices() -> Optional[list[dict]]:
    """Fetch live quotes for the four major US market indices.

    S&P 500 (^GSPC), Dow Jones Industrial Average (^DJI), Nasdaq Composite
    (^IXIC), Russell 2000 (^RUT) — via yfinance Ticker().info per symbol.

    Row schema: symbol, name, price, change_percent, change_dollar.

    Returns the indices that resolved (still a list even if one or two
    symbols individually failed). Returns None only if none of the four
    resolved — e.g. the network is down — never raises.
    """
    return _fetch_index_quotes(US_INDEX_SYMBOLS)


def fetch_world_indices() -> Optional[list[dict]]:
    """Fetch live quotes for six major world market indices.

    Nikkei 225 (^N225, Japan), DAX (^GDAXI, Germany), FTSE 100 (^FTSE, UK),
    Shanghai Composite (000001.SS, China), Hang Seng (^HSI, Hong Kong),
    BSE Sensex (^BSESN, India) — via yfinance Ticker().info per symbol.

    Row schema: symbol, name, price, change_percent, change_dollar.

    Returns the indices that resolved (still a list even if some symbols
    individually failed or are delisted/invalid). Returns None only if none
    of the six resolved — e.g. the network is down — never raises.
    """
    return _fetch_index_quotes(WORLD_INDEX_SYMBOLS)


def _fetch_index_quotes(symbols: dict[str, str]) -> Optional[list[dict]]:
    rows = [row for row in (_fetch_index_quote(s, name) for s, name in symbols.items()) if row]
    return rows if rows else None


def fetch_top_10_crypto() -> Optional[list[dict]]:
    """Fetch live quotes for the top 10 cryptocurrencies by market cap.

    Bitcoin, Ethereum, BNB, Solana, Cardano, XRP, Dogecoin, Avalanche, Shiba
    Inu, Polkadot — via yfinance Ticker().info per symbol (BTC-USD, ETH-USD,
    ... DOT-USD).

    Row schema: symbol (clean — "BTC", not "BTC-USD"), name, price,
    change_percent, change_dollar, market_cap. market_cap is a raw float;
    format it for display ($2.1T, $145B, ...) at the UI layer, same as every
    other numeric field in this module.

    Returns the cryptos that resolved (still a list even if some symbols
    individually failed or are delisted/invalid). Returns None only if none
    of the ten resolved — e.g. the network is down — never raises.
    """
    rows = [
        row for row in (_fetch_crypto_quote(s, name) for s, name in CRYPTO_SYMBOLS.items()) if row
    ]
    return rows if rows else None


def _fetch_crypto_quote(symbol: str, fallback_name: str) -> Optional[dict]:
    try:
        info = yf.Ticker(symbol).info
        price = info.get("regularMarketPrice") if info else None
        if price is None:
            logger.warning("No quote data for crypto %s (delisted or invalid symbol?)", symbol)
            return None
        name = (info.get("longName") or info.get("shortName") or fallback_name).strip()
        market_cap = info.get("marketCap")
        return {
            "symbol": symbol.removesuffix("-USD"),
            "name": name,
            "price": float(price),
            "change_percent": float(info.get("regularMarketChangePercent", 0.0) or 0.0),
            "change_dollar": float(info.get("regularMarketChange", 0.0) or 0.0),
            "market_cap": float(market_cap) if market_cap is not None else None,
        }
    except Exception as exc:
        logger.warning("Could not fetch crypto quote for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Internal: dispatch (yfinance primary, BeautifulSoup fallback)
# ---------------------------------------------------------------------------

def _fetch(
    screener_query: str, markets_path: str, count: int, offset: int = 0
) -> tuple[list[dict], Optional[int]]:
    """Try the yfinance predefined screener; fall back to scraping on failure.

    Returns (rows, total) — total is the real number of stocks Yahoo reports
    matching this screen (from the same response the rows came from), or
    None if that figure wasn't obtainable.
    """
    try:
        rows, total = _from_yfinance(screener_query, count, offset)
        if rows:
            return rows, total
        logger.warning(
            "yfinance screen(%r) returned no rows, falling back to scrape", screener_query
        )
    except Exception as exc:
        logger.warning(
            "yfinance screen(%r) failed (%s), falling back to scrape", screener_query, exc
        )
    return _scrape_markets_page(markets_path, count, offset)


_YFINANCE_PREDEFINED_PAGE_SIZE = 25


def _from_yfinance(screener_query: str, count: int, offset: int = 0) -> tuple[list[dict], Optional[int]]:
    """Fetch up to *count* quotes starting at *offset* from a predefined yfinance screen.

    yfinance silently caps predefined-screen results at 25 quotes per call
    whenever `offset` is passed as a non-None value at all — even offset=0 —
    regardless of what `count` asks for (verified directly against the live
    API). Omitting `offset` entirely for the first page and only passing it
    for later ones was tried first, but that mixes two different underlying
    Yahoo pagination baselines and produces a couple of duplicate rows right
    at the page-1/page-2 boundary (also verified live). So every call —
    including offset=0 — goes through the same fixed 25-row chunked loop,
    keeping the pagination baseline consistent across all pages.
    """
    quotes: list[dict] = []
    total: Optional[int] = None
    current_offset = offset
    while len(quotes) < count:
        result = yf.screen(screener_query, count=_YFINANCE_PREDEFINED_PAGE_SIZE, offset=current_offset)
        if not result:
            break
        total = result.get("total", total)
        batch = result.get("quotes", [])
        if not batch:
            break
        quotes.extend(batch)
        current_offset += len(batch)
        if len(batch) < _YFINANCE_PREDEFINED_PAGE_SIZE:
            break
    return [_normalize_quote(q) for q in quotes[:count]], total


def _normalize_quote(quote: dict) -> dict:
    """Map a raw yfinance screener quote to the common row schema."""
    name = quote.get("longName") or quote.get("shortName") or quote.get("displayName") or quote.get("symbol", "")
    return {
        "symbol": quote.get("symbol", ""),
        "name": name,
        "price": float(quote.get("regularMarketPrice", 0.0) or 0.0),
        "change_percent": float(quote.get("regularMarketChangePercent", 0.0) or 0.0),
        "volume": int(quote.get("regularMarketVolume", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Internal: BeautifulSoup fallback (scrapes finance.yahoo.com/markets/stocks/*)
# ---------------------------------------------------------------------------

def _scrape_markets_page(path: str, count: int, offset: int = 0) -> tuple[list[dict], Optional[int]]:
    url = f"{MARKETS_BASE_URL}/{path}/?count={count}&start={offset}"
    resp = _get(url)
    if resp is None:
        return [], None

    soup = BeautifulSoup(resp.text, "lxml")
    rows: list[dict] = []
    for tr in soup.select('tr[data-testid="data-table-v2-row"]')[:count]:
        row = _parse_row(tr)
        if row:
            rows.append(row)
    return rows, _extract_total(resp.text)


def _extract_total(html: str) -> Optional[int]:
    """Extract the "of N" total-results count Yahoo renders next to the pager."""
    match = re.search(r"\bof\s+([\d,]+)\b", html)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_row(tr) -> Optional[dict]:
    """Parse one <tr> of a Yahoo Finance markets data table into the row schema."""
    symbol_el = tr.select_one("span.symbol")
    name_el = tr.select_one('td[data-testid-cell="companyshortname.raw"] div')
    price_el = tr.select_one('td[data-testid-cell="intradayprice"] span[data-testid="change"]')
    change_el = tr.select_one('td[data-testid-cell="percentchange"] span')
    volume_el = tr.select_one('td[data-testid-cell="dayvolume"] span')

    if symbol_el is None or price_el is None:
        return None

    try:
        return {
            "symbol": symbol_el.get_text(strip=True),
            "name": name_el.get_text(strip=True) if name_el else "",
            "price": _parse_float(price_el.get_text(strip=True)),
            "change_percent": _parse_float(change_el.get_text(strip=True)) if change_el else 0.0,
            "volume": _parse_volume(volume_el.get_text(strip=True)) if volume_el else 0,
        }
    except ValueError as exc:
        logger.warning("Row parse failed for %s: %s", symbol_el.get_text(strip=True), exc)
        return None


def _parse_float(text: str) -> float:
    """Parse strings like '+28.79%', '-1.33', '5.95' into a float."""
    cleaned = text.strip().replace(",", "").replace("%", "").replace("+", "")
    return float(cleaned)


def _parse_volume(text: str) -> int:
    """Parse strings like '55.692M', '858,955', '1.2B' into an int."""
    cleaned = text.strip().replace(",", "")
    if cleaned and cleaned[-1].upper() in _SUFFIX_MULTIPLIERS:
        return int(float(cleaned[:-1]) * _SUFFIX_MULTIPLIERS[cleaned[-1].upper()])
    return int(float(cleaned)) if cleaned else 0


def _get(url: str) -> Optional[requests.Response]:
    try:
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.warning("Request failed: %s — %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Internal: 52-week high/low proximity screener (all-time-high/low proxy)
# ---------------------------------------------------------------------------

_HIGH_LOW_POOL_SIZE = 250
_HIGH_LOW_MIN_MARKET_CAP = 300_000_000
_HIGH_LOW_MIN_AVG_VOLUME = 100_000


def fetch_all_time_high_total() -> Optional[int]:
    """Total number of candidates fetch_all_time_high() can page through, or None on failure."""
    return _high_low_candidate_count("high")


def fetch_all_time_low_total() -> Optional[int]:
    """Total number of candidates fetch_all_time_low() can page through, or None on failure."""
    return _high_low_candidate_count("low")


def _high_low_candidate_count(kind: str) -> Optional[int]:
    try:
        return len(_high_low_candidates(kind))
    except Exception as exc:
        logger.warning("fetch_all_time_%s_total failed: %s", kind, exc)
        return None


def _fetch_high_low_page(kind: str, limit: int, offset: int) -> Optional[list[dict]]:
    """Return one page of the "near 52-week high/low" ranking, or None on failure."""
    if limit <= 0 or offset < 0:
        return []

    try:
        candidates = _high_low_candidates(kind)
        page = candidates[offset: offset + limit]
        if not page:
            return []

        days_map = _days_since_extreme(tuple(q["symbol"] for q in page), kind)
        return [_build_high_low_row(q, kind, days_map) for q in page]
    except Exception as exc:
        logger.warning("fetch_all_time_%s failed (limit=%s, offset=%s): %s", kind, limit, offset, exc)
        return None


def _high_low_candidates(kind: str) -> list[dict]:
    """Candidate quotes with valid 52-week high/low context, sorted nearest-to-extreme first."""
    universe = _fetch_high_low_universe()
    if not universe:
        return []

    change_key = "fiftyTwoWeekHighChangePercent" if kind == "high" else "fiftyTwoWeekLowChangePercent"
    extreme_key = "fiftyTwoWeekHigh" if kind == "high" else "fiftyTwoWeekLow"

    candidates = [
        q for q in universe
        if isinstance(q.get(change_key), (int, float)) and q.get(extreme_key)
    ]
    # Smallest absolute distance from the 52-week extreme sorts first.
    candidates.sort(key=lambda q: abs(q[change_key]))
    return candidates


def _fetch_high_low_universe() -> list[dict]:
    """Broad pool of liquid US equities (by market cap) carrying 52-week high/low context."""
    query = EquityQuery(
        "and",
        [
            EquityQuery("gt", ["intradaymarketcap", _HIGH_LOW_MIN_MARKET_CAP]),
            EquityQuery("gt", ["avgdailyvol3m", _HIGH_LOW_MIN_AVG_VOLUME]),
            EquityQuery("eq", ["region", "us"]),
        ],
    )
    result = yf.screen(query, size=_HIGH_LOW_POOL_SIZE, sortField="intradaymarketcap", sortAsc=False)
    return result.get("quotes", []) if result else []


def _build_high_low_row(quote: dict, kind: str, days_map: dict[str, Optional[int]]) -> dict:
    symbol = quote.get("symbol", "")
    name = quote.get("longName") or quote.get("shortName") or symbol
    row = {
        "symbol": symbol,
        "name": name,
        "price": float(quote.get("regularMarketPrice", 0.0) or 0.0),
        "change_percent": float(quote.get("regularMarketChangePercent", 0.0) or 0.0),
    }
    if kind == "high":
        row["all_time_high_price"] = float(quote.get("fiftyTwoWeekHigh", 0.0) or 0.0)
        row["days_from_high"] = days_map.get(symbol)
    else:
        row["all_time_low_price"] = float(quote.get("fiftyTwoWeekLow", 0.0) or 0.0)
        row["days_from_low"] = days_map.get(symbol)
    return row


def _days_since_extreme(symbols: tuple[str, ...], kind: str) -> dict[str, Optional[int]]:
    """Calendar days since each symbol's max (kind="high") or min (kind="low") close in the last year."""
    if not symbols:
        return {}

    try:
        history = yf.download(
            list(symbols), period="1y", interval="1d",
            group_by="ticker", progress=False, auto_adjust=True, threads=True,
        )
    except Exception as exc:
        logger.warning("Batch history download failed for %d symbols: %s", len(symbols), exc)
        return {}

    now = pd.Timestamp.now(tz="UTC")
    days_map: dict[str, Optional[int]] = {}
    for symbol in symbols:
        try:
            close = history[symbol]["Close"].dropna() if len(symbols) > 1 else history["Close"].dropna()
            if close.empty:
                days_map[symbol] = None
                continue
            extreme_date = pd.Timestamp(close.idxmax() if kind == "high" else close.idxmin())
            extreme_date = (
                extreme_date.tz_localize("UTC") if extreme_date.tzinfo is None
                else extreme_date.tz_convert("UTC")
            )
            days_map[symbol] = (now - extreme_date).days
        except Exception as exc:
            logger.warning("Could not compute days_from_%s for %s: %s", kind, symbol, exc)
            days_map[symbol] = None
    return days_map


# ---------------------------------------------------------------------------
# Internal: US market indices
# ---------------------------------------------------------------------------

def _fetch_index_quote(symbol: str, fallback_name: str) -> Optional[dict]:
    """Fetch one index quote. *fallback_name* is used only when Yahoo's own
    longName/shortName are both empty — some non-US indices (e.g. Shanghai
    Composite, 000001.SS) return real price/change data but no name at all.
    """
    try:
        info = yf.Ticker(symbol).info
        price = info.get("regularMarketPrice") if info else None
        if price is None:
            logger.warning("No quote data for index %s (delisted or invalid symbol?)", symbol)
            return None
        name = (info.get("longName") or info.get("shortName") or fallback_name).strip()
        return {
            "symbol": symbol,
            "name": name,
            "price": float(price),
            "change_percent": float(info.get("regularMarketChangePercent", 0.0) or 0.0),
            "change_dollar": float(info.get("regularMarketChange", 0.0) or 0.0),
        }
    except Exception as exc:
        logger.warning("Could not fetch index quote for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"{'Symbol':<8}{'Name':<32}{'Price':>10}{'Change %':>11}{'Volume':>16}")
    print("-" * 77)
    for row in rows:
        print(
            f"{row['symbol']:<8}{row['name'][:30]:<32}{row['price']:>10.2f}"
            f"{row['change_percent']:>10.2f}%{row['volume']:>16,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Yahoo Finance market-movers scraper")
    parser.add_argument(
        "--test", action="store_true", help="Fetch the first 5 top gainers and print them"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.test:
        rows = fetch_top_gainers(count=5)
        if not rows:
            print("No data returned (yfinance and scrape fallback both failed).")
            return
        _print_table(rows, "Top Gainers")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
