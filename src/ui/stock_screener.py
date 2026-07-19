"""
Streamlit component: live Yahoo Finance market screener.

Every category shares one "Load More" pagination pattern: each click fetches
the next batch (50 rows) and appends it to an accumulating, session_state-held
list rather than replacing the table. Clicking a row returns that row's
ticker symbol so the caller can act on it (e.g. jump to the analysis page).

Caching is two-layered: `_cached_fetch`/`_cached_total` wrap the underlying
scraper calls in @st.cache_data(ttl=300) — a shared, process-wide 5-minute
cache, so concurrent users viewing the same (category, offset) page cost
Yahoo Finance one live request, not one per user. `session_state` sits on
top of that purely to hold each user's own accumulated "Load More" list
across reruns; it has no TTL of its own and isn't a rate-limiting
mechanism — the @st.cache_data layer underneath is what does that job.

Every table also has a "30-Day Trend" sparkline column, fetched in a
decoupled step *after* the main screener rows are already in hand — the
table itself never blocks on price-history data. It uses
st.column_config.LineChartColumn (auto green/red), not a real Plotly
figure: st.dataframe cells can't hold interactive per-cell charts with
their own click handlers. The real interactive Plotly chart
(sparkline_renderer.render_price_detail_modal) opens in a dedicated
st.dialog via the "📈 View Chart" control below each table, not via a
table-cell click — row clicks already mean "load this ticker into
Search & Fetch" and that behavior is left untouched.
"""
from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from scrapers.yahoo_finance_scraper import (
    fetch_top_gainers,
    fetch_top_gainers_total,
    fetch_top_losers,
    fetch_top_losers_total,
    fetch_most_active,
    fetch_most_active_total,
    fetch_52week_gainers,
    fetch_52week_gainers_total,
    fetch_52week_losers,
    fetch_52week_losers_total,
    fetch_all_time_high,
    fetch_all_time_high_total,
    fetch_all_time_low,
    fetch_all_time_low_total,
    fetch_us_indices,
    fetch_world_indices,
    fetch_top_10_crypto,
)
from src.price_history import get_price_history_batch
from src.ui.sparkline_renderer import render_price_detail_modal

_PAGE_SIZE = 50

_VOLUME_COLUMN = ("volume", "Volume", "int")
_TREND_COLUMN = "30-Day Trend"

# category label -> {
#   fetch:         fn(limit, offset) -> Optional[list[dict]]  (positional: row count, then offset)
#   total:         fn() -> Optional[int]                       (real count from the API; None if unknown)
#   extra:         [(row_key, column_label, format_kind), ...] format_kind in
#                  {"int", "price", "market_cap", None}
#   caveat:        optional caption shown above the table
#   paginated:     default True; False hides "Load More" entirely (fixed small row set)
#   price_symbol:  default identity; fn(display_symbol) -> yfinance-queryable
#                  symbol, for categories whose display symbol isn't directly
#                  queryable (crypto strips "-USD" for display)
# }
_SCREENERS: dict[str, dict] = {
    "📈 Top Gainers (Today)": {
        "fetch": fetch_top_gainers, "total": fetch_top_gainers_total, "extra": [_VOLUME_COLUMN],
    },
    "📉 Top Losers (Today)": {
        "fetch": fetch_top_losers, "total": fetch_top_losers_total, "extra": [_VOLUME_COLUMN],
    },
    "🔥 Most Active": {
        "fetch": fetch_most_active, "total": fetch_most_active_total, "extra": [_VOLUME_COLUMN],
    },
    "5️⃣2️⃣ 52 Week Gainers": {
        "fetch": fetch_52week_gainers, "total": fetch_52week_gainers_total, "extra": [_VOLUME_COLUMN],
    },
    "5️⃣2️⃣ 52 Week Losers": {
        "fetch": fetch_52week_losers, "total": fetch_52week_losers_total, "extra": [_VOLUME_COLUMN],
    },
    "📈 All Time High": {
        "fetch": fetch_all_time_high, "total": fetch_all_time_high_total,
        "extra": [
            ("all_time_high_price", "52-Week High ($)", "price"),
            ("days_from_high", "Days Since High", None),
        ],
        "caveat": "52-week basis — Yahoo Finance has no free all-time price history.",
    },
    "📉 All Time Low": {
        "fetch": fetch_all_time_low, "total": fetch_all_time_low_total,
        "extra": [
            ("all_time_low_price", "52-Week Low ($)", "price"),
            ("days_from_low", "Days Since Low", None),
        ],
        "caveat": "52-week basis — Yahoo Finance has no free all-time price history.",
    },
    "🇺🇸 US Market Indices": {
        "fetch": lambda limit, offset: fetch_us_indices(),
        "total": None,
        "extra": [("change_dollar", "Change ($)", "price")],
        "paginated": False,
    },
    "🌍 World Market Indices": {
        "fetch": lambda limit, offset: fetch_world_indices(),
        "total": None,
        "extra": [("change_dollar", "Change ($)", "price")],
        "paginated": False,
    },
    "🪙 Top 10 Cryptocurrency": {
        "fetch": lambda limit, offset: fetch_top_10_crypto(),
        "total": None,
        "extra": [
            ("change_dollar", "Change ($)", "price"),
            ("market_cap", "Market Cap", "market_cap"),
        ],
        "paginated": False,
        # fetch_top_10_crypto() strips "-USD" for display (row["symbol"] is
        # "BTC", not "BTC-USD") — but "BTC" alone is a different, unrelated
        # yfinance ticker, not Bitcoin. Price-history lookups need the real
        # symbol back.
        "price_symbol": lambda s: f"{s}-USD",
    },
}


def render_stock_screener() -> Optional[str]:
    """
    Render the market screener page: category picker, refresh button,
    live Yahoo Finance table with "Load More" pagination.

    Returns
    -------
    The ticker symbol of the row the user selected this run, or None.
    """
    st.title("🧭 Market Screener")
    st.caption("Live market movers from Yahoo Finance. Select a row to analyze that ticker.")

    category = st.sidebar.selectbox("Screener", list(_SCREENERS.keys()), key="screener_category")
    return _render_screener_page(category)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fetch(category: str, limit: int, offset: int) -> Optional[list[dict]]:
    """Shared 5-minute cache across every user session for identical
    (category, offset) requests, so N concurrent users looking at the same
    screener page cost Yahoo Finance one live request, not N.
    """
    return _SCREENERS[category]["fetch"](limit, offset)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_total(category: str) -> Optional[int]:
    total_fn = _SCREENERS[category].get("total")
    return total_fn() if total_fn is not None else None


def _render_screener_page(category: str) -> Optional[str]:
    spec = _SCREENERS[category]
    extra_cols: list[tuple[str, str, Optional[str]]] = spec["extra"]
    paginated: bool = spec.get("paginated", True)
    price_symbol = spec.get("price_symbol", lambda s: s)

    rows_key = f"screener_rows__{category}"
    total_key = f"screener_total__{category}"

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True, key=f"refresh__{category}"):
            st.session_state.pop(rows_key, None)
            st.session_state.pop(total_key, None)
            # Refresh means "force a live re-fetch now" — without this the
            # 5-minute cache above would just hand back the same cached page.
            _cached_fetch.clear()
            _cached_total.clear()

    if rows_key not in st.session_state:
        with st.spinner(f"Fetching {category}…"):
            try:
                first_page = _cached_fetch(category, _PAGE_SIZE, 0)
            except Exception as exc:
                st.error(f"Could not load screener data: {exc}")
                return None
            try:
                total = _cached_total(category)
            except Exception:
                total = None

        if first_page is None:
            st.error(
                "Could not load screener data — Yahoo Finance may be temporarily unreachable."
            )
            return None
        st.session_state[rows_key] = first_page
        st.session_state[total_key] = total

    caption = f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if spec.get("caveat"):
        caption = f"{spec['caveat']} {caption}"
    st.caption(caption)

    rows: list[dict] = st.session_state[rows_key]
    total: Optional[int] = st.session_state[total_key]

    if not rows:
        st.warning("No data available for this category right now. Try **Refresh**.")
        return None

    # Sparkline data is fetched as its own, decoupled step — the table above
    # this point only needed the already-cached screener rows, so it's ready
    # to draw immediately; the "30-Day Trend" column fills in once this
    # (separately cached, batched) fetch completes.
    with st.spinner("Loading price trends…"):
        lookup_symbols = tuple(price_symbol(r["symbol"]) for r in rows)
        raw_sparklines = _fetch_sparkline_series(lookup_symbols)
        sparkline_data = {r["symbol"]: raw_sparklines.get(price_symbol(r["symbol"]), []) for r in rows}

    df = _build_dataframe(rows, extra_cols, sparkline_data)
    selected = _render_selectable_table(
        df, key=f"screener_table__{category}", column_config=_build_column_config(extra_cols)
    )

    _render_chart_picker(category, [r["symbol"] for r in rows], price_symbol)

    if not paginated:
        return selected

    exhausted = total is not None and len(rows) >= total
    load_col, status_col = st.columns([1, 5])
    with load_col:
        clicked = st.button(
            "Load More", use_container_width=True,
            disabled=exhausted, key=f"load_more__{category}",
        )
    with status_col:
        if total is not None:
            st.caption(f"Loaded {len(rows)} of {total} rows" + (" — all loaded" if exhausted else ""))
        else:
            st.caption(f"Loaded {len(rows)} rows")

    if clicked and not exhausted:
        with st.spinner("Loading more…"):
            try:
                next_page = _cached_fetch(category, _PAGE_SIZE, len(rows))
            except Exception as exc:
                st.error(f"Could not load more rows: {exc}")
                next_page = None

        if next_page is None:
            st.error("Could not load more rows — Yahoo Finance may be temporarily unreachable.")
        elif next_page:
            st.session_state[rows_key] = rows + next_page
            if len(next_page) < _PAGE_SIZE:
                # Fewer rows than requested came back: we've reached the real
                # end, even if the earlier total-count call said otherwise.
                st.session_state[total_key] = len(rows) + len(next_page)
            st.rerun()
        else:
            st.session_state[total_key] = len(rows)
            st.rerun()

    return selected


# ---------------------------------------------------------------------------
# Sparkline column + "View Chart" detail
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_sparkline_series(symbols: tuple[str, ...]) -> dict[str, list[float]]:
    """1-hour-cached, batched 30-day close-price series per symbol, for the
    in-table sparkline column. Missing symbols just get no sparkline.
    """
    return get_price_history_batch(symbols)


def _render_chart_picker(category: str, symbols: list[str], price_symbol) -> None:
    """"Select ticker for chart" + "View Chart" -> opens the full price-detail
    modal (st.dialog). A table-cell click can't open a modal directly (see
    sparkline_renderer.py docstring), so this is a deliberate, separate
    control rather than tied to row-selection, which already means
    "load this ticker into Search & Fetch" on these tables.

    *price_symbol* converts the row's display symbol to a real
    yfinance-queryable one (identity for everything except crypto).
    """
    if not symbols:
        return
    pick_col, button_col, _ = st.columns([3, 2, 3])
    with pick_col:
        chosen = st.selectbox(
            "Select ticker for chart", symbols, key=f"chart_pick__{category}", label_visibility="collapsed",
        )
    with button_col:
        if st.button("📈 View Chart", use_container_width=True, key=f"view_chart__{category}"):
            render_price_detail_modal(price_symbol(chosen))


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def _build_dataframe(
    rows: list[dict],
    extra_cols: list[tuple[str, str, Optional[str]]],
    sparkline_data: dict[str, list[float]],
) -> pd.DataFrame:
    records = []
    for r in rows:
        record = {
            "Symbol": r["symbol"],
            "Company Name": r["name"],
            "Current Price": r["price"],
            "Change %": r["change_percent"],
        }
        for row_key, label, format_kind in extra_cols:
            value = r.get(row_key)
            if format_kind == "market_cap":
                value = _format_market_cap(value)
            record[label] = value
        record[_TREND_COLUMN] = sparkline_data.get(r["symbol"]) or []
        records.append(record)
    return pd.DataFrame(records)


def _build_column_config(extra_cols: list[tuple[str, str, Optional[str]]]) -> dict:
    config = {
        "Current Price": st.column_config.NumberColumn(format="$%.2f"),
        "Change %": st.column_config.NumberColumn(format="%.2f%%"),
        _TREND_COLUMN: st.column_config.LineChartColumn(
            _TREND_COLUMN, help="Last 30 trading days' closing price", color="auto",
        ),
    }
    for _row_key, label, format_kind in extra_cols:
        if format_kind == "int":
            config[label] = st.column_config.NumberColumn(format="%d")
        elif format_kind == "price":
            config[label] = st.column_config.NumberColumn(format="$%.2f")
        # "market_cap" is pre-formatted into a display string by
        # _build_dataframe, so it renders as plain text — no NumberColumn.
    return config


def _format_market_cap(value: Optional[float]) -> str:
    """Format a raw market-cap float as e.g. "$1.26T", "$145.20B", "$5.88B"."""
    if value is None:
        return "N/A"
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"${value / threshold:.2f}{suffix}"
    return f"${value:.2f}"


def _render_selectable_table(df: pd.DataFrame, *, key: str, column_config: dict) -> Optional[str]:
    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
        column_config=column_config,
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        symbol = df.iloc[selected_rows[0]]["Symbol"]
        st.success(f"Selected **{symbol}** — jumping to Search & Fetch…")
        return symbol

    return None
