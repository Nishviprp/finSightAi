"""
Streamlit components: a mini Plotly sparkline and a full price-detail modal
for a single ticker, built on src.price_history.

Note on st.dataframe's real limits: st.dataframe cannot embed interactive
Plotly figures as table cells with independent per-cell click handlers —
that's a hard Streamlit capability limit, not a design choice here. The
in-table "30-Day Trend" column (wired in stock_screener.py) uses Streamlit's
own st.column_config.LineChartColumn instead, which supports per-row
green/red auto-coloring but not custom hover text or clicks. The real
interactive Plotly sparkline built here (render_sparkline) is used as a
quick single-symbol preview; the full detail (render_price_detail_modal)
opens in a real st.dialog popup, triggered by a dedicated button rather
than a table-cell click — see stock_screener.py for the wiring.
"""
from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from src.price_history import get_price_history

_UP_COLOR = "#28a745"
_DOWN_COLOR = "#dc3545"
_UP_FILL = "rgba(40, 167, 69, 0.12)"
_DOWN_FILL = "rgba(220, 53, 69, 0.12)"


def render_sparkline(symbol: str) -> Optional[go.Figure]:
    """
    Build a tiny (100px), title-less, axis-less 30-day close-price line
    chart for *symbol* — green if the price rose over the window, red if it
    fell. Hovering a point shows its date and price.

    Returns None if no price history is available for *symbol*.
    """
    history = get_price_history(symbol)
    if not history:
        return None

    dates = [row["date"] for row in history]
    prices = [row["close_price"] for row in history]
    is_up = prices[-1] >= prices[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=prices,
            mode="lines",
            line=dict(color=_UP_COLOR if is_up else _DOWN_COLOR, width=1.5),
            fill="tozeroy",
            fillcolor=_UP_FILL if is_up else _DOWN_FILL,
            hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def render_price_detail_modal(symbol: str) -> None:
    """
    Open a modal dialog showing *symbol*'s full 30-day price chart (title,
    axes, grid, hover) plus min/max/avg price, with a Close button.

    Call this directly in response to a user action (e.g. a button click) —
    st.dialog-decorated functions open the moment they're called.
    """
    _price_detail_dialog(symbol)


@st.dialog("30-Day Price Detail")
def _price_detail_dialog(symbol: str) -> None:
    # symbol is whatever yfinance needs to resolve the quote (e.g.
    # "BTC-USD"); strip a crypto "-USD" suffix only for display text.
    display_symbol = symbol.removesuffix("-USD")
    history = get_price_history(symbol)

    if not history:
        st.error(f"Could not load price history for **{display_symbol}**.")
        if st.button("Close"):
            st.rerun()
        return

    dates = [row["date"] for row in history]
    prices = [row["close_price"] for row in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=prices,
            mode="lines+markers",
            line=dict(color="#4b8bff", width=2),
            marker=dict(size=5),
            name="Close",
            hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{display_symbol} — 30-Day Price History",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    min_price = min(prices)
    max_price = max(prices)
    avg_price = sum(prices) / len(prices)

    c1, c2, c3 = st.columns(3)
    c1.metric("Min", f"${min_price:.2f}")
    c2.metric("Max", f"${max_price:.2f}")
    c3.metric("Avg", f"${avg_price:.2f}")

    if st.button("Close", use_container_width=True):
        st.rerun()
