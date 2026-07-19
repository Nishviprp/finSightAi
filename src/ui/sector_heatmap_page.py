"""
Streamlit page: sector performance heatmap.

The 11 GICS sectors, laid out as a 3x4 grid of colored Plotly markers
(green/yellow/red by today's % change) so each cell gets a real hover
tooltip (30-day trend + top mover) and is independently clickable — a
plain data table can't do either. Clicking a cell opens a modal (via
st.dialog, same pattern as sparkline_renderer.render_price_detail_modal)
with the sector's real top-10 stocks ranked by 30-day gain.
"""
from __future__ import annotations

import datetime
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from src.sector_analytics import get_sector_performance, get_sector_top_stocks

_GRID_COLS = 4
_HEATMAP_BULLISH_THRESHOLD = 2.0  # this page's own color rule — separate from
                                  # sector_analytics' momentum thresholds

_RED = "#dc3545"
_YELLOW = "#ffc107"
_GREEN = "#28a745"


def render_sector_heatmap_page() -> None:
    st.title("🌡️ Sector Heatmap")
    st.caption(
        "Live S&P sector performance via SPDR sector ETFs. Hover a cell for its 30-day trend "
        "and top mover; click a cell for the sector's top 10 stocks."
    )

    with st.spinner("Loading sector performance…"):
        try:
            data = get_sector_performance()
        except Exception as exc:
            st.error(f"Could not load sector performance: {exc}")
            return

    if data is None:
        st.error(
            "Could not load sector performance — Yahoo Finance may be temporarily unreachable."
        )
        return

    sectors = data["sectors"]
    st.caption(f"Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    fig, sector_order = _build_heatmap_figure(sectors)
    event = st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun", selection_mode="points", key="sector_heatmap",
    )

    clicked_sector = _extract_clicked_sector(event)
    if clicked_sector and clicked_sector in sectors:
        _open_drilldown_dialog(clicked_sector, sectors[clicked_sector]["symbol"])

    st.divider()
    _render_legend()


# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------

def _bucket_color(change_percent: float) -> str:
    if change_percent >= _HEATMAP_BULLISH_THRESHOLD:
        return _GREEN
    if change_percent >= 0.0:
        return _YELLOW
    return _RED


def _build_heatmap_figure(sectors: dict[str, dict]) -> tuple[go.Figure, list[str]]:
    sector_order = list(sectors.keys())
    xs, ys, colors, texts, hover_texts, customdata = [], [], [], [], [], []

    for i, name in enumerate(sector_order):
        row_data = sectors[name]
        xs.append(i % _GRID_COLS)
        ys.append(-(i // _GRID_COLS))  # negative row so row 0 renders at the top
        colors.append(_bucket_color(row_data["change_percent"]))
        texts.append(f"{name}<br>{row_data['change_percent']:+.2f}%")

        change_30day = row_data["change_30day"]
        change_30day_text = f"{change_30day:+.2f}%" if change_30day is not None else "N/A"
        hover_texts.append(
            f"<b>{name}</b> ({row_data['symbol']})<br>"
            f"Today: {row_data['change_percent']:+.2f}%<br>"
            f"30-day trend: {change_30day_text}<br>"
            f"Top mover: {row_data['top_stock'] or 'N/A'}<br>"
            f"Momentum: {row_data['momentum'].capitalize()}<br>"
            "<i>Click for top 10 stocks</i>"
        )
        customdata.append(name)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(symbol="square", size=110, color=colors, line=dict(width=2, color="white")),
            text=texts,
            textposition="middle center",
            textfont=dict(color="white", size=13),
            customdata=customdata,
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.7, _GRID_COLS - 0.3]),
        yaxis=dict(visible=False, range=[-2.7, 0.7], scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig, sector_order


def _extract_clicked_sector(event) -> Optional[str]:
    """event.selection is normally an attribute-and-dict-accessible object
    (Streamlit's PlotlySelectionState), but accept a plain dict too — cheap
    robustness against how the selection payload happens to be shaped.
    """
    if not event:
        return None
    selection = event.get("selection") if isinstance(event, dict) else event.selection
    if not selection:
        return None
    points = selection.get("points") if isinstance(selection, dict) else selection.points
    if not points:
        return None
    custom = points[0].get("customdata")
    return custom[0] if custom else None


def _render_legend() -> None:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<span style="color:{_GREEN}">■</span> +{_HEATMAP_BULLISH_THRESHOLD:.0f}% or better', unsafe_allow_html=True)
    c2.markdown(f'<span style="color:{_YELLOW}">■</span> 0% to +{_HEATMAP_BULLISH_THRESHOLD:.0f}%', unsafe_allow_html=True)
    c3.markdown(f'<span style="color:{_RED}">■</span> Negative', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Drill-down modal
# ---------------------------------------------------------------------------

def _open_drilldown_dialog(sector_name: str, etf_symbol: str) -> None:
    _drilldown_dialog(sector_name, etf_symbol)


@st.dialog("Top 10 Stocks")
def _drilldown_dialog(sector_name: str, etf_symbol: str) -> None:
    st.subheader(f"{sector_name} ({etf_symbol})")
    st.caption("Ranked by each stock's own 30-day price change.")

    with st.spinner("Loading top stocks…"):
        try:
            top_stocks = get_sector_top_stocks(sector_name, limit=10)
        except Exception as exc:
            st.error(f"Could not load top stocks: {exc}")
            top_stocks = []

    if not top_stocks:
        st.info("No stock data available for this sector right now.")
    else:
        for rank, stock in enumerate(top_stocks, start=1):
            color = _GREEN if stock["change_30day"] >= 0 else _RED
            st.markdown(
                f"**{rank}. {stock['symbol']}** — {stock['name']}  \n"
                f"${stock['price']:.2f} · "
                f'<span style="color:{color}">{stock["change_30day"]:+.2f}%</span> (30-day)',
                unsafe_allow_html=True,
            )

    if st.button("Close", use_container_width=True):
        st.rerun()
