"""
Streamlit component: 90-day Prophet price forecast overlaid on 2 years of
historical closes, for a single ticker.
"""
from __future__ import annotations

import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.financial_forecast_prophet import forecast_stock_price_prophet


def render_forecast_chart(symbol: str) -> None:
    """Render the historical + forecast price chart for *symbol*."""
    symbol = (symbol or "").upper().strip()
    if not symbol:
        st.info("Select a ticker to see its price forecast.")
        return

    with st.spinner(f"Forecasting {symbol}…"):
        data = forecast_stock_price_prophet(symbol, days_forward=90)

    if data is None:
        st.error(
            f"Could not build a price forecast for **{symbol}** — the symbol may not exist "
            "on Yahoo Finance, there may not be enough price history (Prophet needs roughly "
            "2 years / ~500 trading days), or the data source is temporarily unreachable."
        )
        return

    fig = _build_forecast_figure(symbol, data)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _build_forecast_figure(symbol: str, data: dict) -> go.Figure:
    hist: pd.DataFrame = data["historical_prices"]
    fcst: pd.DataFrame = data["forecast_prices"]

    fig = go.Figure()

    # 1 — historical price (bottom layer)
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist["Close"],
            mode="lines",
            line=dict(color="#4b8bff", width=2),
            name="Historical Close",
            hovertemplate="%{x|%Y-%m-%d}<br>Historical Close: $%{y:.2f}<extra></extra>",
        )
    )

    # 2 — forecast mean
    fig.add_trace(
        go.Scatter(
            x=fcst.index, y=fcst["mean"],
            mode="lines",
            line=dict(color="#fd7e14", width=2, dash="dash"),
            name="Forecast Mean",
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast Mean: $%{y:.2f}<extra></extra>",
        )
    )

    # 3 — confidence band (top layer): invisible upper bound + filled lower bound
    fig.add_trace(
        go.Scatter(
            x=fcst.index, y=fcst["upper"],
            mode="lines",
            line=dict(width=0),
            name="95% Confidence Interval",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Upper bound: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fcst.index, y=fcst["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(253, 126, 20, 0.18)",
            name="95% Confidence Interval",
            hovertemplate="%{x|%Y-%m-%d}<br>Lower bound: $%{y:.2f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=hist.index[-1],
        line_dash="dot",
        line_color="rgba(150,150,150,0.5)",
        annotation_text="Forecast start",
        annotation_position="top",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{symbol} 90-Day Price Forecast"
                f"<br><sup>{data['model_type']} | MAPE: {data['mape']:.2f}%</sup>"
            )
        ),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        height=440,
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig
