"""
Streamlit component: financial profile dashboard for a single ticker.

Wraps src.financial_metrics behind a session-state cache keyed by symbol
(on top of that module's own @st.cache_data(ttl=3600)), so a page rerun
triggered by an unrelated widget doesn't redo the six lookups + chart builds.
"""
from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.financial_metrics import (
    get_analyst_ratings,
    get_balance_sheet,
    get_cash_flow,
    get_company_profile,
    get_earnings_history,
    get_income_statement,
)

_UI_CACHE_TTL = datetime.timedelta(hours=1)


def render_financial_dashboard(symbol: str) -> None:
    """Render the full financial profile dashboard (4 sections) for *symbol*."""
    symbol = (symbol or "").upper().strip()
    if not symbol:
        st.warning("Enter a ticker symbol above to view its financial profile.")
        return

    data = _get_dashboard_data(symbol)

    if data["profile"] is None:
        st.error(
            f"Could not load financial data for **{symbol}** — the symbol may be invalid, "
            "or Yahoo Finance is temporarily unreachable."
        )
        return

    st.caption(f"Metrics last fetched: {data['fetched_at'].strftime('%Y-%m-%d %H:%M:%S')}")

    _render_company_profile(data["profile"])
    st.divider()
    _render_income_statement(data["income"])
    st.divider()
    _render_balance_and_cashflow(data["balance"], data["cashflow"])
    st.divider()
    _render_earnings_and_ratings(data["earnings"], data["ratings"])


# ---------------------------------------------------------------------------
# UI-level cache (session_state, on top of financial_metrics.py's own TTL cache)
# ---------------------------------------------------------------------------

def _get_dashboard_data(symbol: str) -> dict:
    cache_key = f"fin_dashboard_data__{symbol}"
    cached = st.session_state.get(cache_key)
    is_stale = cached is None or (datetime.datetime.now() - cached["fetched_at"]) > _UI_CACHE_TTL

    if is_stale:
        with st.spinner(f"Loading financial profile for {symbol}…"):
            st.session_state[cache_key] = {
                "profile": get_company_profile(symbol),
                "income": get_income_statement(symbol),
                "balance": get_balance_sheet(symbol),
                "cashflow": get_cash_flow(symbol),
                "earnings": get_earnings_history(symbol),
                "ratings": get_analyst_ratings(symbol),
                "fetched_at": datetime.datetime.now(),
            }
    return st.session_state[cache_key]


# ---------------------------------------------------------------------------
# Section 1 — Company profile
# ---------------------------------------------------------------------------

def _render_company_profile(profile: dict) -> None:
    st.subheader("🏢 Company Profile")

    row1 = st.columns(2)
    row1[0].markdown(f"**Name**  \n{profile.get('name') or 'N/A'}")
    row1[1].markdown(f"**Sector**  \n{profile.get('sector') or 'N/A'}")

    row2 = st.columns(2)
    row2[0].markdown(f"**Industry**  \n{profile.get('industry') or 'N/A'}")
    row2[1].markdown(f"**Website**  \n{profile.get('website') or 'N/A'}")

    row3 = st.columns(2)
    employees = profile.get("employee_count")
    with row3[0]:
        st.metric("Employees", f"{employees:,}" if employees else "N/A")
    row3[1].markdown(f"**CEO**  \n{profile.get('ceo') or 'N/A'}")

    if profile.get("business_summary"):
        with st.expander("Business summary"):
            st.write(profile["business_summary"])


# ---------------------------------------------------------------------------
# Section 2 — Income statement
# ---------------------------------------------------------------------------

def _render_income_statement(income_df: Optional[pd.DataFrame]) -> None:
    st.subheader("💰 Income Statement (Last 5 Years)")
    if income_df is None or income_df.empty:
        st.info("No data available.")
        return

    st.dataframe(income_df, use_container_width=True)

    years = list(income_df.columns)[::-1]  # chronological, oldest -> newest
    revenue = [income_df.loc["Revenue", y] for y in years]
    net_income = [income_df.loc["Net Income", y] for y in years]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=revenue, name="Revenue", marker_color="#4b8bff"))
    fig.add_trace(go.Bar(x=years, y=net_income, name="Net Income", marker_color="#28a745"))
    fig.update_layout(
        barmode="group",
        title="Revenue vs Net Income",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 3 — Balance sheet + cash flow
# ---------------------------------------------------------------------------

def _render_balance_and_cashflow(
    balance_df: Optional[pd.DataFrame], cashflow_df: Optional[pd.DataFrame]
) -> None:
    st.subheader("🏦 Balance Sheet & Cash Flow")
    left, right = st.columns(2)

    with left:
        st.markdown("**Balance Sheet**")
        if balance_df is None or balance_df.empty:
            st.info("No data available.")
        else:
            st.dataframe(balance_df, use_container_width=True)
            years = list(balance_df.columns)[::-1]
            equity = [balance_df.loc["Shareholders Equity", y] for y in years]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=years, y=equity, mode="lines+markers", name="Shareholders Equity",
                    line=dict(color="#4b8bff", width=2.5), marker=dict(size=8),
                )
            )
            fig.update_layout(
                title="Shareholders Equity Trend",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Cash Flow**")
        if cashflow_df is None or cashflow_df.empty:
            st.info("No data available.")
        else:
            st.dataframe(cashflow_df, use_container_width=True)
            years = list(cashflow_df.columns)[::-1]
            ocf = [cashflow_df.loc["Operating Cash Flow", y] for y in years]
            fcf = [cashflow_df.loc["Free Cash Flow", y] for y in years]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=years, y=ocf, name="Operating CF", marker_color="#4b8bff"))
            fig.add_trace(go.Bar(x=years, y=fcf, name="Free CF", marker_color="#fd7e14"))
            fig.update_layout(
                barmode="group",
                title="Operating vs Free Cash Flow",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 4 — Earnings history + analyst ratings
# ---------------------------------------------------------------------------

def _surprise_color(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"color: {'#28a745' if value >= 0 else '#dc3545'}; font-weight: 600"


def _render_earnings_and_ratings(earnings: Optional[list[dict]], ratings: Optional[dict]) -> None:
    st.subheader("📅 Earnings & Analyst Ratings")
    left, right = st.columns(2)

    with left:
        st.markdown("**Earnings History (Last 8 Quarters)**")
        if not earnings:
            st.info("No data available.")
        else:
            df = pd.DataFrame(
                [
                    {
                        "Date": e["date"],
                        "EPS Actual": e["eps_actual"],
                        "EPS Estimate": e["eps_estimate"],
                        "Surprise %": e["surprise_percent"],
                    }
                    for e in earnings
                ]
            )
            styled = df.style.map(_surprise_color, subset=["Surprise %"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

    with right:
        st.markdown("**Analyst Ratings**")
        if not ratings:
            st.info("N/A — no analyst coverage data available.")
        else:
            avg = ratings.get("average_rating")
            st.metric(
                "Average Rating (1=Strong Buy, 5=Strong Sell)",
                f"{avg:.2f}" if avg is not None else "N/A",
            )

            buy = ratings.get("buy_count") or 0
            hold = ratings.get("hold_count") or 0
            sell = ratings.get("sell_count") or 0
            if buy + hold + sell > 0:
                pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Buy", "Hold", "Sell"],
                            values=[buy, hold, sell],
                            marker=dict(colors=["#28a745", "#ffa500", "#dc3545"]),
                            hole=0.4,
                        )
                    ]
                )
                pie.update_layout(
                    height=280, margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(pie, use_container_width=True)
            else:
                st.caption("No buy/hold/sell breakdown available.")

            pt_current = ratings.get("price_target_current")
            pt_high = ratings.get("price_target_high")
            pt_low = ratings.get("price_target_low")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current", f"${pt_current:.2f}" if pt_current is not None else "N/A")
            c2.metric("High", f"${pt_high:.2f}" if pt_high is not None else "N/A")
            c3.metric("Low", f"${pt_low:.2f}" if pt_low is not None else "N/A")
