"""
Streamlit page: peer comparison dashboard.

Shows a company against its real, live-identified sector/market-cap peers
(src.peer_comparison.get_peers) across 10 valuation and quality metrics,
color-coded best/worst per column via a pandas Styler (st.dataframe renders
Styler background colors directly — no separate charting library needed for
a per-cell heatmap), plus a cheap/fair/expensive verdict.

Ticker auto-populate: shares the "ticker_input" session_state key with the
Search & Fetch page — whichever ticker the user most recently picked there
(including via a Market Screener row click, which already writes that key)
is this page's default, without needing its own redirect wiring.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src.peer_comparison import (
    METRIC_HIGHER_IS_BETTER,
    compare_peers,
    compute_valuation_verdict,
    get_peers,
    get_symbol_snapshot,
)

_BEST_COLOR = "background-color: rgba(40, 167, 69, 0.35)"
_WORST_COLOR = "background-color: rgba(220, 53, 69, 0.35)"

_VERDICT_RENDER = {
    "UNDERVALUED": ("success", "🟢"),
    "OVERVALUED": ("warning", "🔴"),
    "FAIRLY VALUED": ("info", "🟡"),
}


def render_peer_comparison_page() -> None:
    st.title("🏆 Peer Comparison")
    st.caption(
        "Compare a company against its real sector peers — identified live by sector and "
        "market cap, not a hardcoded list — across 10 valuation and quality metrics."
    )

    if "peer_comparison_ticker" not in st.session_state:
        st.session_state["peer_comparison_ticker"] = st.session_state.get("ticker_input", "AAPL")
    symbol = st.text_input("Ticker symbol", key="peer_comparison_ticker").upper().strip()

    if not symbol:
        st.info("Enter a ticker symbol to compare.")
        return

    with st.spinner(f"Identifying {symbol}'s peers…"):
        try:
            peers = get_peers(symbol)
        except Exception as exc:
            st.error(f"Could not find peers for {symbol}: {exc}")
            return

    if not peers:
        st.warning(
            f"Could not identify peer companies for **{symbol}** — it may be an invalid "
            "ticker, or Yahoo Finance has no sector data on file for it."
        )
        return

    with st.spinner("Loading comparison metrics…"):
        try:
            df = compare_peers(symbol, tuple(peers))
        except Exception as exc:
            st.error(f"Could not build comparison table: {exc}")
            return

    if df.empty or symbol not in df.index:
        st.error(f"No metrics data available for {symbol}.")
        return

    snapshot = get_symbol_snapshot(symbol)
    if snapshot:
        _render_company_overview(snapshot)

    st.divider()
    st.subheader("📊 Peer Metrics Comparison")
    resolved_peers = [s for s in df.index if s != symbol]
    st.caption(f"Peers: {', '.join(resolved_peers)}" if resolved_peers else "No peer data resolved.")
    st.dataframe(_style_comparison_table(df, symbol), use_container_width=True)
    st.caption("Green = best in group, red = worst in group, per column.")

    st.divider()
    _render_verdict(df, symbol)


def _render_company_overview(snapshot: dict) -> None:
    st.subheader(f"{snapshot['name']} ({snapshot['symbol']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sector", snapshot.get("sector") or "N/A")
    c2.metric("Market Cap", _format_market_cap(snapshot.get("market_cap")))
    price = snapshot.get("price")
    c3.metric("Price", f"${price:.2f}" if price is not None else "N/A")
    pe = snapshot.get("P/E Ratio")
    c4.metric("P/E Ratio", f"{pe:.1f}" if pe is not None else "N/A")


def _format_market_cap(value: Optional[float]) -> str:
    if not value:
        return "N/A"
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"${value / threshold:.2f}{suffix}"
    return f"${value:.2f}"


# ---------------------------------------------------------------------------
# Table styling
# ---------------------------------------------------------------------------

def _style_comparison_table(df: pd.DataFrame, symbol: str):
    return (
        df.style
        .apply(_highlight_best_worst, axis=0)
        .apply(_bold_target_row, axis=1, symbol=symbol)
        .format(precision=2, na_rep="N/A")
    )


def _highlight_best_worst(col: pd.Series) -> list[str]:
    higher_is_better = METRIC_HIGHER_IS_BETTER.get(col.name, True)
    valid = col.dropna()
    if valid.empty:
        return ["" for _ in col]

    best_val = valid.max() if higher_is_better else valid.min()
    worst_val = valid.min() if higher_is_better else valid.max()

    styles = []
    for v in col:
        if pd.isna(v) or best_val == worst_val:
            styles.append("")
        elif v == best_val:
            styles.append(_BEST_COLOR)
        elif v == worst_val:
            styles.append(_WORST_COLOR)
        else:
            styles.append("")
    return styles


def _bold_target_row(row: pd.Series, symbol: str) -> list[str]:
    if row.name == symbol:
        return ["font-weight: bold"] * len(row)
    return ["" for _ in row]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _render_verdict(df: pd.DataFrame, symbol: str) -> None:
    st.subheader("💡 Valuation Verdict")
    verdict = compute_valuation_verdict(df, symbol)

    if verdict["verdict"] == "INSUFFICIENT_DATA":
        st.info(f"Not enough data to compute a valuation verdict — {verdict['reason']}")
        return

    method, emoji = _VERDICT_RENDER[verdict["verdict"]]
    getattr(st, method)(f"{emoji} **{symbol} is {verdict['verdict']} vs peers.**  \n{verdict['reason']}")
    st.caption(
        "Heuristic only: cheap = P/E below peer median, paired with above-median revenue "
        "growth = undervalued (and the inverse for overvalued); anything mixed is fairly "
        "valued. Not investment advice."
    )
