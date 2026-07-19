"""
Streamlit page: rule-based technical trading signals (RSI, MACD, moving-
average crossover).

Despite "Trading Signals" branding, there is no AI/ML here — see
src/trading_signals.py's module docstring for exactly what the signal and
confidence score are (and are not). This page repeats the disclaimer
because it's the thing a user actually reads before clicking "Scan All."
"""
from __future__ import annotations

import datetime

import pandas as pd
import streamlit as st

from scrapers.yahoo_finance_scraper import fetch_most_active
from src.trading_signals import compute_signal_accuracy, record_scan, scan_signals

_DEFAULT_SCAN_SIZE = 50
_SIGNAL_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}


def render_signals_page() -> None:
    st.title("📊 Trading Signals")
    st.caption(
        "Rule-based technical signals over live Yahoo Finance data — classical technical "
        "analysis (RSI, MACD, moving averages), not AI/ML, and not investment advice."
    )

    with st.expander("ℹ️ How this works — and what it isn't"):
        st.markdown(
            "- **Signal** (BUY/SELL/HOLD) and **Confidence** (0-100%) come from a fixed, "
            "fully mechanical weighted-vote rule over three indicators — RSI(14), MACD "
            "crossover, and the MA20/MA50 crossover. There is no machine learning or trained "
            "model involved.\n"
            "- **Confidence** measures how strongly those three indicators agree with each "
            "other, not a backtested win probability or a prediction of future returns.\n"
            "- **Accuracy** below is a real, growing log of past BUY/SELL signals checked "
            "against what the price actually did afterward — it starts empty and only "
            "reflects signals this app has actually generated, nothing pre-filled.\n"
            "- Nothing on this page is investment advice."
        )

    if "signals_rows" not in st.session_state:
        st.session_state["signals_rows"] = []
        st.session_state["signals_scanned_at"] = None

    col_scan, col_filter = st.columns([1, 2])
    with col_scan:
        if st.button("🔄 Scan All", use_container_width=True, type="primary"):
            _run_scan()

    rows: list[dict] = st.session_state["signals_rows"]
    scanned_at = st.session_state["signals_scanned_at"]

    if scanned_at:
        st.caption(f"Scanned at: {scanned_at}")

    if not rows:
        st.info(
            f"Click **Scan All** to run a live technical scan across the "
            f"{_DEFAULT_SCAN_SIZE} most active stocks."
        )
    else:
        with col_filter:
            filter_choice = st.radio(
                "Filter", ["All", "BUY only", "SELL only"],
                horizontal=True, label_visibility="collapsed", key="signals_filter",
            )
        filtered = _apply_filter(rows, filter_choice)
        if not filtered:
            st.warning(f"No {filter_choice.replace(' only', '')} signals in the last scan.")
        else:
            st.dataframe(
                _build_signals_dataframe(filtered),
                use_container_width=True, hide_index=True, column_config=_column_config(),
            )

    st.divider()
    _render_accuracy_section()


def _run_scan() -> None:
    with st.spinner(f"Scanning the {_DEFAULT_SCAN_SIZE} most active stocks…"):
        try:
            source_rows = fetch_most_active(count=_DEFAULT_SCAN_SIZE)
        except Exception as exc:
            st.error(f"Could not load the symbol universe to scan: {exc}")
            return

        if not source_rows:
            st.error("Could not load the symbol universe to scan — Yahoo Finance may be unreachable.")
            return

        symbols = tuple(r["symbol"] for r in source_rows)
        names = {r["symbol"]: r["name"] for r in source_rows}
        try:
            rows = scan_signals(symbols, _names=names)
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
            return

    if not rows:
        st.error("Scan completed but no signals could be computed — Yahoo Finance may be unreachable.")
        return

    st.session_state["signals_rows"] = rows
    st.session_state["signals_scanned_at"] = datetime.datetime.now().strftime("%H:%M today")
    record_scan(rows)
    st.rerun()


def _apply_filter(rows: list[dict], filter_choice: str) -> list[dict]:
    if filter_choice == "BUY only":
        return [r for r in rows if r["signal"] == "BUY"]
    if filter_choice == "SELL only":
        return [r for r in rows if r["signal"] == "SELL"]
    return rows


def _build_signals_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": r["symbol"],
                "Company": r["name"],
                "Signal": f"{_SIGNAL_EMOJI.get(r['signal'], '')} {r['signal']}",
                "Confidence": r["confidence"],
                "Price": r["price"],
                "RSI (14d)": r["rsi_14d"],
                "MACD": r["macd_trend"].capitalize(),
                "Reason": r["reason"],
            }
            for r in rows
        ]
    )


def _column_config() -> dict:
    return {
        "Confidence": st.column_config.NumberColumn(format="%.1f%%"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "RSI (14d)": st.column_config.NumberColumn(format="%.1f"),
    }


def _render_accuracy_section() -> None:
    st.subheader("📈 Signal History & Accuracy")

    try:
        accuracy = compute_signal_accuracy()
    except Exception as exc:
        st.warning(f"Could not compute signal accuracy: {exc}")
        return

    if accuracy is None:
        st.info(
            "No signal history old enough to evaluate yet. This starts genuinely empty — "
            "accuracy fills in automatically as BUY/SELL signals from past scans age past a "
            "day. Run **Scan All** periodically to build up a real track record."
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Overall hit rate", f"{accuracy['hit_rate']:.1f}%",
        help=f"{accuracy['total_evaluated']} signals evaluated",
    )
    c2.metric(
        "BUY hit rate",
        f"{accuracy['buy_hit_rate']:.1f}%" if accuracy["buy_hit_rate"] is not None else "N/A",
        help=f"{accuracy['buy_hits']}/{accuracy['buy_total']} BUY signals where price later rose",
    )
    c3.metric(
        "SELL hit rate",
        f"{accuracy['sell_hit_rate']:.1f}%" if accuracy["sell_hit_rate"] is not None else "N/A",
        help=f"{accuracy['sell_hits']}/{accuracy['sell_total']} SELL signals where price later fell",
    )
    st.caption(
        '"Hit" means price moved in the predicted direction by any amount since the signal '
        "was recorded — this system has no price-target concept, so it isn't measuring "
        "whether a specific target was reached. Past accuracy is not a guarantee of future results."
    )
