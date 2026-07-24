"""
Streamlit page: rule-based technical trading signals.

Two independent signal systems, selectable via the mode toggle:
  - Classic:  RSI(14) + MACD + MA20/50 crossover (src.trading_signals.scan_signals)
  - Ensemble: RSI(14) + Bollinger Bands + Stochastic RSI + volume spike
              (src.trading_signals.scan_signals_ensemble)

Despite "Trading Signals" branding, neither is AI/ML — see
src/trading_signals.py's module docstring for exactly what signal and
confidence are (and are not), and src/signal_backtest.py for the real,
honest backtest results behind the Ensemble system (including its real
limits — see that module's docstring on what "accuracy" does and doesn't
mean here). This page repeats the disclaimer because it's the thing a user
actually reads before clicking "Scan All."
"""
from __future__ import annotations

import datetime

import pandas as pd
import streamlit as st

from scrapers.yahoo_finance_scraper import fetch_most_active
from src.trading_signals import (
    ENSEMBLE_VOTE_THRESHOLD,
    compute_signal_accuracy,
    record_scan,
    scan_signals,
    scan_signals_ensemble,
)

_DEFAULT_SCAN_SIZE = 50
_SIGNAL_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
_CLASSIC = "Classic (RSI + MACD + MA)"
_ENSEMBLE = "Ensemble (RSI + Bollinger + StochRSI + Volume)"


def render_signals_page() -> None:
    st.title("📊 Trading Signals")
    st.caption(
        "Rule-based technical signals over live Yahoo Finance data — classical technical "
        "analysis, not AI/ML, and not investment advice."
    )

    mode = st.radio("Signal system", [_CLASSIC, _ENSEMBLE], horizontal=True, key="signals_mode")

    if mode == _CLASSIC:
        with st.expander("ℹ️ How this works — and what it isn't"):
            st.markdown(
                "- **Signal** (BUY/SELL/HOLD) and **Confidence** (0-100%) come from a fixed, "
                "fully mechanical weighted-vote rule over three indicators — RSI(14), MACD "
                "crossover, and the MA20/MA50 crossover. There is no machine learning or "
                "trained model involved.\n"
                "- **Confidence** measures how strongly those three indicators agree with "
                "each other, not a backtested win probability or a prediction of future "
                "returns.\n"
                "- Nothing on this page is investment advice."
            )
    else:
        with st.expander("ℹ️ How this works — and what it isn't"):
            st.markdown(
                "- **Signal** comes from 4 indicators voting: RSI(14) oversold/overbought, "
                "price breaching a Bollinger Band, a Stochastic RSI %K/%D crossover, and a "
                "volume spike (≥2x the 20-day average — this one has no direction of its "
                "own, it just confirms whichever move the other three are already leaning "
                f"toward). BUY/SELL fires once **{ENSEMBLE_VOTE_THRESHOLD} of 4** agree; "
                "otherwise HOLD.\n"
                "- **Real backtest, real limits**: on 5 large-cap stocks over Jan 2023–Jan "
                "2025, this scored ~78% directional accuracy pooled — but a naive "
                '"always predict UP" baseline (zero indicators) scored ~83% over the same '
                "stretch, because it was a strong bull market. The genuine edge measured "
                "was on the **SELL side** (~76% vs a ~69% always-predict-down baseline, and "
                "vs ~56% for the classic system's SELL logic) — BUY accuracy mostly "
                "reflects market drift, not indicator skill. \"Accuracy\" here means price "
                "closed in the predicted direction *at any point* in the next 5 trading "
                "days — no fees, slippage, or execution modeled. This is not a backtested "
                "guarantee of future performance, and nothing here is investment advice.\n"
                "- Fewer signals than Classic by design — it only fires when multiple "
                "independent indicators agree, which is a real trade-off: more selective, "
                "not necessarily more total opportunities."
            )

    if "signals_rows" not in st.session_state:
        st.session_state["signals_rows"] = {}
        st.session_state["signals_scanned_at"] = {}

    col_scan, col_filter = st.columns([1, 2])
    with col_scan:
        if st.button("🔄 Scan All", use_container_width=True, type="primary"):
            _run_scan(mode)

    rows: list[dict] = st.session_state["signals_rows"].get(mode, [])
    scanned_at = st.session_state["signals_scanned_at"].get(mode)

    if scanned_at:
        st.caption(f"Scanned at: {scanned_at}")

    if not rows:
        st.info(
            f"Click **Scan All** to run a live {mode.split(' (')[0].lower()} scan across the "
            f"{_DEFAULT_SCAN_SIZE} most active stocks."
        )
    else:
        with col_filter:
            filter_choice = st.radio(
                "Filter", ["All", "BUY only", "SELL only"],
                horizontal=True, label_visibility="collapsed", key=f"signals_filter_{mode}",
            )
        filtered = _apply_filter(rows, filter_choice)
        if not filtered:
            st.warning(f"No {filter_choice.replace(' only', '')} signals in the last scan.")
        elif mode == _CLASSIC:
            st.dataframe(
                _build_classic_dataframe(filtered),
                use_container_width=True, hide_index=True, column_config=_classic_column_config(),
            )
        else:
            st.dataframe(
                _build_ensemble_dataframe(filtered),
                use_container_width=True, hide_index=True, column_config=_ensemble_column_config(),
            )

    st.divider()
    _render_accuracy_section()


def _run_scan(mode: str) -> None:
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
            if mode == _CLASSIC:
                rows = scan_signals(symbols, _names=names)
            else:
                rows = scan_signals_ensemble(symbols, _names=names)
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
            return

    if not rows:
        st.error("Scan completed but no signals could be computed — Yahoo Finance may be unreachable.")
        return

    st.session_state["signals_rows"][mode] = rows
    st.session_state["signals_scanned_at"][mode] = datetime.datetime.now().strftime("%H:%M today")
    if mode == _CLASSIC:
        record_scan(rows)  # signal-history accuracy tracking only exists for the classic system so far
    st.rerun()


def _apply_filter(rows: list[dict], filter_choice: str) -> list[dict]:
    if filter_choice == "BUY only":
        return [r for r in rows if r["signal"] == "BUY"]
    if filter_choice == "SELL only":
        return [r for r in rows if r["signal"] == "SELL"]
    return rows


def _build_classic_dataframe(rows: list[dict]) -> pd.DataFrame:
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


def _classic_column_config() -> dict:
    return {
        "Confidence": st.column_config.NumberColumn(format="%.1f%%"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "RSI (14d)": st.column_config.NumberColumn(format="%.1f"),
    }


def _build_ensemble_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": r["symbol"],
                "Company": r["name"],
                "Signal": f"{_SIGNAL_EMOJI.get(r['signal'], '')} {r['signal']}",
                "Confidence": r["confidence"],
                "Votes": f"{r['buy_votes']} buy / {r['sell_votes']} sell",
                "Price": r["price"],
                "RSI (14d)": r["rsi_14d"],
                "Bollinger": f"{r['bollinger_lower']:.1f} – {r['bollinger_upper']:.1f}",
                "StochRSI": f"K {r['stoch_k']:.0f} / D {r['stoch_d']:.0f}",
                "EMA Trend": r["ema_trend"].capitalize(),
                "Reason": r["reason"],
            }
            for r in rows
        ]
    )


def _ensemble_column_config() -> dict:
    return {
        "Confidence": st.column_config.NumberColumn(format="%.1f%%"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "RSI (14d)": st.column_config.NumberColumn(format="%.1f"),
    }


def _render_accuracy_section() -> None:
    st.subheader("📈 Signal History & Accuracy")
    st.caption(
        "This live-tracked history is for the **Classic** system only — it logs every "
        "BUY/SELL scan result and checks it against what price actually did afterward. The "
        "Ensemble system's accuracy comes from the historical backtest described in its "
        "expander above (src/signal_backtest.py), not from live tracking yet."
    )

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
