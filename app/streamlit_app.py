"""
FinSight Streamlit Dashboard — self-contained (no FastAPI required).

Runs standalone on Streamlit Cloud or locally with:
    streamlit run app/streamlit_app.py

Three pages:
  1. Search & Fetch  — trigger EDGAR ingestion + AI analysis
  2. Insights        — per-quarter sentiment, risks, guidance cards
  3. Trend Analysis  — Plotly sentiment chart + QuarterComparator narrative
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import streamlit as st

# ── project root on sys.path (must happen before src.* imports) ─────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── page config — MUST be the first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── env / secrets ────────────────────────────────────────────────────────────
# Load .env locally; on Streamlit Cloud read from st.secrets instead.
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error(
            "**GEMINI_API_KEY not set.**\n\n"
            "• **Locally:** add `GEMINI_API_KEY=your_key` to a `.env` file in the project root.\n"
            "• **Streamlit Cloud:** add it under *Settings → Secrets*."
        )
        st.stop()

# ── src imports (after sys.path and env are ready) ───────────────────────────
from src.ingest.edgar import TranscriptFetcher          # noqa: E402
from src.process.store import TranscriptStore           # noqa: E402
from src.analyze.analyzer import TranscriptAnalyzer     # noqa: E402
from src.analyze.comparator import QuarterComparator    # noqa: E402
from src.analyze.models import QuarterTrend as QT       # noqa: E402
from src.ui.stock_screener import render_stock_screener  # noqa: E402
from src.ui.financial_dashboard import render_financial_dashboard  # noqa: E402
from src.ui.forecast_chart import render_forecast_chart  # noqa: E402
from src.ui.signals_page import render_signals_page  # noqa: E402
from src.ui.sector_heatmap_page import render_sector_heatmap_page  # noqa: E402
from src.ui.peer_comparison_page import render_peer_comparison_page  # noqa: E402

# ── cached resource: single shared DB connection ─────────────────────────────
@st.cache_resource
def _store() -> TranscriptStore:
    return TranscriptStore()


# ── cached helper: EDGAR ticker validation (TTL 5 min) ───────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _validate(ticker: str) -> dict:
    """Returns get_supported_info dict; never raises."""
    try:
        return TranscriptFetcher().get_supported_info(ticker)
    except Exception as exc:
        return {"exists": False, "company_name": "", "cik": "", "total_8k_count": 0,
                "_error": str(exc)}


# ── UI helpers ────────────────────────────────────────────────────────────────

def sentiment_color(score: float) -> str:
    if score > 0.2:   return "green"
    if score < -0.2:  return "red"
    return "gray"


def direction_arrow(direction: str) -> str:
    return {"up": "↑", "down": "↓", "flat": "→", "unclear": "~"}.get(direction, "~")


def severity_badge(severity: str) -> str:
    colors = {"high": "#ff4b4b", "medium": "#ffa500", "low": "#4b8bff"}
    color  = colors.get(severity, "#888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.78em;font-weight:600">{severity.upper()}</span>'
    )


def trend_badge(direction: str) -> str:
    cfg = {
        "improving": ("#28a745", "▲ IMPROVING"),
        "declining":  ("#dc3545", "▼ DECLINING"),
        "volatile":   ("#fd7e14", "⚡ VOLATILE"),
        "stable":     ("#6c757d", "● STABLE"),
    }
    color, label = cfg.get(direction, ("#6c757d", direction.upper()))
    return (
        f'<span style="background:{color};color:white;padding:4px 14px;'
        f'border-radius:6px;font-size:1em;font-weight:700">{label}</span>'
    )


def _render_footer() -> None:
    """Attribution footer. Called at the end of every page — and also right
    before every early st.stop(), since st.stop() halts the whole script,
    not just the current page's if/elif branch, and would otherwise skip it.
    """
    st.divider()
    st.caption(
        "📊 Data sourced from Yahoo Finance via yfinance.  \n"
        "Financial metrics analyzed with Gemini AI.  \n"
        "Historical EDGAR data from SEC."
    )


# ── sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("📈 FinSight")
st.sidebar.caption("AI Earnings Call Analyzer")

_NAV_PAGES = [
    "Search & Fetch", "Insights", "Trend Analysis", "Market Screener",
    "Trading Signals", "Sector Heatmap", "Peer Comparison",
]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = _NAV_PAGES[0]
# A widget's session_state key can't be reassigned once that widget has been
# instantiated in the current run (the radio below claims "nav_page" on every
# run). Other pages that want to redirect here stash the target in this
# unbound key instead and call st.rerun(); we apply it before the radio runs.
if "_pending_nav_page" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("_pending_nav_page")
page = st.sidebar.radio("Navigate", _NAV_PAGES, key="nav_page")
st.sidebar.success("● Running (standalone)", icon="✅")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Search & Fetch
# ════════════════════════════════════════════════════════════════════════════

if page == "Search & Fetch":
    st.title("🔍 Search & Fetch Transcripts")
    st.caption("Pull earnings filings from SEC EDGAR and analyse them with Gemini AI.")

    _now         = datetime.datetime.now()
    _cur_year    = _now.year
    _cur_quarter = (_now.month - 1) // 3 + 1

    col1, col2 = st.columns([2, 3])
    with col1:
        if "ticker_input" not in st.session_state:
            st.session_state["ticker_input"] = "AAPL"
        ticker = st.text_input("Ticker symbol", key="ticker_input").upper().strip()

        # Live ticker validation
        if ticker:
            _info = _validate(ticker)
            if _info.get("exists"):
                st.success(f"✓ {_info.get('company_name', ticker)}", icon="✅")
            else:
                st.warning(f"⚠ '{ticker}' not found on EDGAR", icon="⚠️")

    with col2:
        _default_start = max(2019, _cur_year - 2)
        year_range = st.slider(
            "Year range",
            2019, _cur_year,
            (_default_start, _cur_year),
        )
        st.caption(
            f"Searching up to Q{_cur_quarter} {_cur_year} · "
            "Updates automatically each quarter"
        )

    show_financial_profile = st.checkbox("📊 Show Financial Profile", key="show_financial_profile")

    st.divider()

    # ── Fetch ──────────────────────────────────────────────────────────────
    if st.button("📥 Fetch transcripts from EDGAR", use_container_width=True):
        store   = _store()
        fetcher = TranscriptFetcher()
        with st.spinner(f"Fetching {ticker} filings ({year_range[0]}–{year_range[1]})…"):
            try:
                docs, status = fetcher.fetch_by_ticker(
                    ticker, year_range[0], year_range[1]
                )
            except Exception as exc:
                st.error(f"EDGAR fetch failed: {exc}")
                docs, status = [], {}

        if status:
            saved   = sum(1 for d in docs if store.save(d))
            skipped = len(docs) - saved

            c1, c2, c3 = st.columns(3)
            c1.metric("8-Ks checked", status.get("total_filings_checked", 0))
            c2.metric("Saved",        saved)
            c3.metric("Skipped (dup)", skipped)

            if status.get("errors"):
                with st.expander(f"⚠ {len(status['errors'])} warning(s)", expanded=False):
                    for e in status["errors"]:
                        st.caption(e)

            if saved == 0 and len(docs) == 0:
                st.warning(
                    "No transcripts found. "
                    "Try a wider year range or verify the ticker has 8-K filings on EDGAR."
                )
            elif saved == 0:
                st.info("All fetched documents were already in the database (duplicates skipped).")
            else:
                st.success(f"✅ {saved} new transcript(s) saved.")

    st.divider()

    # ── Stored docs table ──────────────────────────────────────────────────
    st.subheader(f"Stored documents for {ticker}")
    store    = _store()
    all_docs = store.get_by_ticker(ticker)
    docs     = [d for d in all_docs if year_range[0] <= (d.get("year") or 0) <= year_range[1]]

    if docs:
        import pandas as pd
        df = pd.DataFrame([
            {
                "Date":    d.get("filed_date", ""),
                "Quarter": f"Q{d.get('quarter','')} {d.get('year','')}",
                "Form":    d.get("form_type", ""),
                "Chars":   len(d.get("raw_text") or ""),
                "URL":     d.get("url", ""),
            }
            for d in docs
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif all_docs:
        st.info(
            f"No stored transcripts in {year_range[0]}–{year_range[1]} for {ticker}. "
            f"{len(all_docs)} document(s) exist outside this range — widen the year range slider to see them."
        )
    else:
        st.info("No transcripts stored yet — press Fetch above.")

    st.divider()

    # ── Analyse ────────────────────────────────────────────────────────────
    st.subheader("🤖 AI Analysis")
    if st.button("Analyze with AI  (Gemini)", use_container_width=True, type="primary"):
        store    = _store()
        all_docs = store.get_by_ticker(ticker)

        if not all_docs:
            st.warning("No transcripts for this ticker — fetch them first.")
        else:
            try:
                analyzer = TranscriptAnalyzer()
            except EnvironmentError as exc:
                st.error(str(exc))
                analyzer = None

            if analyzer:
                analyzed = 0
                skipped  = 0
                prog     = st.progress(0, text="Starting analysis…")

                for i, doc in enumerate(all_docs):
                    q = doc.get("quarter") or 0
                    y = doc.get("year")    or 0
                    prog.progress(
                        (i + 1) / len(all_docs),
                        text=f"Analyzing Q{q} {y}… ({i+1}/{len(all_docs)})",
                    )
                    if q and y and store.insight_exists(ticker, q, y):
                        skipped += 1
                        continue
                    try:
                        insight = analyzer.analyze(doc)
                        store.save_insight(insight)
                        analyzed += 1
                    except Exception as exc:
                        st.warning(f"Q{q} {y} failed: {exc}")

                prog.empty()
                c1, c2 = st.columns(2)
                c1.metric("Newly analyzed", analyzed)
                c2.metric("Already cached", skipped)

                if analyzed:
                    st.success(
                        "✅ Analysis complete — head to the **Insights** page to review results."
                    )
                else:
                    st.info("All documents were already cached.")

    # ── Financial Profile (Yahoo Finance) ────────────────────────────────────
    if show_financial_profile:
        st.divider()
        if ticker:
            try:
                render_financial_dashboard(ticker)
            except Exception as exc:
                st.error(f"Could not load financial profile for {ticker}: {exc}")
        else:
            st.info("Enter a ticker symbol above to view its financial profile.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Insights
# ════════════════════════════════════════════════════════════════════════════

elif page == "Insights":
    st.title("💡 Quarter Insights")

    store   = _store()
    tickers = store.list_tickers()

    if not tickers:
        st.warning("No tickers in the database yet. Go to **Search & Fetch** first.")
        _render_footer()
        st.stop()

    ticker   = st.selectbox("Select ticker", tickers, index=0)
    insights = [i.model_dump() for i in store.get_insights(ticker)]

    if not insights:
        st.info(f"No insights for {ticker} yet — run AI analysis first.")
        _render_footer()
        st.stop()

    for ins in insights:
        sent    = ins.get("sentiment", {})
        score   = sent.get("score", 0.0)
        label   = sent.get("label", "—")
        ratio   = sent.get("rationale", "")
        conf    = sent.get("confidence", 0.0)
        q       = ins.get("quarter", "?")
        yr      = ins.get("year",    "?")
        risks   = ins.get("risks",    [])
        guidance = ins.get("guidance", [])
        color   = sentiment_color(score)

        with st.container():
            st.markdown(f"### Q{q} {yr}")
            cols = st.columns([1, 4])
            with cols[0]:
                delta_str = f"{score:+.2f}"
                st.metric(
                    label="Sentiment",
                    value=label.title(),
                    delta=delta_str,
                    delta_color="normal" if color == "green" else (
                        "inverse" if color == "red" else "off"
                    ),
                )
                st.caption(f"Confidence: {conf:.0%}")
            with cols[1]:
                with st.expander("📝 Rationale", expanded=False):
                    st.write(ratio)

                if risks:
                    st.markdown("**Risk factors**")
                    chips = " &nbsp; ".join(
                        severity_badge(r.get("severity", "medium"))
                        + f" {r.get('text', '')}"
                        for r in risks
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No risk factors extracted.")

                if guidance:
                    st.markdown("**Forward guidance**")
                    for g in guidance:
                        arrow  = direction_arrow(g.get("direction", "unclear"))
                        metric = g.get("metric", "")
                        tf     = g.get("timeframe", "")
                        txt    = g.get("text", "")
                        st.markdown(
                            f"**{arrow} {metric}** ({tf}): {txt}",
                            unsafe_allow_html=False,
                        )

        st.divider()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Trend Analysis
# ════════════════════════════════════════════════════════════════════════════

elif page == "Trend Analysis":
    st.title("📊 Trend Analysis")

    store   = _store()
    tickers = store.list_tickers()

    if not tickers:
        st.warning("No tickers in the database yet. Go to **Search & Fetch** first.")
        _render_footer()
        st.stop()

    ticker   = st.selectbox("Select ticker", tickers, index=0)
    raw_ins  = store.get_insights(ticker)

    if not raw_ins:
        st.info(f"No trend data for {ticker} yet — run AI analysis first.")
        _render_footer()
        st.stop()

    # Build QuarterTrend via comparator, then convert to plain dict for rendering
    try:
        cmp_obj = QuarterComparator()
        qt      = cmp_obj.compare(ticker, raw_ins)
        trend   = qt.model_dump()
    except Exception as exc:
        st.error(f"Could not build trend: {exc}")
        _render_footer()
        st.stop()

    quarters  = trend.get("quarters",        [])
    drift     = trend.get("sentiment_drift", [])
    direction = trend.get("trend_direction", "stable")
    new_risks = trend.get("new_risks",       [])
    dropped   = trend.get("dropped_risks",   [])

    if len(quarters) < 1:
        st.info("Need at least one analyzed quarter.")
        _render_footer()
        st.stop()

    # ── Plotly chart ───────────────────────────────────────────────────────
    import plotly.graph_objects as go

    labels = [f"Q{q['quarter']} {q['year']}" for q in quarters]
    scores = [q["sentiment"]["score"]         for q in quarters]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels, y=scores,
        mode="lines+markers",
        line=dict(color="#4b8bff", width=2.5),
        marker=dict(size=8, color="#4b8bff"),
        name="Sentiment score",
        hovertemplate="%{x}<br>Score: %{y:.2f}<extra></extra>",
    ))

    fig.add_hline(
        y=0, line_dash="dash", line_color="rgba(150,150,150,0.5)",
        annotation_text="Neutral", annotation_position="bottom right",
    )

    if scores:
        min_idx = scores.index(min(scores))
        max_idx = scores.index(max(scores))
        fig.add_trace(go.Scatter(
            x=[labels[min_idx]], y=[scores[min_idx]],
            mode="markers+text",
            marker=dict(color="red", size=12, symbol="circle"),
            text=[f"Low: {scores[min_idx]:+.2f}"],
            textposition="bottom center",
            name="Lowest",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[labels[max_idx]], y=[scores[max_idx]],
            mode="markers+text",
            marker=dict(color="green", size=12, symbol="circle"),
            text=[f"High: {scores[max_idx]:+.2f}"],
            textposition="top center",
            name="Highest",
            showlegend=False,
        ))

    fig.update_layout(
        title=f"{ticker} — Sentiment Trend",
        xaxis_title="Quarter",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1], zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Trend badge ────────────────────────────────────────────────────────
    st.markdown(
        f"Overall trend: {trend_badge(direction)}",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Drift table ────────────────────────────────────────────────────────
    if drift:
        with st.expander("Quarter-over-quarter drift", expanded=False):
            import pandas as pd
            drift_rows = [
                {
                    "From → To": f"{labels[i]} → {labels[i+1]}",
                    "Δ Score":   f"{drift[i]:+.3f}",
                }
                for i in range(len(drift))
            ]
            st.dataframe(pd.DataFrame(drift_rows), hide_index=True, use_container_width=True)

    # ── Narrative summary ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Narrative summary")
    try:
        qt_obj  = QT.model_validate(trend)
        summary = QuarterComparator().summarize_trend(qt_obj)
        st.info(summary)
    except Exception as exc:
        st.warning(f"Could not generate narrative: {exc}")

    # ── New vs resolved risks ──────────────────────────────────────────────
    st.divider()
    st.subheader("Risk changes (latest quarter)")
    r_col, d_col = st.columns(2)

    with r_col:
        st.markdown("#### 🆕 New risks")
        if new_risks:
            for r in new_risks:
                st.markdown(
                    severity_badge(r.get("severity", "medium"))
                    + f" &nbsp; {r.get('text', '')}",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No new risks identified.")

    with d_col:
        st.markdown("#### ✅ Risks resolved")
        if dropped:
            for r in dropped:
                st.markdown(f"~~{r.get('text', '')}~~")
        else:
            st.info("No risks dropped from prior quarter.")

    # ── Price Forecast (Yahoo Finance / ARIMA) ────────────────────────────────
    st.divider()
    st.subheader("🔮 Price Forecast")
    try:
        render_forecast_chart(ticker)
    except Exception as exc:
        st.error(f"Could not load price forecast for {ticker}: {exc}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Market Screener
# ════════════════════════════════════════════════════════════════════════════

elif page == "Market Screener":
    selected_symbol = render_stock_screener()
    if selected_symbol:
        st.session_state["ticker_input"] = selected_symbol
        st.session_state["_pending_nav_page"] = "Search & Fetch"
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Trading Signals
# ════════════════════════════════════════════════════════════════════════════

elif page == "Trading Signals":
    render_signals_page()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Sector Heatmap
# ════════════════════════════════════════════════════════════════════════════

elif page == "Sector Heatmap":
    render_sector_heatmap_page()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Peer Comparison
# ════════════════════════════════════════════════════════════════════════════

elif page == "Peer Comparison":
    render_peer_comparison_page()

# ════════════════════════════════════════════════════════════════════════════
# Footer — attribution, shown at the bottom of every page
# ════════════════════════════════════════════════════════════════════════════

_render_footer()
