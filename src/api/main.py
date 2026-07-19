"""
FinSight FastAPI backend — Week 5 audit.

Endpoints:
  GET  /health
  GET  /tickers
  GET  /validate/{ticker}      ← new: live EDGAR ticker check
  POST /fetch/{ticker}
  POST /analyze/{ticker}
  GET  /insights/{ticker}      ← returns [] instead of 404 when no data
  GET  /trend/{ticker}         ← returns empty stable trend instead of 404
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env from project root before any src imports
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.ingest.edgar import TranscriptFetcher
from src.analyze.analyzer import TranscriptAnalyzer
from src.analyze.comparator import QuarterComparator
from src.process.store import TranscriptStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinSight API",
    description="Earnings call transcript analysis backend",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------

def get_store() -> TranscriptStore:
    return TranscriptStore()


# ---------------------------------------------------------------------------
# Response model (analyze only — fetch returns raw dict)
# ---------------------------------------------------------------------------

class AnalyzeResult(BaseModel):
    ticker: str
    analyzed: int
    skipped_cached: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tickers")
def list_tickers() -> list[str]:
    return get_store().list_tickers()


@app.get("/validate/{ticker}")
def validate_ticker(ticker: str) -> dict[str, Any]:
    """
    Check whether *ticker* exists on EDGAR.

    Returns {"valid": bool, "company_name": str, "message": str}.
    Never raises — errors are returned as {"valid": false, ...}.
    """
    try:
        fetcher = TranscriptFetcher()
        info    = fetcher.get_supported_info(ticker.upper())
        if info["exists"]:
            return {
                "valid":        True,
                "company_name": info["company_name"],
                "message":      f"Found {info['company_name']} ({ticker.upper()})",
            }
        return {
            "valid":        False,
            "company_name": "",
            "message":      f"Ticker {ticker.upper()} not found on EDGAR",
        }
    except Exception as exc:
        logger.warning("validate_ticker error for %s: %s", ticker, exc)
        return {"valid": False, "company_name": "", "message": str(exc)}


@app.post("/fetch/{ticker}")
def fetch_ticker(
    ticker: str, start_year: int = 2022, end_year: int = 2024
) -> dict[str, Any]:
    """
    Fetch 8-K filings from EDGAR for *ticker* and persist new ones.

    Returns the fetch_status dict (from edgar.py) augmented with:
      total_saved  — documents newly written to the DB
      skipped      — documents already present (dedup)
    """
    store   = get_store()
    fetcher = TranscriptFetcher()

    try:
        results, status = fetcher.fetch_by_ticker(ticker.upper(), start_year, end_year)
    except Exception as exc:
        logger.error("EDGAR fetch failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker.upper()}

    saved   = sum(1 for r in results if store.save(r))
    skipped = len(results) - saved

    status["total_saved"] = saved
    status["skipped"]     = skipped
    return status


@app.post("/analyze/{ticker}", response_model=AnalyzeResult)
def analyze_ticker(ticker: str) -> AnalyzeResult:
    """
    Run TranscriptAnalyzer on every un-cached document for *ticker*.
    Already-cached quarters are skipped.
    """
    store = get_store()
    docs  = store.get_by_ticker(ticker.upper())
    if not docs:
        return AnalyzeResult(ticker=ticker.upper(), analyzed=0, skipped_cached=0)

    try:
        analyzer = TranscriptAnalyzer()
    except EnvironmentError as exc:
        logger.error("TranscriptAnalyzer init failed: %s", exc)
        return AnalyzeResult(ticker=ticker.upper(), analyzed=0, skipped_cached=0)

    analyzed = 0
    skipped  = 0

    for doc in docs:
        q = doc.get("quarter") or 0
        y = doc.get("year")    or 0
        if q and y and store.insight_exists(ticker.upper(), q, y):
            skipped += 1
            continue
        try:
            insight = analyzer.analyze(doc)
            store.save_insight(insight)
            analyzed += 1
        except Exception as exc:
            logger.warning(
                "Analysis failed for %s Q%s %s: %s", ticker, q, y, exc
            )

    return AnalyzeResult(
        ticker=ticker.upper(),
        analyzed=analyzed,
        skipped_cached=skipped,
    )


@app.get("/insights/{ticker}")
def get_insights(ticker: str) -> list[dict[str, Any]]:
    """Return all cached TranscriptInsights for *ticker* as JSON dicts.
    Returns an empty list (not 404) when no data exists yet."""
    try:
        store    = get_store()
        insights = store.get_insights(ticker.upper())
        return [i.model_dump() for i in insights]
    except Exception as exc:
        logger.error("get_insights error for %s: %s", ticker, exc)
        return []


@app.get("/trend/{ticker}")
def get_trend(ticker: str) -> dict[str, Any]:
    """
    Run QuarterComparator across all insights for *ticker*.
    Returns an empty stable trend (not 404) when no insights exist yet.
    """
    _empty_trend: dict[str, Any] = {
        "ticker":           ticker.upper(),
        "quarters":         [],
        "sentiment_drift":  [],
        "new_risks":        [],
        "dropped_risks":    [],
        "trend_direction":  "stable",
    }

    try:
        store    = get_store()
        insights = store.get_insights(ticker.upper())
        if not insights:
            return _empty_trend

        cmp   = QuarterComparator()
        trend = cmp.compare(ticker.upper(), insights)
        return trend.model_dump()
    except Exception as exc:
        logger.error("get_trend error for %s: %s", ticker, exc)
        return _empty_trend
