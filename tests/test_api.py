"""
FastAPI endpoint tests — Week 5 audit.
All heavy operations (fetcher, analyzer) are mocked so no live HTTP or API calls occur.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path):
    """Patch TranscriptStore to use a fresh temp DB for every test."""
    from src.process.store import TranscriptStore

    store = TranscriptStore(db_path=tmp_path / "test.db")
    with patch("src.api.main.get_store", return_value=store):
        yield store


@pytest.fixture()
def client(tmp_store):
    """Return a TestClient backed by a temp store."""
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client_with_aapl(tmp_store):
    """
    TestClient whose store already has one AAPL transcript and one insight.
    """
    from src.analyze.models import SentimentResult, TranscriptInsight, RiskFactor
    from src.api.main import app

    # Seed one transcript
    tmp_store.save({
        "ticker": "AAPL", "cik": "320193", "form_type": "8-K",
        "filed_date": "2023-11-02", "quarter": 4, "year": 2023,
        "raw_text": "Apple Q4 2023 earnings. Revenue strong. Operator next question.",
        "cleaned_text": "Apple Q4 2023 earnings.",
        "url": "https://example.com/aapl-q4-2023",
    })

    # Seed one insight
    insight = TranscriptInsight(
        ticker="AAPL", quarter=4, year=2023,
        sentiment=SentimentResult(score=0.45, label="confident", rationale="Strong Q4.", confidence=0.9),
        risks=[RiskFactor(text="Macro slowdown", severity="medium", category="macro")],
        guidance=[],
        summary="AAPL Q4 strong.",
    )
    tmp_store.save_insight(insight)

    with patch("src.api.main.get_store", return_value=tmp_store):
        yield TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body_status_ok(self, client):
        assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /tickers
# ---------------------------------------------------------------------------

class TestTickers:
    def test_empty_store_returns_empty_list(self, client):
        resp = client.get("/tickers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_saved_ticker(self, client_with_aapl):
        resp = client_with_aapl.get("/tickers")
        assert resp.status_code == 200
        assert "AAPL" in resp.json()

    def test_returns_list(self, client):
        resp = client.get("/tickers")
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# /validate/{ticker}
# ---------------------------------------------------------------------------

class TestValidate:
    def _mock_valid_info(self):
        return {
            "exists":         True,
            "company_name":   "Apple Inc.",
            "cik":            "320193",
            "total_8k_count": 50,
        }

    def _mock_invalid_info(self):
        return {
            "exists":         False,
            "company_name":   "",
            "cik":            "",
            "total_8k_count": 0,
        }

    def test_valid_ticker_returns_200(self, client):
        with patch("src.api.main.TranscriptFetcher") as MockFetcher:
            MockFetcher.return_value.get_supported_info.return_value = self._mock_valid_info()
            resp = client.get("/validate/AAPL")
        assert resp.status_code == 200

    def test_valid_ticker_body(self, client):
        with patch("src.api.main.TranscriptFetcher") as MockFetcher:
            MockFetcher.return_value.get_supported_info.return_value = self._mock_valid_info()
            body = client.get("/validate/AAPL").json()
        assert body["valid"] is True
        assert "company_name" in body
        assert "message" in body

    def test_invalid_ticker_valid_false(self, client):
        with patch("src.api.main.TranscriptFetcher") as MockFetcher:
            MockFetcher.return_value.get_supported_info.return_value = self._mock_invalid_info()
            body = client.get("/validate/FAKE123").json()
        assert body["valid"] is False

    def test_validate_returns_200_on_exception(self, client):
        with patch("src.api.main.TranscriptFetcher") as MockFetcher:
            MockFetcher.return_value.get_supported_info.side_effect = RuntimeError("network error")
            resp = client.get("/validate/AAPL")
        assert resp.status_code == 200
        assert resp.json()["valid"] is False


# ---------------------------------------------------------------------------
# /insights/{ticker}
# ---------------------------------------------------------------------------

class TestInsights:
    def test_returns_200_when_no_insights(self, client):
        """No insights → 200 with empty list (not 404)."""
        resp = client.get("/insights/FAKE")
        assert resp.status_code == 200

    def test_returns_empty_list_when_no_insights(self, client):
        body = client.get("/insights/FAKE").json()
        assert body == []

    def test_returns_list_for_aapl(self, client_with_aapl):
        resp = client_with_aapl.get("/insights/AAPL")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 1

    def test_insight_has_required_fields(self, client_with_aapl):
        body = client_with_aapl.get("/insights/AAPL").json()
        first = body[0]
        assert "ticker"    in first
        assert "sentiment" in first
        assert "risks"     in first
        assert "guidance"  in first
        assert "quarter"   in first
        assert "year"      in first

    def test_sentiment_score_in_range(self, client_with_aapl):
        first = client_with_aapl.get("/insights/AAPL").json()[0]
        score = first["sentiment"]["score"]
        assert -1.0 <= score <= 1.0

    def test_ticker_case_insensitive(self, client_with_aapl):
        resp = client_with_aapl.get("/insights/aapl")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /trend/{ticker}
# ---------------------------------------------------------------------------

class TestTrend:
    def test_returns_200_when_no_insights(self, client):
        """No insights → 200 with empty stable trend (not 404)."""
        resp = client.get("/trend/FAKE")
        assert resp.status_code == 200

    def test_returns_stable_trend_when_no_insights(self, client):
        body = client.get("/trend/FAKE").json()
        assert body["trend_direction"] == "stable"
        assert body["quarters"]        == []

    def test_returns_trend_dict(self, client_with_aapl):
        resp = client_with_aapl.get("/trend/AAPL")
        assert resp.status_code == 200
        body = resp.json()
        assert "ticker"          in body
        assert "trend_direction" in body
        assert "sentiment_drift" in body
        assert "quarters"        in body

    def test_trend_direction_valid(self, client_with_aapl):
        body = client_with_aapl.get("/trend/AAPL").json()
        assert body["trend_direction"] in ("improving", "declining", "stable", "volatile")

    def test_single_quarter_stable(self, client_with_aapl):
        body = client_with_aapl.get("/trend/AAPL").json()
        # One quarter → no drift → stable
        assert body["sentiment_drift"] == []
        assert body["trend_direction"] == "stable"


# ---------------------------------------------------------------------------
# POST /fetch/{ticker}  (mocked fetcher)
# ---------------------------------------------------------------------------

class TestFetch:
    def _mock_fetch_result(self):
        """Return (docs, fetch_status) tuple as edgar.py now does."""
        docs = [
            {
                "ticker": "AAPL", "cik": "320193", "form_type": "8-K",
                "filed_date": "2023-08-03", "quarter": 3, "year": 2023,
                "raw_text": "Apple Q3 revenue strong.",
                "url": "https://example.com",
            }
        ]
        status = {
            "ticker":                "AAPL",
            "cik":                   "320193",
            "total_filings_checked": 1,
            "years_searched":        "2023-2023",
            "current_quarter":       "Q2 2026",
            "errors":                [],
        }
        return (docs, status)

    def test_fetch_returns_200(self, client, tmp_store):
        with patch("src.api.main.TranscriptFetcher") as MockFetcher:
            MockFetcher.return_value.fetch_by_ticker.return_value = self._mock_fetch_result()
            resp = client.post("/fetch/AAPL", params={"start_year": 2023, "end_year": 2023})
        assert resp.status_code == 200

    def test_fetch_returns_counts(self, client, tmp_store):
        with patch("src.api.main.TranscriptFetcher") as MockFetcher:
            MockFetcher.return_value.fetch_by_ticker.return_value = self._mock_fetch_result()
            body = client.post("/fetch/AAPL").json()
        assert body["total_filings_checked"] == 1
        assert body["total_saved"]           == 1
        assert body["skipped"]               == 0

    def test_fetch_dedup_on_second_call(self, client, tmp_store):
        with patch("src.api.main.TranscriptFetcher") as MockFetcher:
            MockFetcher.return_value.fetch_by_ticker.return_value = self._mock_fetch_result()
            client.post("/fetch/AAPL")
            body = client.post("/fetch/AAPL").json()
        assert body["total_saved"] == 0
        assert body["skipped"]     == 1

    def test_fetch_returns_200_on_edgar_error(self, client, tmp_store):
        """EDGAR errors must not cause 500 — returns 200 with error key."""
        with patch("src.api.main.TranscriptFetcher") as MockFetcher:
            MockFetcher.return_value.fetch_by_ticker.side_effect = RuntimeError("EDGAR down")
            resp = client.post("/fetch/AAPL")
        assert resp.status_code == 200
        assert "error" in resp.json()


# ---------------------------------------------------------------------------
# POST /analyze/{ticker}  (mocked analyzer)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_200_when_no_docs(self, client):
        """No transcripts → 200 with zeros (not 404)."""
        resp = client.post("/analyze/FAKE")
        assert resp.status_code == 200
        body = resp.json()
        assert body["analyzed"]       == 0
        assert body["skipped_cached"] == 0

    def test_analyze_runs_and_caches(self, client_with_aapl, tmp_store):
        # Add a second doc that has no insight yet
        tmp_store.save({
            "ticker": "AAPL", "cik": "320193", "form_type": "8-K",
            "filed_date": "2023-08-03", "quarter": 3, "year": 2023,
            "raw_text": "Apple Q3 2023 revenue $81B analyst question.",
            "cleaned_text": "Apple Q3 2023.",
            "url": "https://example.com/q3",
        })

        from src.analyze.models import SentimentResult, TranscriptInsight

        mock_insight = TranscriptInsight(
            ticker="AAPL", quarter=3, year=2023,
            sentiment=SentimentResult(score=0.3, label="positive", rationale="Good.", confidence=0.8),
            risks=[], guidance=[], summary="Q3 ok.",
        )

        with patch("src.api.main.TranscriptAnalyzer") as MockAnalyzer:
            MockAnalyzer.return_value.analyze.return_value = mock_insight
            resp = client_with_aapl.post("/analyze/AAPL")

        assert resp.status_code == 200
        body = resp.json()
        assert body["analyzed"]        == 1   # Q3 was new
        assert body["skipped_cached"]  == 1   # Q4 was already cached

    def test_analyze_skips_all_cached(self, client_with_aapl):
        # Q4 2023 is already cached — nothing new to analyze
        with patch("src.api.main.TranscriptAnalyzer") as MockAnalyzer:
            MockAnalyzer.return_value.analyze.return_value = MagicMock()
            resp = client_with_aapl.post("/analyze/AAPL")
        body = resp.json()
        assert body["skipped_cached"] >= 1
