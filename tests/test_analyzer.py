"""
Unit tests for Week-2: LLM analysis engine.
All Claude API calls are mocked — no real HTTP in this suite.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_insight(ticker, quarter, year, score, risks=None, guidance=None):
    """Build a TranscriptInsight without touching the API."""
    from src.analyze.models import (
        GuidanceStatement,
        RiskFactor,
        SentimentResult,
        TranscriptInsight,
    )
    return TranscriptInsight(
        ticker=ticker,
        quarter=quarter,
        year=year,
        sentiment=SentimentResult(
            score=score,
            label="test label",
            rationale="test rationale",
            confidence=0.9,
        ),
        risks=risks or [],
        guidance=guidance or [],
        summary="test summary",
    )


def _make_analyzer_with_mock(mock_response_text: str):
    """
    Return a TranscriptAnalyzer whose _call_llm always returns mock_response_text.
    Patches GEMINI_API_KEY so no real key is needed.
    """
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key"}):
        with patch("google.genai.Client"):
            from src.analyze.analyzer import TranscriptAnalyzer
            analyzer = TranscriptAnalyzer(api_key="fake-gemini-key")
    analyzer._call_llm = MagicMock(return_value=mock_response_text)
    return analyzer


# ---------------------------------------------------------------------------
# XML parsing — SentimentResult
# ---------------------------------------------------------------------------

class TestSentimentXMLParsing:
    SAMPLE_XML = """
<sentiment>
  <score>0.45</score>
  <label>cautiously optimistic</label>
  <rationale>Management cited strong iPhone demand while flagging macro uncertainty.</rationale>
  <confidence>0.88</confidence>
</sentiment>
"""

    def test_score_parsed_correctly(self):
        from src.analyze.analyzer import TranscriptAnalyzer
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        result = analyzer._parse_sentiment_xml(self.SAMPLE_XML)
        assert abs(result.score - 0.45) < 1e-6

    def test_label_parsed(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        result = analyzer._parse_sentiment_xml(self.SAMPLE_XML)
        assert result.label == "cautiously optimistic"

    def test_rationale_nonempty(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        result = analyzer._parse_sentiment_xml(self.SAMPLE_XML)
        assert len(result.rationale) > 10

    def test_confidence_parsed(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        result = analyzer._parse_sentiment_xml(self.SAMPLE_XML)
        assert abs(result.confidence - 0.88) < 1e-6

    def test_invalid_xml_returns_neutral_default(self):
        analyzer = _make_analyzer_with_mock("")
        result = analyzer._parse_sentiment_xml("not xml at all")
        assert result.score == 0.0
        assert result.label == "neutral"
        assert result.confidence == 0.5

    def test_score_clamped_by_pydantic(self):
        """Pydantic should reject scores outside [-1, 1]."""
        from src.analyze.models import SentimentResult
        with pytest.raises(Exception):
            SentimentResult(score=2.5, label="x", rationale="x", confidence=0.5)


# ---------------------------------------------------------------------------
# XML parsing — RiskFactor
# ---------------------------------------------------------------------------

class TestRiskXMLParsing:
    SAMPLE_XML = """
<risks>
  <risk>
    <text>Rising interest rates may compress consumer spending on discretionary electronics.</text>
    <severity>high</severity>
    <category>macro</category>
  </risk>
  <risk>
    <text>Ongoing supply chain constraints in Southeast Asia could limit iPhone production.</text>
    <severity>medium</severity>
    <category>supply_chain</category>
  </risk>
</risks>
"""

    def test_two_risks_parsed(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        risks = analyzer._parse_risks_xml(self.SAMPLE_XML)
        assert len(risks) == 2

    def test_severity_values(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        risks = analyzer._parse_risks_xml(self.SAMPLE_XML)
        severities = {r.severity for r in risks}
        assert "high" in severities
        assert "medium" in severities

    def test_category_parsed(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        risks = analyzer._parse_risks_xml(self.SAMPLE_XML)
        categories = {r.category for r in risks}
        assert "macro" in categories

    def test_empty_risks_tag_returns_empty_list(self):
        analyzer = _make_analyzer_with_mock("<risks></risks>")
        risks = analyzer._parse_risks_xml("<risks></risks>")
        assert risks == []

    def test_invalid_xml_returns_empty_list(self):
        analyzer = _make_analyzer_with_mock("garbage")
        risks = analyzer._parse_risks_xml("garbage")
        assert risks == []


# ---------------------------------------------------------------------------
# XML parsing — GuidanceStatement
# ---------------------------------------------------------------------------

class TestGuidanceXMLParsing:
    SAMPLE_XML = """
<guidance>
  <item>
    <text>Management expects revenue to grow low-to-mid single digits year over year.</text>
    <metric>revenue</metric>
    <direction>up</direction>
    <timeframe>Q2 2024</timeframe>
  </item>
  <item>
    <text>Gross margins expected to remain under pressure due to product mix.</text>
    <metric>gross margin</metric>
    <direction>down</direction>
    <timeframe>Q2 2024</timeframe>
  </item>
</guidance>
"""

    def test_two_items_parsed(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        guidance = analyzer._parse_guidance_xml(self.SAMPLE_XML)
        assert len(guidance) == 2

    def test_direction_up_parsed(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        guidance = analyzer._parse_guidance_xml(self.SAMPLE_XML)
        directions = {g.direction for g in guidance}
        assert "up" in directions
        assert "down" in directions

    def test_metric_parsed(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        guidance = analyzer._parse_guidance_xml(self.SAMPLE_XML)
        metrics = {g.metric for g in guidance}
        assert "revenue" in metrics

    def test_timeframe_parsed(self):
        analyzer = _make_analyzer_with_mock(self.SAMPLE_XML)
        guidance = analyzer._parse_guidance_xml(self.SAMPLE_XML)
        assert all(g.timeframe == "Q2 2024" for g in guidance)

    def test_empty_guidance_returns_empty_list(self):
        analyzer = _make_analyzer_with_mock("<guidance></guidance>")
        result = analyzer._parse_guidance_xml("<guidance></guidance>")
        assert result == []


# ---------------------------------------------------------------------------
# Full analyze() pipeline with mocked Claude
# ---------------------------------------------------------------------------

class TestAnalyzePipeline:
    _SENTIMENT_XML = """
<sentiment>
  <score>0.30</score>
  <label>mildly positive</label>
  <rationale>Steady revenue growth offset by cautious macro commentary.</rationale>
  <confidence>0.80</confidence>
</sentiment>"""

    _RISK_XML = """
<risks>
  <risk><text>Macro slowdown risk</text><severity>high</severity><category>macro</category></risk>
</risks>"""

    _GUIDANCE_XML = """
<guidance>
  <item><text>Expects double-digit services growth.</text><metric>services revenue</metric><direction>up</direction><timeframe>FY2024</timeframe></item>
</guidance>"""

    def _make_analyzer_multi(self):
        """Analyzer whose _call_llm cycles through sentiment→risk→guidance responses."""
        responses = [self._SENTIMENT_XML, self._RISK_XML, self._GUIDANCE_XML]
        counter = {"i": 0}

        def side_effect(prompt, max_tokens=1000):
            resp = responses[counter["i"] % len(responses)]
            counter["i"] += 1
            return resp

        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key"}):
            with patch("google.genai.Client"):
                from src.analyze.analyzer import TranscriptAnalyzer
                analyzer = TranscriptAnalyzer(api_key="fake-gemini-key")
        analyzer._call_llm = MagicMock(side_effect=side_effect)
        return analyzer

    def test_returns_transcript_insight(self):
        from src.analyze.models import TranscriptInsight
        analyzer = self._make_analyzer_multi()
        doc = {
            "ticker": "AAPL", "quarter": 3, "year": 2023,
            "raw_text": "Apple Q3 2023. Revenue was $81B. Operator: next question.",
        }
        result = analyzer.analyze(doc)
        assert isinstance(result, TranscriptInsight)

    def test_ticker_preserved(self):
        analyzer = self._make_analyzer_multi()
        doc = {"ticker": "AAPL", "quarter": 1, "year": 2023,
               "raw_text": "Revenue strong. Operator: question."}
        result = analyzer.analyze(doc)
        assert result.ticker == "AAPL"

    def test_sentiment_populated(self):
        analyzer = self._make_analyzer_multi()
        doc = {"ticker": "AAPL", "quarter": 1, "year": 2023,
               "raw_text": "Revenue strong. Operator: question."}
        result = analyzer.analyze(doc)
        assert result.sentiment.label == "mildly positive"
        assert abs(result.sentiment.score - 0.30) < 1e-6

    def test_risks_populated(self):
        analyzer = self._make_analyzer_multi()
        doc = {"ticker": "AAPL", "quarter": 1, "year": 2023,
               "raw_text": "Revenue strong. Operator: question."}
        result = analyzer.analyze(doc)
        assert len(result.risks) == 1
        assert result.risks[0].severity == "high"

    def test_guidance_populated(self):
        analyzer = self._make_analyzer_multi()
        doc = {"ticker": "AAPL", "quarter": 1, "year": 2023,
               "raw_text": "Revenue strong. Operator: question."}
        result = analyzer.analyze(doc)
        assert len(result.guidance) == 1
        assert result.guidance[0].direction == "up"

    def test_missing_api_key_raises(self):
        # Patch load_dotenv at the dotenv package level so .env is never read,
        # clear os.environ so the key is absent, then verify EnvironmentError is raised.
        with patch("dotenv.main.load_dotenv", return_value=False):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
                    with patch("google.genai.Client"):
                        from src.analyze.analyzer import TranscriptAnalyzer
                        TranscriptAnalyzer()


# ---------------------------------------------------------------------------
# QuarterComparator
# ---------------------------------------------------------------------------

from src.analyze.models import RiskFactor  # noqa: E402

_R = lambda text, sev="medium": RiskFactor(text=text, severity=sev, category="macro")


class TestQuarterComparator:
    """Tests for QuarterComparator using three hardcoded TranscriptInsights."""

    def _make_comparator(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            from src.analyze.comparator import QuarterComparator
            return QuarterComparator(api_key="fake-key")

    def _three_quarters(self):
        """Q1 +0.1, Q2 +0.3, Q3 +0.5  → improving trend, drift=[+0.2, +0.2]"""
        return [
            _make_insight("AAPL", 1, 2023, score=0.1,
                          risks=[_R("Supply chain issues"), _R("FX headwinds")]),
            _make_insight("AAPL", 2, 2023, score=0.3,
                          risks=[_R("FX headwinds"), _R("Rising rates")]),
            _make_insight("AAPL", 3, 2023, score=0.5,
                          risks=[_R("Rising rates"), _R("Regulatory scrutiny")]),
        ]

    def test_compare_returns_quarter_trend(self):
        from src.analyze.models import QuarterTrend
        cmp = self._make_comparator()
        trend = cmp.compare("AAPL", self._three_quarters())
        assert isinstance(trend, QuarterTrend)

    def test_quarters_sorted_by_year_quarter(self):
        cmp = self._make_comparator()
        shuffled = list(reversed(self._three_quarters()))
        trend = cmp.compare("AAPL", shuffled)
        years_quarters = [(q.year, q.quarter) for q in trend.quarters]
        assert years_quarters == sorted(years_quarters)

    def test_sentiment_drift_length(self):
        """drift should have len(quarters)-1 elements."""
        cmp = self._make_comparator()
        trend = cmp.compare("AAPL", self._three_quarters())
        assert len(trend.sentiment_drift) == 2

    def test_sentiment_drift_values(self):
        """Q1→Q2: +0.2, Q2→Q3: +0.2"""
        cmp = self._make_comparator()
        trend = cmp.compare("AAPL", self._three_quarters())
        assert abs(trend.sentiment_drift[0] - 0.2) < 1e-6
        assert abs(trend.sentiment_drift[1] - 0.2) < 1e-6

    def test_trend_direction_improving(self):
        cmp = self._make_comparator()
        trend = cmp.compare("AAPL", self._three_quarters())
        assert trend.trend_direction == "improving"

    def test_trend_direction_declining(self):
        cmp = self._make_comparator()
        insights = [
            _make_insight("AAPL", 1, 2023, score=0.5),
            _make_insight("AAPL", 2, 2023, score=0.3),
            _make_insight("AAPL", 3, 2023, score=0.1),
        ]
        trend = cmp.compare("AAPL", insights)
        assert trend.trend_direction == "declining"

    def test_trend_direction_volatile(self):
        cmp = self._make_comparator()
        insights = [
            _make_insight("AAPL", 1, 2023, score=-0.8),
            _make_insight("AAPL", 2, 2023, score=0.8),
            _make_insight("AAPL", 3, 2023, score=-0.8),
            _make_insight("AAPL", 4, 2023, score=0.8),
        ]
        trend = cmp.compare("AAPL", insights)
        assert trend.trend_direction == "volatile"

    def test_trend_direction_stable(self):
        cmp = self._make_comparator()
        insights = [
            _make_insight("AAPL", 1, 2023, score=0.1),
            _make_insight("AAPL", 2, 2023, score=0.12),
            _make_insight("AAPL", 3, 2023, score=0.09),
        ]
        trend = cmp.compare("AAPL", insights)
        assert trend.trend_direction == "stable"

    def test_new_risks_detected(self):
        """'Regulatory scrutiny' only appears in Q3 → should be new."""
        cmp = self._make_comparator()
        trend = cmp.compare("AAPL", self._three_quarters())
        new_texts = [r.text for r in trend.new_risks]
        assert any("regulatory" in t.lower() for t in new_texts)

    def test_dropped_risks_detected(self):
        """'Supply chain issues' only in Q1 → should be dropped."""
        cmp = self._make_comparator()
        trend = cmp.compare("AAPL", self._three_quarters())
        dropped_texts = [r.text for r in trend.dropped_risks]
        assert any("supply chain" in t.lower() for t in dropped_texts)

    def test_empty_insights_returns_stable(self):
        cmp = self._make_comparator()
        trend = cmp.compare("AAPL", [])
        assert trend.trend_direction == "stable"
        assert trend.sentiment_drift == []

    def test_single_insight_no_drift(self):
        cmp = self._make_comparator()
        trend = cmp.compare("AAPL", [_make_insight("AAPL", 1, 2023, score=0.4)])
        assert trend.sentiment_drift == []
        assert trend.trend_direction == "stable"


# ---------------------------------------------------------------------------
# TranscriptStore — insights table (Week 2 additions)
# ---------------------------------------------------------------------------

class TestInsightsStore:
    def _make_store(self):
        from src.process.store import TranscriptStore
        tmp = tempfile.mktemp(suffix=".db")
        return TranscriptStore(db_path=tmp)

    def test_save_insight_returns_true(self):
        store = self._make_store()
        insight = _make_insight("AAPL", 3, 2023, score=0.4)
        assert store.save_insight(insight) is True

    def test_insight_exists_after_save(self):
        store = self._make_store()
        insight = _make_insight("AAPL", 3, 2023, score=0.4)
        store.save_insight(insight)
        assert store.insight_exists("AAPL", 3, 2023) is True

    def test_insight_not_exists_before_save(self):
        store = self._make_store()
        assert store.insight_exists("MSFT", 1, 2023) is False

    def test_get_insights_returns_saved(self):
        store = self._make_store()
        store.save_insight(_make_insight("AAPL", 1, 2023, score=0.1))
        store.save_insight(_make_insight("AAPL", 2, 2023, score=0.3))
        results = store.get_insights("AAPL")
        assert len(results) == 2

    def test_get_insights_sorted_by_quarter(self):
        store = self._make_store()
        store.save_insight(_make_insight("AAPL", 3, 2023, score=0.5))
        store.save_insight(_make_insight("AAPL", 1, 2023, score=0.1))
        results = store.get_insights("AAPL")
        quarters = [r.quarter for r in results]
        assert quarters == sorted(quarters)

    def test_save_insight_upserts(self):
        """Saving the same ticker/quarter/year twice should update, not duplicate."""
        store = self._make_store()
        store.save_insight(_make_insight("AAPL", 3, 2023, score=0.2))
        store.save_insight(_make_insight("AAPL", 3, 2023, score=0.6))
        results = store.get_insights("AAPL")
        assert len(results) == 1
        assert abs(results[0].sentiment.score - 0.6) < 1e-6

    def test_roundtrip_preserves_risks(self):
        store = self._make_store()
        risks = [_R("Interest rate risk", "high"), _R("FX exposure", "low")]
        insight = _make_insight("AAPL", 2, 2023, score=0.3, risks=risks)
        store.save_insight(insight)
        loaded = store.get_insights("AAPL")[0]
        assert len(loaded.risks) == 2
        assert loaded.risks[0].severity == "high"
