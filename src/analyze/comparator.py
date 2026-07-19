"""
QuarterComparator: derives sentiment trends and risk drift across multiple quarters
for the same ticker, and generates a human-readable trend summary via Claude.
"""
from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.analyze.models import QuarterTrend, RiskFactor, TranscriptInsight
from src.analyze.prompts import TREND_SUMMARY_PROMPT

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

logger = logging.getLogger(__name__)

# Thresholds for trend classification
_DRIFT_IMPROVING  =  0.10   # avg drift above this → improving
_DRIFT_DECLINING  = -0.10   # avg drift below this → declining
_DRIFT_VOLATILE   =  0.20   # std-dev above this   → volatile

# Fuzzy-match threshold: what fraction of words must overlap to call two risks "the same"
_RISK_SIMILARITY_THRESHOLD = 0.40


class QuarterComparator:
    """Compare sentiment and risk factors across multiple earnings quarters."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, ticker: str, insights: list[TranscriptInsight]) -> QuarterTrend:
        """
        Derive sentiment drift and risk delta across *insights* (all for the same ticker).

        insights need not be pre-sorted; they are sorted by (year, quarter) here.
        Returns a fully populated QuarterTrend.
        """
        if not insights:
            return QuarterTrend(ticker=ticker)

        sorted_insights = sorted(insights, key=lambda i: (i.year, i.quarter))

        drift = self._compute_sentiment_drift(sorted_insights)
        direction = self._classify_direction(drift)
        new_risks, dropped_risks = self._compute_risk_delta(sorted_insights)

        return QuarterTrend(
            ticker=ticker,
            quarters=sorted_insights,
            sentiment_drift=drift,
            new_risks=new_risks,
            dropped_risks=dropped_risks,
            trend_direction=direction,
        )

    def summarize_trend(self, trend: QuarterTrend) -> str:
        """
        Generate a one-paragraph human summary of the trend via Claude.
        Falls back to a templated string if the API key is unavailable.
        """
        if not self._api_key:
            return self._fallback_summary(trend)

        try:
            from google import genai
            from google.genai import types as gtypes
            client = genai.Client(api_key=self._api_key)

            scores_str = ", ".join(
                f"Q{q.quarter}/{q.year}: {q.sentiment.score:+.2f}"
                for q in trend.quarters
            )
            new_risk_texts     = "; ".join(r.text for r in trend.new_risks[:5])    or "none"
            dropped_risk_texts = "; ".join(r.text for r in trend.dropped_risks[:5]) or "none"

            prompt = TREND_SUMMARY_PROMPT.format(
                ticker=trend.ticker,
                scores=scores_str,
                trend_direction=trend.trend_direction,
                new_risks=new_risk_texts,
                dropped_risks=dropped_risk_texts,
            )

            resp = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    max_output_tokens=300, temperature=0.2
                ),
            )
            return resp.text.strip()

        except Exception as exc:
            logger.warning("Trend summary API call failed: %s", exc)
            return self._fallback_summary(trend)

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------

    def _compute_sentiment_drift(self, insights: list[TranscriptInsight]) -> list[float]:
        """
        Quarter-over-quarter sentiment delta.
        drift[i] = insights[i+1].sentiment.score - insights[i].sentiment.score
        """
        scores = [i.sentiment.score for i in insights]
        return [round(scores[i + 1] - scores[i], 4) for i in range(len(scores) - 1)]

    def _classify_direction(
        self, drift: list[float]
    ) -> str:
        """
        Map a list of drift values to one of the four trend labels.

        Priority (highest to lowest):
          volatile  → std-dev(drift) > _DRIFT_VOLATILE
          improving → mean(drift) > _DRIFT_IMPROVING
          declining → mean(drift) < _DRIFT_DECLINING
          stable    → otherwise
        """
        if not drift:
            return "stable"

        avg = sum(drift) / len(drift)

        if len(drift) >= 2:
            variance = sum((d - avg) ** 2 for d in drift) / len(drift)
            std = math.sqrt(variance)
        else:
            std = 0.0

        if std > _DRIFT_VOLATILE:
            return "volatile"
        if avg > _DRIFT_IMPROVING:
            return "improving"
        if avg < _DRIFT_DECLINING:
            return "declining"
        return "stable"

    def _compute_risk_delta(
        self, insights: list[TranscriptInsight]
    ) -> tuple[list[RiskFactor], list[RiskFactor]]:
        """
        Compare the latest quarter's risks against all previous quarters' risks.
        Returns (new_risks, dropped_risks) where:
          new_risks     = risks in latest NOT matched in any prior quarter
          dropped_risks = risks in prior quarters NOT matched in latest
        """
        if len(insights) < 2:
            return [], []

        latest_risks = insights[-1].risks
        prior_risks: list[RiskFactor] = []
        for q in insights[:-1]:
            prior_risks.extend(q.risks)

        new_risks = [
            r for r in latest_risks
            if not any(self._risks_similar(r, p) for p in prior_risks)
        ]
        dropped_risks = [
            p for p in prior_risks
            if not any(self._risks_similar(p, r) for r in latest_risks)
        ]
        # Deduplicate dropped risks by text similarity
        deduped_dropped: list[RiskFactor] = []
        for risk in dropped_risks:
            if not any(self._risks_similar(risk, d) for d in deduped_dropped):
                deduped_dropped.append(risk)

        return new_risks, deduped_dropped

    @staticmethod
    def _risks_similar(a: RiskFactor, b: RiskFactor) -> bool:
        """
        Fuzzy word-overlap check: True when the two risk descriptions share enough
        content to be considered the same risk.
        """
        words_a = set(re.sub(r"[^a-z0-9 ]", "", a.text.lower()).split())
        words_b = set(re.sub(r"[^a-z0-9 ]", "", b.text.lower()).split())
        if not words_a or not words_b:
            return False
        overlap = words_a & words_b
        # Jaccard-like: overlap / smaller set
        similarity = len(overlap) / min(len(words_a), len(words_b))
        return similarity >= _RISK_SIMILARITY_THRESHOLD

    @staticmethod
    def _fallback_summary(trend: QuarterTrend) -> str:
        """Plain-text summary when Claude is unavailable."""
        q_count = len(trend.quarters)
        if q_count == 0:
            return f"No data available for {trend.ticker}."
        latest = trend.quarters[-1]
        return (
            f"{trend.ticker} shows a {trend.trend_direction} sentiment trend "
            f"across {q_count} quarter(s). "
            f"Latest quarter (Q{latest.quarter} {latest.year}) scored "
            f"{latest.sentiment.score:+.2f} ({latest.sentiment.label}). "
            f"New risks: {len(trend.new_risks)}. "
            f"Dropped risks: {len(trend.dropped_risks)}."
        )


