"""
TranscriptAnalyzer: drives Gemini API calls to extract sentiment, risks, and guidance
from earnings call transcripts stored by the Week-1 pipeline.

Provider: Google Gemini (gemini-1.5-flash, free tier)
SDK:      google-genai  (google.genai)
Key var:  GEMINI_API_KEY  (read from .env)
"""
from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.analyze.models import (
    GuidanceStatement,
    RiskFactor,
    SentimentResult,
    TranscriptInsight,
)
from src.analyze.prompts import GUIDANCE_PROMPT, RISK_PROMPT, SENTIMENT_PROMPT
from src.process.cleaner import TranscriptCleaner

# Load .env from project root
_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

logger = logging.getLogger(__name__)
console = Console()

_GEMINI_MODEL = "gemini-flash-lite-latest"
_MAX_RETRIES   = 3
_RETRY_BACKOFF = 2  # seconds


class TranscriptAnalyzer:
    """Analyse a single earnings call transcript using the Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        from google import genai  # deferred so tests can mock before import

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Add it to your .env file."
            )
        self._client  = genai.Client(api_key=key)
        self._cleaner = TranscriptCleaner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, transcript_dict: dict) -> TranscriptInsight:
        """
        Full analysis pipeline for one transcript dict (as returned by TranscriptStore).

        Steps:
          1. Clean raw text
          2. Detect sections (prepared_remarks / qa_session)
          3. Call Gemini for sentiment, risks, and guidance
          4. Parse XML responses into Pydantic models
          5. Return TranscriptInsight
        """
        ticker  = transcript_dict.get("ticker", "UNKNOWN").upper()
        quarter = int(transcript_dict.get("quarter") or 0)
        year    = int(transcript_dict.get("year")    or 0)

        raw   = transcript_dict.get("raw_text") or transcript_dict.get("cleaned_text") or ""
        clean = self._cleaner.clean(raw)

        sections = self._cleaner.detect_sections(clean)

        sentiment_src = sections.get("prepared_remarks") or clean
        guidance_src  = sections.get("prepared_remarks") or clean
        risk_src      = clean  # include Q&A

        sentiment_chunk = self._best_chunk(sentiment_src, max_chars=6000)
        guidance_chunk  = self._best_chunk(guidance_src,  max_chars=6000)
        risk_chunk      = self._best_chunk(risk_src,      max_chars=6000)

        sentiment = self._extract_sentiment(sentiment_chunk)
        risks     = self._extract_risks(risk_chunk)
        guidance  = self._extract_guidance(guidance_chunk)

        summary = self._build_summary(sentiment, risks, guidance, ticker, quarter, year)

        return TranscriptInsight(
            ticker=ticker,
            quarter=max(1, min(4, quarter)) if quarter else 1,
            year=year if year else datetime.now().year,
            sentiment=sentiment,
            risks=risks,
            guidance=guidance,
            summary=summary,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_sentiment(self, text: str) -> SentimentResult:
        prompt  = SENTIMENT_PROMPT.format(transcript_chunk=text)
        xml_str = self._call_llm(prompt, max_tokens=512)
        return self._parse_sentiment_xml(xml_str)

    def _extract_risks(self, text: str) -> list[RiskFactor]:
        prompt  = RISK_PROMPT.format(transcript_chunk=text)
        xml_str = self._call_llm(prompt, max_tokens=1024)
        return self._parse_risks_xml(xml_str)

    def _extract_guidance(self, text: str) -> list[GuidanceStatement]:
        prompt  = GUIDANCE_PROMPT.format(transcript_chunk=text)
        xml_str = self._call_llm(prompt, max_tokens=1024)
        return self._parse_guidance_xml(xml_str)

    # ------------------------------------------------------------------
    # XML parsers
    # ------------------------------------------------------------------

    def _parse_sentiment_xml(self, xml_str: str) -> SentimentResult:
        try:
            root = self._safe_parse_xml(xml_str, "sentiment")
            return SentimentResult(
                score=self._get_text(root, "score",      "0.0"),
                label=self._get_text(root, "label",      "neutral"),
                rationale=self._get_text(root, "rationale", ""),
                confidence=self._get_text(root, "confidence","0.8"),
            )
        except Exception as exc:
            logger.warning("Sentiment XML parse failed (%s) — returning neutral default", exc)
            return SentimentResult(score=0.0, label="neutral", rationale=xml_str[:200], confidence=0.5)

    def _parse_risks_xml(self, xml_str: str) -> list[RiskFactor]:
        risks = []
        try:
            root = self._safe_parse_xml(xml_str, "risks")
            for el in root.findall("risk"):
                risks.append(RiskFactor(
                    text=self._get_text(el, "text",     ""),
                    severity=self._get_text(el, "severity","medium"),
                    category=self._get_text(el, "category","general"),
                ))
        except Exception as exc:
            logger.warning("Risks XML parse failed (%s)", exc)
        return risks

    def _parse_guidance_xml(self, xml_str: str) -> list[GuidanceStatement]:
        items = []
        try:
            root = self._safe_parse_xml(xml_str, "guidance")
            for el in root.findall("item"):
                items.append(GuidanceStatement(
                    text=self._get_text(el, "text",      ""),
                    metric=self._get_text(el, "metric",   "unspecified"),
                    direction=self._get_text(el, "direction","unclear"),
                    timeframe=self._get_text(el, "timeframe","unspecified"),
                ))
        except Exception as exc:
            logger.warning("Guidance XML parse failed (%s)", exc)
        return items

    # ------------------------------------------------------------------
    # Gemini API call (retries + logging)
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Send a prompt to Gemini and return the text response.
        Retries up to _MAX_RETRIES times with exponential backoff.
        Logs token usage via rich.
        """
        from google import genai
        from google.genai import types as gtypes

        config = gtypes.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.1,
        )

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=prompt,
                    config=config,
                )
                self._log_usage(response, attempt)
                # response.text can be None for some models/safety filters
                text = response.text
                if text is None:
                    # Fall back to concatenating part texts directly
                    parts = [p.text for p in (response.candidates or [{}])[0].content.parts
                             if hasattr(p, "text") and p.text]
                    text = "\n".join(parts)
                return text or ""
            except Exception as exc:
                exc_str = str(exc)
                # Honour the retryDelay hint from the API (e.g. "Please retry in 49s")
                retry_hint = re.search(r"retry in (\d+)", exc_str)
                wait = int(retry_hint.group(1)) if retry_hint else _RETRY_BACKOFF ** (attempt + 1)
                wait = min(wait, 65)  # cap at 65 s per attempt
                logger.warning(
                    "Gemini API error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, exc_str[:120], wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"Gemini API call failed after {_MAX_RETRIES} attempts")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _best_chunk(self, text: str, max_chars: int = 6000) -> str:
        if len(text) <= max_chars:
            return text
        cutoff = text.rfind("\n\n", 0, max_chars)
        return text[:cutoff if cutoff != -1 else max_chars]

    def _build_summary(
        self,
        sentiment: SentimentResult,
        risks: list[RiskFactor],
        guidance: list[GuidanceStatement],
        ticker: str,
        quarter: int,
        year: int,
    ) -> str:
        high_risks    = [r.text for r in risks    if r.severity == "high"]
        up_guidance   = [g.text for g in guidance if g.direction == "up"]
        down_guidance = [g.text for g in guidance if g.direction == "down"]

        parts = [
            f"{ticker} Q{quarter} {year}: sentiment {sentiment.label} "
            f"(score {sentiment.score:+.2f}, confidence {sentiment.confidence:.0%}).",
        ]
        if high_risks:
            parts.append(f"High-severity risks: {'; '.join(high_risks[:3])}.")
        if up_guidance:
            parts.append(f"Positive guidance: {up_guidance[0]}.")
        if down_guidance:
            parts.append(f"Cautious guidance: {down_guidance[0]}.")
        return " ".join(parts)

    @staticmethod
    def _safe_parse_xml(text: str, root_tag: str) -> ET.Element:
        pattern = rf"<{root_tag}[\s\S]*?</{root_tag}>"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            raise ValueError(f"No <{root_tag}> element found in response")
        return ET.fromstring(m.group(0))

    @staticmethod
    def _get_text(element: ET.Element, tag: str, default: str = "") -> str:
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return default

    def _log_usage(self, response, attempt: int) -> None:
        try:
            meta = response.usage_metadata
            table = Table(show_header=False, box=None, padding=(0, 1))
            # google.genai uses prompt_token_count / candidates_token_count
            table.add_row("[dim]input tokens[/dim]",  str(getattr(meta, "prompt_token_count",     "?")))
            table.add_row("[dim]output tokens[/dim]", str(getattr(meta, "candidates_token_count", "?")))
            if attempt:
                table.add_row("[dim]attempt[/dim]", str(attempt + 1))
            console.print(table)
        except Exception:
            pass  # usage logging is best-effort
