"""
Pydantic v2 output models for FinSight LLM analysis results.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SentimentResult(BaseModel):
    """Sentiment extracted from an earnings call transcript or section."""

    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment polarity: -1 (very negative) → +1 (very positive)",
    )
    label: str = Field(..., description="Short human-readable label, e.g. 'cautiously optimistic'")
    rationale: str = Field(..., description="One-paragraph explanation of the score")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Model confidence in this sentiment estimate",
    )

    @field_validator("score", "confidence", mode="before")
    @classmethod
    def coerce_float(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0


class RiskFactor(BaseModel):
    """A single risk factor extracted from the transcript."""

    text: str = Field(..., description="Verbatim or paraphrased risk statement")
    severity: Literal["low", "medium", "high"] = Field(
        default="medium", description="Assessed severity of the risk"
    )
    category: str = Field(
        default="general",
        description="Risk category, e.g. macro, regulatory, competitive, operational",
    )
    is_new: bool = Field(
        default=False,
        description="True if this risk did not appear in the prior quarter's analysis",
    )


class GuidanceStatement(BaseModel):
    """A forward-looking guidance item extracted from the transcript."""

    text: str = Field(..., description="The guidance statement as given or paraphrased")
    metric: str = Field(
        default="unspecified",
        description="The financial or operational metric being guided (e.g. revenue, EPS, margins)",
    )
    direction: Literal["up", "down", "flat", "unclear"] = Field(
        default="unclear", description="Direction of the guided metric"
    )
    timeframe: str = Field(
        default="unspecified",
        description="The timeframe for the guidance, e.g. 'Q2 2024' or 'full year 2024'",
    )


class TranscriptInsight(BaseModel):
    """Full analysis result for a single earnings call transcript."""

    ticker: str
    quarter: int = Field(..., ge=1, le=4)
    year: int = Field(..., ge=2000, le=2100)
    sentiment: SentimentResult
    risks: list[RiskFactor] = Field(default_factory=list)
    guidance: list[GuidanceStatement] = Field(default_factory=list)
    summary: str = Field(default="", description="One-paragraph executive summary of the call")
    analyzed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class QuarterTrend(BaseModel):
    """Multi-quarter trend analysis for a single ticker."""

    ticker: str
    quarters: list[TranscriptInsight] = Field(default_factory=list)
    sentiment_drift: list[float] = Field(
        default_factory=list,
        description="Quarter-over-quarter sentiment delta: drift[i] = q[i+1].score - q[i].score",
    )
    new_risks: list[RiskFactor] = Field(
        default_factory=list,
        description="Risks appearing in the latest quarter not seen in prior quarters",
    )
    dropped_risks: list[RiskFactor] = Field(
        default_factory=list,
        description="Risks from prior quarters absent from the latest quarter",
    )
    trend_direction: Literal["improving", "declining", "stable", "volatile"] = Field(
        default="stable",
        description="Overall sentiment trajectory across the analysed quarters",
    )
