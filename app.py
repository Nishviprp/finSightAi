"""
HuggingFace Spaces entry point for FinSight.

HF Spaces looks for app.py at the repo root and runs it with Streamlit.
This file:
  1. Reads GEMINI_API_KEY from HF Spaces secrets (os.environ)
  2. Writes it to a temporary .env so the rest of the app picks it up
  3. Seeds the SQLite DB with two baked-in AAPL sample transcripts
     so the demo works without live EDGAR calls on first visit
  4. Delegates to app/streamlit_app.py for the actual UI
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── write .env from HF Spaces secret ────────────────────────────────────────
_gemini_key = os.environ.get("GEMINI_API_KEY", "")
_env_path   = ROOT / ".env"
if _gemini_key and not _env_path.exists():
    _env_path.write_text(f"GEMINI_API_KEY={_gemini_key}\n")

# ── sample transcripts baked-in for cold-start demo ─────────────────────────
_SAMPLE_Q3_2023 = """
Apple Q3 FY2023 Earnings Release

CUPERTINO, CALIFORNIA — Apple today announced financial results for its fiscal
2023 third quarter ended July 1, 2023. The Company posted quarterly revenue of
$81.8 billion, down 1 percent year over year, and quarterly earnings per diluted
share of $1.26, up 5 percent year over year.

"We are happy to report that we had an all-time revenue record in Services during
the quarter and that our installed base of active devices reached an all-time high,"
said Tim Cook, Apple's CEO. "We're currently live in 24 countries and regions with
our savings account, which has received strong interest, and we're looking forward to
sharing more details about our exciting product roadmap at our upcoming Wonderlust event."

Revenue by segment:
  iPhone:     $39.67B (down 2% YoY)
  Mac:        $6.84B  (down 7% YoY)
  iPad:       $5.79B  (down 20% YoY)
  Wearables:  $8.28B  (down 2% YoY)
  Services:   $21.21B (up 8% YoY) — all-time high

Gross margin: 44.5%
Operating income: $23.0B

Guidance for Q4 FY2023:
  - Revenue: similar to Q3 FY2023 year-over-year performance
  - Gross margin: between 44% and 45%
  - Services revenue expected to achieve similar growth rate to Q3

Risks mentioned:
  - Foreign exchange headwinds continue to weigh on international revenue
  - Macro economic uncertainty affecting consumer discretionary spending
  - Regulatory scrutiny in European and Asian markets
  - Supply chain normalisation ongoing, inventory levels stabilising
"""

_SAMPLE_Q4_2023 = """
Apple Q4 FY2023 Earnings Release

CUPERTINO, CALIFORNIA — Apple today announced financial results for its fiscal
2023 fourth quarter ended September 30, 2023. The Company posted quarterly revenue
of $89.5 billion, down 1 percent year over year, and quarterly earnings per diluted
share of $1.46, up 13 percent year over year.

"Today Apple is pleased to report a September quarter revenue record for iPhone and
an all-time revenue record in Services," said Tim Cook, Apple's CEO. "We now have our
strongest lineup of products ever heading into the holiday season, including the
iPhone 15 lineup and our first carbon neutral Apple Watch models."

Revenue by segment:
  iPhone:     $43.81B (up 3% YoY) — September quarter record
  Mac:        $7.61B  (down 34% YoY)
  iPad:       $6.44B  (down 10% YoY)
  Wearables:  $9.32B  (down 3% YoY)
  Services:   $22.31B (up 16% YoY) — all-time record

Gross margin: 45.2%
EPS: $1.46 (up 13% YoY)

Guidance for Q1 FY2024:
  - Revenue: similar growth rate to Q4 FY2023
  - Gross margin: between 45% and 46%
  - Services expected to continue double-digit growth

Risks mentioned:
  - China revenue down 2.5% YoY amid macro softness
  - Rising interest rates compressing consumer electronics spending
  - Regulatory headwinds in EU (Digital Markets Act compliance costs)
  - FX headwinds persist across most international markets
  - Competition intensifying in the wearables segment

Apple's board declared a cash dividend of $0.24 per share, payable November 16, 2023.
"""

_SAMPLE_DOCS = [
    {
        "ticker":       "AAPL",
        "cik":          "320193",
        "form_type":    "8-K",
        "filed_date":   "2023-08-03",
        "quarter":      3,
        "year":         2023,
        "raw_text":     _SAMPLE_Q3_2023.strip(),
        "cleaned_text": _SAMPLE_Q3_2023.strip(),
        "url":          "https://www.sec.gov/Archives/edgar/data/320193/sample-q3-2023",
    },
    {
        "ticker":       "AAPL",
        "cik":          "320193",
        "form_type":    "8-K",
        "filed_date":   "2023-11-02",
        "quarter":      4,
        "year":         2023,
        "raw_text":     _SAMPLE_Q4_2023.strip(),
        "cleaned_text": _SAMPLE_Q4_2023.strip(),
        "url":          "https://www.sec.gov/Archives/edgar/data/320193/sample-q4-2023",
    },
]


def _seed_demo_data() -> None:
    """
    Insert baked-in sample transcripts into the SQLite DB if they don't exist yet.
    Safe to call on every startup — dedup hash prevents double-inserts.
    """
    try:
        from src.process.store import TranscriptStore
        store = TranscriptStore()
        seeded = 0
        for doc in _SAMPLE_DOCS:
            if store.save(doc):
                seeded += 1
        if seeded:
            print(f"[app.py] Seeded {seeded} sample AAPL transcript(s) for demo.")
        else:
            print("[app.py] Demo data already present.")
    except Exception as exc:
        print(f"[app.py] Warning: could not seed demo data: {exc}")


# ── seed on import (runs once at Streamlit startup) ──────────────────────────
_seed_demo_data()

# ── delegate to the actual Streamlit app ─────────────────────────────────────
# HF Spaces runs `streamlit run app.py`, so we exec the real app module
# to avoid duplicating the entire UI here.
exec(open(ROOT / "app" / "streamlit_app.py").read())
