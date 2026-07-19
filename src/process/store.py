"""
SQLite-backed transcript store with deduplication via content hash.
Week-2 addition: insights table for cached LLM analysis results.
"""
import hashlib
import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.analyze.models import TranscriptInsight

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "transcripts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    cik           TEXT,
    form_type     TEXT,
    filed_date    TEXT,
    quarter       INTEGER,
    year          INTEGER,
    raw_text      TEXT,
    cleaned_text  TEXT,
    url           TEXT,
    fetched_at    TEXT    NOT NULL,
    content_hash  TEXT    UNIQUE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ticker       ON transcripts(ticker);
CREATE INDEX IF NOT EXISTS idx_ticker_year  ON transcripts(ticker, year);
CREATE INDEX IF NOT EXISTS idx_hash         ON transcripts(content_hash);

CREATE TABLE IF NOT EXISTS insights (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    quarter       INTEGER NOT NULL,
    year          INTEGER NOT NULL,
    insight_json  TEXT    NOT NULL,
    analyzed_at   TEXT    NOT NULL,
    UNIQUE(ticker, quarter, year)
);
CREATE INDEX IF NOT EXISTS idx_insights_ticker ON insights(ticker);
"""


class TranscriptStore:
    """Persist and query earnings call transcripts in a local SQLite database."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, d: dict) -> bool:
        """
        Insert a transcript dict.  Returns True on insert, False if duplicate.
        Accepts the dict schema produced by TranscriptFetcher / FallbackScraper.
        """
        content_hash = self._make_hash(d)
        if self.exists(content_hash):
            logger.debug("Duplicate transcript skipped: %s", content_hash)
            return False

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transcripts
                    (ticker, cik, form_type, filed_date, quarter, year,
                     raw_text, cleaned_text, url, fetched_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d.get("ticker", ""),
                    d.get("cik"),
                    d.get("form_type"),
                    d.get("filed_date"),
                    d.get("quarter"),
                    d.get("year"),
                    d.get("raw_text"),
                    d.get("cleaned_text"),
                    d.get("url"),
                    now,
                    content_hash,
                ),
            )
        return True

    def get_by_ticker(self, ticker: str) -> list[dict]:
        """Return all transcripts for a ticker, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM transcripts WHERE ticker = ? ORDER BY filed_date DESC",
                (ticker.upper(),),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_quarter(self, ticker: str, year: int, quarter: int) -> list[dict]:
        """Return transcripts matching a specific ticker + fiscal year + quarter."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM transcripts
                   WHERE ticker = ? AND year = ? AND quarter = ?
                   ORDER BY filed_date DESC""",
                (ticker.upper(), year, quarter),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def exists(self, content_hash: str) -> bool:
        """Return True if a record with this hash already exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM transcripts WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return row is not None

    def list_tickers(self) -> list[str]:
        """Return sorted list of distinct tickers in the store."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM transcripts ORDER BY ticker"
            ).fetchall()
        return [r[0] for r in rows]

    def get_stats(self) -> dict:
        """Return summary statistics about the store."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
            tickers = conn.execute(
                "SELECT COUNT(DISTINCT ticker) FROM transcripts"
            ).fetchone()[0]
            years = conn.execute(
                "SELECT MIN(year), MAX(year) FROM transcripts WHERE year > 0"
            ).fetchone()
            latest = conn.execute(
                "SELECT filed_date FROM transcripts ORDER BY filed_date DESC LIMIT 1"
            ).fetchone()
        return {
            "total_transcripts": total,
            "unique_tickers": tickers,
            "year_range": (years[0], years[1]) if years and years[0] else (None, None),
            "latest_filing": latest[0] if latest else None,
            "db_path": str(self.db_path),
        }

    # ------------------------------------------------------------------
    # Insight persistence (Week 2)
    # ------------------------------------------------------------------

    def save_insight(self, insight: "TranscriptInsight") -> bool:
        """
        Upsert a TranscriptInsight into the insights table.
        Returns True on insert/update, False on error.
        """
        from src.analyze.models import TranscriptInsight as TI  # local import avoids circular dep

        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO insights (ticker, quarter, year, insight_json, analyzed_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, quarter, year) DO UPDATE SET
                        insight_json = excluded.insight_json,
                        analyzed_at  = excluded.analyzed_at
                    """,
                    (
                        insight.ticker,
                        insight.quarter,
                        insight.year,
                        insight.model_dump_json(),
                        now,
                    ),
                )
            return True
        except Exception as exc:
            logger.error("save_insight failed: %s", exc)
            return False

    def get_insights(self, ticker: str) -> list["TranscriptInsight"]:
        """Return all cached insights for a ticker, oldest first."""
        from src.analyze.models import TranscriptInsight as TI

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT insight_json FROM insights WHERE ticker = ? ORDER BY year, quarter",
                (ticker.upper(),),
            ).fetchall()

        results = []
        for row in rows:
            try:
                results.append(TI.model_validate_json(row[0]))
            except Exception as exc:
                logger.warning("Could not deserialise insight: %s", exc)
        return results

    def insight_exists(self, ticker: str, quarter: int, year: int) -> bool:
        """Return True if an insight for this ticker/quarter/year is already cached."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM insights WHERE ticker = ? AND quarter = ? AND year = ?",
                (ticker.upper(), quarter, year),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _make_hash(d: dict) -> str:
        """MD5 of ticker + filed_date + first 200 chars of raw_text."""
        ticker = str(d.get("ticker", ""))
        filed_date = str(d.get("filed_date", ""))
        raw_text = str(d.get("raw_text", ""))[:200]
        payload = ticker + filed_date + raw_text
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return dict(row)
