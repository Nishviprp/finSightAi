"""
SQLite-backed log of past trading signals, for tracking real accuracy over
time.

This starts empty and stays empty until signals are actually scanned and
recorded — there is no seeded or backfilled history here. "Accuracy" is
computed only from signals old enough to have a real, observable outcome
(price actually moved or didn't), fetched live; nothing about past
performance is fabricated or estimated. Past directional accuracy is not a
guarantee of future performance.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "signal_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol           TEXT    NOT NULL,
    signal           TEXT    NOT NULL,   -- "BUY" or "SELL" (HOLD isn't tracked — nothing to evaluate)
    confidence       REAL    NOT NULL,
    price_at_signal  REAL    NOT NULL,
    scanned_at       TEXT    NOT NULL    -- ISO8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_signal_symbol      ON signal_history(symbol);
CREATE INDEX IF NOT EXISTS idx_signal_scanned_at  ON signal_history(scanned_at);
"""


class SignalHistoryStore:
    """Persist and evaluate BUY/SELL signals over time."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, rows: list[dict]) -> int:
        """
        Record BUY/SELL rows from a scan_signals()/calculate_signals()
        result (HOLD rows are silently skipped — there's no directional
        outcome to evaluate later). Returns how many rows were recorded.
        """
        actionable = [r for r in rows if r.get("signal") in ("BUY", "SELL")]
        if not actionable:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO signal_history (symbol, signal, confidence, price_at_signal, scanned_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [(r["symbol"], r["signal"], r["confidence"], r["price"], now) for r in actionable],
            )
        return len(actionable)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_recent(self, limit: int = 100) -> list[dict]:
        """Most recently recorded signals, newest first."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM signal_history ORDER BY scanned_at DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_evaluable(self, min_age_days: int = 1) -> list[dict]:
        """Signals recorded at least *min_age_days* ago — old enough that
        "did the price move as predicted" is a real question, not one
        where the answer is still mostly noise from the scan itself.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM signal_history WHERE scanned_at <= ? ORDER BY scanned_at DESC",
                (cutoff,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0]
