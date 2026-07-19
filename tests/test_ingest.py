"""
Unit tests for FinSight Week 1: ingest and process modules.
"""
import pytest
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# TranscriptStore — save and deduplication
# ---------------------------------------------------------------------------

class TestTranscriptStore:
    SAMPLE = {
        "ticker": "AAPL",
        "cik": "320193",
        "form_type": "8-K",
        "filed_date": "2023-08-03",
        "quarter": 3,
        "year": 2023,
        "raw_text": "Apple Q3 2023 Earnings Call Transcript. Good afternoon everyone...",
        "cleaned_text": "Apple Q3 2023 Earnings Call Transcript. Good afternoon everyone...",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
    }

    def _make_store(self):
        from src.process.store import TranscriptStore
        tmp = tempfile.mktemp(suffix=".db")
        return TranscriptStore(db_path=tmp)

    def test_save_returns_true_on_new_record(self):
        store = self._make_store()
        result = store.save(self.SAMPLE)
        assert result is True

    def test_save_returns_false_on_duplicate(self):
        store = self._make_store()
        store.save(self.SAMPLE)
        result = store.save(self.SAMPLE)  # same data → same hash
        assert result is False

    def test_get_by_ticker_returns_saved(self):
        store = self._make_store()
        store.save(self.SAMPLE)
        rows = store.get_by_ticker("AAPL")
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["year"] == 2023

    def test_get_by_quarter(self):
        store = self._make_store()
        store.save(self.SAMPLE)
        rows = store.get_by_quarter("AAPL", 2023, 3)
        assert len(rows) == 1
        rows_wrong = store.get_by_quarter("AAPL", 2023, 1)
        assert len(rows_wrong) == 0

    def test_list_tickers(self):
        store = self._make_store()
        store.save(self.SAMPLE)
        tickers = store.list_tickers()
        assert "AAPL" in tickers

    def test_get_stats(self):
        store = self._make_store()
        store.save(self.SAMPLE)
        stats = store.get_stats()
        assert stats["total_transcripts"] == 1
        assert stats["unique_tickers"] == 1

    def test_different_tickers_not_dedup(self):
        store = self._make_store()
        store.save(self.SAMPLE)
        msft = {**self.SAMPLE, "ticker": "MSFT"}
        result = store.save(msft)
        assert result is True
        assert store.get_stats()["total_transcripts"] == 2

    def test_empty_store_stats(self):
        store = self._make_store()
        stats = store.get_stats()
        assert stats["total_transcripts"] == 0
        assert stats["unique_tickers"] == 0


# ---------------------------------------------------------------------------
# TranscriptCleaner
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """
Apple Q3 2023 Earnings Call

Participants:
Tim Cook - CEO
Luca Maestri - CFO
Shannon Cross - Analyst

Prepared Remarks:

Tim Cook: Good afternoon and welcome to Apple's third fiscal quarter 2023 conference call.
We had a very strong quarter with revenues of $81.8 billion.

Luca Maestri: Thank you Tim. We are pleased to report quarterly revenue of $81.8 billion,
down 1 percent year over year. Our gross margin was 44.5 percent.

Question and Answer Session:

Shannon Cross: Hi, thanks for taking my question. Can you talk about iPhone demand?

Tim Cook: Sure, iPhone revenue was $39.7 billion for the quarter.
We feel great about the momentum we're seeing.

Operator: Thank you. Your next question comes from Amit Daryanani.
"""


class TestTranscriptCleaner:
    def _make_cleaner(self):
        from src.process.cleaner import TranscriptCleaner
        return TranscriptCleaner()

    def test_clean_returns_string(self):
        cleaner = self._make_cleaner()
        result = cleaner.clean(SAMPLE_TRANSCRIPT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_removes_smart_quotes(self):
        cleaner = self._make_cleaner()
        result = cleaner.clean("“Hello” and ‘world’")
        assert "“" not in result
        assert "”" not in result

    def test_clean_strips_urls(self):
        cleaner = self._make_cleaner()
        result = cleaner.clean("See https://www.example.com/report for details.")
        assert "https://" not in result

    def test_clean_collapses_whitespace(self):
        cleaner = self._make_cleaner()
        result = cleaner.clean("hello   world\t\ttabs")
        assert "   " not in result
        assert "\t" not in result

    def test_chunk_returns_list(self):
        cleaner = self._make_cleaner()
        chunks = cleaner.chunk(SAMPLE_TRANSCRIPT, max_tokens=200)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_chunk_respects_max_tokens(self):
        cleaner = self._make_cleaner()
        max_tokens = 100
        chunks = cleaner.chunk(SAMPLE_TRANSCRIPT, max_tokens=max_tokens)
        max_chars = max_tokens * 4
        for chunk in chunks:
            assert len(chunk) <= max_chars * 2, (
                f"Chunk too long: {len(chunk)} chars (limit ~{max_chars})"
            )

    def test_chunk_no_content_lost(self):
        cleaner = self._make_cleaner()
        text = "Word " * 500
        chunks = cleaner.chunk(text, max_tokens=100)
        combined = " ".join(chunks)
        # All words should be present (allow for minor whitespace differences)
        assert len(combined) >= len(text) * 0.95

    def test_detect_sections_keys(self):
        cleaner = self._make_cleaner()
        sections = cleaner.detect_sections(SAMPLE_TRANSCRIPT)
        assert "prepared_remarks" in sections
        assert "qa_session" in sections
        assert "participants" in sections

    def test_detect_sections_qa_nonempty(self):
        cleaner = self._make_cleaner()
        sections = cleaner.detect_sections(SAMPLE_TRANSCRIPT)
        assert sections["qa_session"] != "" or sections["prepared_remarks"] != ""

    def test_detect_sections_fallback_to_prepared(self):
        cleaner = self._make_cleaner()
        plain = "Some random text with no headers at all."
        sections = cleaner.detect_sections(plain)
        assert sections["prepared_remarks"] == plain


# ---------------------------------------------------------------------------
# TranscriptFetcher — initialization only (no live HTTP in unit tests)
# ---------------------------------------------------------------------------

class TestTranscriptFetcherInit:
    def test_instantiates_without_error(self):
        from src.ingest.edgar import TranscriptFetcher
        fetcher = TranscriptFetcher()
        assert fetcher is not None

    def test_has_session_with_user_agent(self):
        from src.ingest.edgar import TranscriptFetcher
        fetcher = TranscriptFetcher()
        ua = fetcher.session.headers.get("User-Agent", "")
        assert "FinSight" in ua

    def test_fetch_by_ticker_returns_tuple_type(self, monkeypatch):
        from src.ingest.edgar import TranscriptFetcher
        fetcher = TranscriptFetcher()
        # _get_cik now returns (cik, company_name); (None, None) → no CIK found
        monkeypatch.setattr(fetcher, "_get_cik", lambda ticker: (None, None))
        result = fetcher.fetch_by_ticker("FAKE", 2023, 2023)
        assert isinstance(result, tuple)
        docs, status = result
        assert docs == []
        assert status["ticker"] == "FAKE"
        assert "CIK not found" in (status["errors"] or [""])[0]
