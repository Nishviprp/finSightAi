"""
Transcript cleaning and chunking utilities.
Normalizes raw text, splits into token-safe chunks, and detects key sections.
"""
import re
from typing import Optional

# Speaker turn pattern: "FirstName LastName:" or "FIRSTNAME LASTNAME:" or "Operator:"
_SPEAKER_RE = re.compile(
    r"^([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+){0,3})\s*[:—]",
    re.MULTILINE,
)

# Section header keywords
_PREPARED_HEADERS = re.compile(
    r"(prepared remarks?|opening remarks?|presentation|opening statement)",
    re.IGNORECASE,
)
_QA_HEADERS = re.compile(
    r"(question[- ]and[- ]answer|q\s*[&a]\s*a|q&a session|q&a|analyst questions)",
    re.IGNORECASE,
)
_PARTICIPANT_HEADERS = re.compile(
    r"(participants|attendees|speakers|executives|analysts present)",
    re.IGNORECASE,
)

# Chars-per-token approximation (conservative GPT-style)
_CHARS_PER_TOKEN = 4


class TranscriptCleaner:
    """Normalize, chunk, and section-detect earnings call transcripts."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, raw_text: str) -> str:
        """Return a normalized plain-text string from raw transcript input."""
        text = raw_text

        # Normalize Unicode dashes / smart quotes
        text = text.replace("—", " - ").replace("–", "-")
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')

        # Remove HTML entities if any slipped through
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)

        # Strip URLs
        text = re.sub(r"https?://\S+", "", text)

        # Remove page numbers / slide references
        text = re.sub(r"\bslide\s+\d+\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bpage\s+\d+\b", "", text, flags=re.IGNORECASE)

        # Collapse multiple spaces / tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse 3+ blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def chunk(self, text: str, max_tokens: int = 2000) -> list[str]:
        """
        Split text into chunks of at most max_tokens, preferring speaker-turn
        boundaries so each chunk is semantically coherent.
        """
        max_chars = max_tokens * _CHARS_PER_TOKEN
        segments = self._split_on_speakers(text)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg)
            # If a single segment exceeds the limit, hard-split it
            if seg_len > max_chars:
                if current_parts:
                    chunks.append("\n".join(current_parts))
                    current_parts = []
                    current_len = 0
                for hard_chunk in self._hard_split(seg, max_chars):
                    chunks.append(hard_chunk)
            elif current_len + seg_len > max_chars and current_parts:
                chunks.append("\n".join(current_parts))
                current_parts = [seg]
                current_len = seg_len
            else:
                current_parts.append(seg)
                current_len += seg_len

        if current_parts:
            chunks.append("\n".join(current_parts))

        return [c.strip() for c in chunks if c.strip()]

    def detect_sections(self, text: str) -> dict[str, str]:
        """
        Identify the three main sections of an earnings call transcript.
        Returns a dict with keys: prepared_remarks, qa_session, participants.
        Values are the extracted text for each section (empty string if not found).
        """
        lines = text.splitlines()
        section_starts: dict[str, int] = {}

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if _PARTICIPANT_HEADERS.search(stripped) and "participants" not in section_starts:
                section_starts["participants"] = i
            elif _PREPARED_HEADERS.search(stripped) and "prepared_remarks" not in section_starts:
                section_starts["prepared_remarks"] = i
            elif _QA_HEADERS.search(stripped) and "qa_session" not in section_starts:
                section_starts["qa_session"] = i

        # Infer ordering and slice
        ordered = sorted(section_starts.items(), key=lambda x: x[1])
        sections: dict[str, str] = {
            "participants": "",
            "prepared_remarks": "",
            "qa_session": "",
        }

        for idx, (name, start) in enumerate(ordered):
            end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(lines)
            sections[name] = "\n".join(lines[start:end]).strip()

        # If no sections detected at all, put everything in prepared_remarks
        if not any(sections.values()):
            sections["prepared_remarks"] = text

        return sections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_on_speakers(self, text: str) -> list[str]:
        """Split text at every speaker-turn boundary."""
        positions = [m.start() for m in _SPEAKER_RE.finditer(text)]
        if not positions:
            return [text]

        segments = []
        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            segments.append(text[pos:end])

        # Prepend any text before the first speaker
        if positions[0] > 0:
            segments.insert(0, text[: positions[0]])

        return segments

    def _hard_split(self, text: str, max_chars: int) -> list[str]:
        """Forcibly split a long segment at sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            if current_len + len(sent) > max_chars and current:
                chunks.append(" ".join(current))
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent)
        if current:
            chunks.append(" ".join(current))
        return chunks
