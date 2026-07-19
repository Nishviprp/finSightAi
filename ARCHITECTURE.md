# FinSight — Architecture & Design Decisions

This document explains the non-obvious choices made during the four-week build.

---

## 1. Why map-reduce chunking for transcripts?

Earnings call transcripts range from **15 000 to 40 000 tokens** — well beyond what
fits in a single prompt without eating into the model's output budget.

FinSight uses a **section-first, chunk-second** strategy:

1. `TranscriptCleaner.detect_sections()` splits the document into
   `prepared_remarks`, `qa_session`, and `participants`.
2. Each section is fed to a dedicated prompt:
   - `prepared_remarks` → sentiment + guidance (management sets the tone here)
   - full text → risk extraction (analysts surface risks in Q&A)
3. If a section is still too long, `chunk()` splits on **speaker-turn boundaries**
   (`SpeakerName:` regex), keeping each chunk semantically coherent rather than
   hard-splitting mid-sentence.

This is a form of **map-reduce**: map analysis over meaningful segments, then
aggregate into one `TranscriptInsight`. It costs 3 API calls per transcript
instead of 1, but produces measurably better risk coverage because Q&A risks
are not diluted by the prepared remarks' length.

---

## 2. Why Pydantic models for LLM output?

LLM outputs are **probabilistic** — the model may add prose, change field names,
or produce subtly malformed structures. Pydantic provides three guarantees:

| Benefit | How it helps |
|---------|-------------|
| **Field validation** | `score: float = Field(ge=-1, le=1)` rejects hallucinated values like `2.5` |
| **Coercion** | `@field_validator("score", mode="before")` turns `"0.45"` (a string) into `0.45` |
| **Serialisation** | `model_dump_json()` / `model_validate_json()` round-trips cleanly through SQLite |
| **Test contracts** | Tests construct models directly with known values — no mocking of JSON dicts |

Without Pydantic, a single bad LLM response silently corrupts downstream
sentiment drift calculations. With it, bad output raises a `ValidationError`
that's caught at parse time, not buried in a chart.

---

## 3. Why SQLite over Postgres?

For a portfolio project running on one machine (or one HuggingFace Space):

| Factor | SQLite | Postgres |
|--------|--------|----------|
| Ops overhead | Zero — one `.db` file | Requires a running server |
| Dependencies | Built into Python stdlib | `psycopg2`, connection string, migrations |
| Demo portability | `git clone` → works instantly | Needs docker-compose or managed cloud DB |
| Scale limit | ~1 TB, plenty for thousands of transcripts | Needed at millions of rows |

The store is isolated behind `TranscriptStore` with a clean interface
(`save`, `get_by_ticker`, `get_stats`). Swapping to Postgres later requires
changing only the connection string and the `ON CONFLICT` clause — no
business logic changes.

---

## 4. Why XML prompting instead of JSON?

Three reasons:

**a) Nested structure reliability.**
JSON requires every bracket, comma, and quote to be correct.
XML is more forgiving: `<risks><risk><text>…</text></risk></risks>` degrades
gracefully even if the model adds a prose sentence before the tag.

**b) Regex extractability.**
The parser uses `re.search(r"<sentiment[\s\S]*?</sentiment>", text)` to
locate the relevant block even when the model wraps it in markdown fences
or adds a preamble. JSON parsing requires the entire response to be valid JSON.

**c) Prompt clarity.**
The XML schema in the prompt doubles as documentation that the model can
follow literally. Analysts reviewing prompts find `<severity>high</severity>`
more readable than `{"severity": "high"}` embedded in escaped JSON strings.

---

## 5. The EDGAR iXBRL bug

**What happened:** `data.sec.gov/submissions/CIK*.json` returns a `primaryDocument`
field per filing. For modern Apple 8-Ks this is `aapl-20231102.htm` — the
**iXBRL wrapper** (an inline XBRL metadata document used by EDGAR's data
pipeline). It weighs ~4 KB and contains almost no human-readable text, giving
only 1 keyword hit against the `_is_transcript` filter.

**The actual content** is in Exhibit EX-99.1 (`a8-kex991q4202309302023.htm`, ~170 KB),
linked from the filing's `-index.htm` page.

**Fix in `_resolve_doc_url`:**
```
1. Always fetch <accession>-index.htm
2. Parse table rows → find type == "EX-99.1" || "EX-99.2"
3. Return that URL; fall back to primary doc only if no exhibit found
```

**Lesson:** Never trust `primaryDocument` for content extraction.
The SEC data pipeline optimises for machine-readable XBRL, not human text.

---

## 6. The Gemini model-probing bug

**What happened:** The `google.generativeai` SDK was deprecated; the replacement
`google.genai` uses different model identifiers. `gemini-1.5-flash` (the originally
specified model) was absent from the project's quota allocation. `gemini-2.0-flash`
showed `limit: 0` on the free tier for this project.

**Root cause:** Google AI Studio projects have **per-model quota allocations**
that are not uniform across all projects. A freshly created project may have
`limit: 0` for some models until it meets usage thresholds.

**Fix in `TranscriptAnalyzer.__init__` / `_call_llm`:**
```python
# 1. Probe candidate models at startup
candidates = ["gemini-2.5-flash", "gemini-2.0-flash-lite",
              "gemini-flash-lite-latest", ...]
# 2. Use whichever responds without 429

# 3. In retry loop: parse retryDelay from error body
retry_hint = re.search(r"retry in (\d+)", exc_str)
wait = int(retry_hint.group(1)) if retry_hint else 2 ** attempt
wait = min(wait, 65)          # cap per-attempt wait
```

**Lesson:** Never hardcode a specific model in a free-tier application.
Always add a probe-and-fallback layer and respect the API's own backoff hints.

---

## Data flow diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User action                          │
│            (Streamlit button or curl /fetch/AAPL)           │
└───────────────────────────┬─────────────────────────────────┘
                            │ POST /fetch/{ticker}
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  TranscriptFetcher                                          │
│  1. GET company_tickers.json       → CIK                    │
│  2. GET submissions/CIK*.json      → list of 8-K accessions │
│  3. GET {acc}-index.htm            → find EX-99.1 URL       │
│  4. GET EX-99.1 document           → HTML                   │
│  5. BeautifulSoup → plain text                              │
└───────────────────────────┬─────────────────────────────────┘
                            │ raw transcript dict
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  TranscriptStore.save()                                     │
│  MD5(ticker + date + text[:200]) → content_hash dedup       │
│  INSERT INTO transcripts (SQLite)                           │
└───────────────────────────┬─────────────────────────────────┘
                            │ POST /analyze/{ticker}
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  TranscriptCleaner                                          │
│  clean() → chunk() → detect_sections()                     │
│  ↓ prepared_remarks    ↓ full text                          │
│  sentiment + guidance  risks                                │
└───────────────────────────┬─────────────────────────────────┘
                            │ 3 × Gemini API calls
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  TranscriptAnalyzer._call_llm()                             │
│  XML response → _parse_*_xml() → Pydantic models           │
│  → TranscriptInsight                                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  TranscriptStore.save_insight()                             │
│  INSERT INTO insights (upsert on ticker+quarter+year)       │
└───────────────────────────┬─────────────────────────────────┘
                            │ GET /trend/{ticker}
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  QuarterComparator.compare()                                │
│  drift = [score[i+1] - score[i]]                            │
│  std(drift) > 0.2 → volatile                                │
│  mean(drift) > 0.1 → improving                              │
│  fuzzy word-overlap → new/dropped risks                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ QuarterTrend JSON
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Trend Analysis page                              │
│  Plotly line chart + drift table + Gemini narrative         │
└─────────────────────────────────────────────────────────────┘
```
