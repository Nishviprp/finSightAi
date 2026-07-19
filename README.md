# FinSight

**AI-powered earnings call and SEC filing analyzer** — turns a stock ticker into a structured, human-readable breakdown of sentiment, risk factors, forward guidance, and quarter-over-quarter trends in under 30 seconds.

## What it does

FinSight pulls a company's real SEC filings directly from EDGAR, runs them through an LLM pipeline, and surfaces what an equity analyst would normally spend hours digging for:

- Sentiment score for the filing/earnings period
- Key risk factors disclosed by the company
- Forward guidance and management commentary
- Quarter-over-quarter trend comparison with drift detection
- A human-readable narrative summary, not just raw scores

## How it works

1. **Ingestion** — Fetches filings from the SEC EDGAR API. Filings are indexed as metadata XML rather than raw text, so the ingestion layer reverse-engineers the filing index to locate and extract the actual earnings content (EX-99.1 exhibits), bypassing iXBRL wrapper formats.
2. **Processing** — Cleans and normalizes extracted filing text for LLM consumption.
3. **Analysis** — Sends processed text to an LLM (Gemini) for sentiment scoring, risk extraction, and guidance summarization.
4. **Comparison** — Compares current quarter results against prior quarters to surface meaningful trend shifts.
5. **Presentation** — Serves results through a FastAPI backend and a 3-page Streamlit dashboard.

## Tech stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **LLM:** Google Gemini
- **Data source:** SEC EDGAR API
- **Caching:** SQLite
- **Testing:** 82 automated tests (fully mocked — no live API calls in the test suite)
- **Deployment:** Hugging Face Spaces, with a self-contained entry point and baked-in demo data for instant exploration

## Try it

- **Live app:** https://finsightaiapp.streamlit.app/

## Development notes

This project was built solo over several weeks. Development was AI-assisted using Claude Code, which accelerated implementation — architecture decisions, prompt design for the LLM analysis pipeline, the EDGAR filing reverse-engineering approach, and evaluation/testing strategy were directed and reviewed by me throughout.

## License

MIT
