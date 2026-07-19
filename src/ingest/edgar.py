"""
SEC EDGAR transcript fetcher — Week 5 audit.

Supports ANY US ticker reliably up to the current quarter:
  - CIK lookup: EDGAR atom feed (primary) → company_tickers.json (fallback)
  - Full pagination: recent filings + all historical batches via filings.files
  - Exhibit priority: EX-99.1 > EX-99.2 > any .htm > primary doc
  - Filter: raw_text > 2000 chars; filing_date <= today
  - Returns tuple[list[dict], fetch_status_dict]
  - get_supported_info(ticker) for live ticker validation
"""
from __future__ import annotations

import re
import time
import logging
from datetime import datetime
from typing import Optional

import warnings
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

HEADERS         = {"User-Agent": "FinSight/1.0 research@finsight.dev"}
SLEEP_BETWEEN   = 0.5
MAX_RETRIES     = 3
MIN_TEXT_CHARS  = 2000


class TranscriptFetcher:
    """Fetch earnings call transcripts from SEC EDGAR 8-K filings."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_by_ticker(
        self, ticker: str, start_year: int, end_year: int
    ) -> tuple[list[dict], dict]:
        """
        Fetch earnings transcripts for *ticker* covering start_year..end_year.

        Returns
        -------
        (results, fetch_status)
            results      – list of transcript dicts ready to persist
            fetch_status – metrics dict with keys:
                           ticker, cik, total_filings_checked,
                           years_searched, current_quarter, errors
        """
        ticker  = ticker.upper()
        now     = datetime.now()
        cq      = (now.month - 1) // 3 + 1

        fetch_status: dict = {
            "ticker":                ticker,
            "cik":                   "",
            "total_filings_checked": 0,
            "years_searched":        f"{start_year}–{end_year}",
            "current_quarter":       f"Q{cq} {now.year}",
            "errors":                [],
        }

        cik, _company_name = self._get_cik(ticker)
        if not cik:
            msg = f"CIK not found for {ticker}"
            logger.warning(msg)
            fetch_status["errors"].append(msg)
            return [], fetch_status

        fetch_status["cik"] = cik

        filings = self._get_8k_filings(cik, start_year, end_year)
        fetch_status["total_filings_checked"] = len(filings)

        results: list[dict] = []
        for filing in filings:
            try:
                doc = self._fetch_filing_document(filing, ticker, cik)
                if doc and len(doc.get("raw_text", "")) >= MIN_TEXT_CHARS:
                    results.append(doc)
            except Exception as exc:
                msg = f"Filing {filing.get('accession_no', '?')}: {exc}"
                logger.warning(msg)
                fetch_status["errors"].append(msg)

        return results, fetch_status

    def get_supported_info(self, ticker: str) -> dict:
        """
        Check whether a ticker exists on EDGAR.

        Returns
        -------
        {
            "exists":       bool,
            "company_name": str,
            "cik":          str,
            "total_8k_count": int,
        }
        """
        ticker = ticker.upper()
        cik, company_name = self._get_cik(ticker)
        if not cik:
            return {"exists": False, "company_name": "", "cik": "", "total_8k_count": 0}

        # Count 8-Ks in the recent batch (fast approximation)
        padded  = cik.zfill(10)
        url     = f"https://data.sec.gov/submissions/CIK{padded}.json"
        resp    = self._get(url)
        count   = 0
        if resp is not None:
            try:
                data  = resp.json()
                forms = data.get("filings", {}).get("recent", {}).get("form", [])
                count = sum(1 for f in forms if f == "8-K")
            except Exception:
                pass

        return {
            "exists":         True,
            "company_name":   company_name or "",
            "cik":            cik,
            "total_8k_count": count,
        }

    # ------------------------------------------------------------------
    # CIK lookup
    # ------------------------------------------------------------------

    def _get_cik(self, ticker: str) -> tuple[Optional[str], Optional[str]]:
        """
        Return (cik_no_leading_zeros, company_name).

        Strategy:
          1. EDGAR atom feed (primary) — also returns company name
          2. company_tickers.json    (fallback)
        """
        # 1 — atom feed -------------------------------------------------------
        atom_url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?company=&CIK={ticker}&type=8-K&action=getcompany&output=atom"
        )
        resp = self._get(atom_url)
        if resp is not None:
            cik, name = self._parse_cik_from_atom(resp.text)
            if cik:
                return cik, name

        # 2 — company_tickers.json fallback -----------------------------------
        url  = "https://www.sec.gov/files/company_tickers.json"
        resp = self._get(url)
        if resp is not None:
            try:
                data = resp.json()
                for entry in data.values():
                    if entry.get("ticker", "").upper() == ticker:
                        return str(entry["cik_str"]), entry.get("title", "")
            except Exception as exc:
                logger.error("company_tickers.json parse error: %s", exc)

        return None, None

    @staticmethod
    def _parse_cik_from_atom(xml_text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract CIK and company name from an EDGAR atom-feed response.

        The feed contains one or more <entry> elements; each has the CIK
        embedded in link hrefs and (sometimes) in namespace-prefixed tags.
        We use regex for robustness against namespace variations.
        """
        cik: Optional[str]  = None
        name: Optional[str] = None

        # CIK in a <*:cik> tag with possible leading zeros
        m = re.search(r"<(?:[A-Za-z]+:)?cik>\s*0*(\d+)\s*</", xml_text, re.IGNORECASE)
        if m:
            cik = m.group(1)

        # CIK embedded in archive URLs: /Archives/edgar/data/320193/
        if not cik:
            m = re.search(r"/Archives/edgar/data/(\d+)/", xml_text)
            if m:
                cik = m.group(1)

        # CIK in query-string links: action=getcompany&CIK=0000320193&
        if not cik:
            m = re.search(r"[?&]CIK=0*(\d+)&", xml_text, re.IGNORECASE)
            if m:
                cik = m.group(1)

        # Company name from <conformed-name> (EDGAR atom feed) or generic company-name tag
        for pat in (
            r"<conformed-name>\s*([^<]+?)\s*</",
            r"<(?:[A-Za-z]+:)?company-name>\s*([^<]+?)\s*</",
        ):
            m = re.search(pat, xml_text, re.IGNORECASE)
            if m:
                name = m.group(1)
                break

        return cik, name

    # ------------------------------------------------------------------
    # Filing list via EDGAR submissions API (with full pagination)
    # ------------------------------------------------------------------

    def _get_8k_filings(
        self, cik: str, start_year: int, end_year: int
    ) -> list[dict]:
        """
        Return 8-K filing metadata for *cik* within start_year..end_year.

        Fetches the primary submissions JSON and ALL historical batch files
        listed under filings.files so we never miss older filings.
        Results are capped at today's date (no future filings).
        """
        padded = cik.zfill(10)
        url    = f"https://data.sec.gov/submissions/CIK{padded}.json"
        resp   = self._get(url)
        if resp is None:
            return []

        try:
            data = resp.json()
        except Exception as exc:
            logger.error("Submissions JSON parse error for CIK %s: %s", cik, exc)
            return []

        filings: list[dict] = []

        # Recent batch (always present)
        recent = data.get("filings", {}).get("recent", {})
        filings.extend(self._extract_filings_from_batch(recent, start_year, end_year))

        # Historical batches (older filings, each is a separate JSON file)
        for file_entry in data.get("filings", {}).get("files", []):
            batch_name = file_entry.get("name", "")
            if not batch_name:
                continue
            batch_url  = f"https://data.sec.gov/submissions/{batch_name}"
            batch_resp = self._get(batch_url)
            if batch_resp is None:
                continue
            try:
                batch_data = batch_resp.json()
                filings.extend(
                    self._extract_filings_from_batch(batch_data, start_year, end_year)
                )
            except Exception as exc:
                logger.warning("Batch %s parse error: %s", batch_name, exc)

        # Drop filings filed in the future (shouldn't happen, but guard anyway)
        today_str = datetime.now().strftime("%Y-%m-%d")
        filings   = [f for f in filings if f["filed_date"] <= today_str]

        logger.info(
            "Found %d 8-K filings for CIK %s (%d–%d)",
            len(filings), cik, start_year, end_year,
        )
        return filings

    @staticmethod
    def _extract_filings_from_batch(
        batch: dict, start_year: int, end_year: int
    ) -> list[dict]:
        """Extract 8-K metadata rows from a submissions batch dict."""
        forms        = batch.get("form",             [])
        filing_dates = batch.get("filingDate",       [])
        accessions   = batch.get("accessionNumber",  [])
        primary_docs = batch.get("primaryDocument",  [])

        filings = []
        for form, date, acc, primary in zip(forms, filing_dates, accessions, primary_docs):
            if form != "8-K":
                continue
            try:
                year = int(str(date)[:4])
            except (ValueError, TypeError):
                continue
            if not (start_year <= year <= end_year):
                continue
            filings.append(
                {
                    "form_type":   form,
                    "filed_date":  date,
                    "accession_no": acc,        # e.g. "0000320193-23-000077"
                    "primary_doc": primary,
                }
            )
        return filings

    # ------------------------------------------------------------------
    # Document fetch
    # ------------------------------------------------------------------

    def _fetch_filing_document(
        self, filing: dict, ticker: str, cik: str
    ) -> Optional[dict]:
        acc_dashes  = filing["accession_no"]
        acc_nodash  = acc_dashes.replace("-", "")
        filed_date  = filing["filed_date"]
        form_type   = filing["form_type"]
        primary_doc = filing.get("primary_doc", "")

        base_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"
        )

        doc_url = self._resolve_doc_url(base_url, acc_dashes, primary_doc)
        if not doc_url:
            return None

        time.sleep(SLEEP_BETWEEN)
        doc_resp = self._get(doc_url)
        if doc_resp is None:
            return None

        raw_text = self._extract_text(doc_resp.text)

        if not self._is_transcript(raw_text):
            return None

        year, quarter = self._parse_quarter(filed_date, raw_text)

        return {
            "ticker":    ticker,
            "cik":       cik,
            "form_type": form_type,
            "filed_date": filed_date,
            "quarter":   quarter,
            "year":      year,
            "raw_text":  raw_text,
            "url":       doc_url,
        }

    def _resolve_doc_url(
        self, base_url: str, acc_dashes: str, primary_doc: str
    ) -> Optional[str]:
        """
        Return the best document URL for this filing.

        Always scans the index first; exhibit priority:
          EX-99.1 > EX-99.2 > any content .htm > primary_doc
        """
        index_url = f"{base_url}/{acc_dashes}-index.htm"
        time.sleep(SLEEP_BETWEEN)
        resp = self._get(index_url)
        if resp is not None:
            doc_url = self._pick_doc_from_htm_index(resp.text, base_url)
            if doc_url:
                return doc_url

        # Fall back to primary document
        if primary_doc and re.search(r"\.(htm|html|txt)$", primary_doc, re.IGNORECASE):
            return f"{base_url}/{primary_doc}"

        return None

    def _pick_doc_from_htm_index(self, html: str, base_url: str) -> Optional[str]:
        """
        Parse a filing index page and return the highest-priority document URL.

        Priority:
          1. EX-99.1
          2. EX-99.2
          3. First non-XBRL .htm/.html/.txt document
        """
        soup = BeautifulSoup(html, "lxml")

        ex991_url: Optional[str] = None
        ex992_url: Optional[str] = None
        fallback_url: Optional[str] = None

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            doc_type = cells[1].get_text(strip=True).upper()
            link     = row.find("a", href=True)
            if not link:
                continue
            href = link["href"]
            name = href.split("/")[-1].lower()

            # Only consider textual documents
            if not re.search(r"\.(htm|html|txt)$", name):
                continue

            full_url = (
                href if href.startswith("http") else f"https://www.sec.gov{href}"
            )

            # Skip inline XBRL wrappers (tiny metadata files)
            if doc_type in ("", "XML", "XBRL", "EX-101.INS", "EX-101.SCH"):
                continue

            if doc_type == "EX-99.1" and ex991_url is None:
                ex991_url = full_url
            elif doc_type in ("EX-99.2", "EX-99") and ex992_url is None:
                ex992_url = full_url
            elif fallback_url is None:
                fallback_url = full_url

        return ex991_url or ex992_url or fallback_url

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _extract_text(self, html: str) -> str:
        """Strip HTML/XML tags and return plain text."""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        text  = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)

    def _is_transcript(self, text: str) -> bool:
        """
        Heuristic: does this text look like earnings-related content?

        Accepts both full call transcripts and earnings press releases (EX-99.1).
        """
        lower = text.lower()
        core       = ["earnings", "revenue", "quarter", "fiscal"]
        transcript = ["operator", "conference call", "q&a", "analyst", "participants"]
        core_hits       = sum(1 for kw in core       if kw in lower)
        transcript_hits = sum(1 for kw in transcript if kw in lower)
        return core_hits >= 2 or (core_hits >= 1 and transcript_hits >= 1)

    @staticmethod
    def _parse_quarter(filed_date: str, text: str) -> tuple[int, int]:
        """Derive (year, quarter) from filing date and/or document text."""
        try:
            dt    = datetime.strptime(filed_date[:10], "%Y-%m-%d")
            year  = dt.year
            month = dt.month
        except Exception:
            return 0, 0

        # Prefer explicit mention in text: "Q3 fiscal 2023" or "Q3 2023"
        m = re.search(r"[Qq](\d)\s*(?:fiscal)?\s*(\d{4})", text)
        if m:
            return int(m.group(2)), int(m.group(1))

        quarter = (month - 1) // 3 + 1
        return year, quarter

    # ------------------------------------------------------------------
    # HTTP helper with retry + exponential back-off
    # ------------------------------------------------------------------

    def _get(self, url: str) -> Optional[requests.Response]:
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
                time.sleep(SLEEP_BETWEEN)
                return resp
            except requests.RequestException as exc:
                wait = 2 ** attempt
                logger.warning(
                    "HTTP error (%s) for %s — retry in %ds", exc, url, wait
                )
                time.sleep(wait)
        logger.error("All retries exhausted for %s", url)
        return None
