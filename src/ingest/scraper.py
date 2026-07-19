"""
Fallback scraper for earnings call transcripts from Seeking Alpha public pages.
Used when EDGAR doesn't have the full transcript text.
"""
import re
import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Patterns that indicate boilerplate / operator text to strip
_BOILERPLATE_PATTERNS = [
    r"this transcript is produced by.*?\n",
    r"seeking alpha.*?transcript",
    r"all rights reserved.*?\n",
    r"questions and answers.*?session",
    r"\[operator instructions\]",
    r"please stand by.*?\n",
    r"your conference is now being recorded",
    r"thank you for standing by",
    r"ladies and gentlemen.*?welcome",
    r"this concludes today.*?conference",
    r"you may now disconnect",
]
_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.DOTALL
)

# Operator / intro lines
_OPERATOR_LINE_RE = re.compile(
    r"^(operator|moderator|coordinator)[:\s].*$", re.IGNORECASE | re.MULTILINE
)


class FallbackScraper:
    """Scrape earnings call transcripts from Seeking Alpha public listing pages."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_seeking_alpha_style(self, ticker: str) -> list[dict]:
        """
        Scrape transcript listings for *ticker* from Seeking Alpha.
        Returns list of dicts with the standard transcript schema.

        Note: Seeking Alpha may block scraping or require login for full text.
        This implementation scrapes the public listing page and attempts to
        extract available transcript previews / summaries.
        """
        ticker = ticker.upper()
        url = f"https://seekingalpha.com/symbol/{ticker}/earnings/transcripts"
        resp = self._get(url)
        if resp is None:
            logger.warning("Could not reach Seeking Alpha for %s", ticker)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        links = self._find_transcript_links(soup, ticker)
        results = []

        for link_info in links[:5]:  # cap at 5 to be polite
            time.sleep(1.0)
            doc = self._fetch_transcript_page(link_info, ticker)
            if doc:
                results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_transcript_links(
        self, soup: BeautifulSoup, ticker: str
    ) -> list[dict]:
        """Extract transcript article links from the listing page."""
        links = []
        base = "https://seekingalpha.com"
        for a in soup.find_all("a", href=True):
            href = a["href"]
            title = a.get_text(strip=True)
            if "earnings-call-transcript" in href or (
                "transcript" in href.lower() and ticker.lower() in href.lower()
            ):
                full_url = href if href.startswith("http") else base + href
                links.append({"url": full_url, "title": title})
        return links

    def _fetch_transcript_page(
        self, link_info: dict, ticker: str
    ) -> Optional[dict]:
        url = link_info["url"]
        title = link_info.get("title", "")
        resp = self._get(url)
        if resp is None:
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        raw_text = self._extract_article_text(soup)
        if not raw_text or len(raw_text) < 200:
            return None

        cleaned = self._strip_boilerplate(raw_text)
        filed_date, year, quarter = self._parse_meta(soup, title)

        return {
            "ticker": ticker,
            "cik": None,
            "form_type": "SA_TRANSCRIPT",
            "filed_date": filed_date,
            "quarter": quarter,
            "year": year,
            "raw_text": cleaned,
            "url": url,
        }

    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        """Pull article body text from a Seeking Alpha article page."""
        # Try common article container selectors
        for selector in [
            "div[data-test-id='article-body']",
            "div.paywall-full-content",
            "article",
            "div#main-content",
            "div.content-body",
        ]:
            container = soup.select_one(selector)
            if container:
                return container.get_text(separator="\n")

        # Fallback: largest <div> by text length
        divs = soup.find_all("div")
        if divs:
            biggest = max(divs, key=lambda d: len(d.get_text()))
            return biggest.get_text(separator="\n")

        return soup.get_text(separator="\n")

    def _strip_boilerplate(self, text: str) -> str:
        """Remove legal boilerplate and operator remarks."""
        text = _BOILERPLATE_RE.sub("", text)
        text = _OPERATOR_LINE_RE.sub("", text)
        # Collapse blank lines
        lines = [l.strip() for l in text.splitlines()]
        deduped = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank:
                    deduped.append("")
                prev_blank = True
            else:
                deduped.append(line)
                prev_blank = False
        return "\n".join(deduped).strip()

    def _parse_meta(
        self, soup: BeautifulSoup, title: str
    ) -> tuple[str, int, int]:
        """Extract date, year, quarter from page meta or title."""
        filed_date = ""
        year = 0
        quarter = 0

        # Try meta publish date
        for meta in soup.find_all("meta"):
            prop = meta.get("property", "") or meta.get("name", "")
            if "publish" in prop.lower() or "date" in prop.lower():
                content = meta.get("content", "")
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})", content)
                if m:
                    filed_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    year = int(m.group(1))
                    month = int(m.group(2))
                    quarter = (month - 1) // 3 + 1
                    break

        # Try to parse quarter from title like "Q3 2023 Earnings"
        m = re.search(r"[Qq](\d)\s+(\d{4})", title)
        if m:
            quarter = int(m.group(1))
            year = int(m.group(2))

        return filed_date, year, quarter

    def _get(self, url: str) -> Optional[requests.Response]:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning("Scrape failed: %s — %s", url, exc)
            return None
