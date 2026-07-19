"""
Financial metrics for a single ticker, sourced entirely from yfinance (free tier).

Every public function is cached with `@st.cache_data(ttl=3600)` — financial
statements and profile data don't change intraday — and never raises: on any
failure (invalid symbol, network error, missing fields) it logs a warning and
returns None instead of propagating the exception.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Company profile
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_profile(symbol: str) -> Optional[dict]:
    """
    Fetch company profile info for *symbol*.

    Returns a dict with keys: symbol, name, sector, industry, website,
    founded, founders, ceo, employee_count, business_summary, city, state,
    country. `founded` and `founders` are always None — yfinance does not
    expose founding date or founder names for any ticker.

    Returns None if the symbol doesn't resolve or the request fails.
    """
    try:
        info = yf.Ticker(symbol).info
        if not info or not (info.get("longName") or info.get("shortName")):
            logger.warning("No company profile found for %s", symbol)
            return None

        ceo = None
        for officer in info.get("companyOfficers") or []:
            if "ceo" in (officer.get("title") or "").lower():
                ceo = officer.get("name")
                break

        return {
            "symbol": symbol.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "founded": None,
            "founders": None,
            "ceo": ceo,
            "employee_count": info.get("fullTimeEmployees"),
            "business_summary": info.get("longBusinessSummary"),
            "city": info.get("city"),
            "state": info.get("state"),
            "country": info.get("country"),
        }
    except Exception as exc:
        logger.warning("get_company_profile failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Financial statements
# ---------------------------------------------------------------------------

def _row(df: pd.DataFrame, label: str) -> pd.Series:
    """Return *df* row *label*, or a NaN-filled Series if the row is absent."""
    if label in df.index:
        return df.loc[label]
    return pd.Series([float("nan")] * len(df.columns), index=df.columns)


def _year_columns(df: pd.DataFrame) -> list[str]:
    return [str(c.year) if hasattr(c, "year") else str(c) for c in df.columns]


@st.cache_data(ttl=3600, show_spinner=False)
def get_income_statement(symbol: str) -> Optional[pd.DataFrame]:
    """
    Annual income statement for *symbol*, last 5 fiscal years (most recent first).

    Rows: Revenue, Operating Income, Net Income, Gross Profit Margin %,
    Operating Margin %, Net Profit Margin %. Columns: fiscal year.

    Returns None if no income statement data is available.
    """
    try:
        raw = yf.Ticker(symbol).income_stmt
        if raw is None or raw.empty:
            logger.warning("No income statement data for %s", symbol)
            return None

        raw = raw.iloc[:, :5]
        revenue = _row(raw, "Total Revenue")
        operating_income = _row(raw, "Operating Income")
        net_income = _row(raw, "Net Income")
        gross_profit = _row(raw, "Gross Profit")

        df = pd.DataFrame(
            {
                "Revenue": revenue,
                "Operating Income": operating_income,
                "Net Income": net_income,
                "Gross Profit Margin %": (gross_profit / revenue * 100).round(2),
                "Operating Margin %": (operating_income / revenue * 100).round(2),
                "Net Profit Margin %": (net_income / revenue * 100).round(2),
            }
        ).T
        df.columns = _year_columns(raw)
        return df
    except Exception as exc:
        logger.warning("get_income_statement failed for %s: %s", symbol, exc)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_balance_sheet(symbol: str) -> Optional[pd.DataFrame]:
    """
    Annual balance sheet for *symbol*, last 5 fiscal years (most recent first).

    Rows: Total Assets, Total Liabilities, Shareholders Equity, Current Ratio,
    Debt-to-Equity Ratio. Columns: fiscal year.

    Returns None if no balance sheet data is available.
    """
    try:
        raw = yf.Ticker(symbol).balance_sheet
        if raw is None or raw.empty:
            logger.warning("No balance sheet data for %s", symbol)
            return None

        raw = raw.iloc[:, :5]
        total_assets = _row(raw, "Total Assets")
        total_liabilities = _row(raw, "Total Liabilities Net Minority Interest")
        equity = _row(raw, "Stockholders Equity")
        current_assets = _row(raw, "Current Assets")
        current_liabilities = _row(raw, "Current Liabilities")
        total_debt = _row(raw, "Total Debt")

        df = pd.DataFrame(
            {
                "Total Assets": total_assets,
                "Total Liabilities": total_liabilities,
                "Shareholders Equity": equity,
                "Current Ratio": (current_assets / current_liabilities).round(2),
                "Debt-to-Equity Ratio": (total_debt / equity).round(2),
            }
        ).T
        df.columns = _year_columns(raw)
        return df
    except Exception as exc:
        logger.warning("get_balance_sheet failed for %s: %s", symbol, exc)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_cash_flow(symbol: str) -> Optional[pd.DataFrame]:
    """
    Annual cash flow statement for *symbol*, last 5 fiscal years (most recent first).

    Rows: Operating Cash Flow, Free Cash Flow, Capital Expenditures.
    Columns: fiscal year.

    Returns None if no cash flow data is available.
    """
    try:
        raw = yf.Ticker(symbol).cashflow
        if raw is None or raw.empty:
            logger.warning("No cash flow data for %s", symbol)
            return None

        raw = raw.iloc[:, :5]
        df = pd.DataFrame(
            {
                "Operating Cash Flow": _row(raw, "Operating Cash Flow"),
                "Free Cash Flow": _row(raw, "Free Cash Flow"),
                "Capital Expenditures": _row(raw, "Capital Expenditure"),
            }
        ).T
        df.columns = _year_columns(raw)
        return df
    except Exception as exc:
        logger.warning("get_cash_flow failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Earnings history
# ---------------------------------------------------------------------------

def _to_float(value) -> Optional[float]:
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_history(symbol: str) -> Optional[list[dict]]:
    """
    Last 8 reported quarterly earnings for *symbol*, most recent first.

    Each entry: {date, eps_estimate, eps_actual, surprise_percent}.
    Future/unreported quarters (no actual EPS yet) are excluded.

    Returns None if no earnings history is available.
    """
    try:
        dates = yf.Ticker(symbol).get_earnings_dates(limit=16)
        if dates is None or dates.empty:
            logger.warning("No earnings history for %s", symbol)
            return None

        reported = dates.dropna(subset=["Reported EPS"]).sort_index(ascending=False).head(8)
        return [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "eps_estimate": _to_float(row.get("EPS Estimate")),
                "eps_actual": _to_float(row.get("Reported EPS")),
                "surprise_percent": _to_float(row.get("Surprise(%)")),
            }
            for idx, row in reported.iterrows()
        ]
    except Exception as exc:
        logger.warning("get_earnings_history failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Analyst ratings
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_analyst_ratings(symbol: str) -> Optional[dict]:
    """
    Analyst coverage summary for *symbol*.

    Returns a dict: average_rating (1=Strong Buy .. 5=Strong Sell), rating_key,
    number_of_analysts, buy_count, hold_count, sell_count, price_target_current,
    price_target_mean, price_target_high, price_target_low.

    Returns None if the symbol doesn't resolve.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or not (info.get("longName") or info.get("shortName")):
            logger.warning("No analyst ratings found for %s", symbol)
            return None

        buy_count = hold_count = sell_count = 0
        try:
            rec = ticker.recommendations
            if rec is not None and not rec.empty:
                latest = rec.iloc[0]
                buy_count = int(latest.get("strongBuy", 0)) + int(latest.get("buy", 0))
                hold_count = int(latest.get("hold", 0))
                sell_count = int(latest.get("strongSell", 0)) + int(latest.get("sell", 0))
        except Exception as exc:
            logger.warning("recommendations lookup failed for %s: %s", symbol, exc)

        price_targets: dict = {}
        try:
            price_targets = ticker.analyst_price_targets or {}
        except Exception as exc:
            logger.warning("price target lookup failed for %s: %s", symbol, exc)

        return {
            "average_rating": info.get("recommendationMean"),
            "rating_key": info.get("recommendationKey"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
            "buy_count": buy_count,
            "hold_count": hold_count,
            "sell_count": sell_count,
            "price_target_current": price_targets.get("current"),
            "price_target_mean": price_targets.get("mean"),
            "price_target_high": price_targets.get("high"),
            "price_target_low": price_targets.get("low"),
        }
    except Exception as exc:
        logger.warning("get_analyst_ratings failed for %s: %s", symbol, exc)
        return None
