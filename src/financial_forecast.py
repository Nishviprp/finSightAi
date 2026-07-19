"""
90-day price forecast for a single ticker, built on ARIMA(1,1,1) over 2 years
of daily closes fetched from yfinance.

Honesty note: this is a statistical trend/uncertainty-band projection, not a
trading signal. ARIMA on a near-random-walk price series converges toward a
flat drift line at longer horizons, and its backtested RMSE over a ~20%
held-out window is typically a high-single- to double-digit percentage of
price for a volatile stock — not a few percent. `rmse` is reported exactly as
computed; nothing here is tuned to look more accurate than the model is.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)

ARIMA_ORDER = (1, 1, 1)
HISTORY_PERIOD = "2y"
TEST_SPLIT_FRACTION = 0.2
CONFIDENCE_LEVEL = 0.95
MIN_HISTORY_POINTS = 60


@st.cache_data(ttl=86400, show_spinner=False)
def forecast_stock_price(symbol: str, days_forward: int = 90) -> Optional[dict]:
    """
    Fit ARIMA(1,1,1) on 2 years of *symbol* daily closes and forecast
    *days_forward* trading days ahead with a confidence band.

    Returns a dict:
        historical_prices – DataFrame, column "Close", indexed by date
                             (last 2 years of daily closes)
        forecast_prices   – DataFrame, columns "mean"/"lower"/"upper",
                             indexed by future business-day date
        model_type        – "ARIMA"
        rmse               – float; RMSE of a backtest that fits on the first
                             80% of history and forecasts the held-out 20%
        confidence_level   – float, 0.95

    Returns None on any failure (invalid symbol, network error, insufficient
    history, model fit failure) — never raises.
    """
    try:
        close = _fetch_close_history(symbol)
        if close is None or len(close) < MIN_HISTORY_POINTS:
            logger.warning("Not enough price history for %s to forecast", symbol)
            return None

        rmse = _backtest_rmse(close)
        if rmse is None:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(close.values, order=ARIMA_ORDER).fit()
            summary = fit.get_forecast(steps=days_forward).summary_frame(
                alpha=1 - CONFIDENCE_LEVEL
            )

        future_dates = pd.bdate_range(
            start=close.index[-1] + pd.Timedelta(days=1), periods=days_forward
        )
        forecast_df = pd.DataFrame(
            {
                "mean": summary["mean"].to_numpy(),
                "lower": summary["mean_ci_lower"].to_numpy(),
                "upper": summary["mean_ci_upper"].to_numpy(),
            },
            index=future_dates,
        )

        return {
            "historical_prices": close.to_frame(name="Close"),
            "forecast_prices": forecast_df,
            "model_type": "ARIMA",
            "rmse": rmse,
            "confidence_level": CONFIDENCE_LEVEL,
        }
    except Exception as exc:
        logger.warning("forecast_stock_price failed for %s: %s", symbol, exc)
        return None


def _fetch_close_history(symbol: str) -> Optional[pd.Series]:
    """Last 2 years of daily closes for *symbol*, tz-naive, ascending by date."""
    hist = yf.Ticker(symbol).history(period=HISTORY_PERIOD, interval="1d")
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna()
    if close.empty:
        return None

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Date"
    return close.sort_index()


def _backtest_rmse(close: pd.Series) -> Optional[float]:
    """Fit ARIMA on the first 80% of *close*, forecast the held-out last 20%, return RMSE."""
    n_test = max(1, int(len(close) * TEST_SPLIT_FRACTION))
    train, test = close.iloc[:-n_test], close.iloc[-n_test:]
    if len(train) < 10 or len(test) == 0:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(train.values, order=ARIMA_ORDER).fit()
            predicted = fit.forecast(steps=len(test))
        return float(np.sqrt(np.mean((predicted - test.values) ** 2)))
    except Exception as exc:
        logger.warning("ARIMA backtest failed: %s", exc)
        return None
