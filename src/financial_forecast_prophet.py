"""
90-day price forecast for a single ticker, built on Facebook Prophet over 2
years of daily closes fetched from yfinance.

Unlike ARIMA(1,1,1) (src/financial_forecast.py), which treats a stock price
as close to a random walk and converges to a flat drift line at longer
horizons, Prophet decomposes the series into trend + weekly seasonality and
extrapolates the trend forward — so its 90-day forecast actually slopes
rather than flattening. That's a genuine difference in what the model
assumes, not evidence either one is "more right": trend-following is exactly
as unfalsifiable a bet on future stock prices as a random walk is. Nothing
here is investment advice.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from prophet import Prophet

logger = logging.getLogger(__name__)

# Prophet/cmdstanpy are chatty at INFO level ("Chain [1] start processing" on
# every fit) — quiet them down to match this project's logging conventions.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

HISTORY_PERIOD = "2y"
TEST_SPLIT_FRACTION = 0.2
CONFIDENCE_LEVEL = 0.95
MIN_HISTORY_POINTS = 60


@st.cache_data(ttl=86400, show_spinner=False)
def forecast_stock_price_prophet(symbol: str, days_forward: int = 90) -> Optional[dict]:
    """
    Fit Prophet on 2 years of *symbol* daily closes and forecast
    *days_forward* trading days ahead with a confidence band.

    Returns a dict:
        historical_prices – DataFrame, column "Close", indexed by date
                             (last 2 years of daily closes)
        forecast_prices   – DataFrame, columns "mean"/"lower"/"upper",
                             indexed by future business-day date
        model_type        – "Prophet"
        mape               – float; mean absolute percentage error (0-100)
                             of a backtest that fits on the first 80% of
                             history and forecasts the held-out 20%
        confidence_level   – float, 0.95

    Returns None on any failure (invalid symbol, network error, insufficient
    history, model fit failure) — never raises.
    """
    try:
        close = _fetch_close_history(symbol)
        if close is None or len(close) < MIN_HISTORY_POINTS:
            logger.warning("Not enough price history for %s to forecast", symbol)
            return None

        mape = _backtest_mape(close)
        if mape is None:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(interval_width=CONFIDENCE_LEVEL)
            model.fit(pd.DataFrame({"ds": close.index, "y": close.to_numpy()}))
            future = model.make_future_dataframe(periods=days_forward, freq="B")
            forecast = model.predict(future)

        future_only = forecast[forecast["ds"] > close.index[-1]].head(days_forward)
        forecast_df = pd.DataFrame(
            {
                "mean": future_only["yhat"].to_numpy(),
                "lower": future_only["yhat_lower"].to_numpy(),
                "upper": future_only["yhat_upper"].to_numpy(),
            },
            index=pd.DatetimeIndex(future_only["ds"]),
        )

        return {
            "historical_prices": close.to_frame(name="Close"),
            "forecast_prices": forecast_df,
            "model_type": "Prophet",
            "mape": mape,
            "confidence_level": CONFIDENCE_LEVEL,
        }
    except Exception as exc:
        logger.warning("forecast_stock_price_prophet failed for %s: %s", symbol, exc)
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


def _backtest_mape(close: pd.Series) -> Optional[float]:
    """Fit Prophet on the first 80% of *close*, forecast the held-out last
    20%, return the mean absolute percentage error (0-100) against actual.
    """
    n_test = max(1, int(len(close) * TEST_SPLIT_FRACTION))
    train, test = close.iloc[:-n_test], close.iloc[-n_test:]
    if len(train) < 10 or len(test) == 0:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(interval_width=CONFIDENCE_LEVEL)
            model.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))
            future = model.make_future_dataframe(periods=len(test), freq="B")
            forecast = model.predict(future)

        predicted = forecast["yhat"].to_numpy()[-len(test):]
        actual = test.to_numpy()
        return float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    except Exception as exc:
        logger.warning("Prophet backtest failed: %s", exc)
        return None
