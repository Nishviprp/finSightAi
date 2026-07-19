"""
Tests for src.financial_forecast_prophet.

Uses a real live call to Yahoo Finance for AAPL, same rationale as
test_financial_forecast.py (the ARIMA version): this is a thin wrapper
whose correctness genuinely depends on live data and a real model fit, not
just its own logic. Each monkeypatched test clears the
@st.cache_data(ttl=86400) cache and uses a distinct symbol, since the cache
is keyed by (symbol, days_forward) and would otherwise mask the mock with a
prior real result.

Note on the MAPE assertion: a live backtest of Prophet on AAPL's last 2
years (fit on the first 80%, forecast the held-out last ~100 days) came
back with MAPE ~7%. That's a normal result for stock-price forecasting at
a ~100-day horizon, not a "good model" — the test asserts MAPE is a finite,
sane percentage (well under 100%) rather than an unrealistic accuracy bar.
"""
import math

import pandas as pd
import pytest

from src import financial_forecast_prophet as fp

AAPL = "AAPL"
INVALID_SYMBOL = "ZZZZZZINVALID"


def _boom_ticker(*_args, **_kwargs):
    raise ConnectionError("simulated network down")


@pytest.fixture(scope="module")
def result():
    return fp.forecast_stock_price_prophet(AAPL, days_forward=90)


class TestForecastStockPriceProphetRealSymbol:
    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_has_expected_keys(self, result):
        expected_keys = {
            "historical_prices", "forecast_prices", "model_type", "mape", "confidence_level",
        }
        assert expected_keys.issubset(result.keys())

    def test_model_type_is_prophet(self, result):
        assert result["model_type"] == "Prophet"

    def test_confidence_level_is_95_percent(self, result):
        assert result["confidence_level"] == pytest.approx(0.95)

    def test_historical_prices_is_dataframe_with_close_column(self, result):
        df = result["historical_prices"]
        assert isinstance(df, pd.DataFrame)
        assert "Close" in df.columns

    def test_historical_prices_covers_roughly_two_years(self, result):
        df = result["historical_prices"]
        assert 400 <= len(df) <= 520

    def test_forecast_prices_is_dataframe_with_expected_columns(self, result):
        df = result["forecast_prices"]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["mean", "lower", "upper"]

    def test_forecast_prices_has_90_rows(self, result):
        assert len(result["forecast_prices"]) == 90

    def test_forecast_index_is_after_historical_index(self, result):
        last_historical_date = result["historical_prices"].index[-1]
        first_forecast_date = result["forecast_prices"].index[0]
        assert first_forecast_date > last_historical_date

    def test_forecast_bounds_bracket_the_mean(self, result):
        df = result["forecast_prices"]
        assert (df["lower"] <= df["mean"]).all()
        assert (df["mean"] <= df["upper"]).all()

    def test_confidence_band_widens_with_horizon(self, result):
        df = result["forecast_prices"]
        first_band = df["upper"].iloc[0] - df["lower"].iloc[0]
        last_band = df["upper"].iloc[-1] - df["lower"].iloc[-1]
        assert last_band > first_band

    def test_mape_is_a_sane_finite_percentage(self, result):
        mape = result["mape"]
        assert isinstance(mape, float)
        assert math.isfinite(mape)
        assert mape > 0
        assert mape < 100  # sanity bound, not a "good model" bar

    def test_forecast_shows_a_trend_not_a_flat_line(self, result):
        # The whole point of switching from ARIMA to Prophet: ARIMA(1,1,1)
        # converges to a near-constant mean by the end of a 90-day horizon
        # (verified in test_financial_forecast.py); Prophet extrapolates
        # its fitted trend instead, so the forecast should move meaningfully
        # over the window rather than flatten out.
        mean = result["forecast_prices"]["mean"]
        first, last = mean.iloc[0], mean.iloc[-1]
        relative_change = abs(last - first) / first
        assert relative_change > 0.01, (
            f"forecast barely moved over 90 days ({first:.2f} -> {last:.2f}), "
            "looks flat rather than trending"
        )


class TestForecastStockPriceProphetCustomHorizon:
    def test_days_forward_controls_row_count(self):
        result = fp.forecast_stock_price_prophet(AAPL, days_forward=30)
        assert len(result["forecast_prices"]) == 30


class TestForecastStockPriceProphetErrorHandling:
    def test_invalid_symbol_returns_none(self):
        assert fp.forecast_stock_price_prophet(INVALID_SYMBOL) is None

    def test_network_down_returns_none(self, monkeypatch):
        fp.forecast_stock_price_prophet.clear()
        monkeypatch.setattr(fp.yf, "Ticker", _boom_ticker)
        assert fp.forecast_stock_price_prophet("NETDOWN") is None

    def test_empty_history_returns_none(self, monkeypatch):
        fp.forecast_stock_price_prophet.clear()

        class EmptyHistoryTicker:
            def __init__(self, *_a, **_kw):
                pass

            def history(self, *_a, **_kw):
                return pd.DataFrame()

        monkeypatch.setattr(fp.yf, "Ticker", EmptyHistoryTicker)
        assert fp.forecast_stock_price_prophet("EMPTYHIST") is None
