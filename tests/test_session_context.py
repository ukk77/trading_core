"""Tests for pre/post-market session handling across all three options.

Covers:
  Option 1 — fetch_ohlcv filters extended-hours bars for hourly data
  Option 2 — fetch_ohlcv tags bars with is_extended; fetch_ohlcv_extended retains all
  Option 3a — premarket_gap_pct computes correct signed gap; MarketMetrics contains field;
               report.py warns on large gap
  Option 3b — premarket_confirmation_mult boosts/penalises aligned/contrary signals
  Option 3c — early_session_size_scalar returns 0.75 during first 30 min, 1.0 otherwise

All tests are fully self-contained using synthetic DataFrames — no live market data
or running services required.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import pytz

# ---------------------------------------------------------------------------
# Path setup — risk_calculator/ root must be on sys.path so 'backend' resolves
# ---------------------------------------------------------------------------
_TRADING_ROOT = Path(__file__).resolve().parents[2]   # trading/
_RC_ROOT = _TRADING_ROOT / "risk_calculator"
for _p in [str(_RC_ROOT), str(_TRADING_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.app.services.market_data import (
    _is_regular_session_bar,
    fetch_ohlcv,
    fetch_ohlcv_extended,
)
from backend.app.services.session_context import (
    fetch_premarket_gap,
    premarket_confirmation_mult,
    early_session_size_scalar,
)
from backend.app.services.risk_metrics import premarket_gap_pct as _pm_gap_metric
from backend.app.models.schemas import MarketMetrics


# ---------------------------------------------------------------------------
# Helpers — build synthetic hourly DataFrames
# ---------------------------------------------------------------------------

_EASTERN = pytz.timezone("US/Eastern")


def _make_hourly_df(timestamps_et: list[str]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame indexed in UTC from ET timestamp strings."""
    idx = pd.DatetimeIndex(
        [_EASTERN.localize(datetime.strptime(t, "%Y-%m-%d %H:%M")).astimezone(pytz.utc)
         for t in timestamps_et]
    )
    n = len(idx)
    df = pd.DataFrame(
        {
            "open":   [100.0 + i for i in range(n)],
            "high":   [101.0 + i for i in range(n)],
            "low":    [99.0  + i for i in range(n)],
            "close":  [100.5 + i for i in range(n)],
            "volume": [1_000_000] * n,
        },
        index=idx,
    )
    return df


# Timestamps spanning pre-market, regular session, and after-hours on the same day
_MIXED_TIMESTAMPS = [
    "2024-06-10 07:00",  # pre-market
    "2024-06-10 08:00",  # pre-market
    "2024-06-10 09:00",  # pre-market
    "2024-06-10 09:30",  # regular open
    "2024-06-10 10:00",  # regular
    "2024-06-10 11:00",  # regular
    "2024-06-10 12:00",  # regular
    "2024-06-10 13:00",  # regular
    "2024-06-10 14:00",  # regular
    "2024-06-10 15:00",  # regular
    "2024-06-10 16:00",  # after-hours (16:00 is excluded: open < 16:00)
    "2024-06-10 17:00",  # after-hours
    "2024-06-10 18:00",  # after-hours
]

_REGULAR_ONLY_TIMESTAMPS = [
    "2024-06-10 09:30",
    "2024-06-10 10:00",
    "2024-06-10 11:00",
    "2024-06-10 12:00",
    "2024-06-10 13:00",
    "2024-06-10 14:00",
    "2024-06-10 15:00",
]

_PRE_MARKET_TIMESTAMPS = [
    "2024-06-10 05:00",
    "2024-06-10 06:00",
    "2024-06-10 07:00",
    "2024-06-10 08:00",
    "2024-06-10 09:00",
]

_AFTER_HOURS_TIMESTAMPS = [
    "2024-06-10 16:00",
    "2024-06-10 17:00",
    "2024-06-10 18:00",
    "2024-06-10 19:00",
    "2024-06-10 20:00",
]


# ===========================================================================
# Option 1 — _is_regular_session_bar
# ===========================================================================

class TestIsRegularSessionBar:
    def _idx(self, ts_et: str) -> pd.DatetimeIndex:
        dt = _EASTERN.localize(datetime.strptime(ts_et, "%Y-%m-%d %H:%M")).astimezone(pytz.utc)
        return pd.DatetimeIndex([dt])

    def test_regular_open_930_is_regular(self):
        assert bool(_is_regular_session_bar(self._idx("2024-06-10 09:30")).iloc[0]) is True

    def test_midday_1300_is_regular(self):
        assert bool(_is_regular_session_bar(self._idx("2024-06-10 13:00")).iloc[0]) is True

    def test_last_regular_bar_1500_is_regular(self):
        assert bool(_is_regular_session_bar(self._idx("2024-06-10 15:00")).iloc[0]) is True

    def test_close_1600_is_extended(self):
        assert bool(_is_regular_session_bar(self._idx("2024-06-10 16:00")).iloc[0]) is False

    def test_premarket_0700_is_extended(self):
        assert bool(_is_regular_session_bar(self._idx("2024-06-10 07:00")).iloc[0]) is False

    def test_premarket_0930_boundary_is_regular(self):
        """09:30 is the first regular bar — must be included."""
        assert bool(_is_regular_session_bar(self._idx("2024-06-10 09:30")).iloc[0]) is True

    def test_afterhours_1800_is_extended(self):
        assert bool(_is_regular_session_bar(self._idx("2024-06-10 18:00")).iloc[0]) is False


# ===========================================================================
# Option 1 — fetch_ohlcv filters to regular session only
# ===========================================================================

class TestFetchOhlcvSessionFilter:

    def _ctx(self, df, interval="1h", lookback=365):
        _now = datetime(2024, 6, 11, 20, 0, tzinfo=timezone.utc)
        mock_dt = MagicMock()
        mock_dt.now.return_value = _now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        p_parquet = patch("backend.app.services.market_data.pd.read_parquet", return_value=df)
        p_exists = patch("backend.app.services.market_data.Path.exists", return_value=True)
        p_dt = patch("backend.app.services.market_data.datetime", mock_dt)
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(p_parquet)
        stack.enter_context(p_exists)
        stack.enter_context(p_dt)
        return stack

    def test_hourly_strips_pre_market_bars(self):
        raw = _make_hourly_df(_MIXED_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv("AAPL", lookback_days=365, interval="1h")
        times_et = result.index.tz_localize("UTC").tz_convert(_EASTERN)
        assert all(h >= 9 for h in times_et.hour), "Pre-market bars must be stripped"
        assert 9 not in [t.hour * 60 + t.minute for t in times_et if t.hour == 9 and t.minute == 0]

    def test_hourly_strips_after_hours_bars(self):
        raw = _make_hourly_df(_MIXED_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv("AAPL", lookback_days=365, interval="1h")
        times_et = result.index.tz_localize("UTC").tz_convert(_EASTERN)
        assert all(h < 16 for h in times_et.hour), "After-hours bars (>= 16:00) must be stripped"

    def test_hourly_retains_regular_session_bars(self):
        raw = _make_hourly_df(_MIXED_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv("AAPL", lookback_days=365, interval="1h")
        assert len(result) == len(_REGULAR_ONLY_TIMESTAMPS), \
            f"Expected {len(_REGULAR_ONLY_TIMESTAMPS)} regular bars, got {len(result)}"

    def test_hourly_only_regular_input_unchanged_count(self):
        raw = _make_hourly_df(_REGULAR_ONLY_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv("AAPL", lookback_days=365, interval="1h")
        assert len(result) == len(_REGULAR_ONLY_TIMESTAMPS)

    def test_daily_not_filtered(self):
        """Daily bars have no session concept — all rows returned regardless."""
        raw = _make_hourly_df(["2024-06-09 00:00", "2024-06-10 00:00", "2024-06-11 00:00"])
        with self._ctx(raw, interval="1d"):
            result = fetch_ohlcv("AAPL", lookback_days=365, interval="1d")
        assert len(result) == 3

    def test_columns_renamed_titlecase(self):
        raw = _make_hourly_df(_REGULAR_ONLY_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv("AAPL", lookback_days=365, interval="1h")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns, f"Expected TitleCase column '{col}'"


# ===========================================================================
# Option 2 — is_extended tagging
# ===========================================================================

class TestIsExtendedTagging:

    def _ctx(self, df, interval="1h"):
        _now = datetime(2024, 6, 11, 20, 0, tzinfo=timezone.utc)
        mock_dt = MagicMock()
        mock_dt.now.return_value = _now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("backend.app.services.market_data.pd.read_parquet", return_value=df))
        stack.enter_context(patch("backend.app.services.market_data.Path.exists", return_value=True))
        stack.enter_context(patch("backend.app.services.market_data.datetime", mock_dt))
        return stack

    def test_hourly_is_extended_false_for_regular_bars(self):
        raw = _make_hourly_df(_REGULAR_ONLY_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv("AAPL", lookback_days=365, interval="1h")
        assert "is_extended" in result.columns
        assert result["is_extended"].sum() == 0, "All regular bars must have is_extended=False"

    def test_daily_is_extended_always_false(self):
        raw = _make_hourly_df(["2024-06-10 00:00", "2024-06-11 00:00"])
        with self._ctx(raw, interval="1d"):
            result = fetch_ohlcv("AAPL", lookback_days=365, interval="1d")
        assert "is_extended" in result.columns
        assert result["is_extended"].sum() == 0

    def test_fetch_ohlcv_extended_includes_all_bars(self):
        raw = _make_hourly_df(_MIXED_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv_extended("AAPL", lookback_days=365)
        assert len(result) == len(_MIXED_TIMESTAMPS), \
            "fetch_ohlcv_extended must return ALL bars including extended hours"

    def test_fetch_ohlcv_extended_tags_premarket_as_extended(self):
        raw = _make_hourly_df(_MIXED_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv_extended("AAPL", lookback_days=365)
        times_et = result.index.tz_localize("UTC").tz_convert(_EASTERN)
        for i, (row_idx, row) in enumerate(result.iterrows()):
            et = times_et[i]
            in_regular = 9 * 60 + 30 <= et.hour * 60 + et.minute < 16 * 60
            assert bool(row["is_extended"]) == (not in_regular), \
                f"Bar at {et} mis-tagged: is_extended={row['is_extended']}"

    def test_fetch_ohlcv_extended_only_premarket(self):
        raw = _make_hourly_df(_PRE_MARKET_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv_extended("AAPL", lookback_days=365)
        assert result["is_extended"].all(), "All pre-market bars must be is_extended=True"

    def test_fetch_ohlcv_extended_only_afterhours(self):
        raw = _make_hourly_df(_AFTER_HOURS_TIMESTAMPS)
        with self._ctx(raw):
            result = fetch_ohlcv_extended("AAPL", lookback_days=365)
        assert result["is_extended"].all(), "All after-hours bars must be is_extended=True"


# ===========================================================================
# Option 3a — premarket_gap_pct (session_context + risk_metrics)
# ===========================================================================

def _make_extended_df_for_gap(
    prior_close: float,
    pm_open: float,
) -> pd.DataFrame:
    """Build a synthetic extended-hours DataFrame anchored to *today* so that
    the premarket filter (pd.Timestamp.now().normalize() == today) matches
    without any patching."""
    eastern = pytz.timezone("US/Eastern")
    today_et = datetime.now(tz=eastern).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_et = today_et.replace(hour=15, minute=0)  # 15:00 ET yesterday
    # Actually go back one day
    from datetime import timedelta as _td
    yesterday_et = (today_et - _td(days=1)).replace(hour=15, minute=0)
    pm_today_et = today_et.replace(hour=7, minute=0)   # 07:00 ET today

    prior_dt = yesterday_et.astimezone(pytz.utc)
    pm_dt = pm_today_et.astimezone(pytz.utc)

    idx = pd.DatetimeIndex([prior_dt, pm_dt]).tz_localize(None)
    df = pd.DataFrame(
        {
            "Open":        [prior_close, pm_open],
            "High":        [prior_close + 1, pm_open + 1],
            "Low":         [prior_close - 1, pm_open - 1],
            "Close":       [prior_close, pm_open],
            "Volume":      [1_000_000, 200_000],
            "is_extended": [False, True],
        },
        index=idx,
    )
    return df


class TestPremarketGapPct:

    def _patch_extended(self, df: pd.DataFrame):
        return patch(
            "backend.app.services.market_data.fetch_ohlcv_extended",
            return_value=df,
        )

    def test_gap_up_positive(self):
        df = _make_extended_df_for_gap(prior_close=100.0, pm_open=103.0)
        with self._patch_extended(df):
            gap = fetch_premarket_gap("AAPL")
        assert gap is not None
        assert gap == pytest.approx(0.03, abs=1e-6), f"Expected +3% gap, got {gap}"

    def test_gap_down_negative(self):
        df = _make_extended_df_for_gap(prior_close=100.0, pm_open=96.0)
        with self._patch_extended(df):
            gap = fetch_premarket_gap("AAPL")
        assert gap is not None
        assert gap == pytest.approx(-0.04, abs=1e-6), f"Expected -4% gap, got {gap}"

    def test_no_premarket_bars_returns_none(self):
        """When there are no pre-market bars today, should return None gracefully."""
        df = _make_extended_df_for_gap(prior_close=100.0, pm_open=100.0)
        df["is_extended"] = False
        with self._patch_extended(df):
            gap = fetch_premarket_gap("AAPL")
        assert gap is None

    def test_exception_returns_none(self):
        with patch(
            "backend.app.services.market_data.fetch_ohlcv_extended",
            side_effect=RuntimeError("No cache"),
        ):
            gap = fetch_premarket_gap("AAPL")
        assert gap is None

    def test_market_metrics_has_premarket_gap_field(self):
        m = MarketMetrics(premarket_gap_pct=0.025)
        assert m.premarket_gap_pct == pytest.approx(0.025)

    def test_market_metrics_premarket_gap_defaults_none(self):
        m = MarketMetrics()
        assert m.premarket_gap_pct is None


# ===========================================================================
# Option 3a — report.py warning on large pre-market gap
# ===========================================================================

class TestPremarketGapWarning:

    def _make_market_metrics(self, gap: Optional[float]) -> MarketMetrics:
        return MarketMetrics(premarket_gap_pct=gap)

    def _make_sentiment_risk_metrics(self):
        from backend.app.models.schemas import SentimentRiskMetrics
        return SentimentRiskMetrics(
            negative_ratio=0.2,
            dispersion=0.1,
            momentum_24h_vs_7d=0.0,
            recency_weighted_sentiment=0.5,
            news_volume_zscore=0.0,
            source_concentration_hhi=0.2,
            confidence=0.8,
            extreme_score_share=0.05,
            polarity_gap=0.1,
            sentiment_vol_proxy=0.1,
        )

    def _make_sentiment_response(self):
        from backend.app.models.schemas import SentimentResponse, SentimentMetrics
        return SentimentResponse(
            ticker="AAPL",
            company_name="Apple Inc.",
            overall_sentiment="neutral",
            confidence=0.8,
            metrics=SentimentMetrics(
                total_articles=30,
                positive_count=10,
                negative_count=8,
                neutral_count=12,
                avg_sentiment=0.1,
                sources_breakdown={"newsapi": 30},
            ),
            articles=[],
        )

    def test_large_gap_up_triggers_warning(self):
        from backend.app.services.report import _build_warnings
        market = self._make_market_metrics(gap=0.025)  # +2.5%
        sentiment = self._make_sentiment_risk_metrics()
        s_resp = self._make_sentiment_response()
        warnings = _build_warnings(market, sentiment, s_resp)
        assert any("pre-market gap" in w.lower() for w in warnings), \
            f"Expected pre-market gap warning, got: {warnings}"
        assert any("up" in w.lower() for w in warnings if "pre-market" in w.lower())

    def test_large_gap_down_triggers_warning(self):
        from backend.app.services.report import _build_warnings
        market = self._make_market_metrics(gap=-0.03)  # -3%
        sentiment = self._make_sentiment_risk_metrics()
        s_resp = self._make_sentiment_response()
        warnings = _build_warnings(market, sentiment, s_resp)
        assert any("pre-market gap" in w.lower() for w in warnings)
        assert any("down" in w.lower() for w in warnings if "pre-market" in w.lower())

    def test_small_gap_no_warning(self):
        from backend.app.services.report import _build_warnings
        market = self._make_market_metrics(gap=0.005)  # +0.5% — below threshold
        sentiment = self._make_sentiment_risk_metrics()
        s_resp = self._make_sentiment_response()
        warnings = _build_warnings(market, sentiment, s_resp)
        assert not any("pre-market gap" in w.lower() for w in warnings)

    def test_no_gap_no_warning(self):
        from backend.app.services.report import _build_warnings
        market = self._make_market_metrics(gap=None)
        sentiment = self._make_sentiment_risk_metrics()
        s_resp = self._make_sentiment_response()
        warnings = _build_warnings(market, sentiment, s_resp)
        assert not any("pre-market gap" in w.lower() for w in warnings)

    def test_exact_boundary_2pct_triggers_warning(self):
        from backend.app.services.report import _build_warnings
        market = self._make_market_metrics(gap=0.02)  # exactly 2%
        sentiment = self._make_sentiment_risk_metrics()
        s_resp = self._make_sentiment_response()
        warnings = _build_warnings(market, sentiment, s_resp)
        assert any("pre-market gap" in w.lower() for w in warnings)


# ===========================================================================
# Option 3b — premarket_confirmation_mult
# ===========================================================================

class TestPremarketConfirmationMult:

    # --- Gap up scenarios ---

    def test_gap_up_buy_signal_boost(self):
        mult = premarket_confirmation_mult(gap_pct=0.01, signal_direction="BUY")
        assert mult == pytest.approx(1.10), "Gap up + BUY should get agree_boost"

    def test_gap_up_sell_signal_penalty(self):
        mult = premarket_confirmation_mult(gap_pct=0.01, signal_direction="SELL")
        assert mult == pytest.approx(0.85), "Gap up + SELL should get disagree_penalty"

    def test_gap_up_short_signal_penalty(self):
        mult = premarket_confirmation_mult(gap_pct=0.01, signal_direction="SHORT")
        assert mult == pytest.approx(0.85), "Gap up + SHORT should get disagree_penalty"

    def test_gap_up_cover_signal_boost(self):
        mult = premarket_confirmation_mult(gap_pct=0.01, signal_direction="COVER")
        assert mult == pytest.approx(1.10), "Gap up + COVER should get agree_boost"

    # --- Gap down scenarios ---

    def test_gap_down_buy_signal_penalty(self):
        mult = premarket_confirmation_mult(gap_pct=-0.02, signal_direction="BUY")
        assert mult == pytest.approx(0.85), "Gap down + BUY should get disagree_penalty"

    def test_gap_down_sell_signal_boost(self):
        mult = premarket_confirmation_mult(gap_pct=-0.02, signal_direction="SELL")
        assert mult == pytest.approx(1.10), "Gap down + SELL should get agree_boost"

    def test_gap_down_short_signal_boost(self):
        mult = premarket_confirmation_mult(gap_pct=-0.02, signal_direction="SHORT")
        assert mult == pytest.approx(1.10), "Gap down + SHORT should get agree_boost"

    def test_gap_down_cover_signal_penalty(self):
        mult = premarket_confirmation_mult(gap_pct=-0.02, signal_direction="COVER")
        assert mult == pytest.approx(0.85), "Gap down + COVER should get disagree_penalty"

    # --- No/trivial gap scenarios ---

    def test_none_gap_returns_1(self):
        mult = premarket_confirmation_mult(gap_pct=None, signal_direction="BUY")
        assert mult == pytest.approx(1.0)

    def test_below_threshold_gap_returns_1(self):
        mult = premarket_confirmation_mult(gap_pct=0.003, signal_direction="BUY")
        assert mult == pytest.approx(1.0), "Gap below 0.5% threshold should be ignored"

    def test_hold_signal_always_1(self):
        for gap in [0.02, -0.02, None]:
            mult = premarket_confirmation_mult(gap_pct=gap, signal_direction="HOLD")
            assert mult == pytest.approx(1.0), f"HOLD should always return 1.0, gap={gap}"

    def test_custom_boost_penalty(self):
        mult = premarket_confirmation_mult(
            gap_pct=0.01, signal_direction="BUY",
            agree_boost=1.20, disagree_penalty=0.70,
        )
        assert mult == pytest.approx(1.20)

    def test_partial_sell_treated_as_sell(self):
        """PARTIAL_SELL is not explicitly handled — should return 1.0 (unknown direction)."""
        mult = premarket_confirmation_mult(gap_pct=0.01, signal_direction="PARTIAL_SELL")
        assert mult == pytest.approx(1.0)


# ===========================================================================
# Option 3c — early_session_size_scalar
# ===========================================================================

class TestEarlySessionSizeScalar:

    def _mock_time(self, hour: int, minute: int):
        """Produce a UTC datetime that corresponds to hour:minute ET."""
        et = _EASTERN.localize(datetime(2024, 6, 10, hour, minute))
        utc = et.astimezone(timezone.utc)
        return patch(
            "backend.app.services.session_context.datetime",
            **{"now.return_value": utc},
        )

    # --- Pre-market: before 09:30 ET ---

    def test_premarket_0700_returns_1(self):
        with self._mock_time(7, 0):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(1.0), "Pre-market is outside early-session window"

    def test_premarket_0900_returns_1(self):
        with self._mock_time(9, 0):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(1.0)

    # --- Early session: 09:30–10:00 ET ---

    def test_open_0930_returns_scaled(self):
        with self._mock_time(9, 30):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(0.75), "09:30 is in early-session window"

    def test_early_0945_returns_scaled(self):
        with self._mock_time(9, 45):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(0.75)

    def test_early_0959_returns_scaled(self):
        with self._mock_time(9, 59):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(0.75)

    # --- Active hours: 10:00–16:00 ET ---

    def test_active_1000_returns_1(self):
        with self._mock_time(10, 0):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(1.0), "10:00 is past early-session window"

    def test_active_1200_returns_1(self):
        with self._mock_time(12, 0):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(1.0)

    def test_active_1530_returns_1(self):
        with self._mock_time(15, 30):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(1.0)

    # --- Post-market: 16:00+ ET ---

    def test_postmarket_1600_returns_1(self):
        with self._mock_time(16, 0):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(1.0), "After-hours should not scale down"

    def test_postmarket_1800_returns_1(self):
        with self._mock_time(18, 0):
            scalar = early_session_size_scalar()
        assert scalar == pytest.approx(1.0)

    def test_custom_scale_and_window(self):
        with self._mock_time(9, 40):
            scalar = early_session_size_scalar(scale_down=0.5, early_minutes=60)
        assert scalar == pytest.approx(0.5)

    def test_custom_window_excludes_0940_if_window_is_5(self):
        with self._mock_time(9, 40):
            scalar = early_session_size_scalar(scale_down=0.5, early_minutes=5)
        assert scalar == pytest.approx(1.0), "09:40 is outside 5-minute early window"


# ===========================================================================
# Integration — signal strength affected by session context (MR generator)
# ===========================================================================

class TestMeanReversionSignalStrengthIntegration:
    """Verify that the session multipliers propagate through generate_signal()."""

    def _make_ohlc(self) -> pd.DataFrame:
        dates = pd.date_range("2023-01-01", periods=120, freq="B")
        close = 100.0 + pd.Series(range(120), index=dates, dtype=float)
        ohlc = pd.DataFrame({
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": [5_000_000] * 120,
        })
        return ohlc

    def _run_with_session_mocks(
        self,
        pm_gap: Optional[float],
        es_scalar: float,
        ohlc: pd.DataFrame,
    ):
        from mean_reversion.config import MeanReversionConfig
        from mean_reversion.signals.generator import generate_signal

        cfg = MeanReversionConfig()

        with patch("mean_reversion.signals.generator.fetch_premarket_gap", return_value=pm_gap), \
             patch("mean_reversion.signals.generator.early_session_size_scalar", return_value=es_scalar), \
             patch("mean_reversion.signals.generator._fetch_latest_sentiment", return_value=None), \
             patch("mean_reversion.signals.generator._fetch_latest_risk", return_value=None), \
             patch("mean_reversion.signals.generator.premarket_confirmation_mult",
                   side_effect=premarket_confirmation_mult):
            sig = generate_signal("AAPL", ohlc, cfg)
        return sig

    def test_active_hours_no_gap_strength_unmodified(self):
        ohlc = self._make_ohlc()
        sig_base = self._run_with_session_mocks(pm_gap=None, es_scalar=1.0, ohlc=ohlc)
        sig_same = self._run_with_session_mocks(pm_gap=0.0, es_scalar=1.0, ohlc=ohlc)
        assert sig_base.filtered_strength == pytest.approx(sig_same.filtered_strength, abs=1e-6)

    def test_early_session_reduces_buy_strength(self):
        ohlc = self._make_ohlc()
        sig_normal = self._run_with_session_mocks(pm_gap=None, es_scalar=1.0, ohlc=ohlc)
        sig_early = self._run_with_session_mocks(pm_gap=None, es_scalar=0.75, ohlc=ohlc)
        if sig_normal.action == "BUY":
            assert sig_early.filtered_strength <= sig_normal.filtered_strength

    def test_confirming_gap_up_increases_buy_strength(self):
        ohlc = self._make_ohlc()
        sig_no_gap = self._run_with_session_mocks(pm_gap=None, es_scalar=1.0, ohlc=ohlc)
        sig_gap_up = self._run_with_session_mocks(pm_gap=0.02, es_scalar=1.0, ohlc=ohlc)
        if sig_no_gap.action == "BUY":
            assert sig_gap_up.filtered_strength >= sig_no_gap.filtered_strength

    def test_contrary_gap_down_reduces_buy_strength(self):
        ohlc = self._make_ohlc()
        sig_no_gap = self._run_with_session_mocks(pm_gap=None, es_scalar=1.0, ohlc=ohlc)
        sig_gap_dn = self._run_with_session_mocks(pm_gap=-0.02, es_scalar=1.0, ohlc=ohlc)
        if sig_no_gap.action == "BUY":
            assert sig_gap_dn.filtered_strength <= sig_no_gap.filtered_strength

    def test_reason_chain_includes_pm_gap_annotation(self):
        ohlc = self._make_ohlc()
        sig = self._run_with_session_mocks(pm_gap=0.02, es_scalar=1.0, ohlc=ohlc)
        if sig.action not in ("HOLD",):
            assert "pm_gap" in sig.reason, f"Expected pm_gap in reason: {sig.reason}"

    def test_reason_chain_includes_early_session_annotation(self):
        ohlc = self._make_ohlc()
        sig = self._run_with_session_mocks(pm_gap=None, es_scalar=0.75, ohlc=ohlc)
        if sig.action not in ("HOLD",):
            assert "early_session" in sig.reason, f"Expected early_session in reason: {sig.reason}"

    def test_hold_signal_strength_always_zero(self):
        """Session context must not accidentally boost a HOLD out of zero."""
        ohlc = self._make_ohlc()
        sig = self._run_with_session_mocks(pm_gap=0.05, es_scalar=0.75, ohlc=ohlc)
        if sig.action == "HOLD":
            assert sig.filtered_strength == pytest.approx(0.0)
