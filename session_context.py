"""Session-context helpers for pre/post-market awareness.

These functions are consumed by strategy signal generators (mean_reversion,
trend_following, volatility_breakout) to:

  1. Retrieve today's pre-market gap from the risk_calculator market_data service.
  2. Compute a pre-market confirmation multiplier for signal strength.
  3. Compute an early-session position-size scalar that reduces size when a run
     is triggered in the first 30 minutes of the regular session.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytz


_EASTERN = pytz.timezone("US/Eastern")
_EARLY_SESSION_MINUTES = 30  # first N minutes after open are "early session"


# ---------- Pre-market gap ----------

def fetch_premarket_gap(ticker: str) -> Optional[float]:
    """Return today's pre-market gap as a signed fraction (or None if unavailable).

    Delegates to ``fetch_ohlcv_extended`` from the shared market_data module so
    there is a single source of truth for extended-hours data.
    """
    try:
        from trading_core.market_data import fetch_ohlcv_extended
        import pandas as pd

        df = fetch_ohlcv_extended(ticker, lookback_days=5)
        if df.empty:
            return None

        today = pd.Timestamp.now().normalize()
        regular_closes = df[~df["is_extended"]]["Close"].dropna()
        premarket_bars = df[
            df["is_extended"] & (pd.to_datetime(df.index).normalize() == today)
        ]

        if premarket_bars.empty or regular_closes.empty:
            return None

        prior_close = float(regular_closes.iloc[-1])
        if prior_close == 0:
            return None

        first_pm_open = float(premarket_bars["Open"].iloc[0])
        return float((first_pm_open - prior_close) / prior_close)
    except Exception:
        return None


# ---------- Confirmation multiplier ----------

def premarket_confirmation_mult(
    gap_pct: Optional[float],
    signal_direction: str,
    agree_boost: float = 1.10,
    disagree_penalty: float = 0.85,
    gap_threshold: float = 0.005,
) -> float:
    """Return a multiplier [0.85, 1.10] based on whether today's pre-market gap
    confirms or contradicts the proposed signal direction.

    Args:
        gap_pct: Signed pre-market gap (positive = gap up, negative = gap down).
        signal_direction: ``"BUY"``, ``"SELL"``, ``"SHORT"``, ``"COVER"``, or ``"HOLD"``.
        agree_boost: Multiplier applied when gap aligns with signal (default 1.10).
        disagree_penalty: Multiplier applied when gap contradicts signal (default 0.85).
        gap_threshold: Minimum absolute gap magnitude to be considered directional (0.5%).

    Returns:
        Float multiplier to apply to ``filtered_strength``.
    """
    if gap_pct is None or abs(gap_pct) < gap_threshold:
        return 1.0

    gap_up = gap_pct > 0
    if signal_direction in ("BUY", "COVER"):
        return agree_boost if gap_up else disagree_penalty
    if signal_direction in ("SELL", "SHORT"):
        return agree_boost if not gap_up else disagree_penalty
    return 1.0


# ---------- Early-session size scalar ----------

def early_session_size_scalar(
    scale_down: float = 0.75,
    early_minutes: int = _EARLY_SESSION_MINUTES,
) -> float:
    """Return a position-size scalar < 1.0 if the current wall-clock time falls
    within the first ``early_minutes`` of the regular NYSE session (09:30 ET),
    otherwise 1.0.

    This reduces exposure to the open-of-day volatility bleed from pre-market
    activity without blocking the trade entirely.
    """
    now_et = datetime.now(timezone.utc).astimezone(_EASTERN)
    open_minutes = 9 * 60 + 30  # 09:30
    current_minutes = now_et.hour * 60 + now_et.minute
    if open_minutes <= current_minutes < open_minutes + early_minutes:
        return scale_down
    return 1.0
