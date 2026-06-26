"""Historical price data fetch from the centralized Parquet Data Lake."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

_DEFAULT_MARKET_DATA = Path(__file__).resolve().parents[1] / "market_data"
MARKET_DATA_DIR = Path(os.environ.get("MARKET_DATA_DIR", str(_DEFAULT_MARKET_DATA)))

_REGULAR_SESSION_START_MINUTES = 9 * 60 + 30   # 09:30 ET
_REGULAR_SESSION_END_MINUTES   = 16 * 60        # 16:00 ET (exclusive)


def _is_regular_session_bar(index_utc: "pd.DatetimeIndex") -> "pd.Series":
    """Return boolean Series: True for bars that fall within regular NYSE session (09:30–16:00 ET)."""
    import pytz
    eastern = pytz.timezone("US/Eastern")
    idx_et = index_utc.tz_convert(eastern)
    minutes = idx_et.hour * 60 + idx_et.minute
    return pd.Series(
        (minutes >= _REGULAR_SESSION_START_MINUTES) & (minutes < _REGULAR_SESSION_END_MINUTES),
        index=index_utc,
    )


def fetch_ohlcv(ticker: str, lookback_days: int = 504, interval: str = "1d") -> pd.DataFrame:
    """Return OHLCV indexed by date (UTC-naive). Reads from local Parquet cache.

    For hourly data:
      - Adds an ``is_extended`` boolean column (True = pre/after-market bar).
      - Filters to regular-session bars only (09:30–16:00 ET) before returning,
        so indicators are never distorted by extended-hours low-volume candles.
      - Pre/post-market bars are retained in a separate ``fetch_ohlcv_extended``
        call when callers explicitly need them (e.g. gap-risk computation).
    """
    if interval == "1d":
        parquet_path = MARKET_DATA_DIR / "daily" / f"{ticker}.parquet"
    elif interval == "1h":
        parquet_path = MARKET_DATA_DIR / "hourly" / f"{ticker}.parquet"
    else:
        raise ValueError(f"Unsupported interval: {interval}")

    if not parquet_path.exists():
        raise RuntimeError(f"No cached data found for '{ticker}' at {parquet_path}. Run data_ingestion.py first.")

    df = pd.read_parquet(parquet_path)

    # Filter by lookback_days — Polygon data is in UTC.
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df.loc[start:end].copy()

    # --- Session tagging & filtering (hourly only) ---
    if interval == "1h":
        regular_mask = _is_regular_session_bar(df.index)
        df["is_extended"] = ~regular_mask
        df = df[regular_mask].copy()
    else:
        df["is_extended"] = False

    # Make timezone-naive to match original yfinance behavior
    df.index = df.index.tz_localize(None)

    # Map lowercase Polygon columns to TitleCase expected by the strategies
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    return df


def fetch_ohlcv_extended(ticker: str, lookback_days: int = 30) -> pd.DataFrame:
    """Return ALL hourly bars (including pre/post-market) with an ``is_extended`` tag.

    Intended exclusively for extended-hours feature computation (gap risk, pre-market
    momentum) — NOT for indicator or signal calculation.
    """
    parquet_path = MARKET_DATA_DIR / "hourly" / f"{ticker}.parquet"
    if not parquet_path.exists():
        raise RuntimeError(f"No cached data found for '{ticker}' at {parquet_path}. Run data_ingestion.py first.")

    df = pd.read_parquet(parquet_path)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df.loc[start:end].copy()

    regular_mask = _is_regular_session_bar(df.index)
    df["is_extended"] = ~regular_mask

    df.index = df.index.tz_localize(None)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    return df

def fetch_risk_free_rate_annual(default: float = 0.04) -> float:
    """Return annualized risk-free rate proxy (^IRX is quoted as % annualized)."""
    try:
        df = yf.download("^IRX", period="1mo", progress=False, threads=False)
        if df is None or df.empty:
            return default
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        last = float(df["Close"].dropna().iloc[-1])
        return last / 100.0
    except Exception:
        return default
