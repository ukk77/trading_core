"""Performance metrics for backtest results.

All functions accept a pandas Series (equity curve, indexed by date string)
and return plain floats or None when data is insufficient.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _log_returns(equity: pd.Series) -> pd.Series:
    prices = pd.to_numeric(equity, errors="coerce").dropna()
    return np.log(prices / prices.shift(1)).dropna()


def total_return(equity: pd.Series) -> Optional[float]:
    """Total percentage return (e.g. 0.35 = +35%)."""
    if len(equity) < 2:
        return None
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start == 0:
        return None
    return float(end / start - 1.0)


def cagr(equity: pd.Series) -> Optional[float]:
    """Compound annual growth rate."""
    if len(equity) < 2:
        return None
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start == 0:
        return None
    n_years = len(equity) / TRADING_DAYS
    if n_years == 0:
        return None
    return float((end / start) ** (1.0 / n_years) - 1.0)


def sharpe_ratio(equity: pd.Series, rf_annual: float = 0.04) -> Optional[float]:
    """Annualised Sharpe ratio."""
    rets = _log_returns(equity)
    if len(rets) < 20:
        return None
    rf_daily = rf_annual / TRADING_DAYS
    excess = rets - rf_daily
    sd = float(excess.std(ddof=1))
    if sd < 1e-10:
        return None
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(equity: pd.Series, rf_annual: float = 0.04) -> Optional[float]:
    """Annualised Sortino ratio (uses downside deviation only)."""
    rets = _log_returns(equity)
    if len(rets) < 20:
        return None
    rf_daily = rf_annual / TRADING_DAYS
    excess = rets - rf_daily
    downside = excess[excess < 0]
    if downside.empty:
        return None
    dd = float(downside.std(ddof=1))
    if dd < 1e-10:
        return None
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative float, e.g. -0.25 = -25%)."""
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def calmar_ratio(equity: pd.Series) -> Optional[float]:
    """CAGR / abs(max drawdown)."""
    c = cagr(equity)
    mdd = max_drawdown(equity)
    if c is None or mdd == 0:
        return None
    return float(c / abs(mdd))


def win_rate(trades_df: pd.DataFrame) -> Optional[float]:
    """Fraction of closed trades that were profitable."""
    if trades_df.empty or "action" not in trades_df.columns:
        return None
    sells = trades_df[trades_df["action"] == "SELL"]
    if sells.empty or "pnl" not in sells.columns:
        return None
    winners = sells[sells["pnl"] > 0]
    return float(len(winners) / len(sells))


def profit_factor(trades_df: pd.DataFrame) -> Optional[float]:
    """Gross profit / gross loss across all closed trades."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return None
    sells = trades_df[(trades_df["action"] == "SELL") & trades_df["pnl"].notna()]
    if sells.empty:
        return None
    gross_profit = sells.loc[sells["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(sells.loc[sells["pnl"] < 0, "pnl"].sum())
    if gross_loss == 0:
        return None
    return float(gross_profit / gross_loss)


def avg_holding_days(trades_df: pd.DataFrame) -> Optional[float]:
    """Average number of calendar days between paired BUY and SELL trades."""
    if trades_df.empty or "date" not in trades_df.columns:
        return None
    buys = trades_df[trades_df["action"] == "BUY"].reset_index(drop=True)
    sells = trades_df[trades_df["action"] == "SELL"].reset_index(drop=True)
    n = min(len(buys), len(sells))
    if n == 0:
        return None
    try:
        buy_dates = pd.to_datetime(buys["date"].iloc[:n])
        sell_dates = pd.to_datetime(sells["date"].iloc[:n])
        holding = (sell_dates.values - buy_dates.values).astype("timedelta64[D]").astype(float)
        return float(holding[holding >= 0].mean()) if len(holding) > 0 else None
    except Exception:
        return None


def alpha_vs_benchmark(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
) -> Optional[float]:
    """CAGR alpha of strategy over benchmark."""
    s = cagr(strategy_equity)
    b = cagr(benchmark_equity)
    if s is None or b is None:
        return None
    return float(s - b)


def compute_all_metrics(
    equity: pd.Series,
    initial_capital: float,
    trades_df: pd.DataFrame,
    benchmarks: Dict[str, pd.Series],
    rf_annual: float = 0.04,
) -> Dict:
    """Compute and return all performance metrics as a flat dictionary."""
    result = {
        "total_return_pct": round((total_return(equity) or 0.0) * 100, 2),
        "cagr_pct": round((cagr(equity) or 0.0) * 100, 2),
        "sharpe": sharpe_ratio(equity, rf_annual),
        "sortino": sortino_ratio(equity, rf_annual),
        "calmar": calmar_ratio(equity),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "final_equity": round(float(equity.iloc[-1]), 2) if not equity.empty else initial_capital,
        "profit_loss": round(
            float(equity.iloc[-1] - initial_capital) if not equity.empty else 0.0, 2
        ),
        "total_trades": int(
            len(trades_df[trades_df["action"] == "BUY"]) if not trades_df.empty else 0
        ),
        "win_rate_pct": round((win_rate(trades_df) or 0.0) * 100, 2),
        "profit_factor": profit_factor(trades_df),
        "avg_holding_days": avg_holding_days(trades_df),
    }

    for bench_name, bench_equity in benchmarks.items():
        alpha = alpha_vs_benchmark(equity, bench_equity)
        result[f"alpha_vs_{bench_name.lower()}_pct"] = (
            round(alpha * 100, 2) if alpha is not None else None
        )
        result[f"benchmark_{bench_name.lower()}_return_pct"] = round(
            (total_return(bench_equity) or 0.0) * 100, 2
        )

    return result
