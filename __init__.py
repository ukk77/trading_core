"""Shared core library for trading strategies."""

from .portfolio import Portfolio, Trade
# Do not import AlpacaBroker by default to avoid 'alpaca' requirement for downstream apps like risk_calculator
from .metrics import (
    total_return,
    cagr,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    profit_factor,
    avg_holding_days,
    alpha_vs_benchmark,
    compute_all_metrics,
)

__all__ = [
    "Portfolio",
    "Trade",
    "total_return",
    "cagr",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "avg_holding_days",
    "alpha_vs_benchmark",
    "compute_all_metrics",
]
