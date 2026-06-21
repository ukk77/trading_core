"""Shared core library for trading strategies."""

from .portfolio import Portfolio, Trade
from .alpaca_broker import AlpacaBroker
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
    "AlpacaBroker",
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
