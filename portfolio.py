"""Simulated portfolio — tracks cash, positions, trades, and equity curve.

All prices are assumed to be adjusted-close (auto_adjust=True from yfinance).
Slippage is applied externally before calling buy/sell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Trade:
    """Record of a single simulated trade."""
    date: str
    ticker: str
    action: str
    shares: int
    price: float
    commission: float
    pnl: Optional[float] = None


class Portfolio:
    """Cash + equity portfolio for backtesting."""

    def __init__(self, initial_capital: float, commission_pct: float = 0.001) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct

        self._positions: Dict[str, int] = {}       # long positions
        self._avg_cost: Dict[str, float] = {}
        self._short_positions: Dict[str, int] = {} # short positions
        self._short_avg_cost: Dict[str, float] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

    def shares_held(self, ticker: str) -> int:
        return self._positions.get(ticker, 0)

    def cost_basis(self, ticker: str) -> float:
        return self._avg_cost.get(ticker, 0.0)

    def holdings_value(self, prices: Dict[str, float]) -> float:
        long_val = sum(self._positions.get(t, 0) * p for t, p in prices.items())
        short_pnl = sum(
            (self._short_avg_cost.get(t, p) - p) * self._short_positions.get(t, 0)
            for t, p in prices.items()
            if self._short_positions.get(t, 0) > 0
        )
        return long_val + short_pnl

    def equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self.holdings_value(prices)

    def is_invested(self, ticker: str) -> bool:
        return self._positions.get(ticker, 0) > 0

    def is_short(self, ticker: str) -> bool:
        return self._short_positions.get(ticker, 0) > 0

    def open_position_count(self) -> int:
        """Total number of open positions (long + short combined)."""
        return len(self._positions) + len(self._short_positions)

    def long_notional(self, prices: Dict[str, float]) -> float:
        """Total long notional value at current prices."""
        return sum(self._positions.get(t, 0) * p for t, p in prices.items())

    def short_notional(self, prices: Dict[str, float]) -> float:
        """Total short notional value at current prices."""
        return sum(self._short_positions.get(t, 0) * p for t, p in prices.items())

    def gross_exposure(self, prices: Dict[str, float]) -> float:
        """Total gross exposure: long notional + short notional."""
        return self.long_notional(prices) + self.short_notional(prices)

    def sector_exposure(
        self, prices: Dict[str, float], sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """Gross exposure in dollars per sector (long + short combined)."""
        exp: Dict[str, float] = {}
        for ticker, shares in self._positions.items():
            sector = sector_map.get(ticker, "Unknown")
            exp[sector] = exp.get(sector, 0.0) + shares * prices.get(ticker, 0.0)
        for ticker, shares in self._short_positions.items():
            sector = sector_map.get(ticker, "Unknown")
            exp[sector] = exp.get(sector, 0.0) + shares * prices.get(ticker, 0.0)
        return exp

    def short_shares(self, ticker: str) -> int:
        return self._short_positions.get(ticker, 0)

    def short_cost(self, ticker: str) -> float:
        return self._short_avg_cost.get(ticker, 0.0)

    def buy(self, ticker: str, shares: int, price: float, date_str: str) -> bool:
        """Execute a buy order. Returns True if at least 1 share was bought."""
        if shares <= 0:
            return False

        commission = price * shares * self.commission_pct
        total_cost = price * shares + commission

        if total_cost > self.cash:
            max_shares = int(self.cash / (price * (1.0 + self.commission_pct)))
            if max_shares <= 0:
                return False
            shares = max_shares
            commission = price * shares * self.commission_pct
            total_cost = price * shares + commission

        self.cash -= total_cost

        prev_shares = self._positions.get(ticker, 0)
        prev_cost = self._avg_cost.get(ticker, 0.0)
        new_shares = prev_shares + shares
        self._avg_cost[ticker] = (prev_cost * prev_shares + price * shares) / new_shares
        self._positions[ticker] = new_shares

        self.trades.append(
            Trade(
                date=date_str,
                ticker=ticker,
                action="BUY",
                shares=shares,
                price=price,
                commission=commission,
            )
        )
        return True

    def sell(self, ticker: str, shares: int, price: float, date_str: str) -> bool:
        """Execute a sell order (partial or full)."""
        held = self._positions.get(ticker, 0)
        if held <= 0:
            return False

        shares = min(shares, held)
        commission = price * shares * self.commission_pct
        proceeds = price * shares - commission
        pnl = (price - self._avg_cost.get(ticker, price)) * shares - commission

        self.cash += proceeds
        remaining = held - shares
        if remaining <= 0:
            self._positions.pop(ticker, None)
            self._avg_cost.pop(ticker, None)
        else:
            self._positions[ticker] = remaining

        self.trades.append(
            Trade(
                date=date_str,
                ticker=ticker,
                action="SELL",
                shares=shares,
                price=price,
                commission=commission,
                pnl=pnl,
            )
        )
        return True

    def sell_all(self, ticker: str, price: float, date_str: str) -> bool:
        """Liquidate the entire long position for a ticker."""
        return self.sell(ticker, self._positions.get(ticker, 0), price, date_str)

    def partial_sell(self, ticker: str, fraction: float, price: float, date_str: str) -> bool:
        """Sell a fraction of the long position (e.g., 0.5 = first tranche)."""
        held = self._positions.get(ticker, 0)
        if held <= 0:
            return False
        shares = max(1, int(held * fraction))
        return self.sell(ticker, shares, price, date_str)

    def short(self, ticker: str, shares: int, price: float, date_str: str) -> bool:
        """Open a short position. Cash receives proceeds. Returns True if executed."""
        if shares <= 0:
            return False
        commission = price * shares * self.commission_pct
        proceeds = price * shares - commission
        self.cash += proceeds

        prev_shares = self._short_positions.get(ticker, 0)
        prev_cost = self._short_avg_cost.get(ticker, 0.0)
        new_shares = prev_shares + shares
        self._short_avg_cost[ticker] = (
            (prev_cost * prev_shares + price * shares) / new_shares
        )
        self._short_positions[ticker] = new_shares
        self.trades.append(
            Trade(date=date_str, ticker=ticker, action="SHORT",
                  shares=shares, price=price, commission=commission)
        )
        return True

    def cover(self, ticker: str, shares: int, price: float, date_str: str) -> bool:
        """Close (cover) a short position. Returns True if executed."""
        held = self._short_positions.get(ticker, 0)
        if held <= 0:
            return False
        shares = min(shares, held)
        commission = price * shares * self.commission_pct
        cost = price * shares + commission
        pnl = (self._short_avg_cost.get(ticker, price) - price) * shares - commission
        self.cash -= cost
        remaining = held - shares
        if remaining <= 0:
            self._short_positions.pop(ticker, None)
            self._short_avg_cost.pop(ticker, None)
        else:
            self._short_positions[ticker] = remaining
        self.trades.append(
            Trade(date=date_str, ticker=ticker, action="COVER",
                  shares=shares, price=price, commission=commission, pnl=pnl)
        )
        return True

    def cover_all(self, ticker: str, price: float, date_str: str) -> bool:
        """Cover the entire short position for a ticker."""
        return self.cover(ticker, self._short_positions.get(ticker, 0), price, date_str)

    def partial_cover(self, ticker: str, fraction: float, price: float, date_str: str) -> bool:
        """Cover a fraction of the short position."""
        held = self._short_positions.get(ticker, 0)
        if held <= 0:
            return False
        shares = max(1, int(held * fraction))
        return self.cover(ticker, shares, price, date_str)

    def accrue_cash_interest(self, daily_rf_rate: float) -> None:
        """Add one day of risk-free interest on uninvested cash."""
        if daily_rf_rate > 0 and self.cash > 0:
            self.cash += self.cash * daily_rf_rate

    def record_equity(self, date_str: str, prices: Dict[str, float]) -> None:
        """Append a daily equity snapshot."""
        eq = self.equity(prices)
        self.equity_curve.append(
            {
                "date": date_str,
                "equity": eq,
                "cash": self.cash,
                "holdings_value": self.holdings_value(prices),
            }
        )

    def equity_series(self) -> pd.Series:
        """Return equity curve as a pandas Series indexed by date string."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve).set_index("date")
        return df["equity"]

    def to_trades_df(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trades:
            return pd.DataFrame(
                columns=["date", "ticker", "action", "shares", "price", "commission", "pnl"]
            )
        return pd.DataFrame([vars(t) for t in self.trades])
