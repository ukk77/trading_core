"""Consecutive loss guard — circuit-breaker for per-ticker losing streaks.

Tracks consecutive losses per ticker and blocks new entries after a
configurable threshold is breached, imposing a cool-off period before
the ticker becomes tradable again.

Usage:
    from trading_core.loss_guard import LossGuard

    guard = LossGuard(max_consecutive=3, cooloff_bars=5)
    guard.record_loss("AAPL")      # after a losing trade closes
    guard.record_win("AAPL")       # resets streak
    guard.tick("AAPL")             # advance cooloff counter each bar

    if guard.is_blocked("AAPL"):
        # Skip new entry for AAPL
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

log = logging.getLogger(__name__)


@dataclass
class _TickerState:
    """Internal state for a single ticker."""
    consecutive_losses: int = 0
    cooloff_remaining: int = 0


@dataclass
class LossGuardConfig:
    """Configuration for the loss guard."""
    enabled: bool = True
    max_consecutive_losses: int = 3
    cooloff_bars: int = 5  # bars to wait after breaching threshold


class LossGuard:
    """Per-ticker consecutive loss tracker with cool-off."""

    def __init__(
        self,
        max_consecutive: int = 3,
        cooloff_bars: int = 5,
        enabled: bool = True,
    ):
        self._max = max_consecutive
        self._cooloff = cooloff_bars
        self._enabled = enabled
        self._state: Dict[str, _TickerState] = {}

    @classmethod
    def from_config(cls, cfg: LossGuardConfig) -> "LossGuard":
        return cls(
            max_consecutive=cfg.max_consecutive_losses,
            cooloff_bars=cfg.cooloff_bars,
            enabled=cfg.enabled,
        )

    def _get(self, ticker: str) -> _TickerState:
        if ticker not in self._state:
            self._state[ticker] = _TickerState()
        return self._state[ticker]

    def record_loss(self, ticker: str) -> None:
        """Record a losing trade close for ticker."""
        if not self._enabled:
            return
        s = self._get(ticker)
        s.consecutive_losses += 1
        if s.consecutive_losses >= self._max:
            s.cooloff_remaining = self._cooloff
            log.info(
                "[loss_guard] %s hit %d consecutive losses — blocking for %d bars",
                ticker, s.consecutive_losses, self._cooloff,
            )

    def record_win(self, ticker: str) -> None:
        """Record a winning trade close — resets the streak."""
        if not self._enabled:
            return
        s = self._get(ticker)
        s.consecutive_losses = 0
        s.cooloff_remaining = 0

    def tick(self, ticker: str) -> None:
        """Advance the cool-off counter by one bar.

        Call this once per bar for each ticker to decrement the cool-off.
        """
        if not self._enabled:
            return
        s = self._get(ticker)
        if s.cooloff_remaining > 0:
            s.cooloff_remaining -= 1
            if s.cooloff_remaining == 0:
                # Cool-off expired — reset loss streak
                s.consecutive_losses = 0
                log.info("[loss_guard] %s cooloff expired — re-enabled", ticker)

    def is_blocked(self, ticker: str) -> bool:
        """Return True if new entries should be blocked for this ticker."""
        if not self._enabled:
            return False
        s = self._get(ticker)
        return s.cooloff_remaining > 0

    def get_streak(self, ticker: str) -> int:
        """Return current consecutive loss count."""
        return self._get(ticker).consecutive_losses

    def get_cooloff_remaining(self, ticker: str) -> int:
        """Return remaining bars in cool-off."""
        return self._get(ticker).cooloff_remaining

    def reset(self, ticker: str) -> None:
        """Fully reset state for a ticker."""
        self._state.pop(ticker, None)

    def reset_all(self) -> None:
        """Clear all tracked state."""
        self._state.clear()
