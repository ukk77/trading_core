"""Regime-adaptive parameter selection for trading strategies.

Reads the current market regime (from the harness regime_log DB) and returns
parameter adjustment multipliers/overrides that each strategy can apply to
its configuration.

Usage:
    from trading_core.regime_params import get_regime_adjustments

    adjustments = get_regime_adjustments(strategy="mean_reversion")
    # adjustments.adx_threshold_mult  -> scale ADX threshold
    # adjustments.position_size_mult  -> scale position sizing
    # adjustments.entry_threshold_mult -> scale entry aggressiveness
"""
from __future__ import annotations

import os
import sqlite3
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

log = logging.getLogger(__name__)

_TRADING_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HARNESS_DB = Path(os.getenv(
    "HARNESS_DB_PATH",
    str(_TRADING_ROOT / "harness" / "harness_trades.db")
))


# Regime labels (mirrors harness/regime.py Regime enum values)
BULL_TREND = "bull_trend"
BEAR_TREND = "bear_trend"
HIGH_VOL = "high_vol"
RANGE_BOUND = "range_bound"


@dataclass
class RegimeAdjustments:
    """Parameter adjustments to apply on top of a strategy's base config.

    All multipliers default to 1.0 (no change). Strategies apply these to
    their config values: adjusted_val = base_val * mult.
    """
    regime: str = RANGE_BOUND
    regime_detected: bool = False

    # Position sizing
    position_size_mult: float = 1.0

    # Entry aggressiveness (lower = more aggressive entry thresholds)
    entry_threshold_mult: float = 1.0

    # Stop loss width (higher = wider stops)
    stop_width_mult: float = 1.0

    # ADX threshold adjustment
    adx_threshold_mult: float = 1.0

    # Sentiment filter strictness (higher = stricter)
    sentiment_strictness_mult: float = 1.0

    # Max risk score adjustment (lower = more conservative)
    max_risk_score_mult: float = 1.0


# ── Regime presets per strategy ──────────────────────────────────────────────

_MR_PRESETS: Dict[str, RegimeAdjustments] = {
    BULL_TREND: RegimeAdjustments(
        regime=BULL_TREND,
        regime_detected=True,
        position_size_mult=0.7,       # MR underperforms in trends — reduce size
        entry_threshold_mult=1.3,     # Require deeper z-score to enter (more selective)
        stop_width_mult=1.2,          # Wider stops (trends overshoot)
        adx_threshold_mult=0.8,       # Tighter ADX filter (block entries sooner)
        sentiment_strictness_mult=1.2,
        max_risk_score_mult=0.9,
    ),
    BEAR_TREND: RegimeAdjustments(
        regime=BEAR_TREND,
        regime_detected=True,
        position_size_mult=0.6,       # Even more cautious — bear trends crush MR
        entry_threshold_mult=1.5,     # Very deep z-score required
        stop_width_mult=1.3,
        adx_threshold_mult=0.7,       # Very strict ADX gate
        sentiment_strictness_mult=1.3,
        max_risk_score_mult=0.8,
    ),
    HIGH_VOL: RegimeAdjustments(
        regime=HIGH_VOL,
        regime_detected=True,
        position_size_mult=0.5,       # Cut size significantly in high vol
        entry_threshold_mult=1.2,     # Slightly deeper entries
        stop_width_mult=1.5,          # Much wider stops (vol expansion)
        adx_threshold_mult=1.0,       # ADX less meaningful in high vol
        sentiment_strictness_mult=0.8, # Sentiment less reliable in panic
        max_risk_score_mult=0.7,      # Conservative risk gate
    ),
    RANGE_BOUND: RegimeAdjustments(
        regime=RANGE_BOUND,
        regime_detected=True,
        position_size_mult=1.2,       # MR thrives in ranges — size up
        entry_threshold_mult=0.9,     # Slightly more aggressive entries
        stop_width_mult=0.9,          # Tighter stops (range-bound = less overshoot)
        adx_threshold_mult=1.1,       # Slightly relaxed ADX
        sentiment_strictness_mult=1.0,
        max_risk_score_mult=1.0,
    ),
}

_TF_PRESETS: Dict[str, RegimeAdjustments] = {
    BULL_TREND: RegimeAdjustments(
        regime=BULL_TREND,
        regime_detected=True,
        position_size_mult=1.3,       # TF thrives in bull trends — size up
        entry_threshold_mult=0.9,     # More aggressive entries
        stop_width_mult=1.1,          # Slightly wider trailing stops
        adx_threshold_mult=0.9,       # Lower bar for ADX (catch early trends)
        sentiment_strictness_mult=0.9,
        max_risk_score_mult=1.1,      # Slightly relaxed risk gate
    ),
    BEAR_TREND: RegimeAdjustments(
        regime=BEAR_TREND,
        regime_detected=True,
        position_size_mult=1.0,       # Normal size (shorts enabled)
        entry_threshold_mult=1.0,
        stop_width_mult=1.2,          # Wider stops in bear (volatility higher)
        adx_threshold_mult=0.9,       # Lower ADX bar (bear trends are sharp)
        sentiment_strictness_mult=1.0,
        max_risk_score_mult=0.9,
    ),
    HIGH_VOL: RegimeAdjustments(
        regime=HIGH_VOL,
        regime_detected=True,
        position_size_mult=0.5,       # Halve position size in high vol
        entry_threshold_mult=1.3,     # Higher bar for entries
        stop_width_mult=1.5,          # Much wider stops
        adx_threshold_mult=1.2,       # Require stronger ADX to confirm trend
        sentiment_strictness_mult=0.8,
        max_risk_score_mult=0.7,
    ),
    RANGE_BOUND: RegimeAdjustments(
        regime=RANGE_BOUND,
        regime_detected=True,
        position_size_mult=0.6,       # TF underperforms in ranges — reduce
        entry_threshold_mult=1.2,     # More selective (avoid whipsaws)
        stop_width_mult=0.8,          # Tighter stops (cut losses in chop)
        adx_threshold_mult=1.3,       # Require very strong ADX to trade
        sentiment_strictness_mult=1.2,
        max_risk_score_mult=0.9,
    ),
}

_VB_PRESETS: Dict[str, RegimeAdjustments] = {
    BULL_TREND: RegimeAdjustments(
        regime=BULL_TREND,
        regime_detected=True,
        position_size_mult=1.1,
        entry_threshold_mult=0.9,     # Breakouts work well in bull trends
        stop_width_mult=1.0,
        adx_threshold_mult=0.9,
        sentiment_strictness_mult=1.0,
        max_risk_score_mult=1.0,
    ),
    BEAR_TREND: RegimeAdjustments(
        regime=BEAR_TREND,
        regime_detected=True,
        position_size_mult=0.5,       # Breakouts often fail in bear
        entry_threshold_mult=1.4,     # Much stricter entry
        stop_width_mult=1.2,
        adx_threshold_mult=1.0,
        sentiment_strictness_mult=1.3,
        max_risk_score_mult=0.7,
    ),
    HIGH_VOL: RegimeAdjustments(
        regime=HIGH_VOL,
        regime_detected=True,
        position_size_mult=0.7,       # High vol = more false breakouts
        entry_threshold_mult=1.2,
        stop_width_mult=1.4,
        adx_threshold_mult=1.0,
        sentiment_strictness_mult=0.9,
        max_risk_score_mult=0.8,
    ),
    RANGE_BOUND: RegimeAdjustments(
        regime=RANGE_BOUND,
        regime_detected=True,
        position_size_mult=1.2,       # Breakouts from consolidation work well
        entry_threshold_mult=0.9,
        stop_width_mult=0.9,
        adx_threshold_mult=1.0,
        sentiment_strictness_mult=1.0,
        max_risk_score_mult=1.0,
    ),
}

_STRATEGY_PRESETS: Dict[str, Dict[str, RegimeAdjustments]] = {
    "mean_reversion": _MR_PRESETS,
    "mr": _MR_PRESETS,
    "trend_following": _TF_PRESETS,
    "tf": _TF_PRESETS,
    "volatility_breakout": _VB_PRESETS,
    "vb": _VB_PRESETS,
}


def _read_latest_regime(db_path: Path = _DEFAULT_HARNESS_DB) -> Optional[str]:
    """Read the most recent regime from the harness regime_log table."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT regime FROM regime_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                return row[0]
    except Exception as e:
        log.debug("Could not read regime_log: %s", e)
    return None


def get_regime_adjustments(
    strategy: str,
    regime_override: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> RegimeAdjustments:
    """Get regime-adaptive parameter adjustments for a strategy.

    Args:
        strategy: Strategy name ("mean_reversion", "mr", "trend_following", "tf",
                  "volatility_breakout", "vb").
        regime_override: If provided, skip DB read and use this regime directly.
        db_path: Path to harness DB. Defaults to standard location.

    Returns:
        RegimeAdjustments with multipliers. Returns neutral (all 1.0) if regime
        cannot be determined or strategy is unknown.
    """
    # Determine current regime
    regime = regime_override
    if regime is None:
        regime = _read_latest_regime(db_path or _DEFAULT_HARNESS_DB)

    if regime is None:
        return RegimeAdjustments(regime_detected=False)

    # Look up presets
    presets = _STRATEGY_PRESETS.get(strategy.lower())
    if presets is None:
        log.debug("Unknown strategy %r for regime params — returning neutral", strategy)
        return RegimeAdjustments(regime=regime, regime_detected=True)

    adjustments = presets.get(regime)
    if adjustments is None:
        log.debug("No preset for regime=%r, strategy=%r", regime, strategy)
        return RegimeAdjustments(regime=regime, regime_detected=True)

    return adjustments
