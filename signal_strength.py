"""Shared signal-strength saturation function (Master Spec § 10 S3).

Problem: MR/TF/VB signal generators previously computed
    raw_str = <several multipliers, each individually >= 1.0 in the common case>
    strength = min(raw_str, 1.0)

Since raw_str is a product of several factors that are frequently >= 1.0
(sentiment agreement, contrarian bonus, sector tailwind, regime sizing), the
hard `min(..., 1.0)` cap collapsed the majority of BUY signals to exactly 1.0.
Downstream, the harness's `confidence_weighted` reconciler saw near-uniform
1.0 inputs, so the `min_confidence_to_act` gate (default 0.55) filtered
almost nothing — confidence carried no information for allocation sizing or
the future A3 meta-labeler.

Fix: replace the hard cap with a smooth, monotonic saturating function that
maps (0, inf) -> (0, 1) without a flat ceiling:

    strength = raw_str / (1.0 + raw_str)

This preserves relative ordering between signals (a raw_str of 4.0 still
scores meaningfully higher than a raw_str of 1.2) while keeping the output
bounded in [0, 1) for use as a confidence/position-sizing scalar. At
raw_str == 1.0, strength == 0.5, so the effective bar to clear the default
0.55 min_confidence_to_act gate is raw_str >= 1.222 — a real filter, not a
rubber stamp.
"""
from __future__ import annotations


def saturate(raw_strength: float) -> float:
    """Smoothly saturate a non-negative raw strength score into [0, 1).

    Args:
        raw_strength: Product of signal-quality multipliers; expected >= 0.
                      Negative inputs are clamped to 0 before saturation.

    Returns:
        raw_strength / (1 + raw_strength), bounded in [0, 1).
    """
    r = max(raw_strength, 0.0)
    return r / (1.0 + r)
