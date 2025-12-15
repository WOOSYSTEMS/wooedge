"""
Entropy Tracker

Universal uncertainty quantification for WooEdge runtime.
Tracks entropy across belief states and provides uncertainty metrics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class UncertaintyLevel(Enum):
    """Categorical uncertainty levels."""
    CERTAIN = "certain"         # H < 0.2 (normalized)
    CONFIDENT = "confident"     # H < 0.4
    MODERATE = "moderate"       # H < 0.6
    UNCERTAIN = "uncertain"     # H < 0.8
    VERY_UNCERTAIN = "very_uncertain"  # H >= 0.8


@dataclass
class EntropySnapshot:
    """Point-in-time entropy measurement."""
    timestamp: int
    entropy: float
    normalized: float
    level: UncertaintyLevel
    trend: float  # Positive = increasing uncertainty


@dataclass
class EntropyTracker:
    """
    Tracks uncertainty over time across belief states.

    Provides:
    - Current entropy measurement
    - Entropy history and trends
    - Categorical uncertainty levels
    - Alerts for uncertainty spikes
    """

    # Configuration
    history_limit: int = 1000
    spike_threshold: float = 0.3  # Alert if entropy jumps by this much
    trend_window: int = 10

    # State
    history: List[EntropySnapshot] = field(default_factory=list)
    _tick: int = 0

    # Level thresholds (normalized entropy)
    LEVEL_THRESHOLDS: Dict[UncertaintyLevel, float] = field(default_factory=lambda: {
        UncertaintyLevel.CERTAIN: 0.2,
        UncertaintyLevel.CONFIDENT: 0.4,
        UncertaintyLevel.MODERATE: 0.6,
        UncertaintyLevel.UNCERTAIN: 0.8,
        UncertaintyLevel.VERY_UNCERTAIN: 1.0,
    })

    def track(self, entropy: float, max_entropy: float) -> EntropySnapshot:
        """
        Record an entropy measurement.

        Args:
            entropy: Raw entropy value (nats)
            max_entropy: Maximum possible entropy for the state space

        Returns:
            EntropySnapshot with analysis
        """
        self._tick += 1

        # Normalize
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        normalized = min(1.0, max(0.0, normalized))

        # Determine level
        level = self._get_level(normalized)

        # Compute trend
        trend = self._compute_trend()

        # Create snapshot
        snapshot = EntropySnapshot(
            timestamp=self._tick,
            entropy=entropy,
            normalized=normalized,
            level=level,
            trend=trend,
        )

        # Store history
        self.history.append(snapshot)
        if len(self.history) > self.history_limit:
            self.history.pop(0)

        return snapshot

    def _get_level(self, normalized: float) -> UncertaintyLevel:
        """Map normalized entropy to categorical level."""
        if normalized < self.LEVEL_THRESHOLDS[UncertaintyLevel.CERTAIN]:
            return UncertaintyLevel.CERTAIN
        elif normalized < self.LEVEL_THRESHOLDS[UncertaintyLevel.CONFIDENT]:
            return UncertaintyLevel.CONFIDENT
        elif normalized < self.LEVEL_THRESHOLDS[UncertaintyLevel.MODERATE]:
            return UncertaintyLevel.MODERATE
        elif normalized < self.LEVEL_THRESHOLDS[UncertaintyLevel.UNCERTAIN]:
            return UncertaintyLevel.UNCERTAIN
        else:
            return UncertaintyLevel.VERY_UNCERTAIN

    def _compute_trend(self) -> float:
        """Compute entropy trend from recent history."""
        if len(self.history) < 2:
            return 0.0

        window = min(self.trend_window, len(self.history))
        recent = [s.normalized for s in self.history[-window:]]

        # Linear regression slope
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))

        return num / den if den > 0 else 0.0

    @property
    def current(self) -> Optional[EntropySnapshot]:
        """Get most recent snapshot."""
        return self.history[-1] if self.history else None

    @property
    def current_level(self) -> UncertaintyLevel:
        """Get current uncertainty level."""
        if not self.history:
            return UncertaintyLevel.VERY_UNCERTAIN
        return self.history[-1].level

    @property
    def current_normalized(self) -> float:
        """Get current normalized entropy."""
        if not self.history:
            return 1.0
        return self.history[-1].normalized

    def is_spike(self) -> bool:
        """Check if there was a recent entropy spike."""
        if len(self.history) < 2:
            return False
        prev = self.history[-2].normalized
        curr = self.history[-1].normalized
        return (curr - prev) > self.spike_threshold

    def is_converging(self) -> bool:
        """Check if entropy is trending down (converging)."""
        if not self.history:
            return False
        return self.history[-1].trend < -0.01

    def is_diverging(self) -> bool:
        """Check if entropy is trending up (diverging)."""
        if not self.history:
            return False
        return self.history[-1].trend > 0.01

    def time_at_level(self, level: UncertaintyLevel) -> int:
        """Count consecutive ticks at or above a level."""
        count = 0
        for snapshot in reversed(self.history):
            if self._level_value(snapshot.level) >= self._level_value(level):
                count += 1
            else:
                break
        return count

    def _level_value(self, level: UncertaintyLevel) -> int:
        """Get numeric value for level comparison."""
        order = [
            UncertaintyLevel.CERTAIN,
            UncertaintyLevel.CONFIDENT,
            UncertaintyLevel.MODERATE,
            UncertaintyLevel.UNCERTAIN,
            UncertaintyLevel.VERY_UNCERTAIN,
        ]
        return order.index(level)

    def average_entropy(self, window: int = 50) -> float:
        """Get average normalized entropy over recent history."""
        if not self.history:
            return 1.0
        recent = self.history[-min(window, len(self.history)):]
        return sum(s.normalized for s in recent) / len(recent)

    def reset(self) -> None:
        """Reset tracker state."""
        self.history.clear()
        self._tick = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state."""
        current = self.current
        return {
            "tick": self._tick,
            "current_entropy": current.entropy if current else None,
            "current_normalized": current.normalized if current else None,
            "current_level": current.level.value if current else None,
            "trend": current.trend if current else None,
            "is_spike": self.is_spike(),
            "is_converging": self.is_converging(),
            "is_diverging": self.is_diverging(),
            "history_length": len(self.history),
        }


def compute_entropy(distribution: Dict[Any, float]) -> float:
    """
    Compute Shannon entropy of a distribution.

    Args:
        distribution: Probability distribution (values should sum to 1)

    Returns:
        Entropy in nats
    """
    h = 0.0
    for p in distribution.values():
        if p > 0:
            h -= p * math.log(p)
    return h


def compute_max_entropy(n_states: int) -> float:
    """
    Compute maximum entropy for n states (uniform distribution).

    Args:
        n_states: Number of possible states

    Returns:
        Maximum entropy in nats
    """
    if n_states <= 1:
        return 0.0
    return math.log(n_states)


def compute_kl_divergence(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    """
    Compute KL divergence D_KL(P || Q).

    Measures how much P diverges from Q.

    Args:
        p: "True" distribution
        q: Reference distribution

    Returns:
        KL divergence (always >= 0)
    """
    kl = 0.0
    for key in p:
        p_val = p[key]
        q_val = q.get(key, 1e-10)  # Small floor to avoid log(0)
        if p_val > 0:
            kl += p_val * math.log(p_val / q_val)
    return kl


def compute_js_divergence(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    """
    Compute Jensen-Shannon divergence.

    Symmetric measure of distribution difference.

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        JS divergence in [0, ln(2)]
    """
    # Compute mixture M = (P + Q) / 2
    all_keys = set(p.keys()) | set(q.keys())
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in all_keys}

    return (compute_kl_divergence(p, m) + compute_kl_divergence(q, m)) / 2
