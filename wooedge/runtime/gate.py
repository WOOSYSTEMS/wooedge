"""
Action Gate

Universal action gating for WooEdge runtime.
Gates actions based on uncertainty, hazard, and configurable rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic
from enum import Enum
from abc import ABC, abstractmethod

from .entropy import UncertaintyLevel


class GateDecision(Enum):
    """Gate decision outcomes."""
    ALLOW = "allow"
    BLOCK = "block"
    DELAY = "delay"
    MODIFY = "modify"  # Allow with modifications (e.g., reduced size)


class ActionCategory(Enum):
    """Categories of actions with different risk profiles."""
    OBSERVE = "observe"      # Safe: just gathering information
    REVERSIBLE = "reversible"  # Low risk: can be undone
    COSTLY = "costly"        # Medium risk: has costs but recoverable
    IRREVERSIBLE = "irreversible"  # High risk: cannot be undone


@dataclass
class GateResult:
    """Result of action gating."""
    decision: GateDecision
    original_action: str
    final_action: str
    reason: str
    uncertainty: float
    hazard: float
    modifiers: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        """Whether action is allowed (possibly modified)."""
        return self.decision in (GateDecision.ALLOW, GateDecision.MODIFY)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate result."""
        return {
            "decision": self.decision.value,
            "original_action": self.original_action,
            "final_action": self.final_action,
            "reason": self.reason,
            "uncertainty": self.uncertainty,
            "hazard": self.hazard,
            "modifiers": self.modifiers,
            "allowed": self.allowed,
        }


@dataclass
class GateRule:
    """A single gating rule."""
    name: str
    condition: Callable[[float, float, str, Dict[str, Any]], bool]
    decision: GateDecision
    reason_template: str
    priority: int = 0  # Higher = checked first


@dataclass
class ActionGate:
    """
    Universal action gating layer.

    Gates actions based on:
    - Uncertainty level (entropy)
    - Hazard score (risk)
    - Action category (reversibility)
    - Custom rules

    Example:
        gate = ActionGate()
        gate.configure(
            uncertainty_threshold=0.7,
            hazard_threshold=0.6,
        )
        result = gate.evaluate("buy", uncertainty=0.8, hazard=0.3)
        if result.allowed:
            execute(result.final_action, **result.modifiers)
    """

    # Thresholds
    uncertainty_threshold: float = 0.7
    hazard_threshold: float = 0.6
    combined_threshold: float = 1.0  # uncertainty + hazard

    # Category-specific thresholds (more restrictive for irreversible)
    category_thresholds: Dict[ActionCategory, float] = field(default_factory=lambda: {
        ActionCategory.OBSERVE: 1.0,       # Always allow observation
        ActionCategory.REVERSIBLE: 0.9,
        ActionCategory.COSTLY: 0.7,
        ActionCategory.IRREVERSIBLE: 0.5,
    })

    # Action-to-category mapping
    action_categories: Dict[str, ActionCategory] = field(default_factory=dict)

    # Custom rules
    rules: List[GateRule] = field(default_factory=list)

    # Fallback actions per category
    fallback_actions: Dict[ActionCategory, str] = field(default_factory=lambda: {
        ActionCategory.OBSERVE: "observe",
        ActionCategory.REVERSIBLE: "hold",
        ActionCategory.COSTLY: "hold",
        ActionCategory.IRREVERSIBLE: "scan",
    })

    # Default fallback
    default_fallback: str = "hold"

    # Size scaling (for actions with magnitude)
    min_size_multiplier: float = 0.25
    size_scale_with_confidence: bool = True

    def configure(
        self,
        uncertainty_threshold: float = None,
        hazard_threshold: float = None,
        combined_threshold: float = None,
        min_size_multiplier: float = None,
    ) -> None:
        """Configure gate thresholds."""
        if uncertainty_threshold is not None:
            self.uncertainty_threshold = uncertainty_threshold
        if hazard_threshold is not None:
            self.hazard_threshold = hazard_threshold
        if combined_threshold is not None:
            self.combined_threshold = combined_threshold
        if min_size_multiplier is not None:
            self.min_size_multiplier = min_size_multiplier

    def register_action(self, action: str, category: ActionCategory) -> None:
        """Register an action with its risk category."""
        self.action_categories[action] = category

    def register_actions(self, actions: Dict[str, ActionCategory]) -> None:
        """Register multiple actions with categories."""
        self.action_categories.update(actions)

    def add_rule(self, rule: GateRule) -> None:
        """Add a custom gating rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)  # Higher priority first

    def evaluate(
        self,
        action: str,
        uncertainty: float,
        hazard: float,
        context: Dict[str, Any] = None,
    ) -> GateResult:
        """
        Evaluate whether an action should be allowed.

        Args:
            action: The proposed action
            uncertainty: Current uncertainty level (0-1 normalized)
            hazard: Current hazard score (0-1)
            context: Additional context for custom rules

        Returns:
            GateResult with decision and reasoning
        """
        context = context or {}
        category = self.action_categories.get(action, ActionCategory.COSTLY)

        # Check custom rules first
        for rule in self.rules:
            if rule.condition(uncertainty, hazard, action, context):
                return GateResult(
                    decision=rule.decision,
                    original_action=action,
                    final_action=self._get_fallback(category) if rule.decision == GateDecision.BLOCK else action,
                    reason=rule.reason_template.format(
                        uncertainty=uncertainty,
                        hazard=hazard,
                        action=action,
                        **context,
                    ),
                    uncertainty=uncertainty,
                    hazard=hazard,
                )

        # Always allow observation actions
        if category == ActionCategory.OBSERVE:
            return GateResult(
                decision=GateDecision.ALLOW,
                original_action=action,
                final_action=action,
                reason="OBSERVE_ALLOWED",
                uncertainty=uncertainty,
                hazard=hazard,
            )

        # Check hazard threshold
        if hazard > self.hazard_threshold:
            return GateResult(
                decision=GateDecision.BLOCK,
                original_action=action,
                final_action=self._get_fallback(category),
                reason=f"HIGH_HAZARD: {hazard:.2f} > {self.hazard_threshold:.2f}",
                uncertainty=uncertainty,
                hazard=hazard,
            )

        # Check uncertainty threshold
        if uncertainty > self.uncertainty_threshold:
            return GateResult(
                decision=GateDecision.DELAY,
                original_action=action,
                final_action="scan" if "scan" in self.action_categories else self._get_fallback(category),
                reason=f"HIGH_UNCERTAINTY: {uncertainty:.2f} > {self.uncertainty_threshold:.2f}",
                uncertainty=uncertainty,
                hazard=hazard,
            )

        # Check combined threshold
        combined = uncertainty + hazard
        if combined > self.combined_threshold:
            return GateResult(
                decision=GateDecision.DELAY,
                original_action=action,
                final_action=self._get_fallback(category),
                reason=f"HIGH_COMBINED: {combined:.2f} > {self.combined_threshold:.2f}",
                uncertainty=uncertainty,
                hazard=hazard,
            )

        # Check category-specific threshold
        cat_threshold = self.category_thresholds.get(category, 0.7)
        if uncertainty > cat_threshold:
            return GateResult(
                decision=GateDecision.DELAY,
                original_action=action,
                final_action=self._get_fallback(category),
                reason=f"CATEGORY_THRESHOLD: {uncertainty:.2f} > {cat_threshold:.2f} for {category.value}",
                uncertainty=uncertainty,
                hazard=hazard,
            )

        # Compute size modifier if scaling is enabled
        modifiers = {}
        if self.size_scale_with_confidence:
            size_mult = self._compute_size_multiplier(uncertainty)
            modifiers["size_multiplier"] = size_mult

        # Allow (possibly modified)
        decision = GateDecision.MODIFY if modifiers else GateDecision.ALLOW
        return GateResult(
            decision=decision,
            original_action=action,
            final_action=action,
            reason=self._get_allow_reason(uncertainty, hazard, category),
            uncertainty=uncertainty,
            hazard=hazard,
            modifiers=modifiers,
        )

    def _get_fallback(self, category: ActionCategory) -> str:
        """Get fallback action for a category."""
        return self.fallback_actions.get(category, self.default_fallback)

    def _compute_size_multiplier(self, uncertainty: float) -> float:
        """Compute position size multiplier based on uncertainty."""
        # Full size at 0 uncertainty, min_size at threshold
        if uncertainty <= 0:
            return 1.0
        norm = min(1.0, uncertainty / self.uncertainty_threshold)
        mult = 1.0 - norm * (1.0 - self.min_size_multiplier)
        return max(self.min_size_multiplier, mult)

    def _get_allow_reason(self, uncertainty: float, hazard: float, category: ActionCategory) -> str:
        """Generate reason string for allowed action."""
        conf = 1.0 - uncertainty
        return f"ALLOWED: conf={conf:.0%}, hazard={hazard:.2f}, cat={category.value}"

    def reset(self) -> None:
        """Reset gate to defaults."""
        self.rules.clear()
        self.action_categories.clear()


class GatePolicy(ABC):
    """
    Abstract base for custom gating policies.

    Apps can implement custom policies beyond simple thresholds.
    """

    @abstractmethod
    def evaluate(
        self,
        action: str,
        uncertainty: float,
        hazard: float,
        context: Dict[str, Any],
    ) -> GateResult:
        """Evaluate action against policy."""
        pass


class ConservativePolicy(GatePolicy):
    """
    Conservative gating policy.

    Only allows actions when very confident and low hazard.
    """

    def __init__(self, uncertainty_max: float = 0.3, hazard_max: float = 0.3):
        self.uncertainty_max = uncertainty_max
        self.hazard_max = hazard_max

    def evaluate(
        self,
        action: str,
        uncertainty: float,
        hazard: float,
        context: Dict[str, Any],
    ) -> GateResult:
        if uncertainty > self.uncertainty_max or hazard > self.hazard_max:
            return GateResult(
                decision=GateDecision.BLOCK,
                original_action=action,
                final_action="hold",
                reason=f"CONSERVATIVE_BLOCK: u={uncertainty:.2f}, h={hazard:.2f}",
                uncertainty=uncertainty,
                hazard=hazard,
            )
        return GateResult(
            decision=GateDecision.ALLOW,
            original_action=action,
            final_action=action,
            reason="CONSERVATIVE_ALLOW",
            uncertainty=uncertainty,
            hazard=hazard,
        )


class AdaptivePolicy(GatePolicy):
    """
    Adaptive gating policy.

    Adjusts thresholds based on recent history.
    """

    def __init__(
        self,
        base_uncertainty: float = 0.6,
        base_hazard: float = 0.5,
        adapt_rate: float = 0.1,
    ):
        self.base_uncertainty = base_uncertainty
        self.base_hazard = base_hazard
        self.adapt_rate = adapt_rate
        self.current_uncertainty_thresh = base_uncertainty
        self.current_hazard_thresh = base_hazard
        self.recent_outcomes: List[bool] = []  # True = good, False = bad

    def record_outcome(self, success: bool) -> None:
        """Record outcome of an allowed action."""
        self.recent_outcomes.append(success)
        if len(self.recent_outcomes) > 50:
            self.recent_outcomes.pop(0)

        # Adapt thresholds
        if len(self.recent_outcomes) >= 10:
            success_rate = sum(self.recent_outcomes[-10:]) / 10
            if success_rate < 0.5:
                # Tighten thresholds
                self.current_uncertainty_thresh *= (1 - self.adapt_rate)
                self.current_hazard_thresh *= (1 - self.adapt_rate)
            elif success_rate > 0.8:
                # Relax thresholds (toward base)
                self.current_uncertainty_thresh += self.adapt_rate * (self.base_uncertainty - self.current_uncertainty_thresh)
                self.current_hazard_thresh += self.adapt_rate * (self.base_hazard - self.current_hazard_thresh)

    def evaluate(
        self,
        action: str,
        uncertainty: float,
        hazard: float,
        context: Dict[str, Any],
    ) -> GateResult:
        if uncertainty > self.current_uncertainty_thresh:
            return GateResult(
                decision=GateDecision.DELAY,
                original_action=action,
                final_action="scan",
                reason=f"ADAPTIVE_DELAY: u={uncertainty:.2f} > {self.current_uncertainty_thresh:.2f}",
                uncertainty=uncertainty,
                hazard=hazard,
            )
        if hazard > self.current_hazard_thresh:
            return GateResult(
                decision=GateDecision.BLOCK,
                original_action=action,
                final_action="hold",
                reason=f"ADAPTIVE_BLOCK: h={hazard:.2f} > {self.current_hazard_thresh:.2f}",
                uncertainty=uncertainty,
                hazard=hazard,
            )
        return GateResult(
            decision=GateDecision.ALLOW,
            original_action=action,
            final_action=action,
            reason="ADAPTIVE_ALLOW",
            uncertainty=uncertainty,
            hazard=hazard,
        )
