"""
WooEdge App Interface

Base class for WooEdge applications.
Apps define their observation schema, action space, and world states.
Runtime handles uncertainty tracking, gating, and execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from enum import Enum

from .belief import BeliefState, LikelihoodModel
from .entropy import EntropyTracker, UncertaintyLevel
from .gate import ActionGate, GateResult, ActionCategory
from .bus import Observation, ObservationSource


T = TypeVar('T')  # World state type


class AppState(Enum):
    """Application lifecycle states."""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AppConfig:
    """
    Application configuration.

    Override defaults when instantiating app.
    """
    # Gating thresholds
    uncertainty_threshold: float = 0.7
    hazard_threshold: float = 0.6

    # Belief settings
    belief_history_limit: int = 100
    decay_rate: float = 0.01  # Belief decay toward uniform

    # Entropy tracking
    entropy_history_limit: int = 1000

    # Execution
    tick_interval: float = 0.1  # Seconds between ticks
    max_ticks: int = 0  # 0 = unlimited

    # Debug
    verbose: bool = False


@dataclass
class AppContext:
    """
    Runtime context passed to app methods.

    Contains current state, belief, and utilities.
    """
    tick: int
    belief: BeliefState
    entropy: EntropyTracker
    gate: ActionGate
    observation: Optional[Observation]
    config: AppConfig

    def is_certain(self) -> bool:
        """Check if belief is certain."""
        return self.entropy.current_level == UncertaintyLevel.CERTAIN

    def is_confident(self) -> bool:
        """Check if belief is at least confident."""
        level = self.entropy.current_level
        return level in (UncertaintyLevel.CERTAIN, UncertaintyLevel.CONFIDENT)

    def is_uncertain(self) -> bool:
        """Check if belief is uncertain or worse."""
        level = self.entropy.current_level
        return level in (UncertaintyLevel.UNCERTAIN, UncertaintyLevel.VERY_UNCERTAIN)

    @property
    def uncertainty(self) -> float:
        """Current normalized uncertainty."""
        return self.entropy.current_normalized

    @property
    def most_likely_state(self) -> Any:
        """Most likely world state."""
        return self.belief.most_likely


class WooEdgeApp(ABC, Generic[T]):
    """
    Base class for WooEdge applications.

    Apps extend this class and implement:
    - world_states: List of possible world states
    - action_space: List of possible actions
    - observation_schema: Dict mapping field names to types
    - compute_likelihoods: How observations update belief
    - decide: Core decision logic

    Example:
        class ThermostatApp(WooEdgeApp[str]):
            world_states = ["comfortable", "too_cold", "too_hot"]
            action_space = ["heat_on", "heat_off", "cool_on", "cool_off", "hold"]
            observation_schema = {"temperature": float, "humidity": float}

            def compute_likelihoods(self, obs, ctx):
                temp = obs.get("temperature", 70)
                return {
                    "comfortable": 1.0 if 68 <= temp <= 72 else 0.1,
                    "too_cold": 1.0 if temp < 68 else 0.1,
                    "too_hot": 1.0 if temp > 72 else 0.1,
                }

            def decide(self, ctx):
                if ctx.most_likely_state == "too_cold":
                    return "heat_on"
                elif ctx.most_likely_state == "too_hot":
                    return "cool_on"
                return "hold"
    """

    # Override in subclass
    world_states: List[T] = []
    action_space: List[str] = []
    observation_schema: Dict[str, type] = {}

    # Action categories (override to customize)
    action_categories: Dict[str, ActionCategory] = {}

    # Default action when gated
    default_action: str = "hold"

    def __init__(self, config: AppConfig = None):
        """
        Initialize app with optional config.

        Args:
            config: App configuration (uses defaults if None)
        """
        self.config = config or AppConfig()
        self.state = AppState.CREATED

        # Validate definition
        if not self.world_states:
            raise ValueError("world_states must be defined")
        if not self.action_space:
            raise ValueError("action_space must be defined")

        # Initialize runtime components
        self._belief: Optional[BeliefState[T]] = None
        self._entropy: Optional[EntropyTracker] = None
        self._gate: Optional[ActionGate] = None
        self._tick = 0

        # Hooks
        self._on_tick_handlers: List[Callable[[AppContext], None]] = []
        self._on_decision_handlers: List[Callable[[str, GateResult], None]] = []

    def initialize(self) -> None:
        """
        Initialize runtime components.

        Called automatically by engine, or manually for standalone use.
        """
        # Create belief state
        self._belief = BeliefState.uniform(
            self.world_states,
            history_limit=self.config.belief_history_limit,
        )

        # Create entropy tracker
        self._entropy = EntropyTracker(
            history_limit=self.config.entropy_history_limit,
        )

        # Create gate
        self._gate = ActionGate(
            uncertainty_threshold=self.config.uncertainty_threshold,
            hazard_threshold=self.config.hazard_threshold,
        )

        # Register action categories
        for action in self.action_space:
            category = self.action_categories.get(action, ActionCategory.COSTLY)
            self._gate.register_action(action, category)

        self._tick = 0
        self.state = AppState.INITIALIZED

        # Call setup hook
        self.on_initialize()

    def on_initialize(self) -> None:
        """
        Hook called after initialization.

        Override to perform app-specific setup.
        """
        pass

    @abstractmethod
    def compute_likelihoods(
        self,
        observation: Observation,
        context: AppContext,
    ) -> Dict[T, float]:
        """
        Compute observation likelihoods P(obs|state) for each world state.

        Args:
            observation: Current observation
            context: App context with belief, entropy, etc.

        Returns:
            Dict mapping each world state to its likelihood
        """
        pass

    @abstractmethod
    def decide(self, context: AppContext) -> str:
        """
        Make a decision given current context.

        This is the app's core logic. Returns what the app WANTS to do.
        The runtime will gate this through the ActionGate.

        Args:
            context: App context with belief, entropy, etc.

        Returns:
            Action from action_space
        """
        pass

    def compute_hazard(self, observation: Observation, context: AppContext) -> float:
        """
        Compute current hazard level.

        Override to define app-specific hazard calculation.

        Args:
            observation: Current observation
            context: App context

        Returns:
            Hazard score in [0, 1]
        """
        # Default: no hazard (override for app-specific hazard)
        return 0.0

    def on_gated(self, original_action: str, result: GateResult, context: AppContext) -> None:
        """
        Hook called when an action is gated (blocked or modified).

        Override to handle gating events.

        Args:
            original_action: What app wanted to do
            result: Gate result with decision and reasoning
            context: App context
        """
        pass

    def on_action(self, action: str, result: GateResult, context: AppContext) -> None:
        """
        Hook called after final action is determined.

        Override to handle action execution.

        Args:
            action: Final action to execute
            result: Gate result
            context: App context
        """
        pass

    def tick(self, observation: Observation) -> GateResult:
        """
        Process one tick of the app.

        1. Update belief from observation
        2. Track entropy
        3. Get app decision
        4. Gate the decision
        5. Return result

        Args:
            observation: Current observation

        Returns:
            GateResult with final action and reasoning
        """
        if self.state not in (AppState.INITIALIZED, AppState.RUNNING):
            raise RuntimeError(f"App not ready: {self.state}")

        self.state = AppState.RUNNING
        self._tick += 1

        # Build context
        context = AppContext(
            tick=self._tick,
            belief=self._belief,
            entropy=self._entropy,
            gate=self._gate,
            observation=observation,
            config=self.config,
        )

        # Update belief
        likelihoods = self.compute_likelihoods(observation, context)
        self._belief.update_bayesian(likelihoods)

        # Optional decay toward uniform
        if self.config.decay_rate > 0:
            self._belief.decay_to_uniform(self.config.decay_rate)

        # Track entropy
        self._entropy.track(
            self._belief.entropy(),
            self._belief.max_entropy(),
        )

        # Call tick handlers
        for handler in self._on_tick_handlers:
            handler(context)

        # Get app decision
        wanted_action = self.decide(context)
        if wanted_action not in self.action_space:
            wanted_action = self.default_action

        # Compute hazard
        hazard = self.compute_hazard(observation, context)

        # Gate the decision
        result = self._gate.evaluate(
            wanted_action,
            uncertainty=self._entropy.current_normalized,
            hazard=hazard,
            context={"observation": observation.data, "tick": self._tick},
        )

        # Handle gating
        if not result.allowed or result.final_action != wanted_action:
            self.on_gated(wanted_action, result, context)

        # Call action hook
        self.on_action(result.final_action, result, context)

        # Call decision handlers
        for handler in self._on_decision_handlers:
            handler(result.final_action, result)

        return result

    def get_belief(self) -> BeliefState[T]:
        """Get current belief state."""
        return self._belief

    def get_entropy(self) -> EntropyTracker:
        """Get entropy tracker."""
        return self._entropy

    def get_gate(self) -> ActionGate:
        """Get action gate."""
        return self._gate

    def on_tick(self, handler: Callable[[AppContext], None]) -> None:
        """Register tick handler."""
        self._on_tick_handlers.append(handler)

    def on_decision(self, handler: Callable[[str, GateResult], None]) -> None:
        """Register decision handler."""
        self._on_decision_handlers.append(handler)

    def pause(self) -> None:
        """Pause the app."""
        self.state = AppState.PAUSED

    def resume(self) -> None:
        """Resume the app."""
        if self.state == AppState.PAUSED:
            self.state = AppState.RUNNING

    def stop(self) -> None:
        """Stop the app."""
        self.state = AppState.STOPPED

    def reset(self) -> None:
        """Reset app state."""
        self._tick = 0
        self._belief = BeliefState.uniform(
            self.world_states,
            history_limit=self.config.belief_history_limit,
        )
        self._entropy.reset()
        self.state = AppState.INITIALIZED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize app state."""
        return {
            "state": self.state.value,
            "tick": self._tick,
            "belief": self._belief.to_dict() if self._belief else None,
            "entropy": self._entropy.to_dict() if self._entropy else None,
            "config": {
                "uncertainty_threshold": self.config.uncertainty_threshold,
                "hazard_threshold": self.config.hazard_threshold,
            },
        }


class SimpleApp(WooEdgeApp[str]):
    """
    Simple app for quick prototyping.

    Define states, actions, and logic via constructor instead of subclassing.

    Example:
        app = SimpleApp(
            states=["on", "off"],
            actions=["turn_on", "turn_off", "hold"],
            schema={"switch": bool},
            likelihood_fn=lambda obs, ctx: {
                "on": 1.0 if obs.get("switch") else 0.1,
                "off": 0.1 if obs.get("switch") else 1.0,
            },
            decide_fn=lambda ctx: "turn_on" if ctx.most_likely_state == "off" else "hold",
        )
    """

    def __init__(
        self,
        states: List[str],
        actions: List[str],
        schema: Dict[str, type],
        likelihood_fn: Callable[[Observation, AppContext], Dict[str, float]],
        decide_fn: Callable[[AppContext], str],
        hazard_fn: Callable[[Observation, AppContext], float] = None,
        config: AppConfig = None,
    ):
        # Set class attributes before super().__init__
        self.world_states = states
        self.action_space = actions
        self.observation_schema = schema
        self._likelihood_fn = likelihood_fn
        self._decide_fn = decide_fn
        self._hazard_fn = hazard_fn

        super().__init__(config)

    def compute_likelihoods(
        self,
        observation: Observation,
        context: AppContext,
    ) -> Dict[str, float]:
        return self._likelihood_fn(observation, context)

    def decide(self, context: AppContext) -> str:
        return self._decide_fn(context)

    def compute_hazard(self, observation: Observation, context: AppContext) -> float:
        if self._hazard_fn:
            return self._hazard_fn(observation, context)
        return 0.0
