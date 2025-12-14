"""
Decision Safety Module

Provides a reusable API for checking whether actions are safe to commit
based on belief uncertainty and environment fork/commit structure.

This module wraps the core WOOEdge belief tracking and commit-gate logic
into a simple interface that can be imported by other programs.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass

from .env import GridWorld, Observation, Action, ACTION_DELTAS, EnvConfig
from .belief import ParticleFilter, BeliefConfig
from .model import TransitionModel, ModelConfig


@dataclass
class SafetyConfig:
    """Configuration for the DecisionSafety module."""
    # Belief configuration
    n_particles: int = 100
    belief_sensor_noise_std: float = 1.0
    belief_resample_threshold: float = 0.3

    # Commit gate thresholds
    entropy_threshold: float = 0.16  # Normalized entropy threshold

    # General
    seed: Optional[int] = None
    mirror_invariant: bool = True


class DecisionSafety:
    """
    Decision Safety Module for uncertainty-gated action commitment.

    This module monitors belief state and determines whether it is safe
    to commit to irreversible actions based on current uncertainty.

    For environments with fork structure (env.has_fork == True):
        - Tracks belief entropy
        - Detects when actions would enter commit zones
        - Returns DELAY when entropy is high and action would commit to trap

    For environments without fork structure:
        - Always returns ALLOW (no irreversible commitments)
        - Still reports entropy for monitoring

    Usage:
        safety = DecisionSafety(env)
        safety.reset(seed=42)

        obs = env.reset()
        safety.observe(obs)

        result = safety.propose(action)
        if result["decision"] == "ALLOW":
            obs, cost, done, info = env.step(action)
            safety.observe(obs)
    """

    def __init__(self,
                 env: GridWorld,
                 config: Optional[SafetyConfig] = None):
        """
        Initialize DecisionSafety module.

        Args:
            env: GridWorld environment to monitor
            config: Optional SafetyConfig for customization
        """
        self.env = env
        self.config = config or SafetyConfig()

        # Environment properties
        self.grid = env.get_grid_copy()
        self.grid_size = env.grid_size
        self.valid_positions = env.get_valid_positions()

        # Fork structure (from environment)
        self.has_fork = getattr(env, 'has_fork', False)
        self.commit_zone_goal: Set[Tuple[int, int]] = set()
        self.commit_zone_trap: Set[Tuple[int, int]] = set()

        if self.has_fork:
            self.commit_zone_goal = env.commit_zone_goal.copy()
            self.commit_zone_trap = env.commit_zone_trap.copy()

        # Belief state (initialized on reset)
        self.belief: Optional[ParticleFilter] = None
        self._initialized = False

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the safety module for a new episode.

        Args:
            seed: Optional random seed for reproducibility
        """
        actual_seed = seed if seed is not None else self.config.seed

        # Initialize belief
        belief_config = BeliefConfig(
            n_particles=self.config.n_particles,
            sensor_noise_std=self.config.belief_sensor_noise_std,
            resample_threshold=self.config.belief_resample_threshold,
            slip_prob=0.15,
            mirror_invariant=self.config.mirror_invariant,
            seed=actual_seed
        )

        self.belief = ParticleFilter(
            self.grid,
            self.valid_positions,
            belief_config
        )

        # Start with uniform belief (maximum uncertainty)
        self.belief.initialize_uniform()
        self._initialized = True

    def observe(self, obs: Observation) -> None:
        """
        Update belief state from observation.

        Args:
            obs: Observation from environment
        """
        if not self._initialized:
            raise RuntimeError("Must call reset() before observe()")

        self.belief.update(obs)

    def get_entropy(self) -> float:
        """Get current belief entropy."""
        if not self._initialized:
            return 1.0
        return self.belief.get_entropy()

    def get_normalized_entropy(self) -> float:
        """Get normalized belief entropy (0-1 scale approximately)."""
        if not self._initialized:
            return 1.0
        entropy = self.belief.get_entropy()
        # Normalize by max possible entropy (log of particle count)
        max_entropy = np.log(self.config.n_particles) + 1e-6
        return entropy / max_entropy

    def _would_enter_trap(self, action: int) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Check if action would move any belief particle into trap zone.

        Returns:
            (would_enter, example_position): Whether action enters trap and an example position
        """
        if not self.has_fork or not self.commit_zone_trap:
            return False, None

        # Sample positions from belief
        positions = self.belief.sample_positions(n=min(30, self.config.n_particles))

        dy, dx = ACTION_DELTAS[Action(action)]

        for pos in positions:
            next_pos = (pos[0] + dy, pos[1] + dx)
            # Check if moving from non-trap to trap
            if pos not in self.commit_zone_trap and next_pos in self.commit_zone_trap:
                return True, next_pos

        return False, None

    def _would_enter_goal(self, action: int) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Check if action would move any belief particle into goal zone.

        Returns:
            (would_enter, example_position): Whether action enters goal zone
        """
        if not self.has_fork or not self.commit_zone_goal:
            return False, None

        positions = self.belief.sample_positions(n=min(30, self.config.n_particles))

        dy, dx = ACTION_DELTAS[Action(action)]

        for pos in positions:
            next_pos = (pos[0] + dy, pos[1] + dx)
            if pos not in self.commit_zone_goal and next_pos in self.commit_zone_goal:
                return True, next_pos

        return False, None

    def propose(self, action: int) -> Dict[str, Any]:
        """
        Evaluate whether an action is safe to commit.

        Args:
            action: Proposed action (0-4: UP, DOWN, LEFT, RIGHT, STAY)

        Returns:
            Dictionary with:
                decision: "ALLOW" | "DELAY" | "ABORT"
                entropy: Current belief entropy
                reason: Human-readable explanation
                suggested_action: Alternative action if DELAY (or None)
                debug: Additional diagnostic information
        """
        if not self._initialized:
            raise RuntimeError("Must call reset() before propose()")

        entropy = self.get_entropy()
        norm_entropy = self.get_normalized_entropy()

        # Validate action
        if not isinstance(action, int) or action < 0 or action > 4:
            return {
                "decision": "ABORT",
                "entropy": entropy,
                "reason": f"Invalid action: {action}. Must be 0-4.",
                "suggested_action": None,
                "debug": {
                    "normalized_entropy": norm_entropy,
                    "has_fork": self.has_fork,
                }
            }

        # Non-fork environments: always ALLOW
        if not self.has_fork:
            return {
                "decision": "ALLOW",
                "entropy": entropy,
                "reason": "No fork structure - all actions are safe",
                "suggested_action": action,
                "debug": {
                    "normalized_entropy": norm_entropy,
                    "has_fork": False,
                }
            }

        # Fork environment: check commit gate
        would_trap, trap_pos = self._would_enter_trap(action)
        would_goal, goal_pos = self._would_enter_goal(action)

        debug = {
            "normalized_entropy": norm_entropy,
            "entropy_threshold": self.config.entropy_threshold,
            "has_fork": True,
            "would_enter_trap": would_trap,
            "would_enter_goal": would_goal,
            "trap_position": trap_pos,
            "goal_position": goal_pos,
        }

        # High entropy + would enter trap = DELAY
        if would_trap and norm_entropy > self.config.entropy_threshold:
            return {
                "decision": "DELAY",
                "entropy": entropy,
                "reason": f"High uncertainty (H={norm_entropy:.3f} > {self.config.entropy_threshold}) "
                          f"and action would enter trap zone at {trap_pos}. "
                          f"Gather more information before committing.",
                "suggested_action": Action.STAY.value,  # Suggest staying to gather info
                "debug": debug,
            }

        # Would enter trap but entropy is low = ALLOW (agent is confident)
        if would_trap and norm_entropy <= self.config.entropy_threshold:
            return {
                "decision": "ALLOW",
                "entropy": entropy,
                "reason": f"Entropy is low (H={norm_entropy:.3f} <= {self.config.entropy_threshold}). "
                          f"Agent is confident, allowing commitment.",
                "suggested_action": action,
                "debug": debug,
            }

        # Would enter goal zone = ALLOW
        if would_goal:
            return {
                "decision": "ALLOW",
                "entropy": entropy,
                "reason": f"Action would enter goal zone at {goal_pos}. Safe to proceed.",
                "suggested_action": action,
                "debug": debug,
            }

        # Not entering any commit zone = ALLOW
        return {
            "decision": "ALLOW",
            "entropy": entropy,
            "reason": "Action does not enter commit zone. Safe to proceed.",
            "suggested_action": action,
            "debug": debug,
        }
