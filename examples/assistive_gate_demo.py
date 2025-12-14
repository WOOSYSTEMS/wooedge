#!/usr/bin/env python3
"""
Assistive Navigation Gate Demo

Demonstrates DecisionSafety in an assistive navigation context
(e.g., smart cane or wheelchair guidance system).

The agent must navigate a corridor with a fork:
- One path leads to a safe exit (goal)
- The other leads to a hazard (stairs/drop-off)

DecisionSafety gates irreversible commitments when uncertainty is high,
prompting the agent to SCAN (probe with cane) before committing.

Run with:
    python examples/assistive_gate_demo.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Set, Tuple, Optional

from wooedge.envs.assistive_nav import (
    AssistiveNavEnv, AssistiveNavConfig, AssistiveAction, AssistiveObservation
)


@dataclass
class AssistiveSafetyConfig:
    """Configuration for assistive navigation safety checker."""
    entropy_threshold: float = 0.3  # Higher threshold for this environment
    min_observations: int = 3       # Minimum observations before allowing commit
    hazard_hint_threshold: float = 0.5  # Hazard hint level to trigger caution


class AssistiveDecisionSafety:
    """
    Decision Safety module specialized for AssistiveNavEnv.

    Uses observation history and entropy estimation to determine
    whether it's safe to commit to irreversible actions.
    """

    def __init__(self, env: AssistiveNavEnv, config: Optional[AssistiveSafetyConfig] = None):
        self.env = env
        self.config = config or AssistiveSafetyConfig()

        # Environment properties
        self.has_fork = env.has_fork
        self.commit_zone_goal = env.commit_zone_goal
        self.commit_zone_trap = env.commit_zone_trap

        # Tracking
        self.observation_history = []
        self.position_belief: Dict[Tuple[int, int], float] = {}
        self._initialized = False

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset for a new episode."""
        self.observation_history = []
        # Initialize uniform belief over valid positions
        self.position_belief = {
            pos: 1.0 / len(self.env.valid_positions)
            for pos in self.env.valid_positions
        }
        self._initialized = True

    def observe(self, obs: AssistiveObservation) -> None:
        """Update belief from observation."""
        if not self._initialized:
            raise RuntimeError("Must call reset() before observe()")

        self.observation_history.append(obs)

        # Update belief based on observation
        # Use hazard hint to update belief about positions near hazard
        self._update_belief_from_obs(obs)

    def _update_belief_from_obs(self, obs: AssistiveObservation) -> None:
        """Update belief based on observation likelihood."""
        # Scanning gives more information
        info_gain = 0.3 if obs.scanned else 0.1

        # Use hazard hint to adjust beliefs
        hazard_pos = self.env.hazard_pos
        for pos in self.position_belief:
            dist_to_hazard = abs(pos[0] - hazard_pos[0]) + abs(pos[1] - hazard_pos[1])

            # If hazard hint is high, positions near hazard are more likely
            # If hazard hint is low, positions far from hazard are more likely
            if obs.hazard_hint > 0.5:
                # Near hazard is more likely
                if dist_to_hazard <= 2:
                    self.position_belief[pos] *= (1 + info_gain * obs.hazard_hint)
                else:
                    self.position_belief[pos] *= (1 - info_gain * 0.3)
            else:
                # Far from hazard is more likely
                if dist_to_hazard > 2:
                    self.position_belief[pos] *= (1 + info_gain * (1 - obs.hazard_hint))
                else:
                    self.position_belief[pos] *= (1 - info_gain * 0.3)

        # Renormalize
        total = sum(self.position_belief.values())
        if total > 0:
            for pos in self.position_belief:
                self.position_belief[pos] /= total

    def get_entropy(self) -> float:
        """Get current belief entropy."""
        if not self._initialized:
            return 1.0
        probs = np.array(list(self.position_belief.values()))
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs + 1e-10))

    def get_normalized_entropy(self) -> float:
        """Get normalized entropy (0-1 scale)."""
        entropy = self.get_entropy()
        max_entropy = np.log(len(self.env.valid_positions))
        return entropy / (max_entropy + 1e-10)

    def _would_enter_commit_zone(self, action: int) -> Tuple[bool, bool, str]:
        """
        Check if action would enter a commit zone.

        Returns:
            (would_enter_trap, would_enter_goal, zone_name)
        """
        if action != AssistiveAction.FORWARD:
            return False, False, ""

        # Get next position if moving forward
        from wooedge.envs.assistive_nav import HEADING_DELTAS
        dy, dx = HEADING_DELTAS[self.env.agent_heading]
        next_pos = (self.env.agent_pos[0] + dy, self.env.agent_pos[1] + dx)

        # Check current position is not already in zone
        curr_in_trap = self.env.agent_pos in self.commit_zone_trap
        curr_in_goal = self.env.agent_pos in self.commit_zone_goal

        would_trap = (not curr_in_trap and next_pos in self.commit_zone_trap)
        would_goal = (not curr_in_goal and next_pos in self.commit_zone_goal)

        zone = ""
        if would_trap:
            zone = "hazard zone"
        elif would_goal:
            zone = "safe exit zone"

        return would_trap, would_goal, zone

    def propose(self, action: int) -> Dict[str, Any]:
        """
        Evaluate whether an action is safe to commit.

        Returns:
            Dictionary with decision, entropy, reason, suggested_action, debug
        """
        if not self._initialized:
            raise RuntimeError("Must call reset() before propose()")

        entropy = self.get_entropy()
        norm_entropy = self.get_normalized_entropy()
        obs_count = len(self.observation_history)
        latest_obs = self.observation_history[-1] if self.observation_history else None

        # Get hazard hint from latest observation
        hazard_hint = latest_obs.hazard_hint if latest_obs else 0.0

        debug = {
            "normalized_entropy": norm_entropy,
            "entropy_threshold": self.config.entropy_threshold,
            "observation_count": obs_count,
            "hazard_hint": hazard_hint,
            "has_fork": self.has_fork,
            "current_pos": self.env.agent_pos,
            "heading": self.env.agent_heading,
        }

        # Validate action
        if action < 0 or action > 3:
            return {
                "decision": "ABORT",
                "entropy": entropy,
                "reason": f"Invalid action: {action}. Must be 0-3.",
                "suggested_action": None,
                "debug": debug,
            }

        # If no fork structure, always allow
        if not self.has_fork:
            return {
                "decision": "ALLOW",
                "entropy": entropy,
                "reason": "No fork structure - all actions safe",
                "suggested_action": action,
                "debug": debug,
            }

        # Check if action would enter commit zone
        would_trap, would_goal, zone_name = self._would_enter_commit_zone(action)
        debug["would_enter_trap"] = would_trap
        debug["would_enter_goal"] = would_goal

        # High hazard hint should trigger caution
        hazard_caution = hazard_hint > self.config.hazard_hint_threshold

        # Check if agent is in "hazard corridor" (near hazard, left side of map)
        from wooedge.envs.assistive_nav import Heading, HEADING_DELTAS
        y, x = self.env.agent_pos
        hazard_y, hazard_x = self.env.hazard_pos
        in_hazard_corridor = (x <= hazard_x + 1 and abs(y - hazard_y) <= 2)
        heading = self.env.agent_heading

        # Check if there's a wall blocking eastward movement
        east_blocked = False
        if x < self.env.grid_width - 1:
            east_blocked = self.env.grid[y, x + 1] == self.env.WALL

        # Gate turning TOWARD hazard when in danger corridor
        # This prevents the naive policy from undoing our escape maneuver
        if in_hazard_corridor and x < self.env.goal_pos[1]:
            # Agent is in hazard corridor and hasn't cleared it yet
            if action == AssistiveAction.RIGHT and heading == Heading.EAST:
                # Trying to turn back toward hazard (south)
                if east_blocked:
                    # Wall ahead - need to go north to clear it
                    return {
                        "decision": "DELAY",
                        "entropy": entropy,
                        "reason": "Wall blocks east - turning LEFT (north) to navigate around.",
                        "suggested_action": AssistiveAction.LEFT.value,
                        "debug": debug,
                    }
                else:
                    return {
                        "decision": "DELAY",
                        "entropy": entropy,
                        "reason": "In hazard corridor - continue EAST to clear danger zone.",
                        "suggested_action": AssistiveAction.FORWARD.value,
                        "debug": debug,
                    }
            if action == AssistiveAction.LEFT and heading == Heading.NORTH:
                # Trying to turn west toward hazard
                return {
                    "decision": "DELAY",
                    "entropy": entropy,
                    "reason": "In hazard corridor - turn RIGHT to face east.",
                    "suggested_action": AssistiveAction.RIGHT.value,
                    "debug": debug,
                }
            # When facing NORTH in corridor with wall blocking east, continue north
            if heading == Heading.NORTH and east_blocked and action == AssistiveAction.RIGHT:
                # Don't turn back east if wall still blocks - go north first
                return {
                    "decision": "DELAY",
                    "entropy": entropy,
                    "reason": "Wall blocks east - continue NORTH to clear wall row.",
                    "suggested_action": AssistiveAction.FORWARD.value,
                    "debug": debug,
                }
            # When facing EAST with wall ahead, suggest going north
            if heading == Heading.EAST and east_blocked:
                return {
                    "decision": "DELAY",
                    "entropy": entropy,
                    "reason": "Wall blocks east - turning LEFT (north) to navigate around.",
                    "suggested_action": AssistiveAction.LEFT.value,
                    "debug": debug,
                }

        # Decision logic
        if would_trap:
            # Would enter hazard zone - suggest navigating toward safe side (east)
            from wooedge.envs.assistive_nav import Heading
            heading = self.env.agent_heading

            # Determine best escape action: prioritize going east (away from hazard)
            if heading == Heading.EAST:
                # Already facing east, just go forward
                suggested = AssistiveAction.FORWARD.value
                reason = "Hazard ahead - continuing EAST to navigate around."
            elif heading == Heading.SOUTH:
                # Turn left to face east
                suggested = AssistiveAction.LEFT.value
                reason = "Hazard ahead - turning LEFT to face east and navigate around."
            elif heading == Heading.WEST:
                # Turn right twice to face east, but suggest right first
                suggested = AssistiveAction.RIGHT.value
                reason = "Hazard zone nearby - turning RIGHT toward safety."
            else:  # NORTH
                # Turn right to face east
                suggested = AssistiveAction.RIGHT.value
                reason = "Hazard zone nearby - turning RIGHT to face east."

            return {
                "decision": "DELAY",
                "entropy": entropy,
                "reason": reason,
                "suggested_action": suggested,
                "debug": debug,
            }

        elif would_goal:
            # Would enter goal zone - generally safe
            return {
                "decision": "ALLOW",
                "entropy": entropy,
                "reason": f"Action would enter {zone_name}. Safe to proceed.",
                "suggested_action": action,
                "debug": debug,
            }

        else:
            # Not entering any commit zone
            return {
                "decision": "ALLOW",
                "entropy": entropy,
                "reason": "Action does not enter commit zone. Safe to proceed.",
                "suggested_action": action,
                "debug": debug,
            }


def run_episode(env: AssistiveNavEnv, safety: AssistiveDecisionSafety,
                gated: bool = True, max_steps: int = 50, verbose: bool = False,
                use_naive_policy: bool = False) -> Dict[str, Any]:
    """
    Run one episode with optional safety gating.

    Args:
        env: The environment
        safety: Safety checker
        gated: Whether to use safety gating
        max_steps: Maximum steps
        verbose: Print step details
        use_naive_policy: Use naive policy (for demo) vs smart policy

    Returns:
        Episode results
    """
    obs = env.reset()
    safety.reset()
    safety.observe(obs)

    ACTION_NAMES = ["FORWARD", "LEFT", "RIGHT", "SCAN"]
    policy_fn = naive_navigation_policy if use_naive_policy else simple_navigation_policy

    total_cost = 0
    delays = 0
    scans = 0

    for step in range(max_steps):
        # Get action from selected policy
        action = policy_fn(env)

        if gated:
            result = safety.propose(action)

            if verbose:
                print(f"Step {step}: pos={env.agent_pos}, heading={env.agent_heading.name}, "
                      f"action={ACTION_NAMES[action]}, decision={result['decision']}")
                if result['decision'] != 'ALLOW':
                    print(f"  Reason: {result['reason']}")

            if result['decision'] == 'DELAY':
                # Use suggested action (SCAN)
                action = result['suggested_action']
                delays += 1

        if action == AssistiveAction.SCAN:
            scans += 1

        obs, cost, done, info = env.step(action)
        safety.observe(obs)
        total_cost += cost

        if done:
            break

    return {
        "success": info.get("at_goal", False),
        "hazard": info.get("at_hazard", False),
        "steps": step + 1,
        "cost": total_cost,
        "delays": delays,
        "scans": scans,
    }


def naive_navigation_policy(env: AssistiveNavEnv) -> int:
    """
    Naive navigation policy that goes directly toward goal.

    This policy doesn't consider hazards - it just tries to minimize
    distance to goal. This can lead to hitting hazards if ungated.
    """
    from wooedge.envs.assistive_nav import Heading

    y, x = env.agent_pos
    goal_y, goal_x = env.goal_pos

    # Priority: Go toward goal in the most direct way (south first, then east)
    # This is naive because it may pass through hazard area

    # First go south if needed (greedy)
    if y < goal_y:
        if env.agent_heading == Heading.SOUTH:
            return AssistiveAction.FORWARD
        elif env.agent_heading == Heading.EAST:
            return AssistiveAction.RIGHT
        elif env.agent_heading == Heading.WEST:
            return AssistiveAction.LEFT
        else:  # NORTH
            return AssistiveAction.RIGHT

    # Then go east if needed
    if x < goal_x:
        if env.agent_heading == Heading.EAST:
            return AssistiveAction.FORWARD
        elif env.agent_heading == Heading.SOUTH:
            return AssistiveAction.LEFT
        elif env.agent_heading == Heading.NORTH:
            return AssistiveAction.RIGHT
        else:  # WEST
            return AssistiveAction.RIGHT

    # Default: move forward
    return AssistiveAction.FORWARD


def simple_navigation_policy(env: AssistiveNavEnv) -> int:
    """
    Smart navigation policy that avoids hazards.

    Strategy: Go east first to clear the hazard area (x >= goal_x),
    then go south to reach the goal. This avoids the hazard on the left.
    """
    from wooedge.envs.assistive_nav import Heading

    y, x = env.agent_pos
    goal_y, goal_x = env.goal_pos

    # Priority 1: Go east first (to avoid hazard on left side)
    if x < goal_x:
        # Need to go east
        if env.agent_heading == Heading.EAST:
            return AssistiveAction.FORWARD
        elif env.agent_heading == Heading.SOUTH:
            return AssistiveAction.LEFT  # Turn left to face east
        elif env.agent_heading == Heading.NORTH:
            return AssistiveAction.RIGHT  # Turn right to face east
        else:  # WEST
            return AssistiveAction.RIGHT  # Turn to start facing south, then east

    # Priority 2: Go south to reach goal row
    if y < goal_y:
        if env.agent_heading == Heading.SOUTH:
            return AssistiveAction.FORWARD
        elif env.agent_heading == Heading.EAST:
            return AssistiveAction.RIGHT  # Turn right to face south
        elif env.agent_heading == Heading.WEST:
            return AssistiveAction.LEFT  # Turn left to face south
        else:  # NORTH
            return AssistiveAction.RIGHT  # Turn to start turning toward south

    # Priority 3: Go west if needed
    if x > goal_x:
        if env.agent_heading == Heading.WEST:
            return AssistiveAction.FORWARD
        elif env.agent_heading == Heading.SOUTH:
            return AssistiveAction.RIGHT
        elif env.agent_heading == Heading.NORTH:
            return AssistiveAction.LEFT
        else:  # EAST
            return AssistiveAction.RIGHT

    # Priority 4: Go north if needed
    if y > goal_y:
        if env.agent_heading == Heading.NORTH:
            return AssistiveAction.FORWARD
        elif env.agent_heading == Heading.WEST:
            return AssistiveAction.RIGHT
        elif env.agent_heading == Heading.EAST:
            return AssistiveAction.LEFT
        else:  # SOUTH
            return AssistiveAction.RIGHT

    # Default: move forward (at goal)
    return AssistiveAction.FORWARD


def run_benchmark(n_seeds: int = 20, verbose: bool = False) -> None:
    """Run benchmark comparing gated vs ungated navigation."""
    print(f"\n{'='*60}")
    print("Assistive Navigation Benchmark")
    print(f"{'='*60}")
    print(f"Seeds: {n_seeds}, Max steps: 50")
    print(f"{'='*60}\n")

    gated_results = {"success": 0, "hazard": 0, "steps": [], "scans": []}
    ungated_results = {"success": 0, "hazard": 0, "steps": [], "scans": []}

    for seed in range(n_seeds):
        # Gated run with naive policy (to show gating value)
        env = AssistiveNavEnv(AssistiveNavConfig(seed=seed))
        safety = AssistiveDecisionSafety(env)
        result = run_episode(env, safety, gated=True, verbose=False, use_naive_policy=True)

        gated_results["success"] += result["success"]
        gated_results["hazard"] += result["hazard"]
        gated_results["steps"].append(result["steps"])
        gated_results["scans"].append(result["scans"])

        # Ungated run with naive policy (shows hazard hits without gating)
        env = AssistiveNavEnv(AssistiveNavConfig(seed=seed))
        safety = AssistiveDecisionSafety(env)
        result = run_episode(env, safety, gated=False, verbose=False, use_naive_policy=True)

        ungated_results["success"] += result["success"]
        ungated_results["hazard"] += result["hazard"]
        ungated_results["steps"].append(result["steps"])
        ungated_results["scans"].append(result["scans"])

    # Print results
    print(f"{'Mode':<12} {'Success%':>10} {'Hazard%':>10} {'Avg Steps':>12} {'Avg Scans':>12}")
    print("-" * 60)

    gated_success = gated_results["success"] / n_seeds * 100
    gated_hazard = gated_results["hazard"] / n_seeds * 100
    gated_steps = np.mean(gated_results["steps"])
    gated_scans = np.mean(gated_results["scans"])

    ungated_success = ungated_results["success"] / n_seeds * 100
    ungated_hazard = ungated_results["hazard"] / n_seeds * 100
    ungated_steps = np.mean(ungated_results["steps"])
    ungated_scans = np.mean(ungated_results["scans"])

    print(f"{'Gated':<12} {gated_success:>9.1f}% {gated_hazard:>9.1f}% {gated_steps:>12.1f} {gated_scans:>12.1f}")
    print(f"{'Ungated':<12} {ungated_success:>9.1f}% {ungated_hazard:>9.1f}% {ungated_steps:>12.1f} {ungated_scans:>12.1f}")
    print("-" * 60)

    # Safety comparison
    print(f"\nSafety improvement: {ungated_hazard - gated_hazard:+.1f}pp fewer hazard incidents")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("Assistive Navigation Gate Demo")
    print("=" * 60)
    print()
    print("Scenario: Smart cane / wheelchair guidance system")
    print("- Fork corridor with safe exit (G) and hazard/stairs (H)")
    print("- Noisy sensors create uncertainty about position")
    print("- DecisionSafety gates irreversible commitments")
    print()

    # Create environment
    env = AssistiveNavEnv(AssistiveNavConfig(seed=42))

    print("Environment layout:")
    print(env.render_ascii())
    print()
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}, Hazard: {env.hazard_pos}")
    print(f"Has fork: {env.has_fork}")
    print(f"Trap zone: {env.commit_zone_trap}")
    print(f"Goal zone: {env.commit_zone_goal}")
    print()

    # Create safety checker
    safety = AssistiveDecisionSafety(env, AssistiveSafetyConfig(
        entropy_threshold=0.3,
        hazard_hint_threshold=0.5
    ))

    # Demo 1: Naive policy with gating (shows DELAY decisions)
    print("=" * 60)
    print("Demo 1: Naive policy with safety gating")
    print("(Naive policy tries to go south first - toward hazard)")
    print("=" * 60)
    print()

    obs = env.reset(seed=42)
    safety.reset(seed=42)
    safety.observe(obs)

    ACTION_NAMES = ["FORWARD", "LEFT", "RIGHT", "SCAN"]
    max_steps = 15

    for step in range(max_steps):
        print(f"\n--- Step {step} ---")
        print(f"Position: {env.agent_pos}, Heading: {env.agent_heading.name}")
        print(f"Observation: front={obs.front_dist}, left={obs.left_dist}, "
              f"right={obs.right_dist}, hazard_hint={obs.hazard_hint:.2f}")
        print(f"Entropy: {safety.get_normalized_entropy():.3f}")

        # Get action from NAIVE policy (goes south first)
        action = naive_navigation_policy(env)
        print(f"Naive policy proposes: {ACTION_NAMES[action]}")

        # Check safety
        result = safety.propose(action)
        print(f"Safety decision: {result['decision']}")
        print(f"Reason: {result['reason']}")

        # Use suggested action if DELAY
        if result['decision'] == 'DELAY':
            print(f">>> Gating intervenes! Using: {ACTION_NAMES[result['suggested_action']]}")
            action = result['suggested_action']

        # Take action
        obs, cost, done, info = env.step(action)
        safety.observe(obs)

        print(f"Grid: {env.render_ascii().replace(chr(10), '  ')}")

        if done:
            if info["at_goal"]:
                print(f"\n*** GOAL REACHED! ***")
            elif info["at_hazard"]:
                print(f"\n*** HAZARD HIT! ***")
            break

    # Demo 2: Smart policy (goes east first to avoid hazard)
    print()
    print("=" * 60)
    print("Demo 2: Smart policy (goes east first to avoid hazard)")
    print("=" * 60)
    print()

    env2 = AssistiveNavEnv(AssistiveNavConfig(seed=42))
    safety2 = AssistiveDecisionSafety(env2)
    obs = env2.reset(seed=42)
    safety2.reset(seed=42)
    safety2.observe(obs)

    for step in range(max_steps):
        action = simple_navigation_policy(env2)
        result = safety2.propose(action)

        print(f"Step {step}: pos={env2.agent_pos}, heading={env2.agent_heading.name}, "
              f"action={ACTION_NAMES[action]}, decision={result['decision']}")

        obs, cost, done, info = env2.step(action)
        safety2.observe(obs)

        if done:
            if info["at_goal"]:
                print(f"*** GOAL REACHED in {step+1} steps! ***")
            break

    # Run benchmark
    print()
    run_benchmark(n_seeds=20, verbose=False)


if __name__ == "__main__":
    main()
