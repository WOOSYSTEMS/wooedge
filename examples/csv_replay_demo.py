#!/usr/bin/env python3
"""
CSV Replay Demo

Demonstrates replaying logged sensor data through a safety checker.
Prints entropy, safety decision, and recommended action at each timestep.

Usage:
    python examples/csv_replay_demo.py
    python examples/csv_replay_demo.py --file path/to/log.csv
    python examples/csv_replay_demo.py --delay 0.5  # 500ms between steps
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.io.csv_replay import CSVReplaySource, ReplayObservation


class ReplaySafetyChecker:
    """
    Simplified safety checker for CSV replay.

    Tracks observation history and computes entropy-based safety decisions
    without requiring a full environment. Compatible with DecisionSafety API.
    """

    def __init__(self, entropy_threshold: float = 0.5, hazard_threshold: float = 0.6):
        """
        Initialize replay safety checker.

        Args:
            entropy_threshold: Entropy above this triggers DELAY
            hazard_threshold: Hazard hint above this triggers caution
        """
        self.entropy_threshold = entropy_threshold
        self.hazard_threshold = hazard_threshold
        self.observation_history: List[ReplayObservation] = []
        self.belief_entropy = 1.0  # Start with max uncertainty

    def reset(self) -> None:
        """Reset for new replay session."""
        self.observation_history = []
        self.belief_entropy = 1.0

    def observe(self, obs: ReplayObservation) -> None:
        """
        Update internal state from observation.

        Args:
            obs: Observation from CSV replay
        """
        self.observation_history.append(obs)

        # Simple entropy model: decreases with observations, increases with hazard proximity
        n_obs = len(self.observation_history)
        base_reduction = 0.05 * n_obs

        # Hazard hint increases uncertainty (approaching danger)
        hazard_factor = obs.hazard_hint * 0.3

        # Distance sensors: closer obstacles = more certain about position
        avg_dist = (obs.front_dist + obs.left_dist + obs.right_dist) / 3
        dist_factor = max(0, 0.1 * (5 - avg_dist))  # Closer = lower entropy

        self.belief_entropy = max(0.1, min(1.0,
            1.0 - base_reduction - dist_factor + hazard_factor
        ))

    def get_entropy(self) -> float:
        """Get current belief entropy."""
        return self.belief_entropy

    def propose(self, action: int = 0) -> Dict[str, Any]:
        """
        Evaluate whether action is safe.

        Args:
            action: Proposed action (0=FORWARD, 1=LEFT, 2=RIGHT, 3=SCAN)

        Returns:
            Decision dict with decision, entropy, reason, suggested_action
        """
        ACTION_NAMES = ["FORWARD", "LEFT", "RIGHT", "SCAN"]

        if not self.observation_history:
            return {
                "decision": "ABORT",
                "entropy": 1.0,
                "reason": "No observations yet",
                "suggested_action": 3,  # SCAN
            }

        latest = self.observation_history[-1]
        entropy = self.belief_entropy

        # High hazard hint = danger nearby
        hazard_danger = latest.hazard_hint > self.hazard_threshold

        # Check if action would be risky
        is_forward = action == 0
        is_commit_action = is_forward and latest.front_dist <= 2

        if hazard_danger and is_commit_action:
            if entropy > self.entropy_threshold:
                return {
                    "decision": "DELAY",
                    "entropy": entropy,
                    "reason": f"High hazard ({latest.hazard_hint:.2f}) with uncertain belief (H={entropy:.2f}). Recommend SCAN.",
                    "suggested_action": 3,  # SCAN
                }
            else:
                return {
                    "decision": "DELAY",
                    "entropy": entropy,
                    "reason": f"Hazard detected ({latest.hazard_hint:.2f}). Recommend turning away.",
                    "suggested_action": 2,  # RIGHT (turn away)
                }

        if entropy > self.entropy_threshold and is_commit_action:
            return {
                "decision": "DELAY",
                "entropy": entropy,
                "reason": f"High uncertainty (H={entropy:.2f}). Recommend gathering more information.",
                "suggested_action": 3,  # SCAN
            }

        return {
            "decision": "ALLOW",
            "entropy": entropy,
            "reason": "Action appears safe",
            "suggested_action": action,
        }


def run_replay(csv_path: str, delay: float = 0.0, verbose: bool = True) -> None:
    """
    Run CSV replay through safety checker.

    Args:
        csv_path: Path to CSV file
        delay: Delay between steps (seconds)
        verbose: Print detailed output
    """
    ACTION_NAMES = ["FORWARD", "LEFT", "RIGHT", "SCAN"]

    print("=" * 60)
    print("CSV Replay Safety Demo")
    print("=" * 60)
    print(f"File: {csv_path}")
    print(f"Delay: {delay}s per step")
    print("=" * 60)
    print()

    # Load CSV
    source = CSVReplaySource(csv_path)
    source.load()

    print(f"Loaded {len(source)} observations")
    if not source.validate_order():
        print("WARNING: Timesteps not in ascending order")
    print()

    # Create safety checker
    safety = ReplaySafetyChecker(
        entropy_threshold=0.4,
        hazard_threshold=0.6
    )
    safety.reset()

    # Replay
    print(f"{'Step':<6} {'Front':<6} {'Left':<6} {'Right':<6} {'Hazard':<8} {'Entropy':<8} {'Decision':<8} {'Action'}")
    print("-" * 70)

    for obs in source:
        safety.observe(obs)

        # Propose FORWARD action (most common movement)
        result = safety.propose(action=0)

        suggested = ACTION_NAMES[result["suggested_action"]]

        print(f"{obs.timestep:<6} {obs.front_dist:<6} {obs.left_dist:<6} {obs.right_dist:<6} "
              f"{obs.hazard_hint:<8.2f} {result['entropy']:<8.2f} {result['decision']:<8} {suggested}")

        if verbose and result["decision"] != "ALLOW":
            print(f"       Reason: {result['reason']}")

        if delay > 0:
            time.sleep(delay)

    print("-" * 70)
    print(f"Replay complete. Final entropy: {safety.get_entropy():.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CSV Replay Safety Demo")
    parser.add_argument("--file", "-f", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data", "sample_nav_log.csv"),
                        help="Path to CSV file")
    parser.add_argument("--delay", "-d", type=float, default=0.0,
                        help="Delay between steps in seconds")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output (no reasons)")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    run_replay(args.file, delay=args.delay, verbose=not args.quiet)


if __name__ == "__main__":
    main()
