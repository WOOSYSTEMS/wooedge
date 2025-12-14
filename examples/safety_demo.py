#!/usr/bin/env python3
"""
Decision Safety Demo

Demonstrates how to use the DecisionSafety module to check
whether actions are safe to commit based on belief uncertainty.

Run with:
    python examples/safety_demo.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.env import GridWorld, EnvConfig, Action
from wooedge.safety import DecisionSafety, SafetyConfig


def main():
    print("=" * 60)
    print("WOOEdge Decision Safety Demo")
    print("=" * 60)
    print()

    # Create environment with fork structure
    env = GridWorld(EnvConfig(
        maze_type="symmetric_fork_trap",
        sensor_noise_prob=0.3,
        slip_prob=0.15,
        seed=42
    ))

    print(f"Environment: symmetric_fork_trap")
    print(f"Has fork structure: {env.has_fork}")
    print(f"Goal zone cells: {len(env.commit_zone_goal)}")
    print(f"Trap zone cells: {len(env.commit_zone_trap)}")
    print()

    # Create safety module
    safety_config = SafetyConfig(
        n_particles=100,
        entropy_threshold=0.16,
        seed=42
    )
    safety = DecisionSafety(env, safety_config)
    safety.reset(seed=42)

    # Reset environment
    obs = env.reset()
    safety.observe(obs)

    print("Starting simulation...")
    print("-" * 60)
    print(f"{'Step':<5} {'Pos':<10} {'Entropy':>8} {'Action':<8} {'Decision':<8} {'Reason'}")
    print("-" * 60)

    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

    # Run a short simulation
    for step in range(15):
        # Get current state
        true_pos = env.agent_pos
        entropy = safety.get_normalized_entropy()

        # Try moving DOWN (toward fork)
        proposed_action = Action.DOWN.value

        # Check if action is safe
        result = safety.propose(proposed_action)

        # Print status
        decision = result["decision"]
        reason = result["reason"][:40] + "..." if len(result["reason"]) > 40 else result["reason"]

        print(f"{step:<5} {str(true_pos):<10} {entropy:>8.3f} "
              f"{ACTION_NAMES[proposed_action]:<8} {decision:<8} {reason}")

        # Use suggested action if DELAY
        if decision == "DELAY":
            actual_action = result["suggested_action"]
        else:
            actual_action = proposed_action

        # Take action
        obs, cost, done, info = env.step(actual_action)
        safety.observe(obs)

        if done:
            print(f"\n>>> GOAL REACHED at step {step + 1} <<<")
            break

    print("-" * 60)
    print()

    # Show final position
    print(f"Final position: {env.agent_pos}")
    print(f"Final entropy: {safety.get_normalized_entropy():.3f}")

    # Demonstrate checking specific actions
    print()
    print("=" * 60)
    print("Checking specific actions at current state:")
    print("=" * 60)

    for action in range(5):
        result = safety.propose(action)
        print(f"  {ACTION_NAMES[action]:<6}: {result['decision']:<6} - {result['reason'][:50]}")

    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()
