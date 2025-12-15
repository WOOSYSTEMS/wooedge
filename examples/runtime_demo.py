#!/usr/bin/env python3
"""
WooEdge Runtime Demo

Demonstrates the WooEdge runtime with a simple thermostat app.
Shows how uncertainty gating prevents hasty decisions.

Usage:
    python examples/runtime_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.runtime import (
    WooEdgeApp,
    WooEdgeEngine,
    MemorySource,
    Observation,
    AppConfig,
    ActionCategory,
    run_simulation,
)


class ThermostatApp(WooEdgeApp[str]):
    """
    Simple thermostat app demonstrating WooEdge runtime.

    World states: comfortable, too_cold, too_hot
    Actions: heat_on, cool_on, hold
    """

    world_states = ["comfortable", "too_cold", "too_hot"]
    action_space = ["heat_on", "cool_on", "hold", "scan"]

    observation_schema = {
        "temperature": float,
        "humidity": float,
    }

    action_categories = {
        "heat_on": ActionCategory.COSTLY,
        "cool_on": ActionCategory.COSTLY,
        "hold": ActionCategory.REVERSIBLE,
        "scan": ActionCategory.OBSERVE,
    }

    def __init__(self, target_temp: float = 70.0, tolerance: float = 2.0):
        super().__init__(AppConfig(
            uncertainty_threshold=0.6,
            hazard_threshold=0.7,
            decay_rate=0.005,
        ))
        self.target_temp = target_temp
        self.tolerance = tolerance

    def compute_likelihoods(self, observation, context):
        """Compute P(obs|state) based on temperature."""
        temp = observation.get("temperature", self.target_temp)

        low = self.target_temp - self.tolerance
        high = self.target_temp + self.tolerance

        return {
            "comfortable": 1.0 if low <= temp <= high else 0.1,
            "too_cold": 1.0 if temp < low else 0.05,
            "too_hot": 1.0 if temp > high else 0.05,
        }

    def decide(self, context):
        """Decide action based on most likely state."""
        state = context.most_likely_state

        if state == "too_cold":
            return "heat_on"
        elif state == "too_hot":
            return "cool_on"
        else:
            return "hold"

    def compute_hazard(self, observation, context):
        """Compute hazard from extreme conditions."""
        temp = observation.get("temperature", self.target_temp)

        # Hazard increases for extreme temperatures
        if temp < 50 or temp > 90:
            return 0.8  # High hazard
        elif temp < 60 or temp > 80:
            return 0.3  # Moderate hazard
        return 0.1  # Low hazard


def main():
    print("=" * 70)
    print("WooEdge Runtime Demo - Thermostat App")
    print("=" * 70)
    print()

    # Create app
    app = ThermostatApp(target_temp=70.0, tolerance=2.0)

    # Scenario 1: Stable cold temperatures
    print("Scenario 1: Stable Cold Temperatures")
    print("-" * 50)

    cold_readings = [
        {"temperature": 65.0, "humidity": 40.0},
        {"temperature": 64.5, "humidity": 41.0},
        {"temperature": 64.0, "humidity": 42.0},
        {"temperature": 63.5, "humidity": 43.0},
        {"temperature": 63.0, "humidity": 44.0},
    ]

    results = run_simulation(app, cold_readings, verbose=True)
    print()

    # Scenario 2: Noisy/uncertain readings
    print("Scenario 2: Noisy Readings (high uncertainty)")
    print("-" * 50)

    app2 = ThermostatApp(target_temp=70.0, tolerance=2.0)

    noisy_readings = [
        {"temperature": 65.0, "humidity": 40.0},  # Cold
        {"temperature": 75.0, "humidity": 50.0},  # Hot
        {"temperature": 68.0, "humidity": 45.0},  # Comfortable
        {"temperature": 72.0, "humidity": 48.0},  # Comfortable
        {"temperature": 66.0, "humidity": 42.0},  # Cold
    ]

    results2 = run_simulation(app2, noisy_readings, verbose=True)

    # Show belief state
    belief = app2.get_belief()
    print(f"\nFinal belief: {belief}")
    print(f"Entropy: {belief.normalized_entropy():.2f}")
    print()

    # Scenario 3: Extreme temperatures (high hazard)
    print("Scenario 3: Extreme Temperatures (high hazard)")
    print("-" * 50)

    app3 = ThermostatApp(target_temp=70.0, tolerance=2.0)

    extreme_readings = [
        {"temperature": 45.0, "humidity": 30.0},  # Extreme cold
        {"temperature": 44.0, "humidity": 29.0},
        {"temperature": 43.0, "humidity": 28.0},
    ]

    results3 = run_simulation(app3, extreme_readings, verbose=True)
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The WooEdge runtime:")
    print("1. Tracks belief over world states (comfortable, too_cold, too_hot)")
    print("2. Measures uncertainty via entropy")
    print("3. Gates actions when uncertainty or hazard is high")
    print("4. Allows apps to define their own observation -> likelihood mapping")
    print()
    print("This is the foundation for running ANY app with uncertainty-awareness.")
    print()


if __name__ == "__main__":
    main()
