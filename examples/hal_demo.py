#!/usr/bin/env python3
"""
WooEdge HAL Demo

Demonstrates the Hardware Abstraction Layer with simulated sensors and actuators.
Shows how apps can be portable across platforms.

Usage:
    python examples/hal_demo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.hal import (
    get_hal,
    DesktopHAL,
    SensorType,
    ActuatorType,
    SimulatedSensor,
    HALConfig,
)
from wooedge.runtime import SimpleApp, run_simulation, Observation


def demo_basic_hal():
    """Demonstrate basic HAL operations."""
    print("=" * 60)
    print("Demo 1: Basic HAL Operations")
    print("=" * 60)

    # Auto-detect platform
    hal = get_hal()
    hal.initialize()

    print(f"Platform: {hal.platform.value}")
    print(f"Time: {hal.get_time():.2f}")
    print(f"Ticks: {hal.get_ticks_ms()} ms")

    # Register simulated sensors
    hal.register_sensor("temperature", SensorType.TEMPERATURE, {
        "simulator": SimulatedSensor.temperature(base=22, variance=3),
        "unit": "C",
    })

    hal.register_sensor("distance", SensorType.DISTANCE, {
        "simulator": SimulatedSensor.distance(min_dist=10, max_dist=200),
        "unit": "cm",
    })

    hal.register_sensor("motion", SensorType.MOTION, {
        "simulator": SimulatedSensor.boolean(probability=0.3),
    })

    # Read sensors
    print("\nSensor Readings:")
    for _ in range(3):
        temp = hal.read_sensor("temperature")
        dist = hal.read_sensor("distance")
        motion = hal.read_sensor("motion")

        print(f"  Temp: {temp.value:.1f}{temp.unit}, "
              f"Distance: {dist.value:.1f}{dist.unit}, "
              f"Motion: {motion.value}")
        hal.sleep_ms(100)

    # Register actuators
    executed = []

    def led_handler(command, **kwargs):
        executed.append(f"LED: {command}")
        return True

    def motor_handler(command, **kwargs):
        speed = kwargs.get("speed", 0)
        executed.append(f"Motor: {command} @ {speed}%")
        return True

    hal.register_actuator("led", ActuatorType.LED, {"handler": led_handler})
    hal.register_actuator("motor", ActuatorType.MOTOR, {"handler": motor_handler})

    # Execute actuator commands
    print("\nActuator Commands:")
    hal.execute("led", "on")
    hal.execute("motor", "forward", speed=50)
    hal.execute("led", "off")
    hal.execute("motor", "stop")

    for cmd in executed:
        print(f"  {cmd}")

    # Storage
    print("\nStorage:")
    hal.store("config", {"threshold": 25, "enabled": True})
    config = hal.load("config")
    print(f"  Stored config: {config}")

    hal.shutdown()
    print()


def demo_hal_with_runtime():
    """Demonstrate HAL integrated with WooEdge runtime."""
    print("=" * 60)
    print("Demo 2: HAL + Runtime Integration")
    print("=" * 60)

    # Create HAL
    hal = get_hal()
    hal.initialize()

    # Setup sensors
    hal.register_sensor("temp", SensorType.TEMPERATURE, {
        "simulator": SimulatedSensor.sequence([
            18, 17, 16, 15,  # Getting cold
            20, 22, 24,      # Warming up
            26, 28, 30,      # Getting hot
        ], loop=False),
        "unit": "C",
    })

    # Create WooEdge app that reads from HAL
    class HVACApp(SimpleApp):
        def __init__(self, hal_ref):
            self.hal = hal_ref
            super().__init__(
                states=["comfortable", "too_cold", "too_hot"],
                actions=["heat", "cool", "idle"],
                schema={"temperature": float},
                likelihood_fn=self._likelihoods,
                decide_fn=self._decide,
            )

        def _likelihoods(self, obs, ctx):
            temp = obs.get("temperature", 20)
            return {
                "comfortable": 1.0 if 18 <= temp <= 24 else 0.1,
                "too_cold": 1.0 if temp < 18 else 0.05,
                "too_hot": 1.0 if temp > 24 else 0.05,
            }

        def _decide(self, ctx):
            state = ctx.most_likely_state
            if state == "too_cold":
                return "heat"
            elif state == "too_hot":
                return "cool"
            return "idle"

    app = HVACApp(hal)

    # Collect observations from HAL
    observations = []
    for _ in range(10):
        reading = hal.read_sensor("temp")
        if reading:
            observations.append({"temperature": reading.value})

    # Run through runtime
    print("\nRunning HVAC control with HAL sensors:")
    results = run_simulation(app, observations, verbose=True)

    hal.shutdown()
    print()


def demo_portable_app():
    """Demonstrate a portable app that works on any platform."""
    print("=" * 60)
    print("Demo 3: Portable App (works on Desktop, ESP32, Cloud)")
    print("=" * 60)

    # This app would work on ANY HAL implementation
    print("""
    # Portable WooEdge App Template

    from wooedge.hal import get_hal, SensorType, ActuatorType
    from wooedge.runtime import WooEdgeApp

    class MyPortableApp(WooEdgeApp):
        world_states = ["safe", "danger"]
        action_space = ["proceed", "stop", "scan"]

        def setup(self, hal):
            # Register sensors - HAL handles platform details
            hal.register_sensor("distance", SensorType.DISTANCE, {
                # On ESP32: {"type": "hcsr04", "trigger_pin": 5, "echo_pin": 18}
                # On Desktop: {"simulator": SimulatedSensor.distance()}
                # On Cloud: {"type": "http", "url": "https://api.sensor.io/dist"}
            })

            hal.register_actuator("motor", ActuatorType.MOTOR, {
                # On ESP32: {"type": "pwm", "pin": 4}
                # On Desktop: {"handler": motor_sim}
                # On Cloud: {"type": "http", "url": "https://api.robot.io/motor"}
            })

        def compute_likelihoods(self, obs, ctx):
            dist = obs.get("distance", 100)
            return {
                "safe": 1.0 if dist > 30 else 0.1,
                "danger": 1.0 if dist < 30 else 0.1,
            }

        def decide(self, ctx):
            if ctx.most_likely_state == "danger":
                return "stop"
            return "proceed"

    # Run on ANY platform
    hal = get_hal()  # Auto-detects: Desktop, ESP32, or Cloud
    hal.initialize()

    app = MyPortableApp()
    app.setup(hal)

    # App runs identically regardless of platform!
    """)

    print("This app runs on:")
    print("  - macOS/Linux/Windows (DesktopHAL)")
    print("  - ESP32/Pico (MicroPythonHAL)")
    print("  - AWS Lambda/Cloud Run (CloudHAL)")
    print()


def main():
    print()
    print("WooEdge Hardware Abstraction Layer Demo")
    print("=" * 60)
    print()

    demo_basic_hal()
    demo_hal_with_runtime()
    demo_portable_app()

    print("=" * 60)
    print("HAL enables: Write Once, Run Anywhere")
    print("=" * 60)


if __name__ == "__main__":
    main()
