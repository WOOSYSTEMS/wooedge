"""
WooEdge Hardware Abstraction Layer (HAL)

Platform-agnostic interface for sensors, actuators, storage, and networking.
Apps write to the HAL interface; platform-specific implementations handle details.

Supported Platforms:
- Desktop (macOS, Linux, Windows)
- MicroPython (ESP32, Raspberry Pi Pico)
- Cloud (AWS Lambda, Google Cloud Run, Azure Functions)

Example:
    from wooedge.hal import get_hal, SensorType, ActuatorType

    # Auto-detect platform
    hal = get_hal()
    hal.initialize()

    # Register sensors and actuators
    hal.register_sensor("temp", SensorType.TEMPERATURE, {"pin": 4})
    hal.register_actuator("led", ActuatorType.LED, {"pin": 2})

    # Use them
    reading = hal.read_sensor("temp")
    hal.execute("led", "on")

    # Store state
    hal.store("config", {"threshold": 25})
"""

from .base import (
    # Core classes
    HAL,
    HALConfig,
    # Enums
    Platform,
    SensorType,
    ActuatorType,
    # Data classes
    SensorReading,
    ActuatorCommand,
    # Factory
    get_hal,
)

from .desktop import (
    DesktopHAL,
    SimulatedSensor,
)

# Cloud HAL (always available)
from .cloud import CloudHAL

# MicroPython HAL - may not work on all platforms
try:
    from .micropython import MicroPythonHAL
except ImportError:
    MicroPythonHAL = None


__all__ = [
    # Core
    "HAL",
    "HALConfig",
    # Enums
    "Platform",
    "SensorType",
    "ActuatorType",
    # Data
    "SensorReading",
    "ActuatorCommand",
    # Factory
    "get_hal",
    # Implementations
    "DesktopHAL",
    "CloudHAL",
    "MicroPythonHAL",
    # Utilities
    "SimulatedSensor",
]
