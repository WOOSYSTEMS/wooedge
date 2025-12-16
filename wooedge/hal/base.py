"""
Hardware Abstraction Layer - Base Interface

Abstract interface for platform-specific implementations.
Apps write to this interface; HAL implementations handle the platform details.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic
from enum import Enum
import time


class Platform(Enum):
    """Supported platforms."""
    DESKTOP = "desktop"          # macOS, Linux, Windows
    MICROPYTHON = "micropython"  # ESP32, Pico, etc.
    RPI = "rpi"                  # Raspberry Pi
    CLOUD = "cloud"              # AWS Lambda, Cloud Run, etc.
    BROWSER = "browser"          # WebAssembly/Pyodide
    MOBILE = "mobile"            # iOS/Android (future)


class SensorType(Enum):
    """Common sensor types."""
    DISTANCE = "distance"        # Ultrasonic, IR, ToF
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"            # PIR, accelerometer
    GPS = "gps"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    GENERIC = "generic"          # Custom sensors


class ActuatorType(Enum):
    """Common actuator types."""
    MOTOR = "motor"
    SERVO = "servo"
    LED = "led"
    RELAY = "relay"
    SPEAKER = "speaker"
    DISPLAY = "display"
    GPIO = "gpio"
    HTTP = "http"                # REST API calls
    MQTT = "mqtt"                # MQTT publish
    GENERIC = "generic"          # Custom actuators


@dataclass
class SensorReading:
    """Universal sensor reading container."""
    sensor_id: str
    sensor_type: SensorType
    value: Any
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ActuatorCommand:
    """Universal actuator command container."""
    actuator_id: str
    actuator_type: ActuatorType
    command: str
    value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HALConfig:
    """HAL configuration."""
    platform: Platform = Platform.DESKTOP
    debug: bool = False
    log_level: str = "INFO"
    storage_path: str = "./data"
    sensors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    actuators: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class HAL(ABC):
    """
    Hardware Abstraction Layer base class.

    Platform-specific implementations inherit from this class.
    Apps interact with HAL through this interface, making them portable.

    Example:
        hal = DesktopHAL()  # or MicroPythonHAL(), CloudHAL(), etc.
        hal.initialize()

        # Read sensors
        reading = hal.read_sensor("temp_1")

        # Execute actions
        hal.execute("motor_1", "forward", speed=50)

        # Store/load state
        hal.store("app_state", {"position": 5})
        state = hal.load("app_state")
    """

    def __init__(self, config: HALConfig = None):
        self.config = config or HALConfig()
        self._initialized = False
        self._sensors: Dict[str, Dict[str, Any]] = {}
        self._actuators: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """Get the platform type."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the HAL.

        Sets up platform-specific resources (GPIO, I2C, network, etc.)
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the HAL.

        Releases platform-specific resources.
        """
        pass

    # ==================== Time ====================

    @abstractmethod
    def get_time(self) -> float:
        """
        Get current time in seconds since epoch.

        Platform-specific: RTC, NTP, or system clock.
        """
        pass

    @abstractmethod
    def get_ticks_ms(self) -> int:
        """
        Get milliseconds since boot/start.

        Useful for timing without RTC.
        """
        pass

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """
        Sleep for specified seconds.

        Platform-specific: may yield to other tasks.
        """
        pass

    @abstractmethod
    def sleep_ms(self, ms: int) -> None:
        """Sleep for specified milliseconds."""
        pass

    # ==================== Sensors ====================

    def register_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        config: Dict[str, Any] = None,
    ) -> None:
        """
        Register a sensor with the HAL.

        Args:
            sensor_id: Unique identifier for the sensor
            sensor_type: Type of sensor
            config: Platform-specific configuration (pins, addresses, etc.)
        """
        self._sensors[sensor_id] = {
            "type": sensor_type,
            "config": config or {},
        }
        self._on_sensor_registered(sensor_id, sensor_type, config)

    def _on_sensor_registered(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        config: Dict[str, Any],
    ) -> None:
        """Hook for platform-specific sensor setup. Override in subclass."""
        pass

    @abstractmethod
    def read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """
        Read a value from a sensor.

        Args:
            sensor_id: The sensor to read from

        Returns:
            SensorReading or None if unavailable
        """
        pass

    def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read all registered sensors."""
        readings = {}
        for sensor_id in self._sensors:
            reading = self.read_sensor(sensor_id)
            if reading:
                readings[sensor_id] = reading
        return readings

    # ==================== Actuators ====================

    def register_actuator(
        self,
        actuator_id: str,
        actuator_type: ActuatorType,
        config: Dict[str, Any] = None,
    ) -> None:
        """
        Register an actuator with the HAL.

        Args:
            actuator_id: Unique identifier for the actuator
            actuator_type: Type of actuator
            config: Platform-specific configuration
        """
        self._actuators[actuator_id] = {
            "type": actuator_type,
            "config": config or {},
        }
        self._on_actuator_registered(actuator_id, actuator_type, config)

    def _on_actuator_registered(
        self,
        actuator_id: str,
        actuator_type: ActuatorType,
        config: Dict[str, Any],
    ) -> None:
        """Hook for platform-specific actuator setup. Override in subclass."""
        pass

    @abstractmethod
    def execute(
        self,
        actuator_id: str,
        command: str,
        **kwargs,
    ) -> bool:
        """
        Execute a command on an actuator.

        Args:
            actuator_id: The actuator to command
            command: The command (e.g., "forward", "on", "set")
            **kwargs: Command-specific parameters

        Returns:
            True if successful
        """
        pass

    # ==================== Storage ====================

    @abstractmethod
    def store(self, key: str, value: Any) -> bool:
        """
        Store a value persistently.

        Platform-specific: filesystem, flash, S3, etc.

        Args:
            key: Storage key
            value: Value to store (must be JSON-serializable)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def load(self, key: str, default: Any = None) -> Any:
        """
        Load a value from storage.

        Args:
            key: Storage key
            default: Default value if key not found

        Returns:
            Stored value or default
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from storage."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all storage keys with optional prefix filter."""
        pass

    # ==================== Networking ====================

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if network is available."""
        pass

    @abstractmethod
    def http_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        body: Any = None,
        timeout: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Request headers
            body: Request body (will be JSON-encoded if dict)
            timeout: Request timeout in seconds

        Returns:
            Dict with 'status', 'headers', 'body' or None on failure
        """
        pass

    # ==================== Events ====================

    def on(self, event: str, callback: Callable) -> None:
        """
        Register an event callback.

        Events: "sensor_reading", "error", "connected", "disconnected"
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception:
                pass  # Don't let callback errors break HAL

    # ==================== Utilities ====================

    @abstractmethod
    def get_free_memory(self) -> int:
        """Get free memory in bytes."""
        pass

    @abstractmethod
    def get_cpu_freq(self) -> int:
        """Get CPU frequency in Hz."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get HAL information."""
        return {
            "platform": self.platform.value,
            "initialized": self._initialized,
            "sensors": list(self._sensors.keys()),
            "actuators": list(self._actuators.keys()),
            "free_memory": self.get_free_memory(),
            "cpu_freq": self.get_cpu_freq(),
        }

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.shutdown()


def get_hal(platform: Platform = None, config: HALConfig = None) -> HAL:
    """
    Factory function to get appropriate HAL for current platform.

    Auto-detects platform if not specified.
    """
    if platform is None:
        platform = _detect_platform()

    if platform == Platform.DESKTOP:
        from .desktop import DesktopHAL
        return DesktopHAL(config)
    elif platform == Platform.MICROPYTHON:
        from .micropython import MicroPythonHAL
        return MicroPythonHAL(config)
    elif platform == Platform.CLOUD:
        from .cloud import CloudHAL
        return CloudHAL(config)
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def _detect_platform() -> Platform:
    """Auto-detect current platform."""
    try:
        import sys
        if sys.implementation.name == "micropython":
            return Platform.MICROPYTHON
    except:
        pass

    try:
        import os
        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            return Platform.CLOUD
        if os.environ.get("K_SERVICE"):  # Cloud Run
            return Platform.CLOUD
    except:
        pass

    return Platform.DESKTOP
