"""
Desktop HAL Implementation

HAL for macOS, Linux, and Windows.
Uses filesystem for storage, threading for async, and requests for HTTP.
"""

from __future__ import annotations
import os
import sys
import json
import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from .base import (
    HAL,
    HALConfig,
    Platform,
    SensorType,
    ActuatorType,
    SensorReading,
)


logger = logging.getLogger(__name__)


class DesktopHAL(HAL):
    """
    Desktop HAL for macOS, Linux, Windows.

    Features:
    - Filesystem-based storage (JSON files)
    - Threading-based async operations
    - HTTP via urllib or requests
    - Simulated sensors for testing
    - Callback-based actuators

    Example:
        hal = DesktopHAL()
        hal.initialize()

        # Register simulated sensor
        hal.register_sensor("temp", SensorType.TEMPERATURE, {
            "simulator": lambda: 20 + random.random() * 5
        })

        # Read sensor
        reading = hal.read_sensor("temp")
        print(f"Temperature: {reading.value}")

        # Store state
        hal.store("config", {"threshold": 25})
    """

    def __init__(self, config: HALConfig = None):
        super().__init__(config)
        self._start_time = time.time()
        self._storage_path: Optional[Path] = None
        self._simulators: Dict[str, Callable] = {}
        self._actuator_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    @property
    def platform(self) -> Platform:
        return Platform.DESKTOP

    def initialize(self) -> None:
        """Initialize desktop HAL."""
        if self._initialized:
            return

        # Setup storage directory
        self._storage_path = Path(self.config.storage_path).resolve()
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)

        self._initialized = True
        logger.info(f"DesktopHAL initialized (storage: {self._storage_path})")
        self.emit("initialized")

    def shutdown(self) -> None:
        """Shutdown desktop HAL."""
        self._initialized = False
        self.emit("shutdown")
        logger.info("DesktopHAL shutdown")

    # ==================== Time ====================

    def get_time(self) -> float:
        """Get current Unix timestamp."""
        return time.time()

    def get_ticks_ms(self) -> int:
        """Get milliseconds since HAL start."""
        return int((time.time() - self._start_time) * 1000)

    def sleep(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        time.sleep(seconds)

    def sleep_ms(self, ms: int) -> None:
        """Sleep for specified milliseconds."""
        time.sleep(ms / 1000.0)

    # ==================== Sensors ====================

    def _on_sensor_registered(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        config: Dict[str, Any],
    ) -> None:
        """Setup sensor simulator if provided."""
        if "simulator" in config:
            self._simulators[sensor_id] = config["simulator"]
        elif "file" in config:
            # Read sensor value from file
            self._simulators[sensor_id] = lambda f=config["file"]: self._read_file_sensor(f)

    def _read_file_sensor(self, filepath: str) -> Any:
        """Read sensor value from file."""
        try:
            with open(filepath, 'r') as f:
                return float(f.read().strip())
        except:
            return None

    def read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """Read a sensor value."""
        if sensor_id not in self._sensors:
            logger.warning(f"Unknown sensor: {sensor_id}")
            return None

        sensor_info = self._sensors[sensor_id]
        sensor_type = sensor_info["type"]
        config = sensor_info["config"]

        # Get value from simulator or config
        value = None
        if sensor_id in self._simulators:
            try:
                value = self._simulators[sensor_id]()
            except Exception as e:
                logger.error(f"Sensor simulator error: {e}")
                return None
        elif "value" in config:
            value = config["value"]
        else:
            logger.warning(f"No value source for sensor: {sensor_id}")
            return None

        reading = SensorReading(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            value=value,
            unit=config.get("unit", ""),
            metadata=config.get("metadata", {}),
        )

        self.emit("sensor_reading", reading)
        return reading

    def set_sensor_value(self, sensor_id: str, value: Any) -> None:
        """
        Manually set a sensor value (for testing).

        Args:
            sensor_id: Sensor to set
            value: Value to set
        """
        if sensor_id in self._sensors:
            self._sensors[sensor_id]["config"]["value"] = value

    def set_sensor_simulator(self, sensor_id: str, simulator: Callable) -> None:
        """
        Set a simulator function for a sensor.

        Args:
            sensor_id: Sensor to simulate
            simulator: Callable that returns sensor value
        """
        self._simulators[sensor_id] = simulator

    # ==================== Actuators ====================

    def _on_actuator_registered(
        self,
        actuator_id: str,
        actuator_type: ActuatorType,
        config: Dict[str, Any],
    ) -> None:
        """Setup actuator handler if provided."""
        if "handler" in config:
            self._actuator_handlers[actuator_id] = config["handler"]

    def execute(
        self,
        actuator_id: str,
        command: str,
        **kwargs,
    ) -> bool:
        """Execute an actuator command."""
        if actuator_id not in self._actuators:
            logger.warning(f"Unknown actuator: {actuator_id}")
            return False

        actuator_info = self._actuators[actuator_id]
        actuator_type = actuator_info["type"]

        # Log the command
        logger.debug(f"Execute: {actuator_id}.{command}({kwargs})")

        # Call handler if registered
        if actuator_id in self._actuator_handlers:
            try:
                result = self._actuator_handlers[actuator_id](command, **kwargs)
                self.emit("actuator_executed", {
                    "actuator_id": actuator_id,
                    "command": command,
                    "kwargs": kwargs,
                    "result": result,
                })
                return result if isinstance(result, bool) else True
            except Exception as e:
                logger.error(f"Actuator handler error: {e}")
                return False

        # Default: just log (for testing)
        self.emit("actuator_executed", {
            "actuator_id": actuator_id,
            "command": command,
            "kwargs": kwargs,
        })
        return True

    def set_actuator_handler(self, actuator_id: str, handler: Callable) -> None:
        """
        Set a handler function for an actuator.

        Args:
            actuator_id: Actuator to handle
            handler: Callable(command, **kwargs) -> bool
        """
        self._actuator_handlers[actuator_id] = handler

    # ==================== Storage ====================

    def _get_storage_file(self, key: str) -> Path:
        """Get storage file path for key."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._storage_path / f"{safe_key}.json"

    def store(self, key: str, value: Any) -> bool:
        """Store value to JSON file."""
        if not self._initialized:
            logger.error("HAL not initialized")
            return False

        try:
            filepath = self._get_storage_file(key)
            with self._lock:
                with open(filepath, 'w') as f:
                    json.dump(value, f, indent=2)
            logger.debug(f"Stored: {key}")
            return True
        except Exception as e:
            logger.error(f"Storage error: {e}")
            return False

    def load(self, key: str, default: Any = None) -> Any:
        """Load value from JSON file."""
        if not self._initialized:
            logger.error("HAL not initialized")
            return default

        try:
            filepath = self._get_storage_file(key)
            if not filepath.exists():
                return default
            with self._lock:
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Load error: {e}")
            return default

    def delete(self, key: str) -> bool:
        """Delete storage file."""
        try:
            filepath = self._get_storage_file(key)
            if filepath.exists():
                filepath.unlink()
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all storage keys."""
        if not self._initialized:
            return []

        keys = []
        for filepath in self._storage_path.glob("*.json"):
            key = filepath.stem
            if key.startswith(prefix):
                keys.append(key)
        return keys

    # ==================== Networking ====================

    def is_connected(self) -> bool:
        """Check network connectivity."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False

    def http_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        body: Any = None,
        timeout: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request using urllib."""
        try:
            import urllib.request
            import urllib.error

            headers = headers or {}
            headers.setdefault("User-Agent", "WooEdge/1.0")

            # Encode body
            data = None
            if body is not None:
                if isinstance(body, dict):
                    data = json.dumps(body).encode('utf-8')
                    headers.setdefault("Content-Type", "application/json")
                elif isinstance(body, str):
                    data = body.encode('utf-8')
                else:
                    data = body

            request = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": response.read().decode('utf-8'),
                }

        except urllib.error.HTTPError as e:
            return {
                "status": e.code,
                "headers": dict(e.headers),
                "body": e.read().decode('utf-8'),
            }
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return None

    # ==================== Utilities ====================

    def get_free_memory(self) -> int:
        """Get free memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            # Fallback: use resource module on Unix
            try:
                import resource
                # This is a rough estimate
                return 1024 * 1024 * 1024  # 1GB default
            except:
                return 0

    def get_cpu_freq(self) -> int:
        """Get CPU frequency in Hz."""
        try:
            import psutil
            freq = psutil.cpu_freq()
            return int(freq.current * 1_000_000) if freq else 0
        except ImportError:
            return 0

    def get_os_info(self) -> Dict[str, str]:
        """Get OS information."""
        return {
            "system": sys.platform,
            "python": sys.version,
            "implementation": sys.implementation.name,
        }


class SimulatedSensor:
    """
    Helper class for creating simulated sensors.

    Example:
        temp_sensor = SimulatedSensor.temperature(base=20, variance=5)
        hal.register_sensor("temp", SensorType.TEMPERATURE, {
            "simulator": temp_sensor
        })
    """

    @staticmethod
    def constant(value: Any) -> Callable:
        """Return a constant value."""
        return lambda: value

    @staticmethod
    def random_float(min_val: float, max_val: float) -> Callable:
        """Return random float in range."""
        import random
        return lambda: min_val + random.random() * (max_val - min_val)

    @staticmethod
    def temperature(base: float = 20.0, variance: float = 5.0) -> Callable:
        """Simulate temperature with random variance."""
        import random
        return lambda: base + (random.random() - 0.5) * 2 * variance

    @staticmethod
    def distance(min_dist: float = 10.0, max_dist: float = 200.0) -> Callable:
        """Simulate distance sensor."""
        import random
        return lambda: min_dist + random.random() * (max_dist - min_dist)

    @staticmethod
    def boolean(probability: float = 0.5) -> Callable:
        """Simulate boolean sensor."""
        import random
        return lambda: random.random() < probability

    @staticmethod
    def sequence(values: List[Any], loop: bool = True) -> Callable:
        """Return values in sequence."""
        index = [0]
        def get_next():
            if index[0] >= len(values):
                if loop:
                    index[0] = 0
                else:
                    return values[-1] if values else None
            val = values[index[0]]
            index[0] += 1
            return val
        return get_next

    @staticmethod
    def sine_wave(
        amplitude: float = 1.0,
        period: float = 10.0,
        offset: float = 0.0,
    ) -> Callable:
        """Simulate sine wave sensor."""
        import math
        start_time = time.time()
        def get_value():
            t = time.time() - start_time
            return offset + amplitude * math.sin(2 * math.pi * t / period)
        return get_value
