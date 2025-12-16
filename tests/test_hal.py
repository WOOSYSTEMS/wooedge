"""
Tests for WooEdge Hardware Abstraction Layer.

Tests the HAL components:
- DesktopHAL
- CloudHAL
- Sensors and Actuators
- Storage
"""

import pytest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.hal import (
    get_hal,
    Platform,
    SensorType,
    ActuatorType,
    HALConfig,
    DesktopHAL,
    CloudHAL,
    SimulatedSensor,
    SensorReading,
)


class TestDesktopHAL:
    """Tests for DesktopHAL."""

    @pytest.fixture
    def hal(self, tmp_path):
        """Create HAL with temp storage."""
        config = HALConfig(storage_path=str(tmp_path))
        hal = DesktopHAL(config)
        hal.initialize()
        yield hal
        hal.shutdown()

    def test_initialize(self, hal):
        """Test HAL initializes correctly."""
        assert hal._initialized
        assert hal.platform == Platform.DESKTOP

    def test_time_functions(self, hal):
        """Test time functions."""
        t1 = hal.get_time()
        ms1 = hal.get_ticks_ms()

        hal.sleep_ms(10)

        t2 = hal.get_time()
        ms2 = hal.get_ticks_ms()

        assert t2 >= t1
        assert ms2 > ms1

    def test_register_sensor(self, hal):
        """Test sensor registration."""
        hal.register_sensor("temp", SensorType.TEMPERATURE, {
            "simulator": lambda: 25.0,
            "unit": "C",
        })

        assert "temp" in hal._sensors

    def test_read_sensor_with_simulator(self, hal):
        """Test reading sensor with simulator."""
        hal.register_sensor("temp", SensorType.TEMPERATURE, {
            "simulator": lambda: 25.5,
            "unit": "C",
        })

        reading = hal.read_sensor("temp")

        assert reading is not None
        assert reading.sensor_id == "temp"
        assert reading.value == 25.5
        assert reading.unit == "C"

    def test_read_sensor_unknown(self, hal):
        """Test reading unknown sensor returns None."""
        reading = hal.read_sensor("unknown")
        assert reading is None

    def test_set_sensor_value(self, hal):
        """Test manually setting sensor value."""
        hal.register_sensor("manual", SensorType.GENERIC, {})
        hal.set_sensor_value("manual", 42)

        reading = hal.read_sensor("manual")
        assert reading.value == 42

    def test_register_actuator(self, hal):
        """Test actuator registration."""
        hal.register_actuator("led", ActuatorType.LED, {"pin": 2})
        assert "led" in hal._actuators

    def test_execute_actuator(self, hal):
        """Test actuator execution."""
        executed = []

        def handler(command, **kwargs):
            executed.append((command, kwargs))
            return True

        hal.register_actuator("motor", ActuatorType.MOTOR, {
            "handler": handler,
        })

        result = hal.execute("motor", "forward", speed=50)

        assert result
        assert len(executed) == 1
        assert executed[0] == ("forward", {"speed": 50})

    def test_execute_unknown_actuator(self, hal):
        """Test executing unknown actuator returns False."""
        result = hal.execute("unknown", "test")
        assert not result

    def test_storage_store_load(self, hal):
        """Test storage store and load."""
        data = {"key": "value", "number": 42}

        assert hal.store("test_data", data)
        loaded = hal.load("test_data")

        assert loaded == data

    def test_storage_load_default(self, hal):
        """Test loading non-existent key returns default."""
        result = hal.load("nonexistent", default="default_value")
        assert result == "default_value"

    def test_storage_delete(self, hal):
        """Test storage delete."""
        hal.store("to_delete", {"data": 1})
        assert hal.load("to_delete") is not None

        hal.delete("to_delete")
        assert hal.load("to_delete") is None

    def test_storage_list_keys(self, hal):
        """Test listing storage keys."""
        hal.store("key1", 1)
        hal.store("key2", 2)
        hal.store("other", 3)

        keys = hal.list_keys()
        assert "key1" in keys
        assert "key2" in keys
        assert "other" in keys

        keys_prefix = hal.list_keys(prefix="key")
        assert "key1" in keys_prefix
        assert "key2" in keys_prefix
        assert "other" not in keys_prefix

    def test_get_info(self, hal):
        """Test getting HAL info."""
        hal.register_sensor("s1", SensorType.TEMPERATURE, {})
        hal.register_actuator("a1", ActuatorType.LED, {})

        info = hal.get_info()

        assert info["platform"] == "desktop"
        assert info["initialized"]
        assert "s1" in info["sensors"]
        assert "a1" in info["actuators"]

    def test_event_callbacks(self, hal):
        """Test event callbacks."""
        events = []

        hal.on("sensor_reading", lambda data: events.append(("sensor", data)))
        hal.on("actuator_executed", lambda data: events.append(("actuator", data)))

        hal.register_sensor("temp", SensorType.TEMPERATURE, {
            "simulator": lambda: 20.0,
        })
        hal.read_sensor("temp")

        hal.register_actuator("led", ActuatorType.LED, {})
        hal.execute("led", "on")

        assert len(events) >= 2
        assert events[0][0] == "sensor"
        assert events[1][0] == "actuator"


class TestSimulatedSensor:
    """Tests for SimulatedSensor helpers."""

    def test_constant(self):
        """Test constant simulator."""
        sim = SimulatedSensor.constant(42)
        assert sim() == 42
        assert sim() == 42

    def test_random_float(self):
        """Test random float simulator."""
        sim = SimulatedSensor.random_float(0, 100)

        for _ in range(10):
            value = sim()
            assert 0 <= value <= 100

    def test_temperature(self):
        """Test temperature simulator."""
        sim = SimulatedSensor.temperature(base=20, variance=5)

        for _ in range(10):
            value = sim()
            assert 15 <= value <= 25

    def test_boolean(self):
        """Test boolean simulator."""
        sim = SimulatedSensor.boolean(probability=1.0)
        assert sim() == True

        sim = SimulatedSensor.boolean(probability=0.0)
        assert sim() == False

    def test_sequence(self):
        """Test sequence simulator."""
        sim = SimulatedSensor.sequence([1, 2, 3], loop=True)

        assert sim() == 1
        assert sim() == 2
        assert sim() == 3
        assert sim() == 1  # Loops

    def test_sequence_no_loop(self):
        """Test sequence without loop."""
        sim = SimulatedSensor.sequence([1, 2], loop=False)

        assert sim() == 1
        assert sim() == 2
        assert sim() == 2  # Stays at last value


class TestCloudHAL:
    """Tests for CloudHAL."""

    @pytest.fixture
    def hal(self):
        """Create CloudHAL with memory storage."""
        os.environ["WOOEDGE_STORAGE_BACKEND"] = "memory"
        hal = CloudHAL()
        hal.initialize()
        yield hal
        hal.shutdown()

    def test_initialize(self, hal):
        """Test CloudHAL initializes."""
        assert hal._initialized
        assert hal.platform == Platform.CLOUD

    def test_memory_storage(self, hal):
        """Test memory storage backend."""
        hal.store("key", {"data": "value"})
        result = hal.load("key")

        assert result == {"data": "value"}

    def test_env_sensor(self, hal):
        """Test environment variable sensor."""
        os.environ["TEST_SENSOR_VALUE"] = "123"

        hal.register_sensor("env_test", SensorType.GENERIC, {
            "type": "env",
            "env_var": "TEST_SENSOR_VALUE",
            "value_type": "int",
        })

        reading = hal.read_sensor("env_test")
        assert reading.value == 123

        del os.environ["TEST_SENSOR_VALUE"]

    def test_mock_sensor(self, hal):
        """Test mock sensor."""
        hal.register_sensor("mock", SensorType.GENERIC, {
            "type": "mock",
            "value": 42,
        })

        reading = hal.read_sensor("mock")
        assert reading.value == 42

    def test_mock_actuator(self, hal):
        """Test mock actuator."""
        hal.register_actuator("mock", ActuatorType.GENERIC, {
            "type": "mock",
        })

        result = hal.execute("mock", "test", param=123)
        assert result

    def test_is_connected(self, hal):
        """Test is_connected (always True for cloud)."""
        assert hal.is_connected()


class TestGetHAL:
    """Tests for get_hal factory function."""

    def test_get_desktop_hal(self):
        """Test getting DesktopHAL."""
        hal = get_hal(Platform.DESKTOP)
        assert isinstance(hal, DesktopHAL)

    def test_get_cloud_hal(self):
        """Test getting CloudHAL."""
        hal = get_hal(Platform.CLOUD)
        assert isinstance(hal, CloudHAL)

    def test_auto_detect(self):
        """Test auto-detection (should return Desktop on test machine)."""
        hal = get_hal()
        assert isinstance(hal, DesktopHAL)


class TestHALContextManager:
    """Test HAL context manager support."""

    def test_context_manager(self, tmp_path):
        """Test using HAL as context manager."""
        config = HALConfig(storage_path=str(tmp_path))

        with DesktopHAL(config) as hal:
            assert hal._initialized
            hal.store("test", 123)

        # After exiting, HAL should be shutdown
        assert not hal._initialized


class TestSensorReading:
    """Tests for SensorReading dataclass."""

    def test_creation(self):
        """Test creating SensorReading."""
        reading = SensorReading(
            sensor_id="temp",
            sensor_type=SensorType.TEMPERATURE,
            value=25.5,
            unit="C",
        )

        assert reading.sensor_id == "temp"
        assert reading.value == 25.5
        assert reading.unit == "C"

    def test_to_dict(self):
        """Test serialization."""
        reading = SensorReading(
            sensor_id="temp",
            sensor_type=SensorType.TEMPERATURE,
            value=25.5,
            unit="C",
        )

        d = reading.to_dict()

        assert d["sensor_id"] == "temp"
        assert d["value"] == 25.5
        assert d["sensor_type"] == "temperature"
