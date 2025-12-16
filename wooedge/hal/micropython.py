"""
MicroPython HAL Implementation

HAL for ESP32, Raspberry Pi Pico, and other MicroPython boards.
Uses flash storage, machine module for GPIO, and urequests for HTTP.

NOTE: This module is designed to run on MicroPython. It will not
work correctly on standard Python without mocking the micropython modules.
"""

from __future__ import annotations

# MicroPython compatibility - these imports may fail on CPython
try:
    import machine
    import utime
    import ujson as json
    import gc
    MICROPYTHON = True
except ImportError:
    # Fallback for testing on CPython
    import time as utime
    import json
    MICROPYTHON = False

from typing import Dict, List, Any, Optional, Callable

# Import base - may need to handle differently on MicroPython
try:
    from .base import (
        HAL,
        HALConfig,
        Platform,
        SensorType,
        ActuatorType,
        SensorReading,
    )
except ImportError:
    # On MicroPython, may need relative import adjustment
    from base import (
        HAL,
        HALConfig,
        Platform,
        SensorType,
        ActuatorType,
        SensorReading,
    )


class MicroPythonHAL(HAL):
    """
    MicroPython HAL for ESP32, Pico, and similar boards.

    Features:
    - Flash filesystem storage
    - GPIO pin control
    - I2C/SPI sensor support
    - WiFi networking
    - Low memory footprint

    Example (ESP32):
        hal = MicroPythonHAL()
        hal.initialize()

        # Register ultrasonic sensor on pins
        hal.register_sensor("distance", SensorType.DISTANCE, {
            "type": "hcsr04",
            "trigger_pin": 5,
            "echo_pin": 18,
        })

        # Read sensor
        reading = hal.read_sensor("distance")
        print(f"Distance: {reading.value} cm")

        # Control LED
        hal.register_actuator("led", ActuatorType.LED, {"pin": 2})
        hal.execute("led", "on")
    """

    def __init__(self, config: HALConfig = None):
        super().__init__(config)
        self._start_ticks = 0
        self._pins: Dict[str, Any] = {}  # Pin objects
        self._i2c: Optional[Any] = None
        self._spi: Optional[Any] = None
        self._wifi: Optional[Any] = None

    @property
    def platform(self) -> Platform:
        return Platform.MICROPYTHON

    def initialize(self) -> None:
        """Initialize MicroPython HAL."""
        if self._initialized:
            return

        if MICROPYTHON:
            self._start_ticks = utime.ticks_ms()

            # Run garbage collection
            gc.collect()

            # Initialize WiFi if configured
            if self.config.sensors.get("wifi") or self.config.actuators.get("wifi"):
                self._init_wifi()
        else:
            self._start_ticks = int(utime.time() * 1000)

        self._initialized = True
        self.emit("initialized")

    def _init_wifi(self) -> None:
        """Initialize WiFi connection."""
        if not MICROPYTHON:
            return

        try:
            import network
            self._wifi = network.WLAN(network.STA_IF)
            self._wifi.active(True)
        except Exception as e:
            print(f"WiFi init error: {e}")

    def connect_wifi(self, ssid: str, password: str, timeout: int = 10) -> bool:
        """
        Connect to WiFi network.

        Args:
            ssid: Network SSID
            password: Network password
            timeout: Connection timeout in seconds

        Returns:
            True if connected
        """
        if not MICROPYTHON or not self._wifi:
            return False

        try:
            self._wifi.connect(ssid, password)

            # Wait for connection
            start = utime.ticks_ms()
            while not self._wifi.isconnected():
                if utime.ticks_diff(utime.ticks_ms(), start) > timeout * 1000:
                    return False
                utime.sleep_ms(100)

            self.emit("connected", self._wifi.ifconfig())
            return True
        except Exception as e:
            print(f"WiFi connect error: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown MicroPython HAL."""
        # Release pins
        for pin_id, pin in self._pins.items():
            try:
                if MICROPYTHON:
                    pin.value(0)  # Set low
            except:
                pass

        self._initialized = False
        self.emit("shutdown")

    # ==================== Time ====================

    def get_time(self) -> float:
        """Get current time (may not be accurate without RTC/NTP)."""
        if MICROPYTHON:
            return utime.time()
        return utime.time()

    def get_ticks_ms(self) -> int:
        """Get milliseconds since boot."""
        if MICROPYTHON:
            return utime.ticks_ms()
        return int(utime.time() * 1000) - self._start_ticks

    def sleep(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        if MICROPYTHON:
            utime.sleep(seconds)
        else:
            utime.sleep(seconds)

    def sleep_ms(self, ms: int) -> None:
        """Sleep for specified milliseconds."""
        if MICROPYTHON:
            utime.sleep_ms(ms)
        else:
            utime.sleep(ms / 1000.0)

    # ==================== Sensors ====================

    def _on_sensor_registered(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        config: Dict[str, Any],
    ) -> None:
        """Setup sensor hardware."""
        if not MICROPYTHON:
            return

        sensor_hw_type = config.get("type", "").lower()

        if sensor_hw_type == "hcsr04":
            # HC-SR04 ultrasonic sensor
            self._setup_hcsr04(sensor_id, config)
        elif sensor_hw_type == "dht11" or sensor_hw_type == "dht22":
            # DHT temperature/humidity sensor
            self._setup_dht(sensor_id, config)
        elif sensor_hw_type == "adc":
            # Analog sensor
            self._setup_adc(sensor_id, config)
        elif sensor_hw_type == "digital":
            # Digital input
            self._setup_digital_input(sensor_id, config)

    def _setup_hcsr04(self, sensor_id: str, config: Dict[str, Any]) -> None:
        """Setup HC-SR04 ultrasonic sensor."""
        trigger_pin = config.get("trigger_pin")
        echo_pin = config.get("echo_pin")

        if trigger_pin is None or echo_pin is None:
            return

        self._pins[f"{sensor_id}_trigger"] = machine.Pin(trigger_pin, machine.Pin.OUT)
        self._pins[f"{sensor_id}_echo"] = machine.Pin(echo_pin, machine.Pin.IN)

    def _setup_dht(self, sensor_id: str, config: Dict[str, Any]) -> None:
        """Setup DHT sensor."""
        try:
            import dht
            pin = config.get("pin")
            sensor_type = config.get("type", "dht11").lower()

            if pin is None:
                return

            pin_obj = machine.Pin(pin)
            if sensor_type == "dht22":
                self._pins[sensor_id] = dht.DHT22(pin_obj)
            else:
                self._pins[sensor_id] = dht.DHT11(pin_obj)
        except ImportError:
            print("DHT module not available")

    def _setup_adc(self, sensor_id: str, config: Dict[str, Any]) -> None:
        """Setup ADC sensor."""
        pin = config.get("pin")
        if pin is not None:
            self._pins[sensor_id] = machine.ADC(machine.Pin(pin))

    def _setup_digital_input(self, sensor_id: str, config: Dict[str, Any]) -> None:
        """Setup digital input."""
        pin = config.get("pin")
        pull = config.get("pull", "none").lower()

        if pin is None:
            return

        if pull == "up":
            self._pins[sensor_id] = machine.Pin(pin, machine.Pin.IN, machine.Pin.PULL_UP)
        elif pull == "down":
            self._pins[sensor_id] = machine.Pin(pin, machine.Pin.IN, machine.Pin.PULL_DOWN)
        else:
            self._pins[sensor_id] = machine.Pin(pin, machine.Pin.IN)

    def read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """Read sensor value."""
        if sensor_id not in self._sensors:
            return None

        sensor_info = self._sensors[sensor_id]
        sensor_type = sensor_info["type"]
        config = sensor_info["config"]

        value = None
        unit = ""

        if not MICROPYTHON:
            # Return mock value for testing
            value = config.get("mock_value", 0)
            unit = config.get("unit", "")
        else:
            sensor_hw_type = config.get("type", "").lower()

            if sensor_hw_type == "hcsr04":
                value = self._read_hcsr04(sensor_id)
                unit = "cm"
            elif sensor_hw_type in ("dht11", "dht22"):
                value = self._read_dht(sensor_id, config.get("reading", "temperature"))
                unit = "C" if config.get("reading") != "humidity" else "%"
            elif sensor_hw_type == "adc":
                value = self._read_adc(sensor_id, config)
                unit = config.get("unit", "")
            elif sensor_hw_type == "digital":
                value = self._pins[sensor_id].value()
                unit = ""

        if value is None:
            return None

        return SensorReading(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            value=value,
            unit=unit,
        )

    def _read_hcsr04(self, sensor_id: str) -> Optional[float]:
        """Read HC-SR04 ultrasonic sensor."""
        trigger = self._pins.get(f"{sensor_id}_trigger")
        echo = self._pins.get(f"{sensor_id}_echo")

        if not trigger or not echo:
            return None

        try:
            # Send trigger pulse
            trigger.value(0)
            utime.sleep_us(2)
            trigger.value(1)
            utime.sleep_us(10)
            trigger.value(0)

            # Wait for echo
            timeout = utime.ticks_us()
            while echo.value() == 0:
                if utime.ticks_diff(utime.ticks_us(), timeout) > 30000:
                    return None

            start = utime.ticks_us()
            while echo.value() == 1:
                if utime.ticks_diff(utime.ticks_us(), start) > 30000:
                    return None

            duration = utime.ticks_diff(utime.ticks_us(), start)

            # Calculate distance (speed of sound = 343 m/s)
            distance = (duration * 0.0343) / 2
            return round(distance, 1)
        except Exception as e:
            print(f"HCSR04 read error: {e}")
            return None

    def _read_dht(self, sensor_id: str, reading: str) -> Optional[float]:
        """Read DHT sensor."""
        sensor = self._pins.get(sensor_id)
        if not sensor:
            return None

        try:
            sensor.measure()
            if reading == "humidity":
                return sensor.humidity()
            return sensor.temperature()
        except Exception as e:
            print(f"DHT read error: {e}")
            return None

    def _read_adc(self, sensor_id: str, config: Dict[str, Any]) -> Optional[float]:
        """Read ADC sensor."""
        adc = self._pins.get(sensor_id)
        if not adc:
            return None

        try:
            raw = adc.read()
            # Scale if configured
            min_val = config.get("min", 0)
            max_val = config.get("max", 4095)
            scale_min = config.get("scale_min", 0)
            scale_max = config.get("scale_max", 100)

            # Linear interpolation
            scaled = scale_min + (raw - min_val) / (max_val - min_val) * (scale_max - scale_min)
            return round(scaled, 2)
        except Exception as e:
            print(f"ADC read error: {e}")
            return None

    # ==================== Actuators ====================

    def _on_actuator_registered(
        self,
        actuator_id: str,
        actuator_type: ActuatorType,
        config: Dict[str, Any],
    ) -> None:
        """Setup actuator hardware."""
        if not MICROPYTHON:
            return

        actuator_hw_type = config.get("type", actuator_type.value).lower()

        if actuator_hw_type in ("led", "gpio", "relay"):
            pin = config.get("pin")
            if pin is not None:
                self._pins[actuator_id] = machine.Pin(pin, machine.Pin.OUT)
        elif actuator_hw_type == "pwm":
            pin = config.get("pin")
            freq = config.get("freq", 1000)
            if pin is not None:
                pwm = machine.PWM(machine.Pin(pin))
                pwm.freq(freq)
                self._pins[actuator_id] = pwm
        elif actuator_hw_type == "servo":
            pin = config.get("pin")
            if pin is not None:
                pwm = machine.PWM(machine.Pin(pin))
                pwm.freq(50)  # 50Hz for servo
                self._pins[actuator_id] = pwm

    def execute(
        self,
        actuator_id: str,
        command: str,
        **kwargs,
    ) -> bool:
        """Execute actuator command."""
        if actuator_id not in self._actuators:
            return False

        actuator_info = self._actuators[actuator_id]
        actuator_type = actuator_info["type"]
        config = actuator_info["config"]

        if not MICROPYTHON:
            # Mock execution for testing
            self.emit("actuator_executed", {
                "actuator_id": actuator_id,
                "command": command,
                "kwargs": kwargs,
            })
            return True

        try:
            pin = self._pins.get(actuator_id)
            if not pin:
                return False

            hw_type = config.get("type", actuator_type.value).lower()

            if hw_type in ("led", "gpio", "relay"):
                if command == "on":
                    pin.value(1)
                elif command == "off":
                    pin.value(0)
                elif command == "toggle":
                    pin.value(1 - pin.value())
                elif command == "set":
                    pin.value(kwargs.get("value", 0))

            elif hw_type == "pwm":
                if command == "set":
                    duty = kwargs.get("duty", 0)
                    pin.duty(int(duty))
                elif command == "off":
                    pin.duty(0)

            elif hw_type == "servo":
                if command == "set":
                    angle = kwargs.get("angle", 90)
                    # Convert angle to duty cycle (0-180 -> 26-128 for most servos)
                    duty = int(26 + (angle / 180) * 102)
                    pin.duty(duty)

            self.emit("actuator_executed", {
                "actuator_id": actuator_id,
                "command": command,
                "kwargs": kwargs,
            })
            return True

        except Exception as e:
            print(f"Actuator execute error: {e}")
            return False

    # ==================== Storage ====================

    def store(self, key: str, value: Any) -> bool:
        """Store value to flash filesystem."""
        try:
            filename = f"/data/{key}.json"

            # Ensure directory exists
            try:
                import os
                os.mkdir("/data")
            except:
                pass

            with open(filename, 'w') as f:
                json.dump(value, f)
            return True
        except Exception as e:
            print(f"Store error: {e}")
            return False

    def load(self, key: str, default: Any = None) -> Any:
        """Load value from flash filesystem."""
        try:
            filename = f"/data/{key}.json"
            with open(filename, 'r') as f:
                return json.load(f)
        except:
            return default

    def delete(self, key: str) -> bool:
        """Delete from flash filesystem."""
        try:
            import os
            os.remove(f"/data/{key}.json")
            return True
        except:
            return False

    def list_keys(self, prefix: str = "") -> List[str]:
        """List storage keys."""
        try:
            import os
            files = os.listdir("/data")
            keys = []
            for f in files:
                if f.endswith(".json"):
                    key = f[:-5]
                    if key.startswith(prefix):
                        keys.append(key)
            return keys
        except:
            return []

    # ==================== Networking ====================

    def is_connected(self) -> bool:
        """Check WiFi connection."""
        if not MICROPYTHON or not self._wifi:
            return False
        return self._wifi.isconnected()

    def http_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        body: Any = None,
        timeout: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request."""
        if not MICROPYTHON:
            return None

        try:
            import urequests

            headers = headers or {}
            data = None
            if body is not None:
                if isinstance(body, dict):
                    data = json.dumps(body)
                    headers.setdefault("Content-Type", "application/json")
                else:
                    data = str(body)

            if method.upper() == "GET":
                response = urequests.get(url, headers=headers)
            elif method.upper() == "POST":
                response = urequests.post(url, headers=headers, data=data)
            elif method.upper() == "PUT":
                response = urequests.put(url, headers=headers, data=data)
            elif method.upper() == "DELETE":
                response = urequests.delete(url, headers=headers)
            else:
                return None

            result = {
                "status": response.status_code,
                "body": response.text,
            }
            response.close()
            return result

        except Exception as e:
            print(f"HTTP error: {e}")
            return None

    # ==================== Utilities ====================

    def get_free_memory(self) -> int:
        """Get free memory in bytes."""
        if MICROPYTHON:
            gc.collect()
            return gc.mem_free()
        return 0

    def get_cpu_freq(self) -> int:
        """Get CPU frequency in Hz."""
        if MICROPYTHON:
            try:
                return machine.freq()
            except:
                return 0
        return 0

    def gc_collect(self) -> int:
        """Run garbage collection, return free memory."""
        if MICROPYTHON:
            gc.collect()
            return gc.mem_free()
        return 0
