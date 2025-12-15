"""
Serial Replay Source

Reads live sensor data from a serial port and emits observations compatible
with DecisionSafety.observe(). Enables real-time safety checking.

Supported line formats (no header):
    front,left,right,hazard           (4 fields, timestep auto-incremented)
    timestep,front,left,right,hazard  (5 fields, timestep from line)

Example:
    3,2,4,0.1
    2,2,4,0.15
    1,2,4,0.8
"""

from typing import Iterator, Optional, Generator
from .csv_replay import ReplayObservation

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


def parse_line(line: str, timestep: Optional[int] = None) -> Optional[ReplayObservation]:
    """
    Parse a serial line into ReplayObservation.

    Accepts two formats:
        - 4 fields: front,left,right,hazard (uses provided timestep or 0)
        - 5 fields: timestep,front,left,right,hazard (uses timestep from line)

    Args:
        line: Raw line from serial
        timestep: Fallback timestep for 4-field format (default 0)

    Returns:
        ReplayObservation if valid, None if malformed
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split(',')

    try:
        if len(parts) >= 5:
            # Format: timestep,front,left,right,hazard
            return ReplayObservation(
                timestep=int(parts[0]),
                front_dist=int(parts[1]),
                left_dist=int(parts[2]),
                right_dist=int(parts[3]),
                hazard_hint=float(parts[4]),
                scanned=False
            )
        elif len(parts) >= 4:
            # Format: front,left,right,hazard
            return ReplayObservation(
                timestep=timestep if timestep is not None else 0,
                front_dist=int(parts[0]),
                left_dist=int(parts[1]),
                right_dist=int(parts[2]),
                hazard_hint=float(parts[3]),
                scanned=False
            )
        else:
            return None
    except (ValueError, IndexError):
        return None


class SerialReplaySource:
    """
    Reads live sensor data from serial port for DecisionSafety replay.

    Usage:
        source = SerialReplaySource("/dev/ttyUSB0", baud=115200)
        for obs in source:
            safety.observe(obs)
            result = safety.propose(action)
            if result["decision"] == "DELAY":
                break  # Stop on safety concern

    Supported line formats (no header):
        - front,left,right,hazard (4 fields, timestep auto-incremented)
        - timestep,front,left,right,hazard (5 fields, timestep from line)
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        """
        Initialize serial replay source.

        Args:
            port: Serial port path (e.g., /dev/cu.usbserial-10)
            baud: Baud rate (default 115200)
            timeout: Read timeout in seconds (default 1.0)
        """
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial is required. Install with: pip install pyserial")

        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial: Optional[serial.Serial] = None
        self._timestep = 0

    def open(self) -> None:
        """Open serial connection."""
        if self._serial is not None and self._serial.is_open:
            return
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=self.timeout
        )
        self._timestep = 0

    def close(self) -> None:
        """Close serial connection."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def read_one(self) -> Optional[ReplayObservation]:
        """
        Read and parse one observation from serial.

        Returns:
            ReplayObservation if valid line received, None on timeout/error
        """
        if self._serial is None or not self._serial.is_open:
            self.open()

        try:
            line = self._serial.readline().decode('utf-8', errors='ignore')
            obs = parse_line(line, self._timestep)
            if obs is not None:
                self._timestep += 1
            return obs
        except (serial.SerialException, OSError):
            return None

    def __iter__(self) -> Generator[ReplayObservation, None, None]:
        """
        Iterate over observations from serial port.

        Yields observations until connection is closed or interrupted.
        Skips malformed lines silently.
        """
        self.open()
        try:
            while True:
                obs = self.read_one()
                if obs is not None:
                    yield obs
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def __enter__(self) -> "SerialReplaySource":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @property
    def is_open(self) -> bool:
        """Check if serial connection is open."""
        return self._serial is not None and self._serial.is_open
