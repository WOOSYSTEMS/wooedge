"""
Tests for Serial Replay Source.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.io.serial_replay import parse_line, SerialReplaySource
from wooedge.io.csv_replay import ReplayObservation


class TestParseLine:
    """Tests for parse_line function."""

    def test_parse_valid_line(self):
        """Test parsing a valid sensor line."""
        obs = parse_line("3,2,4,0.5", timestep=0)
        assert obs is not None
        assert obs.front_dist == 3
        assert obs.left_dist == 2
        assert obs.right_dist == 4
        assert obs.hazard_hint == 0.5
        assert obs.timestep == 0

    def test_parse_line_with_whitespace(self):
        """Test parsing line with leading/trailing whitespace."""
        obs = parse_line("  3,2,4,0.5\n", timestep=1)
        assert obs is not None
        assert obs.front_dist == 3
        assert obs.timestep == 1

    def test_parse_line_with_carriage_return(self):
        """Test parsing line with carriage return."""
        obs = parse_line("3,2,4,0.5\r\n", timestep=2)
        assert obs is not None
        assert obs.front_dist == 3

    def test_parse_empty_line_returns_none(self):
        """Test that empty line returns None."""
        assert parse_line("", timestep=0) is None
        assert parse_line("   ", timestep=0) is None
        assert parse_line("\n", timestep=0) is None

    def test_parse_incomplete_line_returns_none(self):
        """Test that incomplete line returns None."""
        assert parse_line("3,2", timestep=0) is None
        assert parse_line("3,2,4", timestep=0) is None

    def test_parse_malformed_line_returns_none(self):
        """Test that malformed line returns None."""
        assert parse_line("abc,def,ghi,jkl", timestep=0) is None
        assert parse_line("3,2,4,abc", timestep=0) is None
        assert parse_line("not,valid,data,here", timestep=0) is None

    def test_parse_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        obs = parse_line("3,2,4,0.5,extra,fields", timestep=0)
        assert obs is not None
        assert obs.front_dist == 3
        assert obs.hazard_hint == 0.5

    def test_parse_float_distances(self):
        """Test parsing with float values for distances (truncated to int)."""
        obs = parse_line("3,2,4,0.5", timestep=0)
        assert obs is not None
        assert isinstance(obs.front_dist, int)

    def test_timestep_increments(self):
        """Test that timestep is set correctly."""
        obs1 = parse_line("3,2,4,0.5", timestep=0)
        obs2 = parse_line("3,2,4,0.5", timestep=5)
        obs3 = parse_line("3,2,4,0.5", timestep=100)
        assert obs1.timestep == 0
        assert obs2.timestep == 5
        assert obs3.timestep == 100


class TestSerialReplaySource:
    """Tests for SerialReplaySource with mocked serial."""

    @patch('wooedge.io.serial_replay.serial')
    def test_open_creates_connection(self, mock_serial_module):
        """Test that open() creates serial connection."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True

        source = SerialReplaySource("/dev/ttyUSB0", baud=9600)
        source.open()

        mock_serial_module.Serial.assert_called_once_with(
            port="/dev/ttyUSB0",
            baudrate=9600,
            timeout=1.0
        )

    @patch('wooedge.io.serial_replay.serial')
    def test_close_closes_connection(self, mock_serial_module):
        """Test that close() closes serial connection."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True

        source = SerialReplaySource("/dev/ttyUSB0")
        source.open()
        source.close()

        mock_serial.close.assert_called_once()

    @patch('wooedge.io.serial_replay.serial')
    def test_read_one_parses_line(self, mock_serial_module):
        """Test that read_one() parses a valid line."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True
        mock_serial.readline.return_value = b"3,2,4,0.5\n"

        source = SerialReplaySource("/dev/ttyUSB0")
        obs = source.read_one()

        assert obs is not None
        assert obs.front_dist == 3
        assert obs.left_dist == 2
        assert obs.right_dist == 4
        assert obs.hazard_hint == 0.5
        assert obs.timestep == 0

    @patch('wooedge.io.serial_replay.serial')
    def test_read_one_increments_timestep(self, mock_serial_module):
        """Test that timestep increments with each valid read."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True
        mock_serial.readline.return_value = b"3,2,4,0.5\n"

        source = SerialReplaySource("/dev/ttyUSB0")
        obs1 = source.read_one()
        obs2 = source.read_one()
        obs3 = source.read_one()

        assert obs1.timestep == 0
        assert obs2.timestep == 1
        assert obs3.timestep == 2

    @patch('wooedge.io.serial_replay.serial')
    def test_read_one_skips_malformed(self, mock_serial_module):
        """Test that read_one() returns None for malformed lines."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True
        mock_serial.readline.return_value = b"invalid\n"

        source = SerialReplaySource("/dev/ttyUSB0")
        obs = source.read_one()

        assert obs is None

    @patch('wooedge.io.serial_replay.serial')
    def test_read_one_handles_empty_line(self, mock_serial_module):
        """Test that read_one() handles empty lines."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True
        mock_serial.readline.return_value = b"\n"

        source = SerialReplaySource("/dev/ttyUSB0")
        obs = source.read_one()

        assert obs is None

    @patch('wooedge.io.serial_replay.serial')
    def test_context_manager(self, mock_serial_module):
        """Test context manager opens and closes connection."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True

        with SerialReplaySource("/dev/ttyUSB0") as source:
            assert source.is_open

        mock_serial.close.assert_called()

    @patch('wooedge.io.serial_replay.serial')
    def test_is_open_property(self, mock_serial_module):
        """Test is_open property reflects connection state."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True

        source = SerialReplaySource("/dev/ttyUSB0")
        assert not source.is_open

        source.open()
        assert source.is_open

        mock_serial.is_open = False
        assert not source.is_open

    @patch('wooedge.io.serial_replay.serial')
    def test_default_baud_rate(self, mock_serial_module):
        """Test default baud rate is 115200."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True

        source = SerialReplaySource("/dev/ttyUSB0")
        source.open()

        mock_serial_module.Serial.assert_called_with(
            port="/dev/ttyUSB0",
            baudrate=115200,
            timeout=1.0
        )


class TestSerialReplayCLI:
    """Tests for replay_serial CLI command."""

    def test_cli_help_shows_options(self):
        """Test that CLI help shows all options."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_serial", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--baud" in result.stdout
        assert "--delay" in result.stdout
        assert "--quiet" in result.stdout

    def test_cli_requires_port(self):
        """Test that CLI requires --port argument."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_serial"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "port" in result.stderr.lower()
