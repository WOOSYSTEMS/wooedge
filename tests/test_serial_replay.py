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

    # 4-field format tests (front,left,right,hazard)

    def test_parse_4field_format(self):
        """Test parsing 4-field format: front,left,right,hazard."""
        obs = parse_line("3,2,4,0.5")
        assert obs is not None
        assert obs.front_dist == 3
        assert obs.left_dist == 2
        assert obs.right_dist == 4
        assert obs.hazard_hint == 0.5
        assert obs.timestep == 0  # Default when not provided

    def test_parse_4field_with_explicit_timestep(self):
        """Test 4-field format with explicit timestep argument."""
        obs = parse_line("3,2,4,0.5", timestep=5)
        assert obs is not None
        assert obs.front_dist == 3
        assert obs.timestep == 5

    def test_parse_4field_with_whitespace(self):
        """Test parsing 4-field line with leading/trailing whitespace."""
        obs = parse_line("  3,2,4,0.5\n", timestep=1)
        assert obs is not None
        assert obs.front_dist == 3
        assert obs.timestep == 1

    def test_parse_4field_with_carriage_return(self):
        """Test parsing 4-field line with carriage return."""
        obs = parse_line("3,2,4,0.5\r\n", timestep=2)
        assert obs is not None
        assert obs.front_dist == 3

    # 5-field format tests (timestep,front,left,right,hazard)

    def test_parse_5field_format(self):
        """Test parsing 5-field format: timestep,front,left,right,hazard."""
        obs = parse_line("10,3,2,4,0.5")
        assert obs is not None
        assert obs.timestep == 10
        assert obs.front_dist == 3
        assert obs.left_dist == 2
        assert obs.right_dist == 4
        assert obs.hazard_hint == 0.5

    def test_parse_5field_ignores_timestep_arg(self):
        """Test that 5-field format uses line timestep, not argument."""
        obs = parse_line("10,3,2,4,0.5", timestep=99)
        assert obs is not None
        assert obs.timestep == 10  # From line, not argument

    def test_parse_5field_with_whitespace(self):
        """Test parsing 5-field line with whitespace."""
        obs = parse_line("  5,3,2,4,0.5\n")
        assert obs is not None
        assert obs.timestep == 5
        assert obs.front_dist == 3

    def test_parse_5field_csv_compatible(self):
        """Test that 5-field format matches CSV schema."""
        # CSV schema: timestep,front_dist,left_dist,right_dist,hazard_hint
        obs = parse_line("0,5,3,2,0.12")
        assert obs is not None
        assert obs.timestep == 0
        assert obs.front_dist == 5
        assert obs.left_dist == 3
        assert obs.right_dist == 2
        assert obs.hazard_hint == 0.12

    # Error handling tests

    def test_parse_empty_line_returns_none(self):
        """Test that empty line returns None."""
        assert parse_line("") is None
        assert parse_line("   ") is None
        assert parse_line("\n") is None

    def test_parse_incomplete_line_returns_none(self):
        """Test that incomplete line returns None."""
        assert parse_line("3,2") is None
        assert parse_line("3,2,4") is None

    def test_parse_malformed_line_returns_none(self):
        """Test that malformed line returns None."""
        assert parse_line("abc,def,ghi,jkl") is None
        assert parse_line("3,2,4,abc") is None
        assert parse_line("not,valid,data,here") is None

    def test_parse_extra_fields_ignored(self):
        """Test that fields beyond 5 are ignored."""
        obs = parse_line("0,3,2,4,0.5,extra,fields")
        assert obs is not None
        assert obs.timestep == 0
        assert obs.front_dist == 3
        assert obs.hazard_hint == 0.5

    def test_parse_float_distances(self):
        """Test parsing with integer distances."""
        obs = parse_line("3,2,4,0.5")
        assert obs is not None
        assert isinstance(obs.front_dist, int)

    def test_timestep_from_argument(self):
        """Test that timestep argument works for 4-field format."""
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
            "/dev/ttyUSB0", 9600, timeout=1.0
        )

    @patch('wooedge.io.serial_replay.serial')
    def test_open_stores_handle(self, mock_serial_module):
        """Test that open() stores the serial handle on self._ser."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True

        source = SerialReplaySource("/dev/ttyUSB0")
        assert source._ser is None
        source.open()
        assert source._ser is mock_serial

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
            "/dev/ttyUSB0", 115200, timeout=1.0
        )

    @patch('wooedge.io.serial_replay.serial')
    def test_read_one_timeout_returns_none(self, mock_serial_module):
        """Test that read_one() returns None on timeout (empty bytes)."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True
        mock_serial.readline.return_value = b""  # Timeout

        source = SerialReplaySource("/dev/ttyUSB0")
        obs = source.read_one()

        assert obs is None
        mock_serial.readline.assert_called_once()

    @patch('wooedge.io.serial_replay.serial')
    def test_iteration_yields_observations(self, mock_serial_module):
        """Test that iteration yields observations as they arrive."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True
        # Simulate: valid line, timeout, valid line
        mock_serial.readline.side_effect = [
            b"3,2,4,0.1\n",
            b"",  # Timeout - should continue
            b"5,4,3,0.2\n",
        ]

        source = SerialReplaySource("/dev/ttyUSB0")
        source.open()

        # Read observations directly
        obs1 = source.read_one()
        obs2 = source.read_one()  # Timeout, returns None
        obs3 = source.read_one()

        assert obs1 is not None
        assert obs1.front_dist == 3
        assert obs2 is None  # Timeout
        assert obs3 is not None
        assert obs3.front_dist == 5
        # Verify handle was used
        assert mock_serial.readline.call_count == 3

    @patch('wooedge.io.serial_replay.serial')
    def test_read_5field_format(self, mock_serial_module):
        """Test reading 5-field format with embedded timestep."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True
        mock_serial.readline.return_value = b"42,3,2,4,0.5\n"

        source = SerialReplaySource("/dev/ttyUSB0")
        obs = source.read_one()

        assert obs is not None
        assert obs.timestep == 42  # From line, not auto-increment
        assert obs.front_dist == 3
        assert obs.hazard_hint == 0.5

    @patch('wooedge.io.serial_replay.serial')
    def test_mixed_formats(self, mock_serial_module):
        """Test reading mixed 4-field and 5-field formats."""
        mock_serial = Mock()
        mock_serial_module.Serial.return_value = mock_serial
        mock_serial.is_open = True
        # First 4-field, then 5-field, then 4-field
        mock_serial.readline.side_effect = [
            b"3,2,4,0.1\n",      # 4-field, timestep=0
            b"99,5,4,3,0.2\n",   # 5-field, timestep=99
            b"1,1,1,0.3\n",      # 4-field, timestep=1
        ]

        source = SerialReplaySource("/dev/ttyUSB0")
        obs1 = source.read_one()
        obs2 = source.read_one()
        obs3 = source.read_one()

        assert obs1.timestep == 0   # Auto-increment
        assert obs1.front_dist == 3
        assert obs2.timestep == 99  # From line
        assert obs2.front_dist == 5
        assert obs3.timestep == 2   # Auto-increment continues
        assert obs3.front_dist == 1


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
