"""
Tests for CLI replay_csv command.
"""

import pytest
import subprocess
import sys
import os


class TestCLIReplayCSV:
    """Tests for replay_csv CLI command."""

    def test_replay_csv_default_file(self):
        """Test replay_csv runs with default file and exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "CSV Replay Safety Demo" in result.stdout
        assert "Loaded" in result.stdout

    def test_replay_csv_with_file_argument(self):
        """Test replay_csv with explicit --file argument."""
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "data", "sample_nav_log.csv"
        )
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv", "--file", csv_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "sample_nav_log.csv" in result.stdout

    def test_replay_csv_with_short_file_flag(self):
        """Test replay_csv with -f short flag."""
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "data", "sample_nav_log.csv"
        )
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv", "-f", csv_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "Loaded 10 observations" in result.stdout

    def test_replay_csv_with_delay(self):
        """Test replay_csv with --delay argument."""
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv", "--delay", "0.0"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "Delay: 0.0s" in result.stdout

    def test_replay_csv_with_short_delay_flag(self):
        """Test replay_csv with -d short flag."""
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv", "-d", "0.0"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0

    def test_replay_csv_quiet_mode(self):
        """Test replay_csv with --quiet flag suppresses reasons."""
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv", "--quiet"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        # Quiet mode should not show "Reason:" lines
        assert "Reason:" not in result.stdout

    def test_replay_csv_short_quiet_flag(self):
        """Test replay_csv with -q short flag."""
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv", "-q"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "Reason:" not in result.stdout

    def test_replay_csv_nonexistent_file_fails(self):
        """Test replay_csv with non-existent file exits with error."""
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv", "--file", "/nonexistent/file.csv"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode != 0

    def test_replay_csv_help(self):
        """Test replay_csv help displays correctly."""
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "replay_csv", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "--file" in result.stdout
        assert "--delay" in result.stdout
        assert "--quiet" in result.stdout
