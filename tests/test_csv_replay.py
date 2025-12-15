"""
Tests for CSV Replay Source.
"""

import pytest
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.io.csv_replay import CSVReplaySource, ReplayObservation


class TestCSVReplaySource:
    """Tests for CSVReplaySource."""

    def test_parse_valid_csv(self):
        """Test parsing a valid CSV file."""
        csv_content = """timestep,front_dist,left_dist,right_dist,hazard_hint
0,5,3,2,0.1
1,4,3,2,0.2
2,3,2,2,0.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            filepath = f.name

        try:
            source = CSVReplaySource(filepath)
            source.load()

            assert len(source) == 3
            assert source[0].timestep == 0
            assert source[0].front_dist == 5
            assert source[0].hazard_hint == 0.1
            assert source[2].timestep == 2
            assert source[2].hazard_hint == 0.5
        finally:
            os.unlink(filepath)

    def test_replay_order_preserved(self):
        """Test that observations are returned in CSV row order."""
        csv_content = """timestep,front_dist,left_dist,right_dist,hazard_hint
0,5,3,2,0.1
1,4,3,2,0.2
2,3,2,2,0.3
3,2,2,2,0.4
4,1,2,2,0.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            filepath = f.name

        try:
            source = CSVReplaySource(filepath)

            timesteps = []
            for obs in source:
                timesteps.append(obs.timestep)

            assert timesteps == [0, 1, 2, 3, 4], "Observations not in correct order"
        finally:
            os.unlink(filepath)

    def test_validate_order_ascending(self):
        """Test validation of ascending timestep order."""
        csv_content = """timestep,front_dist,left_dist,right_dist,hazard_hint
0,5,3,2,0.1
1,4,3,2,0.2
2,3,2,2,0.3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            filepath = f.name

        try:
            source = CSVReplaySource(filepath)
            assert source.validate_order() == True
        finally:
            os.unlink(filepath)

    def test_validate_order_out_of_order(self):
        """Test validation catches out-of-order timesteps."""
        csv_content = """timestep,front_dist,left_dist,right_dist,hazard_hint
0,5,3,2,0.1
2,4,3,2,0.2
1,3,2,2,0.3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            filepath = f.name

        try:
            source = CSVReplaySource(filepath)
            assert source.validate_order() == False
        finally:
            os.unlink(filepath)

    def test_missing_column_raises_error(self):
        """Test that missing required columns raise ValueError."""
        csv_content = """timestep,front_dist,left_dist
0,5,3
1,4,3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            filepath = f.name

        try:
            source = CSVReplaySource(filepath)
            with pytest.raises(ValueError, match="missing required columns"):
                source.load()
        finally:
            os.unlink(filepath)

    def test_observation_to_tuple(self):
        """Test ReplayObservation.to_tuple() for hashing compatibility."""
        obs = ReplayObservation(
            timestep=5,
            front_dist=3,
            left_dist=2,
            right_dist=4,
            hazard_hint=0.55,
            scanned=True
        )

        expected = (3, 2, 4, 0.6, True)  # hazard_hint rounded to 0.1
        assert obs.to_tuple() == expected

    def test_iteration_multiple_times(self):
        """Test that source can be iterated multiple times."""
        csv_content = """timestep,front_dist,left_dist,right_dist,hazard_hint
0,5,3,2,0.1
1,4,3,2,0.2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            filepath = f.name

        try:
            source = CSVReplaySource(filepath)

            # First iteration
            first_pass = list(source)
            # Second iteration
            second_pass = list(source)

            assert len(first_pass) == len(second_pass) == 2
            assert first_pass[0].timestep == second_pass[0].timestep
        finally:
            os.unlink(filepath)
