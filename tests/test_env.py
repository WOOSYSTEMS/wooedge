"""Tests for the environment module."""

import numpy as np
import pytest

from wooedge.env import (
    GridWorld, EnvConfig, Action, ACTION_DELTAS,
    Observation, compute_true_distances, compute_local_patch
)


class TestGridWorld:
    """Tests for GridWorld environment."""

    def test_initialization(self):
        """Test basic initialization."""
        env = GridWorld(EnvConfig(seed=42))

        assert env.grid_size == 12
        assert env.grid.shape == (12, 12)
        assert env.agent_pos is not None
        assert env.goal_pos is not None
        assert env.agent_pos != env.goal_pos

    def test_deterministic_with_seed(self):
        """Test that same seed produces same environment."""
        env1 = GridWorld(EnvConfig(seed=123))
        env2 = GridWorld(EnvConfig(seed=123))

        np.testing.assert_array_equal(env1.grid, env2.grid)
        assert env1.agent_pos == env2.agent_pos
        assert env1.goal_pos == env2.goal_pos

    def test_different_seeds(self):
        """Test that different seeds produce different environments."""
        env1 = GridWorld(EnvConfig(seed=1))
        env2 = GridWorld(EnvConfig(seed=2))

        # Grids should be different (with very high probability)
        assert not np.array_equal(env1.grid, env2.grid)

    def test_walls_on_border(self):
        """Test that border is all walls."""
        env = GridWorld(EnvConfig(seed=42))

        # Top and bottom borders
        assert np.all(env.grid[0, :] == 1)
        assert np.all(env.grid[-1, :] == 1)

        # Left and right borders
        assert np.all(env.grid[:, 0] == 1)
        assert np.all(env.grid[:, -1] == 1)

    def test_reset(self):
        """Test environment reset."""
        env = GridWorld(EnvConfig(seed=42))
        original_pos = env.agent_pos

        # Take some steps
        env.step(Action.DOWN)
        env.step(Action.RIGHT)

        assert env.agent_pos != original_pos

        # Reset
        obs = env.reset()

        assert isinstance(obs, Observation)
        assert env.steps_taken == 0
        assert not env.done

    def test_step_basic(self):
        """Test basic step functionality."""
        env = GridWorld(EnvConfig(seed=42, slip_prob=0.0))
        initial_pos = env.agent_pos

        obs, cost, done, info = env.step(Action.STAY)

        assert isinstance(obs, Observation)
        assert isinstance(cost, float)
        assert isinstance(done, bool)
        assert env.agent_pos == initial_pos
        assert info["true_pos"] == env.agent_pos

    def test_step_movement(self):
        """Test that movement works correctly."""
        # Create environment with no slip
        env = GridWorld(EnvConfig(seed=42, slip_prob=0.0))

        # Find a position where we can move down
        env.reset()
        initial_y, initial_x = env.agent_pos

        # Try to move down
        obs, cost, done, info = env.step(Action.DOWN)

        # Position should change if there's no wall below
        expected_y = initial_y + 1
        if env.grid[expected_y, initial_x] != 1:  # Not a wall
            assert env.agent_pos == (expected_y, initial_x)
        else:
            assert env.agent_pos == (initial_y, initial_x)

    def test_wall_collision(self):
        """Test that agent can't walk through walls."""
        env = GridWorld(EnvConfig(seed=42, slip_prob=0.0))

        # Try to walk into border wall repeatedly
        for _ in range(20):
            env.step(Action.UP)

        # Agent should be stopped by wall
        assert env.agent_pos[0] >= 1  # Can't be in top border

    def test_observation_structure(self):
        """Test observation structure."""
        env = GridWorld(EnvConfig(seed=42))
        obs = env.get_observation()

        assert obs.local_patch.shape == (3, 3)
        assert obs.distance_sensors.shape == (4,)
        assert isinstance(obs.noisy, bool)

    def test_true_observation(self):
        """Test noise-free observation."""
        env = GridWorld(EnvConfig(seed=42))
        obs = env.get_true_observation()

        assert not obs.noisy

    def test_goal_reached(self):
        """Test goal detection."""
        env = GridWorld(EnvConfig(seed=42, slip_prob=0.0))

        # Manually set agent to goal
        env.agent_pos = env.goal_pos
        _, cost, done, _ = env.step(Action.STAY)

        assert done
        assert cost == 0.0

    def test_valid_positions(self):
        """Test get_valid_positions."""
        env = GridWorld(EnvConfig(seed=42))
        valid = env.get_valid_positions()

        for pos in valid:
            assert env.is_valid_position(pos)
            assert env.grid[pos] != 1


class TestActions:
    """Tests for action handling."""

    def test_action_deltas(self):
        """Test action delta values."""
        assert ACTION_DELTAS[Action.UP] == (-1, 0)
        assert ACTION_DELTAS[Action.DOWN] == (1, 0)
        assert ACTION_DELTAS[Action.LEFT] == (0, -1)
        assert ACTION_DELTAS[Action.RIGHT] == (0, 1)
        assert ACTION_DELTAS[Action.STAY] == (0, 0)

    def test_all_actions_covered(self):
        """Test all actions have deltas."""
        for action in Action:
            assert action in ACTION_DELTAS


class TestObservationHelpers:
    """Tests for observation helper functions."""

    def test_compute_true_distances(self):
        """Test distance computation."""
        # Create simple grid with walls on edges
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        # Test from center
        distances = compute_true_distances(grid, (2, 2))

        assert distances[0] == 1  # Up
        assert distances[1] == 1  # Down
        assert distances[2] == 1  # Left
        assert distances[3] == 1  # Right

    def test_compute_local_patch(self):
        """Test local patch computation."""
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[0, :] = 1  # Top wall

        patch = compute_local_patch(grid, (1, 2))

        assert patch.shape == (3, 3)
        assert patch[0, 1] == 1  # Wall above


class TestEnvConfig:
    """Tests for environment configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnvConfig()

        assert config.grid_size == 12
        assert config.wall_density == 0.15
        assert 0 <= config.obs_noise_prob <= 1
        assert config.slip_prob == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnvConfig(
            grid_size=20,
            wall_density=0.3,
            slip_prob=0.2
        )

        env = GridWorld(config)

        assert env.grid_size == 20
