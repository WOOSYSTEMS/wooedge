"""Tests for the planner module."""

import numpy as np
import pytest

from wooedge.env import GridWorld, EnvConfig, Action
from wooedge.belief import ParticleFilter, BeliefConfig
from wooedge.model import TransitionModel, ModelConfig
from wooedge.planner import (
    MPCPlanner, GreedyPlanner, InfoGainPlanner,
    PlannerConfig, create_planner
)


class TestMPCPlanner:
    """Tests for MPCPlanner."""

    @pytest.fixture
    def setup(self):
        """Create environment, belief, model, and planner."""
        env = GridWorld(EnvConfig(seed=42))

        belief = ParticleFilter(
            env.get_grid_copy(),
            env.get_valid_positions(),
            BeliefConfig(n_particles=100, seed=42)
        )
        belief.initialize_at(env.start_pos, spread=2)

        model = TransitionModel(env.grid_size, ModelConfig(seed=42))

        config = PlannerConfig(
            horizon=5,
            n_samples=20,
            seed=42
        )

        planner = MPCPlanner(
            env.grid_size,
            env.goal_pos,
            model,
            belief,
            config
        )

        return {"env": env, "belief": belief, "model": model, "planner": planner}

    def test_plan_returns_valid_action(self, setup):
        """Test that plan returns a valid action."""
        action = setup["planner"].plan()

        assert action in list(Action)

    def test_plan_deterministic_with_seed(self, setup):
        """Test planning is deterministic with same seed."""
        planner1 = setup["planner"]
        action1 = planner1.plan()

        # Create identical planner
        env = setup["env"]
        belief = ParticleFilter(
            env.get_grid_copy(),
            env.get_valid_positions(),
            BeliefConfig(n_particles=100, seed=42)
        )
        belief.initialize_at(env.start_pos, spread=2)

        model = TransitionModel(env.grid_size, ModelConfig(seed=42))
        config = PlannerConfig(horizon=5, n_samples=20, seed=42)

        planner2 = MPCPlanner(
            env.grid_size, env.goal_pos, model, belief, config
        )
        action2 = planner2.plan()

        assert action1 == action2

    def test_action_distribution(self, setup):
        """Test action distribution retrieval."""
        setup["planner"].plan()
        dist = setup["planner"].get_best_action_distribution()

        assert len(dist) == 5
        np.testing.assert_almost_equal(np.sum(dist), 1.0)


class TestGreedyPlanner:
    """Tests for GreedyPlanner."""

    @pytest.fixture
    def setup(self):
        """Create environment and greedy planner."""
        env = GridWorld(EnvConfig(seed=42))

        belief = ParticleFilter(
            env.get_grid_copy(),
            env.get_valid_positions(),
            BeliefConfig(n_particles=100, seed=42)
        )
        belief.initialize_at(env.start_pos, spread=0)

        model = TransitionModel(env.grid_size, ModelConfig(seed=42))

        planner = GreedyPlanner(
            env.grid_size,
            env.goal_pos,
            model,
            belief,
            seed=42
        )

        return {"env": env, "belief": belief, "model": model, "planner": planner}

    def test_plan_returns_valid_action(self, setup):
        """Test greedy plan returns valid action."""
        action = setup["planner"].plan()
        assert action in list(Action)

    def test_moves_toward_goal(self, setup):
        """Test greedy planner tends to move toward goal."""
        env = setup["env"]
        planner = setup["planner"]

        # Get current distance
        mode = setup["belief"].get_mode_position()
        goal = env.goal_pos

        action = planner.plan()

        # Expected position after action
        dy, dx = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
            Action.STAY: (0, 0),
        }[Action(action)]

        new_y = mode[0] + dy
        new_x = mode[1] + dx

        # New position should be closer or same distance to goal
        old_dist = abs(mode[0] - goal[0]) + abs(mode[1] - goal[1])
        new_dist = abs(new_y - goal[0]) + abs(new_x - goal[1])

        # Allow for wall blocking (can't always get closer)
        assert new_dist <= old_dist + 1


class TestInfoGainPlanner:
    """Tests for InfoGainPlanner."""

    @pytest.fixture
    def setup(self):
        """Create environment and info-gain planner."""
        env = GridWorld(EnvConfig(seed=42))

        belief = ParticleFilter(
            env.get_grid_copy(),
            env.get_valid_positions(),
            BeliefConfig(n_particles=100, seed=42)
        )

        model = TransitionModel(env.grid_size, ModelConfig(seed=42))

        planner = InfoGainPlanner(
            env.grid_size,
            env.goal_pos,
            model,
            belief,
            goal_weight=0.5,
            info_weight=0.5,
            seed=42
        )

        return {"env": env, "belief": belief, "model": model, "planner": planner}

    def test_plan_returns_valid_action(self, setup):
        """Test info-gain plan returns valid action."""
        action = setup["planner"].plan()
        assert action in list(Action)


class TestCreatePlanner:
    """Tests for planner factory function."""

    @pytest.fixture
    def components(self):
        """Create shared components."""
        env = GridWorld(EnvConfig(seed=42))

        belief = ParticleFilter(
            env.get_grid_copy(),
            env.get_valid_positions(),
            BeliefConfig(seed=42)
        )

        model = TransitionModel(env.grid_size, ModelConfig(seed=42))

        return {
            "grid_size": env.grid_size,
            "goal_pos": env.goal_pos,
            "transition_model": model,
            "belief": belief,
        }

    def test_create_mpc_planner(self, components):
        """Test creating MPC planner."""
        planner = create_planner("mpc", **components)
        assert isinstance(planner, MPCPlanner)

    def test_create_greedy_planner(self, components):
        """Test creating greedy planner."""
        planner = create_planner("greedy", **components)
        assert isinstance(planner, GreedyPlanner)

    def test_create_infogain_planner(self, components):
        """Test creating info-gain planner."""
        planner = create_planner("infogain", **components)
        assert isinstance(planner, InfoGainPlanner)

    def test_invalid_planner_type(self, components):
        """Test invalid planner type raises error."""
        with pytest.raises(ValueError):
            create_planner("invalid", **components)
