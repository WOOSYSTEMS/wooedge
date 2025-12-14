"""Tests for the agent module."""

import numpy as np
import pytest

from wooedge.env import GridWorld, EnvConfig, Action
from wooedge.agent import (
    WOOEdgeAgent, AgentConfig, AgentState,
    run_episode, compare_planners
)


class TestWOOEdgeAgent:
    """Tests for WOOEdgeAgent."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return GridWorld(EnvConfig(seed=42))

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return WOOEdgeAgent(AgentConfig(
            n_particles=100,
            planner_type="mpc",
            planning_horizon=5,
            n_samples=20,
            seed=42
        ))

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.belief is None
        assert agent.model is None
        assert agent.planner is None
        assert agent.state.step == 0

    def test_setup(self, agent, env):
        """Test agent setup."""
        agent.setup(env, know_start=True)

        assert agent.belief is not None
        assert agent.model is not None
        assert agent.planner is not None
        assert agent.grid is not None
        assert agent.goal_pos == env.goal_pos

    def test_act_returns_valid_action(self, agent, env):
        """Test act returns valid action."""
        agent.setup(env)
        obs = env.get_observation()

        action = agent.act(obs)

        assert action in list(Action)

    def test_act_updates_state(self, agent, env):
        """Test act updates agent state."""
        agent.setup(env)
        obs = env.get_observation()

        assert agent.state.step == 0

        agent.act(obs)

        assert agent.state.step == 1
        assert len(agent.state.entropy_history) == 1
        assert len(agent.state.action_history) == 1

    def test_record_step(self, agent, env):
        """Test step recording."""
        agent.setup(env)
        obs = env.get_observation()
        agent.act(obs)

        agent.record_step(env.agent_pos, 1.5)

        assert len(agent.state.position_history) == 1
        assert len(agent.state.cost_history) == 1
        assert agent.state.total_cost == 1.5

    def test_get_belief_distribution(self, agent, env):
        """Test belief distribution retrieval."""
        agent.setup(env)

        belief = agent.get_belief_distribution()

        assert belief.shape == (env.grid_size, env.grid_size)
        np.testing.assert_almost_equal(np.sum(belief), 1.0)

    def test_get_learned_map(self, agent, env):
        """Test learned map retrieval."""
        agent.setup(env)

        learned_map = agent.get_learned_map()

        assert learned_map.shape == (env.grid_size, env.grid_size)

    def test_get_statistics(self, agent, env):
        """Test statistics retrieval."""
        agent.setup(env)
        obs = env.get_observation()
        agent.act(obs)

        stats = agent.get_statistics()

        assert "steps" in stats
        assert "total_cost" in stats
        assert "mean_entropy" in stats

    def test_multiple_steps(self, agent, env):
        """Test multiple consecutive steps."""
        agent.setup(env)

        for _ in range(10):
            obs = env.get_observation()
            action = agent.act(obs)
            obs, cost, done, info = env.step(action)
            agent.record_step(info["true_pos"], cost)

            if done:
                break

        assert agent.state.step <= 10
        assert len(agent.state.entropy_history) == agent.state.step


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()

        assert config.n_particles == 500
        assert config.planner_type == "mpc"
        assert config.planning_horizon == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            n_particles=200,
            planner_type="greedy",
            planning_horizon=5
        )

        assert config.n_particles == 200
        assert config.planner_type == "greedy"
        assert config.planning_horizon == 5


class TestAgentState:
    """Tests for AgentState."""

    def test_default_state(self):
        """Test default state initialization."""
        state = AgentState()

        assert state.step == 0
        assert state.total_cost == 0.0
        assert len(state.entropy_history) == 0
        assert len(state.cost_history) == 0


class TestRunEpisode:
    """Tests for run_episode function."""

    def test_run_episode_success(self):
        """Test running a full episode."""
        env = GridWorld(EnvConfig(seed=42))
        agent = WOOEdgeAgent(AgentConfig(
            n_particles=100,
            planning_horizon=5,
            n_samples=20,
            seed=42
        ))

        results = run_episode(env, agent, max_steps=50)

        assert "success" in results
        assert "steps" in results
        assert "total_cost" in results
        assert "final_entropy" in results
        assert results["steps"] <= 50

    def test_run_episode_max_steps(self):
        """Test episode terminates at max steps."""
        env = GridWorld(EnvConfig(seed=42, grid_size=20))  # Larger grid
        agent = WOOEdgeAgent(AgentConfig(
            n_particles=50,
            planner_type="greedy",
            seed=42
        ))

        results = run_episode(env, agent, max_steps=10)

        assert results["steps"] == 10

    def test_run_episode_verbose(self, capsys):
        """Test verbose output."""
        env = GridWorld(EnvConfig(seed=42))
        agent = WOOEdgeAgent(AgentConfig(
            n_particles=50,
            seed=42
        ))

        run_episode(env, agent, max_steps=5, verbose=True)

        captured = capsys.readouterr()
        assert "Step" in captured.out


class TestComparePlanners:
    """Tests for compare_planners function."""

    def test_compare_planners_basic(self):
        """Test basic planner comparison."""
        env_config = EnvConfig(
            grid_size=8,
            seed=42
        )

        agent_configs = {
            "mpc": AgentConfig(
                planner_type="mpc",
                n_particles=50,
                planning_horizon=3,
                n_samples=10,
                seed=42
            ),
            "greedy": AgentConfig(
                planner_type="greedy",
                n_particles=50,
                seed=42
            ),
        }

        results = compare_planners(
            env_config,
            agent_configs,
            n_episodes=2,
            max_steps=30
        )

        assert "mpc" in results
        assert "greedy" in results
        assert "success_rate" in results["mpc"]
        assert "mean_steps" in results["mpc"]
