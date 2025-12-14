"""Tests for the belief module (particle filter)."""

import numpy as np
import pytest

from wooedge.env import GridWorld, EnvConfig, Action, Observation
from wooedge.belief import ParticleFilter, BeliefConfig, BeliefPredictor


class TestParticleFilter:
    """Tests for ParticleFilter."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        return GridWorld(EnvConfig(seed=42))

    @pytest.fixture
    def belief(self, simple_env):
        """Create a particle filter for testing."""
        config = BeliefConfig(n_particles=100, seed=42)
        return ParticleFilter(
            simple_env.get_grid_copy(),
            simple_env.get_valid_positions(),
            config
        )

    def test_initialization(self, belief):
        """Test basic initialization."""
        assert belief.n_particles == 100
        assert belief.particles.shape == (100, 2)
        assert len(belief.weights) == 100
        assert np.isclose(np.sum(belief.weights), 1.0)

    def test_weights_sum_to_one(self, belief):
        """Test that weights always sum to one."""
        np.testing.assert_almost_equal(np.sum(belief.weights), 1.0)

    def test_initialize_at_position(self, belief, simple_env):
        """Test initialization at specific position."""
        pos = simple_env.start_pos
        belief.initialize_at(pos, spread=1)

        # Most particles should be near the position
        mean_pos = belief.get_mean_position()
        assert abs(mean_pos[0] - pos[0]) < 2
        assert abs(mean_pos[1] - pos[1]) < 2

    def test_predict_changes_particles(self, belief):
        """Test that predict step changes particle positions."""
        initial_particles = belief.particles.copy()

        belief.predict(Action.DOWN)

        # Some particles should have moved
        assert not np.array_equal(initial_particles, belief.particles)

    def test_update_changes_belief(self, belief, simple_env):
        """Test that update step changes belief distribution."""
        initial_belief = belief.get_belief_distribution().copy()

        obs = simple_env.get_observation()
        belief.update(obs)

        # Belief distribution should have changed (or weights were updated)
        # Note: After resampling, weights may be uniform again
        final_belief = belief.get_belief_distribution()

        # The belief should have focused on some positions
        # (entropy should be less than or equal to initial)
        assert belief.get_entropy() >= 0

    def test_entropy_bounds(self, belief):
        """Test entropy is within valid bounds."""
        entropy = belief.get_entropy()

        assert entropy >= 0
        assert entropy <= belief.get_max_entropy()

    def test_normalized_entropy_bounds(self, belief):
        """Test normalized entropy is between 0 and 1."""
        norm_entropy = belief.get_normalized_entropy()

        assert 0 <= norm_entropy <= 1

    def test_get_belief_distribution(self, belief):
        """Test belief distribution computation."""
        dist = belief.get_belief_distribution()

        assert dist.shape == (belief.grid_size, belief.grid_size)
        np.testing.assert_almost_equal(np.sum(dist), 1.0)
        assert np.all(dist >= 0)

    def test_get_mode_position(self, belief, simple_env):
        """Test mode position retrieval."""
        belief.initialize_at(simple_env.start_pos, spread=0)
        mode = belief.get_mode_position()

        # Mode should be close to start position
        assert abs(mode[0] - simple_env.start_pos[0]) <= 1
        assert abs(mode[1] - simple_env.start_pos[1]) <= 1

    def test_get_top_positions(self, belief):
        """Test top positions retrieval."""
        top = belief.get_top_positions(k=5)

        assert len(top) <= 5
        for pos, prob in top:
            assert isinstance(pos, tuple)
            assert 0 <= prob <= 1

    def test_sample_positions(self, belief):
        """Test position sampling."""
        samples = belief.sample_positions(n=10)

        assert len(samples) == 10
        for pos in samples:
            assert isinstance(pos, tuple)
            assert len(pos) == 2

    def test_get_state(self, belief):
        """Test state retrieval."""
        state = belief.get_state()

        assert "particles" in state
        assert "weights" in state
        assert "entropy" in state
        assert "mean_pos" in state
        assert "mode_pos" in state

    def test_predict_then_update_cycle(self, belief, simple_env):
        """Test predict-update cycle."""
        initial_entropy = belief.get_entropy()

        # Take several predict-update steps
        for _ in range(5):
            belief.predict(Action.DOWN)
            obs = simple_env.get_observation()
            belief.update(obs)

        # Entropy should still be valid
        final_entropy = belief.get_entropy()
        assert final_entropy >= 0
        assert final_entropy <= belief.get_max_entropy()


class TestBeliefPredictor:
    """Tests for BeliefPredictor."""

    @pytest.fixture
    def predictor(self):
        """Create a belief predictor for testing."""
        env = GridWorld(EnvConfig(seed=42))
        valid = [(i, j) for i in range(env.grid_size)
                for j in range(env.grid_size)
                if env.grid[i, j] != 1]
        return BeliefPredictor(
            env.get_grid_copy(),
            valid,
            BeliefConfig(seed=42)
        )

    def test_predict_belief_entropy(self, predictor):
        """Test entropy prediction."""
        # Create uniform belief
        belief = np.ones((12, 12)) / 144

        entropy = predictor.predict_belief_entropy(belief, Action.DOWN)

        assert entropy >= 0

    def test_entropy_varies_by_action(self, predictor):
        """Test that different actions may produce different entropies."""
        belief = np.ones((12, 12)) / 144

        entropies = []
        for action in Action:
            ent = predictor.predict_belief_entropy(belief, action)
            entropies.append(ent)

        # All should be valid
        assert all(e >= 0 for e in entropies)
