"""Tests for the model module (transition learning)."""

import numpy as np
import pytest

from wooedge.env import GridWorld, EnvConfig, Action
from wooedge.model import TransitionModel, ModelConfig, DynamicsPredictor


class TestTransitionModel:
    """Tests for TransitionModel."""

    @pytest.fixture
    def model(self):
        """Create a transition model for testing."""
        return TransitionModel(12, ModelConfig(seed=42))

    def test_initialization(self, model):
        """Test basic initialization."""
        assert model.grid_size == 12
        assert model.total_updates == 0
        assert model.wall_belief.shape == (12, 12)

    def test_wall_belief_prior(self, model):
        """Test wall belief is initialized with prior."""
        assert np.all(model.wall_belief == model.config.wall_prior)

    def test_update_increases_counts(self, model):
        """Test that update increases transition counts."""
        assert model.total_updates == 0

        model.update((5, 5), Action.DOWN, (6, 5))

        assert model.total_updates == 1
        assert len(model.transition_counts) > 0

    def test_visit_counts_updated(self, model):
        """Test visit counts are updated."""
        assert model.visit_counts[6, 5] == 0

        model.update((5, 5), Action.DOWN, (6, 5))

        assert model.visit_counts[6, 5] == 1

    def test_multiple_updates_same_transition(self, model):
        """Test multiple updates for same transition."""
        for _ in range(5):
            model.update((5, 5), Action.DOWN, (6, 5))

        assert model.total_updates == 5
        assert model.transition_counts[(5, 5, Action.DOWN, 6, 5)] == 5

    def test_get_transition_prob(self, model):
        """Test transition probability retrieval."""
        # Before any updates
        prob = model.get_transition_prob((5, 5), Action.DOWN, (6, 5))
        assert 0 <= prob <= 1

        # After updates
        for _ in range(10):
            model.update((5, 5), Action.DOWN, (6, 5))

        prob = model.get_transition_prob((5, 5), Action.DOWN, (6, 5))
        assert prob > 0.5  # Should be high after many observations

    def test_get_expected_next_pos(self, model):
        """Test expected next position."""
        pos = model.get_expected_next_pos((5, 5), Action.DOWN)

        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_sample_next_pos(self, model):
        """Test next position sampling."""
        pos = model.sample_next_pos((5, 5), Action.DOWN)

        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert 0 <= pos[0] < model.grid_size
        assert 0 <= pos[1] < model.grid_size

    def test_exploration_bonus(self, model):
        """Test exploration bonus computation."""
        # Unvisited position should have high bonus
        bonus_unvisited = model.get_exploration_bonus((5, 5))

        # Visit the position
        model.update((4, 5), Action.DOWN, (5, 5))

        # Visited position should have lower bonus
        bonus_visited = model.get_exploration_bonus((5, 5))

        assert bonus_visited < bonus_unvisited

    def test_uncertainty_decreases_with_experience(self, model):
        """Test that uncertainty decreases with more observations."""
        uncertainty_initial = model.get_uncertainty((5, 5), Action.DOWN)

        for _ in range(10):
            model.update((5, 5), Action.DOWN, (6, 5))

        uncertainty_final = model.get_uncertainty((5, 5), Action.DOWN)

        assert uncertainty_final < uncertainty_initial

    def test_wall_belief_updated_from_observation(self, model):
        """Test wall belief update from observation."""
        initial_wall_belief = model.wall_belief.copy()

        # Create observation showing empty cells
        patch = np.zeros((3, 3), dtype=np.int8)
        model.update((5, 5), Action.DOWN, (6, 5), patch)

        # Wall belief should have changed
        assert not np.array_equal(initial_wall_belief, model.wall_belief)

    def test_model_confidence_increases(self, model):
        """Test model confidence increases with experience."""
        initial_conf = model.get_model_confidence()

        # Add many updates
        for i in range(50):
            model.update((5, i % 10 + 1), Action.DOWN, (6, i % 10 + 1))

        final_conf = model.get_model_confidence()

        assert final_conf > initial_conf

    def test_get_statistics(self, model):
        """Test statistics retrieval."""
        model.update((5, 5), Action.DOWN, (6, 5))

        stats = model.get_statistics()

        assert "total_updates" in stats
        assert "unique_transitions" in stats
        assert "cells_visited" in stats
        assert "model_confidence" in stats
        assert stats["total_updates"] == 1

    def test_initialize_from_grid(self, model):
        """Test initialization from known grid."""
        env = GridWorld(EnvConfig(seed=42))
        grid = env.get_grid_copy()

        model.initialize_from_grid(grid)

        # Wall cells should have high belief
        for y in range(model.grid_size):
            for x in range(model.grid_size):
                if grid[y, x] == 1:
                    assert model.wall_belief[y, x] > 0.9


class TestDynamicsPredictor:
    """Tests for DynamicsPredictor."""

    @pytest.fixture
    def predictor(self):
        """Create a dynamics predictor."""
        model = TransitionModel(12, ModelConfig(seed=42))
        return DynamicsPredictor(model)

    def test_predict_next_positions(self, predictor):
        """Test next position prediction."""
        predictions = predictor.predict_next_positions(
            (5, 5), Action.DOWN, n_samples=20
        )

        assert len(predictions) > 0
        total_prob = sum(prob for _, prob in predictions)
        np.testing.assert_almost_equal(total_prob, 1.0)

    def test_rollout(self, predictor):
        """Test action sequence rollout."""
        actions = [Action.DOWN, Action.DOWN, Action.RIGHT]
        trajectory = predictor.rollout((5, 5), actions)

        assert len(trajectory) == len(actions) + 1
        assert trajectory[0] == (5, 5)

    def test_rollout_stays_in_bounds(self, predictor):
        """Test rollout stays within grid bounds."""
        actions = [Action.DOWN] * 20
        trajectory = predictor.rollout((5, 5), actions)

        for pos in trajectory:
            assert 0 <= pos[0] < predictor.model.grid_size
            assert 0 <= pos[1] < predictor.model.grid_size
