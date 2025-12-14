"""
Online Transition Model Learning

Learns P(s'|s,a) online from experience using:
- Tabular transition counts with Dirichlet prior smoothing
- Optional map discovery (learning which cells are walls)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from .env import Action, ACTION_DELTAS


@dataclass
class ModelConfig:
    """Configuration for the learned model."""
    prior_alpha: float = 1.0  # Dirichlet prior (1.0 = uniform)
    wall_prior: float = 0.2  # Prior probability of a cell being a wall
    observation_weight: float = 0.8  # Weight for observation-based wall learning
    seed: Optional[int] = None


class TransitionModel:
    """
    Online learned transition model.

    Maintains:
    - Transition counts for P(s'|s,a) learning
    - Map belief (probability each cell is a wall)
    - Visit counts for exploration bonus
    """

    def __init__(self,
                 grid_size: int,
                 config: Optional[ModelConfig] = None):
        """
        Initialize transition model.

        Args:
            grid_size: Size of the grid
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.grid_size = grid_size
        self.n_actions = len(Action)

        # Transition counts: counts[y1, x1, a, y2, x2] = count of (s,a)->s' transitions
        # We use a sparse representation for efficiency
        self.transition_counts: Dict[Tuple, float] = {}

        # Map belief: probability each cell is a wall
        # Initialized with prior
        self.wall_belief = np.full(
            (grid_size, grid_size),
            self.config.wall_prior
        )

        # Visit counts for exploration bonus
        self.visit_counts = np.zeros((grid_size, grid_size))

        # Total transitions observed from each (s, a) pair
        self.sa_counts: Dict[Tuple[int, int, int, int], float] = {}

        # Learning statistics
        self.total_updates = 0

    def update(self,
               prev_pos: Tuple[int, int],
               action: int,
               new_pos: Tuple[int, int],
               observation_patch: Optional[np.ndarray] = None) -> None:
        """
        Update model from observed transition.

        Args:
            prev_pos: Previous position (y, x)
            action: Action taken
            new_pos: Resulting position (y, x)
            observation_patch: Optional 3x3 observation patch for map learning
        """
        y1, x1 = prev_pos
        y2, x2 = new_pos
        action = int(action)

        # Update transition counts
        key = (y1, x1, action, y2, x2)
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

        # Update (s, a) counts
        sa_key = (y1, x1, action)
        self.sa_counts[sa_key] = self.sa_counts.get(sa_key, 0) + 1

        # Update visit counts
        self.visit_counts[y2, x2] += 1

        # Update map belief from observation
        if observation_patch is not None:
            self._update_map_belief(new_pos, observation_patch)

        self.total_updates += 1

    def _update_map_belief(self,
                           pos: Tuple[int, int],
                           patch: np.ndarray) -> None:
        """
        Update wall belief from observation patch.

        Uses Bayesian update with observation likelihood.
        """
        y, x = pos
        weight = self.config.observation_weight

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx

                if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
                    continue

                observed_wall = (patch[dy + 1, dx + 1] == 1)

                # Bayesian update
                # P(wall|obs) ∝ P(obs|wall) * P(wall)
                prior = self.wall_belief[ny, nx]

                if observed_wall:
                    # Observation says wall
                    likelihood_wall = 0.9  # True positive rate
                    likelihood_empty = 0.1  # False positive rate
                else:
                    # Observation says empty
                    likelihood_wall = 0.1  # False negative rate
                    likelihood_empty = 0.9  # True negative rate

                # Posterior
                p_wall = likelihood_wall * prior
                p_empty = likelihood_empty * (1 - prior)

                posterior = p_wall / (p_wall + p_empty + 1e-10)

                # Weighted update (blend with current belief)
                self.wall_belief[ny, nx] = (
                    weight * posterior + (1 - weight) * self.wall_belief[ny, nx]
                )

        # Mark current position as definitely not a wall
        self.wall_belief[y, x] = 0.01  # Small epsilon for numerical stability

    def get_transition_prob(self,
                            pos: Tuple[int, int],
                            action: int,
                            next_pos: Tuple[int, int]) -> float:
        """
        Get learned transition probability P(next_pos | pos, action).

        Uses Dirichlet smoothing over observed counts.
        """
        y1, x1 = pos
        y2, x2 = next_pos
        action = int(action)

        # Get counts
        key = (y1, x1, action, y2, x2)
        count = self.transition_counts.get(key, 0)

        # Total count for this (s, a) pair
        sa_key = (y1, x1, action)
        total = self.sa_counts.get(sa_key, 0)

        # Number of possible next states (approximation)
        n_possible = 5  # Can stay or move to 4 neighbors

        # Dirichlet smoothing
        alpha = self.config.prior_alpha
        prob = (count + alpha) / (total + n_possible * alpha)

        return prob

    def get_expected_next_pos(self,
                              pos: Tuple[int, int],
                              action: int) -> Tuple[int, int]:
        """
        Get expected next position given action.

        If no experience, use physics-based prediction with wall belief.
        """
        y, x = pos
        action = Action(action)
        dy, dx = ACTION_DELTAS[action]

        new_y, new_x = y + dy, x + dx

        # Check bounds
        if not (0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size):
            return pos

        # Use wall belief to determine if movement is possible
        wall_prob = self.wall_belief[new_y, new_x]

        if wall_prob > 0.5:
            # Likely a wall, stay in place
            return pos
        else:
            return (new_y, new_x)

    def sample_next_pos(self,
                        pos: Tuple[int, int],
                        action: int,
                        slip_prob: float = 0.1) -> Tuple[int, int]:
        """
        Sample next position from learned model.

        Falls back to physics-based model with learned wall belief.
        """
        y, x = pos
        action = Action(action)

        # Check if we have enough experience for this (s, a)
        sa_key = (y, x, int(action))
        if self.sa_counts.get(sa_key, 0) >= 5:
            # Sample from learned distribution
            return self._sample_from_learned(pos, int(action))

        # Otherwise use physics-based model
        # Apply slip
        if action != Action.STAY and self.rng.random() < slip_prob:
            slip_action = self.rng.integers(0, 4)
            action = Action(slip_action)

        dy, dx = ACTION_DELTAS[action]
        new_y, new_x = y + dy, x + dx

        # Check bounds and wall belief
        if not (0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size):
            return pos

        wall_prob = self.wall_belief[new_y, new_x]

        # Stochastic wall check based on belief
        if self.rng.random() < wall_prob:
            return pos
        else:
            return (new_y, new_x)

    def _sample_from_learned(self,
                             pos: Tuple[int, int],
                             action: int) -> Tuple[int, int]:
        """Sample from learned transition distribution."""
        y1, x1 = pos

        # Collect all observed next states
        next_states = []
        counts = []

        for key, count in self.transition_counts.items():
            if key[0] == y1 and key[1] == x1 and key[2] == action:
                next_states.append((key[3], key[4]))
                counts.append(count + self.config.prior_alpha)

        if not next_states:
            # No experience, use default
            return self.get_expected_next_pos(pos, action)

        # Add smoothing for unobserved staying in place
        if pos not in next_states:
            next_states.append(pos)
            counts.append(self.config.prior_alpha)

        # Normalize and sample
        probs = np.array(counts) / np.sum(counts)
        idx = self.rng.choice(len(next_states), p=probs)

        return next_states[idx]

    def get_exploration_bonus(self, pos: Tuple[int, int]) -> float:
        """
        Get exploration bonus for a position.

        Higher bonus for less-visited positions.
        """
        y, x = pos
        visits = self.visit_counts[y, x]

        # UCB-style bonus: 1 / sqrt(visits + 1)
        return 1.0 / np.sqrt(visits + 1)

    def get_uncertainty(self, pos: Tuple[int, int], action: int) -> float:
        """
        Get uncertainty about transition from (pos, action).

        Based on count of observations.
        """
        y, x = pos
        sa_key = (y, x, int(action))
        count = self.sa_counts.get(sa_key, 0)

        # Higher uncertainty for less-observed state-action pairs
        return 1.0 / (count + 1)

    def get_learned_map(self) -> np.ndarray:
        """
        Get the learned map (wall belief).

        Returns 2D array where higher values = more likely wall.
        """
        return self.wall_belief.copy()

    def get_visit_map(self) -> np.ndarray:
        """Get visit count map."""
        return self.visit_counts.copy()

    def get_model_confidence(self) -> float:
        """
        Get overall confidence in the learned model.

        Based on total updates and coverage.
        """
        # Coverage: fraction of cells visited
        visited = np.sum(self.visit_counts > 0)
        total_cells = self.grid_size * self.grid_size
        coverage = visited / total_cells

        # Experience: total updates with diminishing returns
        experience = 1 - np.exp(-self.total_updates / 100)

        return 0.5 * coverage + 0.5 * experience

    def get_statistics(self) -> Dict:
        """Get model statistics."""
        return {
            "total_updates": self.total_updates,
            "unique_transitions": len(self.transition_counts),
            "unique_sa_pairs": len(self.sa_counts),
            "cells_visited": int(np.sum(self.visit_counts > 0)),
            "model_confidence": self.get_model_confidence(),
            "mean_visit_count": float(np.mean(self.visit_counts)),
            "max_visit_count": int(np.max(self.visit_counts)),
        }

    def initialize_from_grid(self, grid: np.ndarray) -> None:
        """
        Initialize wall belief from known grid (for testing/debugging).

        Normally the agent doesn't have this info.
        """
        self.wall_belief = (grid == 1).astype(float)
        self.wall_belief = np.clip(self.wall_belief, 0.01, 0.99)


class DynamicsPredictor:
    """
    Helper for predicting dynamics during planning.

    Combines learned model with physics-based priors.
    """

    def __init__(self,
                 transition_model: TransitionModel,
                 slip_prob: float = 0.1):
        self.model = transition_model
        self.slip_prob = slip_prob
        self.grid_size = transition_model.grid_size

    def predict_next_positions(self,
                               pos: Tuple[int, int],
                               action: int,
                               n_samples: int = 10) -> List[Tuple[Tuple[int, int], float]]:
        """
        Predict distribution over next positions.

        Returns list of (position, probability) tuples.
        """
        # Sample multiple times
        pos_counts: Dict[Tuple[int, int], int] = {}

        for _ in range(n_samples):
            next_pos = self.model.sample_next_pos(pos, action, self.slip_prob)
            pos_counts[next_pos] = pos_counts.get(next_pos, 0) + 1

        # Convert to probabilities
        result = []
        for next_pos, count in pos_counts.items():
            result.append((next_pos, count / n_samples))

        return result

    def rollout(self,
                start_pos: Tuple[int, int],
                actions: List[int]) -> List[Tuple[int, int]]:
        """
        Rollout a sequence of actions from start position.

        Returns trajectory of positions.
        """
        trajectory = [start_pos]
        pos = start_pos

        for action in actions:
            pos = self.model.sample_next_pos(pos, action, self.slip_prob)
            trajectory.append(pos)

        return trajectory
