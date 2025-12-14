"""
Belief State Module - Particle Filter Implementation

Maintains a probabilistic belief over the agent's position using
a particle filter with:
- Distance-only observation model (ambiguous)
- Reduced resampling aggressiveness to prevent belief collapse
- Higher noise tolerance for sustained uncertainty
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from .env import (
    Observation, Action, ACTION_DELTAS,
    compute_true_distances
)


@dataclass
class BeliefConfig:
    """Configuration for belief state."""
    n_particles: int = 1000  # More particles for better coverage
    resample_threshold: float = 0.3  # Lower threshold = less aggressive resampling
    sensor_noise_prob: float = 0.3  # Expected sensor noise probability
    sensor_noise_std: float = 1.0  # Spread for distance likelihood
    slip_prob: float = 0.15  # Expected action slip probability
    min_weight: float = 1e-6  # Minimum particle weight to prevent collapse
    mirror_invariant: bool = False  # Sort W/E and N/S distances
    seed: Optional[int] = None


def make_mirror_invariant(distances: np.ndarray) -> np.ndarray:
    """Make distances mirror-invariant by sorting pairs."""
    up, down, left, right = distances
    return np.array([
        min(up, down), max(up, down),
        min(left, right), max(left, right)
    ], dtype=np.float32)


class ParticleFilter:
    """
    Particle filter for belief state estimation.

    Optimized for ambiguous distance-only sensors with:
    - Soft observation likelihood (handles noise gracefully)
    - Conservative resampling (maintains diversity)
    - Entropy-preserving updates
    """

    def __init__(self,
                 grid: np.ndarray,
                 valid_positions: List[Tuple[int, int]],
                 config: Optional[BeliefConfig] = None):
        """
        Initialize particle filter.

        Args:
            grid: The environment grid (for computing expected observations)
            valid_positions: List of valid positions in the grid
            config: Belief configuration
        """
        self.config = config or BeliefConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.grid = grid
        self.grid_size = grid.shape[0]
        self.valid_positions = valid_positions
        self.n_valid = len(valid_positions)

        # Initialize particles uniformly over valid positions
        self.n_particles = self.config.n_particles
        self.particles = np.zeros((self.n_particles, 2), dtype=np.int32)
        self.weights = np.ones(self.n_particles) / self.n_particles

        self._initialize_uniform()

        # Cache for position-to-index lookup
        self.pos_to_idx = {pos: i for i, pos in enumerate(valid_positions)}

        # Precompute expected distances for all valid positions
        self._precompute_distances()

    def _precompute_distances(self) -> None:
        """Precompute distance sensor readings for all valid positions."""
        self.expected_distances = {}
        for pos in self.valid_positions:
            dist = compute_true_distances(self.grid, pos)
            if self.config.mirror_invariant:
                dist = make_mirror_invariant(dist)
            self.expected_distances[pos] = dist

    def _initialize_uniform(self) -> None:
        """Initialize particles uniformly over valid positions."""
        indices = self.rng.integers(0, self.n_valid, size=self.n_particles)
        for i, idx in enumerate(indices):
            self.particles[i] = self.valid_positions[idx]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def initialize_at(self, pos: Tuple[int, int], spread: int = 3) -> None:
        """
        Initialize particles near a specific position with spread.

        Higher spread = more initial uncertainty.
        """
        nearby = [p for p in self.valid_positions
                  if abs(p[0] - pos[0]) + abs(p[1] - pos[1]) <= spread]

        if not nearby:
            nearby = self.valid_positions

        indices = self.rng.integers(0, len(nearby), size=self.n_particles)
        for i, idx in enumerate(indices):
            self.particles[i] = nearby[idx]

        self.weights = np.ones(self.n_particles) / self.n_particles

    def initialize_uniform(self) -> None:
        """Initialize belief uniformly (maximum uncertainty)."""
        self._initialize_uniform()

    def predict(self, action: int) -> None:
        """
        Predict step: propagate particles through motion model.

        Applies action with slip probability.
        """
        action = Action(action)
        dy, dx = ACTION_DELTAS[action]

        for i in range(self.n_particles):
            # Apply slip with probability
            if action != Action.STAY and self.rng.random() < self.config.slip_prob:
                # Random slip
                slip_action = self.rng.integers(0, 4)  # Exclude STAY
                dy_slip, dx_slip = list(ACTION_DELTAS.values())[slip_action]
                new_y = self.particles[i, 0] + dy_slip
                new_x = self.particles[i, 1] + dx_slip
            else:
                new_y = self.particles[i, 0] + dy
                new_x = self.particles[i, 1] + dx

            # Check validity
            if self._is_valid(new_y, new_x):
                self.particles[i] = [new_y, new_x]
            # Otherwise particle stays in place

    def update(self, observation: Observation) -> None:
        """
        Update step: reweight particles based on observation likelihood.

        Uses ONLY distance sensors (no local patch).
        """
        likelihoods = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            pos = tuple(self.particles[i])
            likelihoods[i] = self._observation_likelihood(pos, observation)

        # Apply minimum weight to prevent collapse
        likelihoods = np.maximum(likelihoods, self.config.min_weight)

        # Normalize
        total = np.sum(likelihoods)
        if total > 1e-10:
            self.weights = likelihoods / total
        else:
            # If all likelihoods are very low, maintain diversity
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Only resample if effective sample size is very low
        ess = self._effective_sample_size()
        if ess < self.config.resample_threshold * self.n_particles:
            self._resample_with_diversity()

    def _observation_likelihood(self,
                                pos: Tuple[int, int],
                                observation: Observation) -> float:
        """
        Compute likelihood of observation given position.

        Uses soft Gaussian-like likelihood for distance sensors.
        """
        # Get expected distances for this position
        if pos in self.expected_distances:
            expected_dist = self.expected_distances[pos]
        else:
            expected_dist = compute_true_distances(self.grid, pos)
            if self.config.mirror_invariant:
                expected_dist = make_mirror_invariant(expected_dist)

        # Distance sensor likelihood
        # Use soft matching that handles ±1 noise gracefully
        observed = observation.distance_sensors
        diff = np.abs(observed - expected_dist)

        # Likelihood decreases with distance, but slowly
        # This prevents belief collapse on noisy observations
        sigma = self.config.sensor_noise_std
        likelihood = np.exp(-0.5 * np.sum(diff ** 2) / (sigma ** 2))

        # Add noise floor to prevent zero likelihood
        likelihood = max(likelihood, self.config.min_weight)

        return likelihood

    def _is_valid(self, y: int, x: int) -> bool:
        """Check if position is valid."""
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return False
        return self.grid[y, x] != 1  # Not a wall

    def _effective_sample_size(self) -> float:
        """Compute effective sample size."""
        return 1.0 / (np.sum(self.weights ** 2) + 1e-10)

    def _resample_with_diversity(self) -> None:
        """
        Resample with diversity preservation.

        Adds some random particles to maintain exploration.
        """
        n_keep = int(0.9 * self.n_particles)  # Keep 90% from resampling
        n_random = self.n_particles - n_keep  # Add 10% random particles

        # Systematic resampling for kept particles
        cumsum = np.cumsum(self.weights)
        step = 1.0 / n_keep
        start = self.rng.uniform(0, step)
        points = start + step * np.arange(n_keep)

        indices = np.searchsorted(cumsum, points)
        indices = np.clip(indices, 0, self.n_particles - 1)

        # Create new particle set
        new_particles = np.zeros_like(self.particles)
        new_particles[:n_keep] = self.particles[indices]

        # Add random particles for diversity
        random_indices = self.rng.integers(0, self.n_valid, size=n_random)
        for i, idx in enumerate(random_indices):
            new_particles[n_keep + i] = self.valid_positions[idx]

        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles

    def get_belief_distribution(self) -> np.ndarray:
        """
        Get belief as probability distribution over grid.

        Returns:
            2D array of probabilities for each grid cell
        """
        belief = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.n_particles):
            y, x = self.particles[i]
            belief[y, x] += self.weights[i]

        # Normalize
        total = np.sum(belief)
        if total > 0:
            belief /= total

        return belief

    def get_entropy(self) -> float:
        """
        Compute entropy of belief distribution.

        Higher entropy = more uncertainty.
        """
        belief = self.get_belief_distribution()
        belief_flat = belief.flatten()

        # Filter out zeros to avoid log(0)
        belief_nonzero = belief_flat[belief_flat > 1e-10]

        if len(belief_nonzero) == 0:
            return 0.0

        return -np.sum(belief_nonzero * np.log(belief_nonzero))

    def get_max_entropy(self) -> float:
        """Get maximum possible entropy (uniform over valid positions)."""
        return np.log(self.n_valid)

    def get_normalized_entropy(self) -> float:
        """Get entropy normalized to [0, 1]."""
        max_ent = self.get_max_entropy()
        if max_ent == 0:
            return 0.0
        return self.get_entropy() / max_ent

    def get_mean_position(self) -> Tuple[float, float]:
        """Get weighted mean position of belief."""
        mean_y = np.sum(self.particles[:, 0] * self.weights)
        mean_x = np.sum(self.particles[:, 1] * self.weights)
        return (mean_y, mean_x)

    def get_mode_position(self) -> Tuple[int, int]:
        """Get most likely position (mode of belief)."""
        belief = self.get_belief_distribution()
        idx = np.argmax(belief)
        return (idx // self.grid_size, idx % self.grid_size)

    def get_top_positions(self, k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """Get top k most likely positions with their probabilities."""
        belief = self.get_belief_distribution()

        # Get flattened indices sorted by probability
        flat_belief = belief.flatten()
        top_indices = np.argsort(flat_belief)[-k:][::-1]

        result = []
        for idx in top_indices:
            y, x = idx // self.grid_size, idx % self.grid_size
            prob = flat_belief[idx]
            if prob > 0:
                result.append(((y, x), prob))

        return result

    def sample_positions(self, n: int = 10) -> List[Tuple[int, int]]:
        """Sample positions from current belief."""
        indices = self.rng.choice(
            self.n_particles,
            size=n,
            p=self.weights
        )
        return [tuple(self.particles[i]) for i in indices]

    def get_position_probability(self, pos: Tuple[int, int]) -> float:
        """Get probability of being at a specific position."""
        belief = self.get_belief_distribution()
        return belief[pos[0], pos[1]]

    def get_spread(self) -> float:
        """
        Get spatial spread of belief (standard deviation).

        Higher spread = more uncertainty about location.
        """
        mean_y, mean_x = self.get_mean_position()

        var_y = np.sum(self.weights * (self.particles[:, 0] - mean_y) ** 2)
        var_x = np.sum(self.weights * (self.particles[:, 1] - mean_x) ** 2)

        return np.sqrt(var_y + var_x)

    def get_state(self) -> Dict:
        """Get full belief state for debugging."""
        return {
            "particles": self.particles.copy(),
            "weights": self.weights.copy(),
            "entropy": self.get_entropy(),
            "normalized_entropy": self.get_normalized_entropy(),
            "mean_pos": self.get_mean_position(),
            "mode_pos": self.get_mode_position(),
            "ess": self._effective_sample_size(),
            "spread": self.get_spread(),
        }


class BeliefPredictor:
    """
    Helper class to predict future belief states without modifying
    the actual belief. Used for planning.
    """

    def __init__(self,
                 grid: np.ndarray,
                 valid_positions: List[Tuple[int, int]],
                 config: Optional[BeliefConfig] = None):
        self.grid = grid
        self.grid_size = grid.shape[0]
        self.valid_positions = valid_positions
        self.config = config or BeliefConfig()

    def predict_belief_entropy(self,
                               current_belief: np.ndarray,
                               action: int,
                               n_samples: int = 50) -> float:
        """
        Estimate entropy of belief after taking action.

        Uses sampling to approximate the expected entropy.
        """
        rng = np.random.default_rng()
        action = Action(action)
        dy, dx = ACTION_DELTAS[action]

        # Sample positions from current belief
        flat_belief = current_belief.flatten()
        flat_belief = flat_belief / (np.sum(flat_belief) + 1e-10)

        # Propagate samples through motion model
        new_belief = np.zeros_like(current_belief)

        for _ in range(n_samples):
            idx = rng.choice(len(flat_belief), p=flat_belief)
            y, x = idx // self.grid_size, idx % self.grid_size

            # Apply action with slip
            if action != Action.STAY and rng.random() < self.config.slip_prob:
                slip_action = rng.integers(0, 4)
                dy_new, dx_new = list(ACTION_DELTAS.values())[slip_action]
            else:
                dy_new, dx_new = dy, dx

            new_y, new_x = y + dy_new, x + dx_new

            # Check validity
            if (0 <= new_y < self.grid_size and
                0 <= new_x < self.grid_size and
                self.grid[new_y, new_x] != 1):
                new_belief[new_y, new_x] += 1
            else:
                new_belief[y, x] += 1

        # Normalize and compute entropy
        new_belief = new_belief / (np.sum(new_belief) + 1e-10)
        nonzero = new_belief[new_belief > 1e-10]

        if len(nonzero) == 0:
            return 0.0

        return -np.sum(nonzero * np.log(nonzero))

    def predict_spread(self,
                       current_belief: np.ndarray,
                       action: int,
                       n_samples: int = 50) -> float:
        """Estimate belief spread after taking action."""
        rng = np.random.default_rng()
        action = Action(action)
        dy, dx = ACTION_DELTAS[action]

        flat_belief = current_belief.flatten()
        flat_belief = flat_belief / (np.sum(flat_belief) + 1e-10)

        positions = []
        for _ in range(n_samples):
            idx = rng.choice(len(flat_belief), p=flat_belief)
            y, x = idx // self.grid_size, idx % self.grid_size

            if action != Action.STAY and rng.random() < self.config.slip_prob:
                slip_action = rng.integers(0, 4)
                dy_new, dx_new = list(ACTION_DELTAS.values())[slip_action]
            else:
                dy_new, dx_new = dy, dx

            new_y = max(0, min(self.grid_size - 1, y + dy_new))
            new_x = max(0, min(self.grid_size - 1, x + dx_new))

            if self.grid[new_y, new_x] != 1:
                positions.append((new_y, new_x))
            else:
                positions.append((y, x))

        if not positions:
            return 0.0

        positions = np.array(positions)
        mean_y, mean_x = np.mean(positions, axis=0)
        var = np.mean((positions[:, 0] - mean_y) ** 2 + (positions[:, 1] - mean_x) ** 2)

        return np.sqrt(var)
