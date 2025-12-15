"""
Belief State Management

Universal belief state for any WooEdge application.
Tracks probability distributions over world states with Bayesian updates.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TypeVar, Generic
from abc import ABC, abstractmethod


T = TypeVar('T')  # World state type


@dataclass
class BeliefState(Generic[T]):
    """
    Universal belief state over world states.

    Maintains a probability distribution over possible world states
    and provides Bayesian update mechanics.

    Example:
        states = ["sunny", "cloudy", "rainy"]
        belief = BeliefState.uniform(states)
        belief.update({"sunny": 0.8, "cloudy": 0.15, "rainy": 0.05})
    """

    # Probability distribution over states
    distribution: Dict[T, float] = field(default_factory=dict)

    # History of distributions (for trend analysis)
    history: List[Dict[T, float]] = field(default_factory=list)
    history_limit: int = 100

    # Metadata
    update_count: int = 0

    @classmethod
    def uniform(cls, states: List[T], history_limit: int = 100) -> BeliefState[T]:
        """Create belief with uniform distribution over states."""
        n = len(states)
        if n == 0:
            raise ValueError("states cannot be empty")
        prob = 1.0 / n
        return cls(
            distribution={s: prob for s in states},
            history_limit=history_limit,
        )

    @classmethod
    def from_priors(cls, priors: Dict[T, float], history_limit: int = 100) -> BeliefState[T]:
        """Create belief from prior distribution."""
        total = sum(priors.values())
        if total <= 0:
            raise ValueError("priors must sum to positive value")
        normalized = {k: v / total for k, v in priors.items()}
        return cls(distribution=normalized, history_limit=history_limit)

    @property
    def states(self) -> List[T]:
        """Get list of possible states."""
        return list(self.distribution.keys())

    @property
    def most_likely(self) -> T:
        """Get most likely state."""
        return max(self.distribution, key=self.distribution.get)

    @property
    def most_likely_prob(self) -> float:
        """Get probability of most likely state."""
        return self.distribution[self.most_likely]

    def prob(self, state: T) -> float:
        """Get probability of a specific state."""
        return self.distribution.get(state, 0.0)

    def entropy(self) -> float:
        """
        Compute Shannon entropy of the belief distribution.

        Returns:
            Entropy in nats. 0 = certain, ln(n) = maximum uncertainty.
        """
        h = 0.0
        for p in self.distribution.values():
            if p > 0:
                h -= p * math.log(p)
        return h

    def max_entropy(self) -> float:
        """Maximum possible entropy for this state space."""
        n = len(self.distribution)
        return math.log(n) if n > 1 else 0.0

    def normalized_entropy(self) -> float:
        """
        Entropy normalized to [0, 1].

        Returns:
            0 = certain, 1 = maximum uncertainty (uniform).
        """
        max_h = self.max_entropy()
        if max_h == 0:
            return 0.0
        return self.entropy() / max_h

    def update_bayesian(self, likelihoods: Dict[T, float]) -> None:
        """
        Bayesian update: P(state|obs) ∝ P(obs|state) * P(state)

        Args:
            likelihoods: P(observation|state) for each state
        """
        # Save history
        self._save_history()

        # Compute posteriors
        posteriors = {}
        total = 0.0
        for state in self.distribution:
            prior = self.distribution[state]
            likelihood = likelihoods.get(state, 0.001)  # Small floor
            posterior = prior * likelihood
            posteriors[state] = posterior
            total += posterior

        # Normalize
        if total > 0:
            for state in posteriors:
                posteriors[state] /= total
        else:
            # Fallback to uniform
            n = len(posteriors)
            for state in posteriors:
                posteriors[state] = 1.0 / n

        self.distribution = posteriors
        self.update_count += 1

    def update_direct(self, new_distribution: Dict[T, float]) -> None:
        """
        Direct update with new distribution (e.g., from external model).

        Args:
            new_distribution: New probability distribution (will be normalized)
        """
        self._save_history()

        total = sum(new_distribution.values())
        if total > 0:
            self.distribution = {k: v / total for k, v in new_distribution.items()
                                if k in self.distribution}
        self.update_count += 1

    def blend(self, other: Dict[T, float], weight: float = 0.5) -> None:
        """
        Blend current belief with another distribution.

        Args:
            other: Distribution to blend with
            weight: Weight of other distribution (0-1)
        """
        self._save_history()

        weight = max(0.0, min(1.0, weight))
        blended = {}
        for state in self.distribution:
            p_self = self.distribution[state]
            p_other = other.get(state, 0.0)
            blended[state] = (1 - weight) * p_self + weight * p_other

        # Normalize
        total = sum(blended.values())
        if total > 0:
            for state in blended:
                blended[state] /= total

        self.distribution = blended
        self.update_count += 1

    def decay_to_uniform(self, rate: float = 0.01) -> None:
        """
        Decay belief toward uniform (for forgetting/uncertainty growth).

        Args:
            rate: Decay rate (0-1). Higher = faster decay to uniform.
        """
        n = len(self.distribution)
        uniform_prob = 1.0 / n

        for state in self.distribution:
            p = self.distribution[state]
            self.distribution[state] = p * (1 - rate) + uniform_prob * rate

    def _save_history(self) -> None:
        """Save current distribution to history."""
        self.history.append(self.distribution.copy())
        if len(self.history) > self.history_limit:
            self.history.pop(0)

    def entropy_trend(self, window: int = 10) -> float:
        """
        Compute trend in entropy over recent history.

        Returns:
            Positive = uncertainty increasing, Negative = converging.
        """
        if len(self.history) < 2:
            return 0.0

        recent = self.history[-min(window, len(self.history)):]
        entropies = []
        for dist in recent:
            h = 0.0
            for p in dist.values():
                if p > 0:
                    h -= p * math.log(p)
            entropies.append(h)

        if len(entropies) < 2:
            return 0.0

        # Simple linear trend
        n = len(entropies)
        x_mean = (n - 1) / 2
        y_mean = sum(entropies) / n

        num = sum((i - x_mean) * (entropies[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))

        return num / den if den > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize belief state."""
        return {
            "distribution": {str(k): v for k, v in self.distribution.items()},
            "entropy": self.entropy(),
            "normalized_entropy": self.normalized_entropy(),
            "most_likely": str(self.most_likely),
            "update_count": self.update_count,
        }

    def __repr__(self) -> str:
        top = self.most_likely
        prob = self.distribution[top]
        h = self.normalized_entropy()
        return f"BeliefState({top}={prob:.1%}, H={h:.2f})"


class LikelihoodModel(ABC, Generic[T]):
    """
    Abstract base for computing observation likelihoods.

    Apps implement this to define P(observation|state).
    """

    @abstractmethod
    def compute(self, observation: Any, state: T) -> float:
        """
        Compute likelihood P(observation|state).

        Args:
            observation: The observation data
            state: The world state to evaluate

        Returns:
            Likelihood value (not necessarily normalized)
        """
        pass

    def compute_all(self, observation: Any, states: List[T]) -> Dict[T, float]:
        """Compute likelihoods for all states."""
        return {state: self.compute(observation, state) for state in states}


class GaussianLikelihood(LikelihoodModel[T]):
    """
    Gaussian likelihood model for continuous observations.

    Each state has expected feature values; likelihood decreases
    with distance from expected values.
    """

    def __init__(self, state_features: Dict[T, Dict[str, float]], scales: Dict[str, float] = None):
        """
        Args:
            state_features: Expected feature values for each state
            scales: Scaling factors for each feature (higher = tighter)
        """
        self.state_features = state_features
        self.scales = scales or {}

    def compute(self, observation: Dict[str, float], state: T) -> float:
        """Compute Gaussian likelihood based on feature distances."""
        if state not in self.state_features:
            return 0.001

        expected = self.state_features[state]
        likelihood = 1.0

        for feature, expected_val in expected.items():
            observed_val = observation.get(feature, expected_val)
            scale = self.scales.get(feature, 1.0)
            distance = abs(observed_val - expected_val)
            likelihood *= math.exp(-distance * scale)

        return likelihood + 0.001  # Floor to prevent zero
