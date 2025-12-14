"""
Model Predictive Control (MPC) Planner

Plans actions using:
- Rollout-based evaluation of action sequences
- Random shooting with optional CEM refinement
- Cost function combining goal distance, action cost, and uncertainty

The planner trades off:
- Reaching the goal (low cost)
- Reducing uncertainty (information gain / curiosity)
- Avoiding commitment when mislocalized
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass

from .env import Action, ACTION_DELTAS
from .model import TransitionModel, DynamicsPredictor
from .belief import ParticleFilter, BeliefPredictor


@dataclass
class PlannerConfig:
    """Configuration for the MPC planner."""
    horizon: int = 12  # Planning horizon
    n_samples: int = 150  # Number of action sequences to sample
    n_belief_samples: int = 30  # Number of belief samples for rollouts
    n_elite: int = 15  # Number of elite samples for CEM
    cem_iterations: int = 2  # CEM refinement iterations
    goal_weight: float = 1.0  # Weight for goal distance
    action_cost: float = 0.1  # Cost per action
    stay_cost: float = 0.05  # Cost for staying
    uncertainty_weight: float = 0.5  # Weight for uncertainty/entropy term
    exploration_weight: float = 0.3  # Weight for exploration bonus
    info_gain_weight: float = 0.4  # Weight for information gain
    spread_penalty: float = 0.3  # Penalty for high belief spread
    discount: float = 0.95  # Discount factor for future costs
    pessimism: float = 0.2  # Weight on worst-case belief sample
    # Risk gate: penalize fork-commit actions when entropy is high
    risk_gate_enabled: bool = False
    risk_gate_entropy_threshold: float = 0.5  # Normalized entropy threshold
    risk_gate_penalty: float = 5.0  # Penalty for committing when uncertain
    # Commit risk gate: specific to commit zone entry
    commit_gate_enabled: bool = False
    commit_gate_entropy_threshold: float = 0.16  # H_thresh for commit gate
    commit_gate_penalty: float = 100.0  # Large penalty for entering commit zone when uncertain
    seed: Optional[int] = None


class MPCPlanner:
    """
    Model Predictive Control planner using rollout-based evaluation.

    Uses random shooting with CEM refinement to find good action sequences.
    Key feature: considers belief uncertainty and prefers actions that
    either make progress OR reduce uncertainty (information gain).
    """

    def __init__(self,
                 grid_size: int,
                 goal_pos: Tuple[int, int],
                 transition_model: TransitionModel,
                 belief: ParticleFilter,
                 config: Optional[PlannerConfig] = None):
        """
        Initialize MPC planner.
        """
        self.config = config or PlannerConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.grid_size = grid_size
        self.goal_pos = goal_pos
        self.model = transition_model
        self.belief = belief

        self.dynamics = DynamicsPredictor(transition_model)

        # Action probabilities for sampling (initially uniform)
        self.action_probs = np.ones((self.config.horizon, 5)) / 5

        # Commit zones (set by agent)
        self.commit_zone_goal: Set[Tuple[int, int]] = set()
        self.commit_zone_trap: Set[Tuple[int, int]] = set()

    def set_commit_zones(self, goal_zone: Set[Tuple[int, int]], trap_zone: Set[Tuple[int, int]]) -> None:
        """Set commit zones for commit gate evaluation."""
        self.commit_zone_goal = goal_zone
        self.commit_zone_trap = trap_zone

    def plan(self) -> int:
        """
        Plan and return the best action.

        Uses random shooting with CEM refinement, considering uncertainty.
        """
        # Sample starting positions from belief
        n_belief_samples = min(self.config.n_belief_samples, self.belief.n_particles)
        start_positions = self.belief.sample_positions(n=n_belief_samples)

        # Get current belief entropy for information gain calculation
        current_entropy = self.belief.get_entropy()
        current_spread = self.belief.get_spread()

        # Generate action sequences
        sequences = self._sample_action_sequences(self.config.n_samples)

        # Evaluate each sequence
        costs = []
        for seq in sequences:
            cost = self._evaluate_sequence(
                seq, start_positions, current_entropy, current_spread
            )
            costs.append(cost)

        costs = np.array(costs)

        # CEM refinement
        for _ in range(self.config.cem_iterations):
            elite_indices = np.argsort(costs)[:self.config.n_elite]
            elite_sequences = [sequences[i] for i in elite_indices]

            self._update_action_probs(elite_sequences)

            sequences = self._sample_action_sequences(self.config.n_samples)

            costs = []
            for seq in sequences:
                cost = self._evaluate_sequence(
                    seq, start_positions, current_entropy, current_spread
                )
                costs.append(cost)
            costs = np.array(costs)

        # Get best first action
        best_idx = np.argmin(costs)
        best_sequence = sequences[best_idx]

        # Reset action probs
        self.action_probs = np.ones((self.config.horizon, 5)) / 5

        return best_sequence[0]

    def _sample_action_sequences(self, n: int) -> List[List[int]]:
        """Sample n action sequences using current action probabilities."""
        sequences = []
        for _ in range(n):
            seq = []
            for t in range(self.config.horizon):
                action = self.rng.choice(5, p=self.action_probs[t])
                seq.append(action)
            sequences.append(seq)
        return sequences

    def _update_action_probs(self, elite_sequences: List[List[int]]) -> None:
        """Update action probabilities from elite sequences (CEM update)."""
        if not elite_sequences:
            return

        for t in range(self.config.horizon):
            counts = np.zeros(5)
            for seq in elite_sequences:
                counts[seq[t]] += 1
            counts += 0.1  # Smoothing
            self.action_probs[t] = counts / np.sum(counts)

    def _evaluate_sequence(self,
                           action_sequence: List[int],
                           start_positions: List[Tuple[int, int]],
                           current_entropy: float,
                           current_spread: float) -> float:
        """
        Evaluate an action sequence considering uncertainty.

        Uses both expected cost AND worst-case cost for robustness.
        Applies risk gate penalty when entropy is high and sequence commits to a fork.
        """
        costs = []
        end_positions = []

        for start_pos in start_positions:
            cost = self._rollout_cost(start_pos, action_sequence)
            costs.append(cost)
            # Track where each rollout ends up
            end_pos = self._simulate_sequence(start_pos, action_sequence)
            end_positions.append(end_pos)

        costs = np.array(costs)

        # Combine expected and worst-case (pessimistic planning)
        expected_cost = np.mean(costs)
        worst_cost = np.max(costs)
        variance_cost = np.std(costs)

        # High spread = uncertain about location = be more cautious
        spread_penalty = self.config.spread_penalty * current_spread

        # Risk gate: penalize committing to a direction when entropy is high
        risk_gate_penalty = 0.0
        if self.config.risk_gate_enabled:
            normalized_entropy = current_entropy / (np.log(len(start_positions) + 1) + 1e-6)
            if normalized_entropy > self.config.risk_gate_entropy_threshold:
                # Check if end positions are spread out (committing to different outcomes)
                if len(end_positions) > 1:
                    end_y = [p[0] for p in end_positions]
                    end_x = [p[1] for p in end_positions]
                    position_variance = np.var(end_y) + np.var(end_x)
                    # High variance = different outcomes from different belief samples
                    # This means the sequence commits to a fork-specific outcome
                    if position_variance > 2.0:  # Significant spread in outcomes
                        risk_gate_penalty = self.config.risk_gate_penalty * (position_variance / 10.0)

        # Commit gate: rollout penalty for entering TRAP commit zone when entropy is high
        # Applied during rollout evaluation, not just first action
        # Only active if commit zones are defined (fork mazes only)
        commit_gate_penalty = 0.0
        if self.config.commit_gate_enabled and len(action_sequence) > 0 and len(self.commit_zone_trap) > 0:
            # Check normalized entropy against commit threshold
            normalized_entropy = current_entropy / (np.log(len(start_positions) + 1) + 1e-6)
            if normalized_entropy > self.config.commit_gate_entropy_threshold:
                # Check if rollout from any belief sample enters trap zone
                trap_entry_count = 0
                for start_pos in start_positions:
                    pos = start_pos
                    for action in action_sequence:
                        next_pos = self.model.sample_next_pos(pos, action, slip_prob=0.0)
                        # Penalize if entering trap zone from non-trap position
                        if pos not in self.commit_zone_trap and next_pos in self.commit_zone_trap:
                            trap_entry_count += 1
                            break  # One trap entry per rollout is enough
                        pos = next_pos

                # Apply penalty proportional to fraction of rollouts hitting trap
                if trap_entry_count > 0:
                    trap_fraction = trap_entry_count / len(start_positions)
                    commit_gate_penalty = self.config.commit_gate_penalty * trap_fraction

        # Prefer lower variance in outcomes when uncertain
        total_cost = (
            (1 - self.config.pessimism) * expected_cost +
            self.config.pessimism * worst_cost +
            0.1 * variance_cost +
            spread_penalty +
            risk_gate_penalty +
            commit_gate_penalty
        )

        return total_cost

    def _simulate_sequence(self,
                          start_pos: Tuple[int, int],
                          action_sequence: List[int]) -> Tuple[int, int]:
        """Simulate action sequence and return final position."""
        pos = start_pos
        for action in action_sequence:
            pos = self.model.sample_next_pos(pos, action, slip_prob=0.0)
        return pos

    def _rollout_cost(self,
                      start_pos: Tuple[int, int],
                      action_sequence: List[int]) -> float:
        """
        Compute cost of executing action sequence from start position.

        Includes:
        - Distance to goal (discounted)
        - Action cost
        - Exploration bonus
        - Information gain approximation
        """
        pos = start_pos
        total_cost = 0.0
        discount = 1.0

        visited = set()

        for t, action in enumerate(action_sequence):
            # Simulate next position
            next_pos = self.model.sample_next_pos(pos, action, slip_prob=0.15)

            # Goal distance cost
            dist_cost = self._manhattan_distance(next_pos, self.goal_pos)
            dist_cost *= self.config.goal_weight

            # Action cost
            if action == Action.STAY:
                act_cost = self.config.stay_cost
            else:
                act_cost = self.config.action_cost

            # Exploration bonus (encourages visiting new places)
            exploration = self.model.get_exploration_bonus(next_pos)
            exploration_cost = -self.config.exploration_weight * exploration

            # Information gain: reward visiting "lookout" positions
            # that would help localize
            info_gain = 0.0
            if next_pos not in visited:
                # Approximate info gain by model uncertainty
                info_gain = self.model.get_uncertainty(pos, action)
            info_cost = -self.config.info_gain_weight * info_gain

            visited.add(next_pos)

            # Combine costs
            step_cost = dist_cost + act_cost + exploration_cost + info_cost
            total_cost += discount * step_cost

            discount *= self.config.discount
            pos = next_pos

            # Early termination if goal reached
            if pos == self.goal_pos:
                # Bonus for reaching goal
                total_cost -= 10.0 * discount
                break

        return total_cost

    def _manhattan_distance(self,
                            pos1: Tuple[int, int],
                            pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_best_action_distribution(self) -> np.ndarray:
        """Get distribution over best actions (after planning)."""
        return self.action_probs[0].copy()


class GreedyPlanner:
    """
    Simple greedy baseline planner.

    Always moves toward the goal based on mode of belief.
    This commits to potentially wrong beliefs and fails when mislocalized.
    """

    def __init__(self,
                 grid_size: int,
                 goal_pos: Tuple[int, int],
                 transition_model: TransitionModel,
                 belief: ParticleFilter,
                 seed: Optional[int] = None):
        self.grid_size = grid_size
        self.goal_pos = goal_pos
        self.model = transition_model
        self.belief = belief
        self.rng = np.random.default_rng(seed)

    def plan(self) -> int:
        """
        Return greedy action toward goal from believed position.

        CRITICAL: Uses mode of belief, which can be wrong!
        When mislocalized, this leads to suboptimal/wrong actions.
        """
        # Use mode of belief as position estimate (can be wrong!)
        estimated_pos = self.belief.get_mode_position()

        gy, gx = self.goal_pos
        y, x = estimated_pos

        # Simple greedy: pick action that minimizes distance to goal
        best_action = Action.STAY
        best_dist = self._manhattan_distance(estimated_pos, self.goal_pos)

        for action in Action:
            if action == Action.STAY:
                continue

            dy, dx = ACTION_DELTAS[action]
            new_y, new_x = y + dy, x + dx

            # Check if valid based on wall belief
            if not (0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size):
                continue

            wall_prob = self.model.wall_belief[new_y, new_x]
            if wall_prob > 0.7:
                continue

            new_dist = abs(new_y - gy) + abs(new_x - gx)
            if new_dist < best_dist:
                best_dist = new_dist
                best_action = action

        return best_action

    def _manhattan_distance(self,
                            pos1: Tuple[int, int],
                            pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class InfoGainPlanner:
    """
    Information-gain focused planner.

    Balances goal progress with uncertainty reduction.
    Better than greedy when localization matters.
    """

    def __init__(self,
                 grid_size: int,
                 goal_pos: Tuple[int, int],
                 transition_model: TransitionModel,
                 belief: ParticleFilter,
                 goal_weight: float = 0.4,
                 info_weight: float = 0.6,
                 seed: Optional[int] = None):
        self.grid_size = grid_size
        self.goal_pos = goal_pos
        self.model = transition_model
        self.belief = belief
        self.goal_weight = goal_weight
        self.info_weight = info_weight
        self.rng = np.random.default_rng(seed)

        self.belief_predictor = BeliefPredictor(
            self.model.get_learned_map(),
            [(i, j) for i in range(grid_size) for j in range(grid_size)
             if self.model.wall_belief[i, j] < 0.5]
        )

    def plan(self) -> int:
        """Plan action balancing goal progress and information gain."""
        current_belief = self.belief.get_belief_distribution()
        current_entropy = self.belief.get_entropy()
        current_spread = self.belief.get_spread()

        best_action = Action.STAY
        best_value = float('-inf')

        # Sample multiple positions from belief for robust estimation
        sampled_positions = self.belief.sample_positions(n=20)

        for action in Action:
            # Estimate entropy after action
            expected_entropy = self.belief_predictor.predict_belief_entropy(
                current_belief, action, n_samples=30
            )

            # Information gain (entropy reduction is good)
            info_gain = current_entropy - expected_entropy

            # Expected goal progress across belief samples
            goal_progress = 0.0
            for pos in sampled_positions:
                dy, dx = ACTION_DELTAS[action]
                new_y = max(0, min(self.grid_size - 1, pos[0] + dy))
                new_x = max(0, min(self.grid_size - 1, pos[1] + dx))

                old_dist = self._manhattan_distance(pos, self.goal_pos)
                new_dist = self._manhattan_distance((new_y, new_x), self.goal_pos)
                goal_progress += (old_dist - new_dist)

            goal_progress /= len(sampled_positions)

            # Combined value
            value = (self.goal_weight * goal_progress +
                    self.info_weight * info_gain)

            # Bonus for reducing spread when uncertain
            if current_spread > 2.0:
                value += 0.2 * info_gain

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _manhattan_distance(self,
                            pos1: Tuple[int, int],
                            pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class UncertaintyAwareMPCPlanner(MPCPlanner):
    """
    MPC planner without uncertainty awareness.

    This is MPC but ignores belief uncertainty - uses only expected cost.
    Should perform worse than full MPC when mislocalization occurs.
    """

    def __init__(self,
                 grid_size: int,
                 goal_pos: Tuple[int, int],
                 transition_model: TransitionModel,
                 belief: ParticleFilter,
                 config: Optional[PlannerConfig] = None):
        # Disable uncertainty terms
        modified_config = config or PlannerConfig()
        modified_config.uncertainty_weight = 0.0
        modified_config.exploration_weight = 0.0
        modified_config.info_gain_weight = 0.0
        modified_config.spread_penalty = 0.0
        modified_config.pessimism = 0.0

        super().__init__(grid_size, goal_pos, transition_model, belief, modified_config)


def create_planner(planner_type: str,
                   grid_size: int,
                   goal_pos: Tuple[int, int],
                   transition_model: TransitionModel,
                   belief: ParticleFilter,
                   config: Optional[PlannerConfig] = None,
                   seed: Optional[int] = None):
    """
    Factory function to create planners.

    Args:
        planner_type: "mpc", "mpc_no_uncert", "greedy", or "infogain"
    """
    if planner_type == "mpc":
        return MPCPlanner(
            grid_size, goal_pos, transition_model, belief, config
        )
    elif planner_type == "mpc_no_uncert":
        return UncertaintyAwareMPCPlanner(
            grid_size, goal_pos, transition_model, belief, config
        )
    elif planner_type == "greedy":
        return GreedyPlanner(
            grid_size, goal_pos, transition_model, belief, seed
        )
    elif planner_type == "infogain":
        return InfoGainPlanner(
            grid_size, goal_pos, transition_model, belief, seed=seed
        )
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")
