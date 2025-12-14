"""
WOOEdge Agent - Main Agent Implementation

Combines:
- Belief state tracking (particle filter)
- Online dynamics learning (transition model)
- Model predictive control planning

The agent follows the predict -> act -> update -> improve cycle.
Tracks mislocalization events and entropy for analysis.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field

from .env import GridWorld, Observation, Action, EnvConfig
from .belief import ParticleFilter, BeliefConfig
from .model import TransitionModel, ModelConfig
from .planner import MPCPlanner, GreedyPlanner, create_planner, PlannerConfig


@dataclass
class AgentConfig:
    """Configuration for the WOOEdge agent."""
    # Belief configuration
    n_particles: int = 1000
    belief_sensor_noise_std: float = 1.0
    belief_resample_threshold: float = 0.3

    # Model configuration
    model_prior_alpha: float = 1.0
    model_wall_prior: float = 0.2

    # Planner configuration
    planner_type: str = "mpc"  # "mpc", "mpc_no_uncert", "greedy", or "infogain"
    planning_horizon: int = 12
    n_samples: int = 150
    n_belief_samples: int = 30  # Number of belief samples for MPC rollouts
    goal_weight: float = 1.0
    uncertainty_weight: float = 0.5
    exploration_weight: float = 0.3
    info_gain_weight: float = 0.4
    spread_penalty: float = 0.3  # Penalty for high belief spread
    pessimism: float = 0.2  # Weight on worst-case outcomes

    # General
    seed: Optional[int] = None
    uniform_init: bool = True  # Start with uniform belief (max uncertainty)
    mirror_invariant: bool = False  # Use mirror-invariant observations
    risk_gate_enabled: bool = False  # Enable risk gate for fork-commit penalty
    risk_gate_entropy_threshold: float = 0.5  # Entropy threshold for risk gate
    risk_gate_penalty: float = 5.0  # Penalty for committing when uncertain
    commit_gate_enabled: bool = False  # Enable commit zone gate
    commit_gate_entropy_threshold: float = 0.16  # H_thresh for commit gate
    commit_gate_penalty: float = 100.0  # Large penalty for entering commit zone when uncertain


@dataclass
class AgentState:
    """Current state of the agent for tracking."""
    step: int = 0
    total_cost: float = 0.0
    entropy_history: List[float] = field(default_factory=list)
    spread_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    action_history: List[int] = field(default_factory=list)
    belief_mode_history: List[Tuple[int, int]] = field(default_factory=list)

    # Mislocalization tracking
    mislocalization_events: int = 0
    mislocalization_steps: List[int] = field(default_factory=list)
    belief_correct_history: List[bool] = field(default_factory=list)
    distance_error_history: List[float] = field(default_factory=list)

    # Vantage point tracking
    vantage_visits: int = 0
    vantage_visit_steps: List[int] = field(default_factory=list)

    # Commit zone tracking
    commit_step: Optional[int] = None  # Step when first entered commit zone
    visited_vantage_before_commit: bool = False  # Did agent visit vantage before committing?
    wrong_commit: bool = False  # True if committed to trap zone


class WOOEdgeAgent:
    """
    World-model Online Edge Intelligence Agent.

    A predictive agent that:
    1. Maintains belief over its position (particle filter)
    2. Learns environment dynamics online
    3. Plans actions using MPC with uncertainty awareness
    4. Tracks mislocalization for analysis
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize agent with configuration."""
        self.config = config or AgentConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Components (initialized in setup)
        self.belief: Optional[ParticleFilter] = None
        self.model: Optional[TransitionModel] = None
        self.planner = None

        # Environment info
        self.grid: Optional[np.ndarray] = None
        self.grid_size: int = 0
        self.goal_pos: Tuple[int, int] = (0, 0)
        self.valid_positions: List[Tuple[int, int]] = []

        # State tracking
        self.state = AgentState()
        self.last_action: Optional[int] = None
        self.last_belief_mode: Optional[Tuple[int, int]] = None
        self.was_mislocalized: bool = False

    def setup(self,
              env: GridWorld,
              know_start: bool = False) -> None:
        """
        Set up agent for a new episode.

        Args:
            env: The environment
            know_start: Whether agent knows its starting position
                       (False = start with uniform belief for max uncertainty)
        """
        self.grid = env.get_grid_copy()
        self.grid_size = env.grid_size
        self.goal_pos = env.goal_pos
        self.valid_positions = env.get_valid_positions()
        self.lookout_positions = env.lookout_positions.copy()
        # Fork-related attributes (only defined for fork mazes)
        self.has_fork = getattr(env, 'has_fork', False)
        self.commit_zone_goal = env.commit_zone_goal.copy() if self.has_fork else set()
        self.commit_zone_trap = env.commit_zone_trap.copy() if self.has_fork else set()

        # Initialize belief
        belief_config = BeliefConfig(
            n_particles=self.config.n_particles,
            sensor_noise_std=self.config.belief_sensor_noise_std,
            resample_threshold=self.config.belief_resample_threshold,
            slip_prob=0.15,
            mirror_invariant=self.config.mirror_invariant,
            seed=self.config.seed
        )

        self.belief = ParticleFilter(
            self.grid,
            self.valid_positions,
            belief_config
        )

        if know_start and not self.config.uniform_init:
            # Initialize belief near start position
            self.belief.initialize_at(env.start_pos, spread=3)
        else:
            # Start with uniform belief (maximum uncertainty)
            self.belief.initialize_uniform()

        # Initialize transition model
        model_config = ModelConfig(
            prior_alpha=self.config.model_prior_alpha,
            wall_prior=self.config.model_wall_prior,
            seed=self.config.seed
        )

        self.model = TransitionModel(self.grid_size, model_config)

        # Initialize planner
        planner_config = PlannerConfig(
            horizon=self.config.planning_horizon,
            n_samples=self.config.n_samples,
            n_belief_samples=self.config.n_belief_samples,
            goal_weight=self.config.goal_weight,
            uncertainty_weight=self.config.uncertainty_weight,
            exploration_weight=self.config.exploration_weight,
            info_gain_weight=self.config.info_gain_weight,
            spread_penalty=self.config.spread_penalty,
            pessimism=self.config.pessimism,
            risk_gate_enabled=self.config.risk_gate_enabled,
            risk_gate_entropy_threshold=self.config.risk_gate_entropy_threshold,
            risk_gate_penalty=self.config.risk_gate_penalty,
            commit_gate_enabled=self.config.commit_gate_enabled,
            commit_gate_entropy_threshold=self.config.commit_gate_entropy_threshold,
            commit_gate_penalty=self.config.commit_gate_penalty,
            seed=self.config.seed
        )

        self.planner = create_planner(
            self.config.planner_type,
            self.grid_size,
            self.goal_pos,
            self.model,
            self.belief,
            planner_config,
            self.config.seed
        )

        # Set commit zones for commit gate (only for fork mazes)
        if self.has_fork and hasattr(self.planner, 'set_commit_zones'):
            self.planner.set_commit_zones(self.commit_zone_goal, self.commit_zone_trap)

        # Reset state tracking
        self.state = AgentState()
        self.last_action = None
        self.last_belief_mode = None
        self.was_mislocalized = False

    def act(self, observation: Observation) -> int:
        """
        Main agent loop: observe -> update -> plan -> act.

        Args:
            observation: Current observation from environment

        Returns:
            Action to take
        """
        # 1. Update belief from observation
        self.belief.update(observation)

        # 2. Record belief state
        current_entropy = self.belief.get_normalized_entropy()
        current_spread = self.belief.get_spread()
        current_mode = self.belief.get_mode_position()

        self.state.entropy_history.append(current_entropy)
        self.state.spread_history.append(current_spread)
        self.state.belief_mode_history.append(current_mode)

        # 3. Update transition model from last transition
        if self.last_action is not None and self.last_belief_mode is not None:
            self.model.update(
                self.last_belief_mode,
                self.last_action,
                current_mode,
                None  # No local patch in new observation model
            )

        # 4. Plan next action
        action = self.planner.plan()

        # 5. Predict (update belief for next step)
        self.belief.predict(action)

        # 6. Record for next iteration
        self.last_action = action
        self.last_belief_mode = current_mode
        self.state.action_history.append(action)
        self.state.step += 1

        return action

    def record_step(self,
                    true_pos: Tuple[int, int],
                    cost: float) -> None:
        """
        Record step information for analysis.

        Args:
            true_pos: Actual position from environment
            cost: Step cost
        """
        self.state.position_history.append(true_pos)
        self.state.cost_history.append(cost)
        self.state.total_cost += cost

        # Track vantage point visits
        if hasattr(self, 'lookout_positions') and true_pos in self.lookout_positions:
            self.state.vantage_visits += 1
            self.state.vantage_visit_steps.append(self.state.step)

        # Track commit zone entry (only for fork mazes)
        if self.has_fork and self.state.commit_step is None:  # Haven't committed yet
            if true_pos in self.commit_zone_goal:
                self.state.commit_step = self.state.step
                self.state.visited_vantage_before_commit = self.state.vantage_visits > 0
                self.state.wrong_commit = False  # Committed to goal zone
            elif true_pos in self.commit_zone_trap:
                self.state.commit_step = self.state.step
                self.state.visited_vantage_before_commit = self.state.vantage_visits > 0
                self.state.wrong_commit = True  # Committed to trap zone

        # Check mislocalization
        if self.last_belief_mode is not None:
            # Distance between believed and true position
            dist_error = (abs(self.last_belief_mode[0] - true_pos[0]) +
                         abs(self.last_belief_mode[1] - true_pos[1]))
            self.state.distance_error_history.append(dist_error)

            # Track if belief mode matches true position
            correct = (self.last_belief_mode == true_pos)
            self.state.belief_correct_history.append(correct)

            # Detect mislocalization event (mode is far from true position)
            is_mislocalized = dist_error >= 3

            # Count new mislocalization events (transition from localized to mislocalized)
            if is_mislocalized and not self.was_mislocalized:
                self.state.mislocalization_events += 1
                self.state.mislocalization_steps.append(self.state.step)

            self.was_mislocalized = is_mislocalized

    def get_belief_distribution(self) -> np.ndarray:
        """Get current belief distribution over positions."""
        if self.belief is None:
            return np.zeros((self.grid_size, self.grid_size))
        return self.belief.get_belief_distribution()

    def get_learned_map(self) -> np.ndarray:
        """Get learned wall belief map."""
        if self.model is None:
            return np.zeros((self.grid_size, self.grid_size))
        return self.model.get_learned_map()

    def get_visit_map(self) -> np.ndarray:
        """Get visit count map."""
        if self.model is None:
            return np.zeros((self.grid_size, self.grid_size))
        return self.model.get_visit_map()

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the agent's performance."""
        stats = {
            "steps": self.state.step,
            "total_cost": self.state.total_cost,
            "mean_entropy": np.mean(self.state.entropy_history) if self.state.entropy_history else 0,
            "final_entropy": self.state.entropy_history[-1] if self.state.entropy_history else 0,
            "entropy_trend": self._compute_entropy_trend(),
            "mean_spread": np.mean(self.state.spread_history) if self.state.spread_history else 0,
            "mislocalization_events": self.state.mislocalization_events,
        }

        if self.state.belief_correct_history:
            stats["belief_accuracy"] = np.mean(self.state.belief_correct_history)

        if self.state.distance_error_history:
            stats["mean_distance_error"] = np.mean(self.state.distance_error_history)
            stats["max_distance_error"] = max(self.state.distance_error_history)

        # Entropy in first 20 steps
        if len(self.state.entropy_history) >= 20:
            stats["entropy_first_20"] = np.mean(self.state.entropy_history[:20])

        if self.model is not None:
            stats["model_stats"] = self.model.get_statistics()

        return stats

    def _compute_entropy_trend(self) -> str:
        """Compute trend of entropy over time."""
        if len(self.state.entropy_history) < 10:
            return "insufficient_data"

        n = len(self.state.entropy_history)
        first_quarter = np.mean(self.state.entropy_history[:n//4])
        last_quarter = np.mean(self.state.entropy_history[-n//4:])

        if last_quarter < first_quarter * 0.7:
            return "decreasing"
        elif last_quarter > first_quarter * 1.3:
            return "increasing"
        else:
            return "stable"

    def get_mislocalization_rate(self) -> float:
        """Get fraction of steps where agent was mislocalized."""
        if not self.state.distance_error_history:
            return 0.0
        mislocalized = sum(1 for d in self.state.distance_error_history if d >= 3)
        return mislocalized / len(self.state.distance_error_history)


def run_episode(env: GridWorld,
                agent: WOOEdgeAgent,
                max_steps: int = 300,
                verbose: bool = False) -> Dict:
    """
    Run a complete episode.

    Args:
        env: Environment
        agent: Agent
        max_steps: Maximum steps before termination
        verbose: Print step information

    Returns:
        Episode results dictionary
    """
    # Setup
    obs = env.reset()
    agent.setup(env, know_start=False)  # Start with uniform belief

    done = False
    step = 0

    while not done and step < max_steps:
        # Agent acts
        action = agent.act(obs)

        # Environment steps
        obs, cost, done, info = env.step(action)

        # Record
        agent.record_step(info["true_pos"], cost)

        if verbose:
            mode = agent.belief.get_mode_position()
            true = info["true_pos"]
            entropy = agent.state.entropy_history[-1]
            dist_err = agent.state.distance_error_history[-1] if agent.state.distance_error_history else 0
            print(f"Step {step}: action={Action(action).name}, "
                  f"true={true}, mode={mode}, "
                  f"entropy={entropy:.3f}, dist_err={dist_err}")

        step += 1

    # Compile results
    success = done and env.agent_pos == env.goal_pos
    results = {
        "success": success,
        "steps": step,
        "total_cost": agent.state.total_cost,
        "goal_reached": done,
        "final_entropy": agent.state.entropy_history[-1] if agent.state.entropy_history else 1.0,
        "mean_entropy": np.mean(agent.state.entropy_history) if agent.state.entropy_history else 1.0,
        "belief_accuracy": np.mean(agent.state.belief_correct_history) if agent.state.belief_correct_history else 0.0,
        "entropy_trend": agent._compute_entropy_trend(),
        "mislocalization_events": agent.state.mislocalization_events,
        "mislocalization_rate": agent.get_mislocalization_rate(),
    }

    # Entropy in first N steps
    if len(agent.state.entropy_history) >= 20:
        results["entropy_first_20"] = np.mean(agent.state.entropy_history[:20])
    else:
        results["entropy_first_20"] = np.mean(agent.state.entropy_history) if agent.state.entropy_history else 1.0

    if len(agent.state.entropy_history) >= 15:
        results["entropy_first_15"] = np.mean(agent.state.entropy_history[:15])
    else:
        results["entropy_first_15"] = np.mean(agent.state.entropy_history) if agent.state.entropy_history else 1.0

    if agent.model is not None:
        results["model_confidence"] = agent.model.get_model_confidence()

    if agent.state.distance_error_history:
        results["mean_distance_error"] = np.mean(agent.state.distance_error_history)

    # Vantage point visits
    results["vantage_visits"] = agent.state.vantage_visits

    # Commit zone tracking (only for fork mazes)
    results["has_fork"] = getattr(agent, 'has_fork', False)
    if results["has_fork"]:
        results["commit_step"] = agent.state.commit_step
        results["visited_vantage_before_commit"] = agent.state.visited_vantage_before_commit
        results["wrong_commit"] = agent.state.wrong_commit
    else:
        # N/A for non-fork mazes
        results["commit_step"] = None
        results["visited_vantage_before_commit"] = None
        results["wrong_commit"] = None

    return results


def compare_planners(env_config: EnvConfig,
                     agent_configs: Dict[str, AgentConfig],
                     n_episodes: int = 10,
                     max_steps: int = 300) -> Dict[str, Dict]:
    """
    Compare different planner configurations.

    Args:
        env_config: Environment configuration
        agent_configs: Dictionary of agent configurations to compare
        n_episodes: Number of episodes per configuration
        max_steps: Maximum steps per episode

    Returns:
        Dictionary of results for each configuration
    """
    results = {}

    for name, agent_config in agent_configs.items():
        print(f"\nTesting {name}...")
        episode_results = []

        for ep in range(n_episodes):
            # Create fresh environment and agent
            env = GridWorld(EnvConfig(
                grid_size=env_config.grid_size,
                maze_type=env_config.maze_type,
                sensor_noise_prob=env_config.sensor_noise_prob,
                slip_prob=env_config.slip_prob,
                trap_corridors=env_config.trap_corridors,
                seed=env_config.seed + ep if env_config.seed is not None else ep
            ))

            # Create agent with updated seed
            agent_dict = {k: v for k, v in agent_config.__dict__.items() if k != 'seed'}
            agent_seed = agent_config.seed + ep if agent_config.seed is not None else ep
            agent = WOOEdgeAgent(AgentConfig(**agent_dict, seed=agent_seed))

            result = run_episode(env, agent, max_steps)
            episode_results.append(result)

            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep + 1}/{n_episodes}")

        # Aggregate results
        results[name] = {
            "success_rate": np.mean([r["success"] for r in episode_results]),
            "mean_steps": np.mean([r["steps"] for r in episode_results]),
            "mean_cost": np.mean([r["total_cost"] for r in episode_results]),
            "mean_final_entropy": np.mean([r["final_entropy"] for r in episode_results]),
            "mean_entropy_first_20": np.mean([r["entropy_first_20"] for r in episode_results]),
            "mean_belief_accuracy": np.mean([r["belief_accuracy"] for r in episode_results]),
            "total_mislocalization_events": sum([r["mislocalization_events"] for r in episode_results]),
            "mean_mislocalization_rate": np.mean([r["mislocalization_rate"] for r in episode_results]),
            "episodes": episode_results,
        }

    return results
