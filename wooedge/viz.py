"""
Visualization Module

Provides:
- ASCII grid display
- Optional matplotlib visualization
- Belief heatmap display
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import sys

from .env import GridWorld, Action


# ANSI color codes for terminal
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def supports_color() -> bool:
    """Check if terminal supports colors."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


class ASCIIRenderer:
    """
    ASCII-based visualization for the gridworld.
    """

    # Cell characters
    EMPTY = "·"
    WALL = "█"
    GOAL = "G"
    START = "S"
    AGENT = "A"
    BELIEF_LOW = "░"
    BELIEF_MED = "▒"
    BELIEF_HIGH = "▓"

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and supports_color()

    def render_grid(self,
                    env: GridWorld,
                    belief: Optional[np.ndarray] = None,
                    show_belief: bool = True,
                    step: int = 0,
                    action: Optional[int] = None,
                    cost: float = 0.0,
                    entropy: float = 0.0) -> str:
        """
        Render the grid as ASCII art.

        Args:
            env: The environment
            belief: Belief distribution (optional)
            show_belief: Whether to show belief overlay
            step: Current step number
            action: Last action taken
            cost: Current cost
            entropy: Current entropy

        Returns:
            String representation of the grid
        """
        grid = env.get_grid_copy()
        grid_size = env.grid_size
        agent_pos = env.agent_pos
        goal_pos = env.goal_pos

        lines = []

        # Header
        header = f"Step: {step}"
        if action is not None:
            header += f" | Action: {Action(action).name}"
        header += f" | Cost: {cost:.2f} | Entropy: {entropy:.3f}"
        lines.append(header)
        lines.append("=" * len(header))

        # Column numbers
        col_nums = "  " + "".join([f"{i%10}" for i in range(grid_size)])
        lines.append(col_nums)

        # Grid rows
        for y in range(grid_size):
            row = f"{y%10} "
            for x in range(grid_size):
                cell = self._get_cell_char(
                    grid, y, x, agent_pos, goal_pos,
                    belief, show_belief
                )
                row += cell
            lines.append(row)

        # Legend
        lines.append("")
        lines.append(f"Legend: {self._colorize(self.AGENT, Colors.CYAN)}=Agent "
                    f"{self._colorize(self.GOAL, Colors.GREEN)}=Goal "
                    f"{self._colorize(self.WALL, Colors.WHITE)}=Wall "
                    f"{self.BELIEF_LOW}/{self.BELIEF_MED}/{self.BELIEF_HIGH}=Belief")

        return "\n".join(lines)

    def _get_cell_char(self,
                       grid: np.ndarray,
                       y: int,
                       x: int,
                       agent_pos: Tuple[int, int],
                       goal_pos: Tuple[int, int],
                       belief: Optional[np.ndarray],
                       show_belief: bool) -> str:
        """Get character for a single cell."""
        # Priority: agent > goal > wall > belief > empty
        if (y, x) == agent_pos:
            return self._colorize(self.AGENT, Colors.CYAN + Colors.BOLD)

        if (y, x) == goal_pos:
            return self._colorize(self.GOAL, Colors.GREEN + Colors.BOLD)

        if grid[y, x] == 1:  # Wall
            return self._colorize(self.WALL, Colors.DIM)

        # Show belief if available
        if show_belief and belief is not None:
            prob = belief[y, x]
            if prob > 0.2:
                return self._colorize(self.BELIEF_HIGH, Colors.YELLOW)
            elif prob > 0.05:
                return self._colorize(self.BELIEF_MED, Colors.YELLOW)
            elif prob > 0.01:
                return self._colorize(self.BELIEF_LOW, Colors.DIM)

        return self._colorize(self.EMPTY, Colors.DIM)

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def render_belief_heatmap(self, belief: np.ndarray) -> str:
        """
        Render belief distribution as ASCII heatmap.
        """
        grid_size = belief.shape[0]
        lines = []

        lines.append("Belief Heatmap:")
        lines.append("-" * (grid_size + 2))

        # Normalize belief
        max_belief = np.max(belief)
        if max_belief > 0:
            normalized = belief / max_belief
        else:
            normalized = belief

        # Intensity levels
        levels = " ░▒▓█"

        for y in range(grid_size):
            row = "|"
            for x in range(grid_size):
                level = int(normalized[y, x] * (len(levels) - 1))
                row += levels[level]
            row += "|"
            lines.append(row)

        lines.append("-" * (grid_size + 2))

        return "\n".join(lines)

    def render_model_map(self, wall_belief: np.ndarray) -> str:
        """
        Render learned wall belief as ASCII map.
        """
        grid_size = wall_belief.shape[0]
        lines = []

        lines.append("Learned Map (wall probability):")
        lines.append("-" * (grid_size + 2))

        for y in range(grid_size):
            row = "|"
            for x in range(grid_size):
                prob = wall_belief[y, x]
                if prob > 0.8:
                    row += "█"
                elif prob > 0.5:
                    row += "▓"
                elif prob > 0.3:
                    row += "▒"
                elif prob > 0.1:
                    row += "░"
                else:
                    row += " "
            row += "|"
            lines.append(row)

        lines.append("-" * (grid_size + 2))

        return "\n".join(lines)


class MatplotlibRenderer:
    """
    Matplotlib-based visualization (optional).
    """

    def __init__(self):
        self.fig = None
        self.axes = None
        self._check_matplotlib()

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            return True
        except ImportError:
            self.plt = None
            return False

    def is_available(self) -> bool:
        """Check if matplotlib visualization is available."""
        return self.plt is not None

    def setup(self, grid_size: int) -> None:
        """Set up matplotlib figure."""
        if not self.is_available():
            return

        self.fig, self.axes = self.plt.subplots(1, 3, figsize=(15, 5))
        self.plt.ion()  # Interactive mode

    def render(self,
               grid: np.ndarray,
               agent_pos: Tuple[int, int],
               goal_pos: Tuple[int, int],
               belief: np.ndarray,
               wall_belief: np.ndarray,
               visit_counts: np.ndarray,
               step: int = 0) -> None:
        """
        Render visualization using matplotlib.
        """
        if not self.is_available() or self.fig is None:
            return

        # Clear axes
        for ax in self.axes:
            ax.clear()

        # Plot 1: Grid with agent and goal
        ax1 = self.axes[0]
        display_grid = grid.copy().astype(float)
        display_grid[agent_pos] = 2
        display_grid[goal_pos] = 3

        ax1.imshow(display_grid, cmap='viridis', interpolation='nearest')
        ax1.set_title(f'Grid (Step {step})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Plot 2: Belief distribution
        ax2 = self.axes[1]
        im2 = ax2.imshow(belief, cmap='hot', interpolation='nearest',
                         vmin=0, vmax=np.max(belief) + 0.01)
        ax2.set_title('Belief Distribution')
        ax2.plot(agent_pos[1], agent_pos[0], 'g*', markersize=15, label='True pos')
        ax2.legend()

        # Plot 3: Learned map
        ax3 = self.axes[2]
        ax3.imshow(wall_belief, cmap='gray', interpolation='nearest',
                   vmin=0, vmax=1)
        ax3.set_title('Learned Wall Belief')

        self.plt.tight_layout()
        self.plt.pause(0.1)

    def close(self) -> None:
        """Close matplotlib figure."""
        if self.is_available() and self.fig is not None:
            self.plt.close(self.fig)


def print_statistics(stats: Dict, prefix: str = "") -> None:
    """Print statistics in a formatted way."""
    print(f"\n{prefix}{'='*50}")
    print(f"{prefix}Results Summary")
    print(f"{prefix}{'='*50}")

    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"{prefix}  {k}: {v:.4f}")
                else:
                    print(f"{prefix}  {k}: {v}")
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        else:
            print(f"{prefix}{key}: {value}")

    print(f"{prefix}{'='*50}\n")


def print_benchmark_results(results: Dict[str, Dict]) -> None:
    """Print comparison benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)

    # Header
    print(f"\n{'Planner':<15} {'Success%':>10} {'Avg Steps':>10} "
          f"{'Avg Cost':>10} {'Entropy':>10} {'Accuracy':>10}")
    print("-"*70)

    for name, result in results.items():
        print(f"{name:<15} "
              f"{result['success_rate']*100:>9.1f}% "
              f"{result['mean_steps']:>10.1f} "
              f"{result['mean_cost']:>10.2f} "
              f"{result['mean_final_entropy']:>10.3f} "
              f"{result['mean_belief_accuracy']*100:>9.1f}%")

    print("-"*70)
    print()


def animate_episode(env: GridWorld,
                    agent,
                    max_steps: int = 200,
                    delay: float = 0.1,
                    use_matplotlib: bool = False) -> Dict:
    """
    Run and animate an episode.

    Args:
        env: Environment
        agent: Agent
        max_steps: Maximum steps
        delay: Delay between frames (seconds)
        use_matplotlib: Use matplotlib instead of ASCII

    Returns:
        Episode results
    """
    import time

    ascii_renderer = ASCIIRenderer(use_colors=True)
    mpl_renderer = MatplotlibRenderer() if use_matplotlib else None

    if mpl_renderer and mpl_renderer.is_available():
        mpl_renderer.setup(env.grid_size)

    obs = env.reset()
    agent.setup(env, know_start=True)

    done = False
    step = 0
    total_cost = 0.0

    while not done and step < max_steps:
        # Agent acts
        action = agent.act(obs)

        # Environment steps
        obs, cost, done, info = env.step(action)
        agent.record_step(info["true_pos"], cost)
        total_cost += cost

        # Get current state
        belief = agent.get_belief_distribution()
        entropy = agent.state.entropy_history[-1] if agent.state.entropy_history else 1.0

        # Render
        if use_matplotlib and mpl_renderer and mpl_renderer.is_available():
            mpl_renderer.render(
                env.get_grid_copy(),
                env.agent_pos,
                env.goal_pos,
                belief,
                agent.get_learned_map(),
                agent.get_visit_map(),
                step
            )
        else:
            # Clear screen (ANSI escape)
            print("\033[2J\033[H", end="")

            output = ascii_renderer.render_grid(
                env, belief, True, step, action, total_cost, entropy
            )
            print(output)

            # Also show learned map
            if step > 0 and step % 20 == 0:
                print()
                print(ascii_renderer.render_model_map(agent.get_learned_map()))

        time.sleep(delay)
        step += 1

    if mpl_renderer:
        mpl_renderer.close()

    # Final results
    success = done and env.agent_pos == env.goal_pos
    return {
        "success": success,
        "steps": step,
        "total_cost": total_cost,
    }
