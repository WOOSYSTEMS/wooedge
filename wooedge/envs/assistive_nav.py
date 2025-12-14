"""
Assistive Navigation Environment

A simulation environment for testing DecisionSafety in an assistive
navigation context (e.g., smart cane or wheelchair guidance system).

The environment models a corridor with:
- A fork where one path leads to a safe exit (goal)
- Another path leads to a hazard (stairs/drop-off)
- Partial observability: sensors provide noisy local information
- SCAN action to reduce uncertainty before committing

Layout (8x6 grid):
```
########
#S     #
#  ### #
#  #H  #
#  # G #
########
```
S = Start, G = Goal (safe exit), H = Hazard (stairs)
The corridor splits - left leads to hazard, right leads to goal.
"""

import numpy as np
from typing import Tuple, Set, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import IntEnum


class AssistiveAction(IntEnum):
    """Actions available in the assistive navigation environment."""
    FORWARD = 0  # Move forward (relative to current heading)
    LEFT = 1     # Turn left 90 degrees
    RIGHT = 2    # Turn right 90 degrees
    SCAN = 3     # Probe/scan to reduce uncertainty


class Heading(IntEnum):
    """Agent heading direction."""
    NORTH = 0  # Up (decreasing y)
    EAST = 1   # Right (increasing x)
    SOUTH = 2  # Down (increasing y)
    WEST = 3   # Left (decreasing x)


# Heading deltas: (dy, dx)
HEADING_DELTAS = {
    Heading.NORTH: (-1, 0),
    Heading.EAST: (0, 1),
    Heading.SOUTH: (1, 0),
    Heading.WEST: (0, -1),
}


@dataclass
class AssistiveObservation:
    """Observation from the assistive navigation environment."""
    front_dist: int      # Distance to obstacle in front
    left_dist: int       # Distance to obstacle on left
    right_dist: int      # Distance to obstacle on right
    hazard_hint: float   # Noisy hint about nearby hazard (0-1)
    scanned: bool        # Whether last action was SCAN (higher reliability)

    def to_tuple(self) -> Tuple[int, int, int, float, bool]:
        """Convert to tuple for hashing."""
        return (self.front_dist, self.left_dist, self.right_dist,
                round(self.hazard_hint, 1), self.scanned)


@dataclass
class AssistiveNavConfig:
    """Configuration for the assistive navigation environment."""
    sensor_noise: float = 0.3      # Probability of distance sensor noise
    hazard_noise: float = 0.4      # Noise in hazard hint (far from hazard)
    hazard_reliable_dist: int = 2  # Distance at which hazard hint becomes reliable
    scan_noise_reduction: float = 0.7  # Noise reduction factor when scanning
    seed: Optional[int] = None


class AssistiveNavEnv:
    """
    Assistive Navigation Environment for testing DecisionSafety.

    Models a corridor navigation scenario where:
    - Agent must reach a safe exit (goal)
    - One path leads to hazard (stairs/drop-off)
    - Observations are noisy and locally ambiguous
    - SCAN action improves observation quality

    Implements the standard env interface: reset(), step(), observe()
    Compatible with DecisionSafety module.
    """

    # Grid constants
    EMPTY = 0
    WALL = 1
    HAZARD = 2
    GOAL = 3
    START = 4

    def __init__(self, config: Optional[AssistiveNavConfig] = None):
        """Initialize the assistive navigation environment."""
        self.config = config or AssistiveNavConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Build the grid (8x6)
        self.grid_height = 8
        self.grid_width = 6
        self.grid = self._build_grid()

        # Find key positions
        self.start_pos = self._find_cell(self.START)
        self.goal_pos = self._find_cell(self.GOAL)
        self.hazard_pos = self._find_cell(self.HAZARD)

        # Define commit zones (fork structure)
        self.has_fork = True
        self.commit_zone_goal: Set[Tuple[int, int]] = set()
        self.commit_zone_trap: Set[Tuple[int, int]] = set()
        self.fork_split_y: Optional[int] = None
        self._define_commit_zones()

        # Agent state
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.agent_heading: Heading = Heading.SOUTH
        self.last_action_was_scan: bool = False
        self.done: bool = False
        self.steps: int = 0

        # Valid positions for belief
        self.valid_positions = self._get_valid_positions()

    def _build_grid(self) -> np.ndarray:
        """
        Build the navigation grid.

        Layout (8x8):
        ########
        #S.....#
        #..#...#
        #.H#.G.#
        #..#...#
        #......#
        #......#
        ########

        S = Start, H = Hazard (left path), G = Goal (right path)
        The central column (#) creates a fork - left leads to hazard, right to goal.
        """
        self.grid_height = 8
        self.grid_width = 8
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)

        # Walls (border)
        grid[0, :] = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0] = self.WALL
        grid[:, -1] = self.WALL

        # Central divider creating the fork (partial wall)
        grid[2, 3] = self.WALL
        grid[3, 3] = self.WALL
        grid[4, 3] = self.WALL

        # Key positions
        grid[1, 1] = self.START   # Start position (top-left)
        grid[3, 2] = self.HAZARD  # Hazard (stairs) - left of divider
        grid[3, 5] = self.GOAL    # Goal (safe exit) - right of divider

        return grid

    def _find_cell(self, cell_type: int) -> Tuple[int, int]:
        """Find position of a cell type."""
        positions = np.where(self.grid == cell_type)
        if len(positions[0]) > 0:
            return (int(positions[0][0]), int(positions[1][0]))
        raise ValueError(f"Cell type {cell_type} not found in grid")

    def _get_valid_positions(self) -> List[Tuple[int, int]]:
        """Get list of valid (non-wall) positions."""
        positions = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x] != self.WALL:
                    positions.append((y, x))
        return positions

    def _define_commit_zones(self) -> None:
        """
        Define commit zones for the fork structure.

        Commit zones mark irreversible choices:
        - Goal zone: cells leading to/at the safe exit (right of divider)
        - Trap zone: cells leading to/at the hazard (left of divider)
        """
        # Fork split is at x=3 (the dividing wall column)
        self.fork_split_y = 3  # Actually x-based split

        # Trap zone: hazard cell and adjacent cells on left side
        self.commit_zone_trap.add(self.hazard_pos)  # (3, 2)
        self.commit_zone_trap.add((2, 2))  # Above hazard
        self.commit_zone_trap.add((4, 2))  # Below hazard
        self.commit_zone_trap.add((3, 1))  # Left of hazard

        # Goal zone: goal cell and adjacent cells on right side
        self.commit_zone_goal.add(self.goal_pos)    # (3, 5)
        self.commit_zone_goal.add((2, 5))  # Above goal
        self.commit_zone_goal.add((4, 5))  # Below goal
        self.commit_zone_goal.add((3, 4))  # Left of goal
        self.commit_zone_goal.add((3, 6))  # Right of goal

    def reset(self, seed: Optional[int] = None) -> AssistiveObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Optional random seed

        Returns:
            Initial observation
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = self.start_pos
        self.agent_heading = Heading.SOUTH
        self.last_action_was_scan = False
        self.done = False
        self.steps = 0

        return self.observe()

    def observe(self) -> AssistiveObservation:
        """
        Get current observation from the environment.

        Returns:
            AssistiveObservation with sensor readings
        """
        # Apply noise reduction if last action was SCAN
        noise_mult = self.config.scan_noise_reduction if self.last_action_was_scan else 1.0

        # Distance sensors (front, left, right relative to heading)
        front_dist = self._sense_distance(self.agent_heading, noise_mult)
        left_dist = self._sense_distance(Heading((self.agent_heading - 1) % 4), noise_mult)
        right_dist = self._sense_distance(Heading((self.agent_heading + 1) % 4), noise_mult)

        # Hazard hint (noisy unless close)
        hazard_hint = self._sense_hazard(noise_mult)

        return AssistiveObservation(
            front_dist=front_dist,
            left_dist=left_dist,
            right_dist=right_dist,
            hazard_hint=hazard_hint,
            scanned=self.last_action_was_scan
        )

    def _sense_distance(self, heading: Heading, noise_mult: float) -> int:
        """
        Sense distance to obstacle in given direction.

        Args:
            heading: Direction to sense
            noise_mult: Noise multiplier (lower = less noise)

        Returns:
            Distance to nearest obstacle (may be noisy)
        """
        dy, dx = HEADING_DELTAS[heading]
        true_dist = 0
        y, x = self.agent_pos

        while True:
            y += dy
            x += dx
            if (y < 0 or y >= self.grid_height or
                x < 0 or x >= self.grid_width or
                self.grid[y, x] == self.WALL):
                break
            true_dist += 1

        # Add noise
        if self.rng.random() < self.config.sensor_noise * noise_mult:
            # Noisy reading: ±1 with clamp
            noise = self.rng.choice([-1, 0, 1])
            return max(0, true_dist + noise)

        return true_dist

    def _sense_hazard(self, noise_mult: float) -> float:
        """
        Sense nearby hazard (noisy unless close).

        Args:
            noise_mult: Noise multiplier

        Returns:
            Hazard hint value (0-1, higher = more likely hazard nearby)
        """
        # Manhattan distance to hazard
        dist_to_hazard = (abs(self.agent_pos[0] - self.hazard_pos[0]) +
                         abs(self.agent_pos[1] - self.hazard_pos[1]))

        # True signal based on proximity
        if dist_to_hazard <= self.config.hazard_reliable_dist:
            # Close to hazard: reliable signal
            true_signal = 1.0 - (dist_to_hazard / (self.config.hazard_reliable_dist + 1))
        else:
            # Far from hazard: weak signal
            true_signal = 0.2

        # Add noise (more noise when far from hazard)
        if dist_to_hazard > self.config.hazard_reliable_dist:
            noise_level = self.config.hazard_noise * noise_mult
        else:
            noise_level = 0.1 * noise_mult

        noise = self.rng.normal(0, noise_level)
        return np.clip(true_signal + noise, 0, 1)

    def step(self, action: int) -> Tuple[AssistiveObservation, float, bool, Dict[str, Any]]:
        """
        Take an action in the environment.

        Args:
            action: Action to take (0-3: FORWARD, LEFT, RIGHT, SCAN)

        Returns:
            Tuple of (observation, cost, done, info)
        """
        action = AssistiveAction(action)
        cost = 1.0  # Base step cost
        self.steps += 1

        # Remember if this was a scan action
        self.last_action_was_scan = (action == AssistiveAction.SCAN)

        if action == AssistiveAction.FORWARD:
            # Move forward in current heading direction
            dy, dx = HEADING_DELTAS[self.agent_heading]
            new_y, new_x = self.agent_pos[0] + dy, self.agent_pos[1] + dx

            # Check if valid move
            if (0 <= new_y < self.grid_height and
                0 <= new_x < self.grid_width and
                self.grid[new_y, new_x] != self.WALL):
                self.agent_pos = (new_y, new_x)

        elif action == AssistiveAction.LEFT:
            # Turn left (counterclockwise)
            self.agent_heading = Heading((self.agent_heading - 1) % 4)

        elif action == AssistiveAction.RIGHT:
            # Turn right (clockwise)
            self.agent_heading = Heading((self.agent_heading + 1) % 4)

        elif action == AssistiveAction.SCAN:
            # Scan action - no movement, just better observation next
            cost = 0.5  # Scanning is cheaper than moving

        # Check termination
        if self.grid[self.agent_pos] == self.GOAL:
            self.done = True
            cost = -10.0  # Reward for reaching goal
        elif self.grid[self.agent_pos] == self.HAZARD:
            self.done = True
            cost = 50.0  # High cost for hitting hazard

        info = {
            "true_pos": self.agent_pos,
            "heading": self.agent_heading,
            "at_goal": self.grid[self.agent_pos] == self.GOAL,
            "at_hazard": self.grid[self.agent_pos] == self.HAZARD,
            "in_commit_zone_goal": self.agent_pos in self.commit_zone_goal,
            "in_commit_zone_trap": self.agent_pos in self.commit_zone_trap,
        }

        return self.observe(), cost, self.done, info

    def get_grid_copy(self) -> np.ndarray:
        """Get a copy of the grid."""
        return self.grid.copy()

    def render_ascii(self) -> str:
        """Render the environment as ASCII art."""
        symbols = {
            self.EMPTY: '.',
            self.WALL: '#',
            self.HAZARD: 'H',
            self.GOAL: 'G',
            self.START: 'S',
        }
        heading_symbols = {
            Heading.NORTH: '^',
            Heading.EAST: '>',
            Heading.SOUTH: 'v',
            Heading.WEST: '<',
        }

        lines = []
        for y in range(self.grid_height):
            row = ""
            for x in range(self.grid_width):
                if (y, x) == self.agent_pos:
                    row += heading_symbols[self.agent_heading]
                else:
                    row += symbols.get(self.grid[y, x], '?')
            lines.append(row)

        return '\n'.join(lines)
