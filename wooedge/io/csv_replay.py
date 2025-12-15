"""
CSV Replay Source

Reads sensor logs from CSV files and emits observations compatible with
DecisionSafety.observe(). Enables offline replay of real-world data.

CSV Schema:
    timestep,front_dist,left_dist,right_dist,hazard_hint

Example:
    0,3,2,4,0.1
    1,2,2,4,0.15
    2,1,2,4,0.8
"""

import csv
from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple


@dataclass
class ReplayObservation:
    """
    Observation from CSV replay.

    Compatible with AssistiveObservation for DecisionSafety.observe().
    """
    front_dist: int
    left_dist: int
    right_dist: int
    hazard_hint: float
    scanned: bool = False
    timestep: int = 0

    def to_tuple(self) -> Tuple[int, int, int, float, bool]:
        """Convert to tuple for hashing (matches AssistiveObservation)."""
        return (self.front_dist, self.left_dist, self.right_dist,
                round(self.hazard_hint, 1), self.scanned)


class CSVReplaySource:
    """
    Reads CSV sensor logs and emits observations for DecisionSafety replay.

    Usage:
        source = CSVReplaySource("sensor_log.csv")
        for obs in source:
            safety.observe(obs)
            result = safety.propose(action)

    The CSV must have header: timestep,front_dist,left_dist,right_dist,hazard_hint
    """

    def __init__(self, filepath: str):
        """
        Initialize CSV replay source.

        Args:
            filepath: Path to CSV file with sensor data
        """
        self.filepath = filepath
        self._observations: List[ReplayObservation] = []
        self._loaded = False

    def load(self) -> None:
        """Load all observations from CSV file."""
        self._observations = []

        with open(self.filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)

            # Validate header
            required = {'timestep', 'front_dist', 'left_dist', 'right_dist', 'hazard_hint'}
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {self.filepath} has no header")

            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for row in reader:
                obs = ReplayObservation(
                    timestep=int(row['timestep']),
                    front_dist=int(row['front_dist']),
                    left_dist=int(row['left_dist']),
                    right_dist=int(row['right_dist']),
                    hazard_hint=float(row['hazard_hint']),
                    scanned=row.get('scanned', 'false').lower() == 'true'
                )
                self._observations.append(obs)

        self._loaded = True

    def __iter__(self) -> Iterator[ReplayObservation]:
        """Iterate over observations in order."""
        if not self._loaded:
            self.load()
        return iter(self._observations)

    def __len__(self) -> int:
        """Return number of observations."""
        if not self._loaded:
            self.load()
        return len(self._observations)

    def __getitem__(self, index: int) -> ReplayObservation:
        """Get observation by index."""
        if not self._loaded:
            self.load()
        return self._observations[index]

    @property
    def observations(self) -> List[ReplayObservation]:
        """Get all observations as list."""
        if not self._loaded:
            self.load()
        return self._observations

    def get_timesteps(self) -> List[int]:
        """Get list of timesteps in replay order."""
        if not self._loaded:
            self.load()
        return [obs.timestep for obs in self._observations]

    def validate_order(self) -> bool:
        """Check that timesteps are in ascending order."""
        if not self._loaded:
            self.load()

        for i in range(1, len(self._observations)):
            if self._observations[i].timestep < self._observations[i-1].timestep:
                return False
        return True
