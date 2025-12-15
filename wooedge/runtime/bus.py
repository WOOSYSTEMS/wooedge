"""
Observation Bus

Universal observation ingestion layer for WooEdge runtime.
Abstracts data sources (sensors, APIs, files, streams) into a unified interface.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Iterator, TypeVar, Generic
from abc import ABC, abstractmethod
from enum import Enum
import time
import queue
import threading


T = TypeVar('T')


class SourceType(Enum):
    """Types of observation sources."""
    SENSOR = "sensor"      # Hardware sensors (serial, GPIO, etc.)
    API = "api"            # External APIs (REST, WebSocket)
    FILE = "file"          # File-based (CSV, logs)
    STREAM = "stream"      # Live streams
    MEMORY = "memory"      # In-memory (testing, simulation)
    CALLBACK = "callback"  # Push-based callbacks


@dataclass
class Observation:
    """
    Universal observation container.

    Wraps any observation data with metadata.
    """
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    source_type: SourceType = SourceType.MEMORY
    sequence: int = 0

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from observation data."""
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize observation."""
        return {
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "source_type": self.source_type.value,
            "sequence": self.sequence,
        }


class ObservationSource(ABC):
    """
    Abstract base for observation sources.

    Apps implement this to define how observations are produced.
    """

    def __init__(self, name: str, source_type: SourceType):
        self.name = name
        self.source_type = source_type
        self._sequence = 0

    @abstractmethod
    def read(self) -> Optional[Observation]:
        """
        Read a single observation.

        Returns None if no observation available (non-blocking).
        """
        pass

    def read_blocking(self, timeout: float = None) -> Optional[Observation]:
        """
        Read with optional blocking.

        Default implementation polls. Override for efficient blocking.
        """
        start = time.time()
        while True:
            obs = self.read()
            if obs is not None:
                return obs
            if timeout is not None and (time.time() - start) > timeout:
                return None
            time.sleep(0.01)  # Small sleep to avoid busy-wait

    def _make_observation(self, data: Dict[str, Any]) -> Observation:
        """Helper to create observation with metadata."""
        self._sequence += 1
        return Observation(
            data=data,
            timestamp=time.time(),
            source=self.name,
            source_type=self.source_type,
            sequence=self._sequence,
        )

    def open(self) -> None:
        """Open/initialize the source. Override if needed."""
        pass

    def close(self) -> None:
        """Close/cleanup the source. Override if needed."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class MemorySource(ObservationSource):
    """
    In-memory observation source for testing.

    Pre-load observations and read them sequentially.
    """

    def __init__(self, name: str = "memory", observations: List[Dict[str, Any]] = None):
        super().__init__(name, SourceType.MEMORY)
        self._observations = list(observations) if observations else []
        self._index = 0

    def add(self, data: Dict[str, Any]) -> None:
        """Add an observation to the queue."""
        self._observations.append(data)

    def read(self) -> Optional[Observation]:
        if self._index >= len(self._observations):
            return None
        data = self._observations[self._index]
        self._index += 1
        return self._make_observation(data)

    def reset(self) -> None:
        """Reset to beginning."""
        self._index = 0


class CallbackSource(ObservationSource):
    """
    Callback-based observation source.

    External code pushes observations via callback.
    """

    def __init__(self, name: str = "callback", buffer_size: int = 100):
        super().__init__(name, SourceType.CALLBACK)
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)

    def push(self, data: Dict[str, Any]) -> bool:
        """
        Push an observation (called by external code).

        Returns False if buffer is full.
        """
        try:
            self._queue.put_nowait(self._make_observation(data))
            return True
        except queue.Full:
            return False

    def read(self) -> Optional[Observation]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def read_blocking(self, timeout: float = None) -> Optional[Observation]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class TransformSource(ObservationSource):
    """
    Wrapper that transforms observations from another source.
    """

    def __init__(
        self,
        source: ObservationSource,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]],
        name: str = None,
    ):
        super().__init__(name or f"transform:{source.name}", source.source_type)
        self._source = source
        self._transform = transform

    def read(self) -> Optional[Observation]:
        obs = self._source.read()
        if obs is None:
            return None
        transformed_data = self._transform(obs.data)
        return self._make_observation(transformed_data)

    def open(self) -> None:
        self._source.open()

    def close(self) -> None:
        self._source.close()


@dataclass
class ObservationBus:
    """
    Central observation bus for WooEdge runtime.

    Manages multiple observation sources and provides unified access.

    Example:
        bus = ObservationBus()
        bus.register(sensor_source)
        bus.register(api_source)

        for obs in bus.stream():
            process(obs)
    """

    sources: Dict[str, ObservationSource] = field(default_factory=dict)
    _handlers: List[Callable[[Observation], None]] = field(default_factory=list)
    _running: bool = False

    def register(self, source: ObservationSource) -> None:
        """Register an observation source."""
        self.sources[source.name] = source

    def unregister(self, name: str) -> None:
        """Unregister a source by name."""
        if name in self.sources:
            self.sources[name].close()
            del self.sources[name]

    def on_observation(self, handler: Callable[[Observation], None]) -> None:
        """Register a handler for all observations."""
        self._handlers.append(handler)

    def read_all(self) -> List[Observation]:
        """
        Read one observation from each source (non-blocking).

        Returns list of available observations.
        """
        observations = []
        for source in self.sources.values():
            obs = source.read()
            if obs is not None:
                observations.append(obs)
                self._notify_handlers(obs)
        return observations

    def read_any(self) -> Optional[Observation]:
        """
        Read one observation from any source (non-blocking).

        Returns first available observation.
        """
        for source in self.sources.values():
            obs = source.read()
            if obs is not None:
                self._notify_handlers(obs)
                return obs
        return None

    def read_from(self, source_name: str) -> Optional[Observation]:
        """Read from a specific source."""
        source = self.sources.get(source_name)
        if source is None:
            return None
        obs = source.read()
        if obs is not None:
            self._notify_handlers(obs)
        return obs

    def stream(self, poll_interval: float = 0.01) -> Iterator[Observation]:
        """
        Stream observations from all sources.

        Yields observations as they become available.
        """
        self._running = True
        try:
            while self._running:
                obs = self.read_any()
                if obs is not None:
                    yield obs
                else:
                    time.sleep(poll_interval)
        finally:
            self._running = False

    def stream_blocking(self, timeout: float = 1.0) -> Iterator[Observation]:
        """
        Stream with blocking reads.

        More efficient than polling for sources that support blocking.
        """
        self._running = True
        try:
            while self._running:
                for source in self.sources.values():
                    obs = source.read_blocking(timeout=timeout / len(self.sources))
                    if obs is not None:
                        self._notify_handlers(obs)
                        yield obs
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop streaming."""
        self._running = False

    def open_all(self) -> None:
        """Open all registered sources."""
        for source in self.sources.values():
            source.open()

    def close_all(self) -> None:
        """Close all registered sources."""
        for source in self.sources.values():
            source.close()

    def _notify_handlers(self, obs: Observation) -> None:
        """Notify all handlers of new observation."""
        for handler in self._handlers:
            try:
                handler(obs)
            except Exception:
                pass  # Don't let handler errors break the bus

    def __enter__(self):
        self.open_all()
        return self

    def __exit__(self, *args):
        self.close_all()


class SchemaValidator:
    """
    Validates observations against a schema.

    Ensures observations have required fields with correct types.
    """

    def __init__(self, schema: Dict[str, type]):
        """
        Args:
            schema: Dict mapping field names to expected types
        """
        self.schema = schema

    def validate(self, obs: Observation) -> bool:
        """Check if observation matches schema."""
        for field_name, expected_type in self.schema.items():
            if field_name not in obs.data:
                return False
            if not isinstance(obs.data[field_name], expected_type):
                return False
        return True

    def filter(self, obs: Observation) -> Optional[Observation]:
        """Return observation only if valid."""
        return obs if self.validate(obs) else None

    def coerce(self, obs: Observation) -> Observation:
        """Coerce observation data to match schema (best effort)."""
        coerced = {}
        for field_name, expected_type in self.schema.items():
            value = obs.data.get(field_name)
            if value is None:
                # Use default for type
                if expected_type == float:
                    coerced[field_name] = 0.0
                elif expected_type == int:
                    coerced[field_name] = 0
                elif expected_type == str:
                    coerced[field_name] = ""
                elif expected_type == bool:
                    coerced[field_name] = False
                else:
                    coerced[field_name] = None
            else:
                try:
                    coerced[field_name] = expected_type(value)
                except (ValueError, TypeError):
                    coerced[field_name] = value

        # Include extra fields not in schema
        for key, value in obs.data.items():
            if key not in coerced:
                coerced[key] = value

        return Observation(
            data=coerced,
            timestamp=obs.timestamp,
            source=obs.source,
            source_type=obs.source_type,
            sequence=obs.sequence,
        )
