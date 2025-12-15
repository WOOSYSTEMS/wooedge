"""
WooEdge Runtime Engine

Central orchestrator for WooEdge applications.
Manages app lifecycle, observation routing, and execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Iterator, Type
from enum import Enum
import time
import threading
import logging

from .app import WooEdgeApp, AppState, AppConfig, AppContext
from .bus import ObservationBus, Observation, ObservationSource, MemorySource
from .belief import BeliefState
from .entropy import EntropyTracker
from .gate import GateResult


logger = logging.getLogger(__name__)


class EngineState(Enum):
    """Engine lifecycle states."""
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class EngineConfig:
    """
    Runtime engine configuration.
    """
    # Execution
    tick_interval: float = 0.1  # Seconds between ticks
    max_ticks: int = 0  # 0 = unlimited

    # Threading
    async_mode: bool = False  # Run in background thread

    # Logging
    log_level: int = logging.INFO
    verbose: bool = False

    # Error handling
    stop_on_error: bool = False


@dataclass
class TickResult:
    """Result of a single engine tick."""
    tick: int
    timestamp: float
    app_name: str
    observation: Optional[Observation]
    gate_result: Optional[GateResult]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "app_name": self.app_name,
            "observation": self.observation.to_dict() if self.observation else None,
            "gate_result": self.gate_result.to_dict() if self.gate_result else None,
            "error": self.error,
        }


class WooEdgeEngine:
    """
    WooEdge Runtime Engine.

    Manages the lifecycle and execution of WooEdge apps.

    Example:
        engine = WooEdgeEngine()
        engine.register_app(my_app, "main")
        engine.register_source(sensor_source)

        # Run for 100 ticks
        for result in engine.run(max_ticks=100):
            print(f"Tick {result.tick}: {result.gate_result.final_action}")

        # Or run in background
        engine.start()
        time.sleep(10)
        engine.stop()
    """

    def __init__(self, config: EngineConfig = None):
        """
        Initialize the engine.

        Args:
            config: Engine configuration
        """
        self.config = config or EngineConfig()
        self.state = EngineState.CREATED

        # Components
        self.apps: Dict[str, WooEdgeApp] = {}
        self.bus = ObservationBus()

        # Routing: source -> app mapping
        self._routes: Dict[str, str] = {}  # source_name -> app_name
        self._default_app: Optional[str] = None

        # State
        self._tick = 0
        self._start_time: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Hooks
        self._on_tick: List[Callable[[TickResult], None]] = []
        self._on_error: List[Callable[[Exception], None]] = []
        self._on_start: List[Callable[[], None]] = []
        self._on_stop: List[Callable[[], None]] = []

        # History
        self._history: List[TickResult] = []
        self._history_limit = 1000

    def register_app(self, app: WooEdgeApp, name: str, default: bool = True) -> None:
        """
        Register an app with the engine.

        Args:
            app: The WooEdge app
            name: Unique name for the app
            default: Set as default app for unrouted observations
        """
        if name in self.apps:
            raise ValueError(f"App '{name}' already registered")

        self.apps[name] = app
        if default or self._default_app is None:
            self._default_app = name

        logger.info(f"Registered app: {name}")

    def unregister_app(self, name: str) -> None:
        """Unregister an app."""
        if name in self.apps:
            del self.apps[name]
            if self._default_app == name:
                self._default_app = next(iter(self.apps), None)

    def register_source(self, source: ObservationSource, route_to: str = None) -> None:
        """
        Register an observation source.

        Args:
            source: The observation source
            route_to: App name to route observations to (uses default if None)
        """
        self.bus.register(source)
        if route_to:
            self._routes[source.name] = route_to

        logger.info(f"Registered source: {source.name} -> {route_to or 'default'}")

    def unregister_source(self, name: str) -> None:
        """Unregister a source."""
        self.bus.unregister(name)
        self._routes.pop(name, None)

    def route(self, source_name: str, app_name: str) -> None:
        """Set routing from source to app."""
        if source_name not in self.bus.sources:
            raise ValueError(f"Unknown source: {source_name}")
        if app_name not in self.apps:
            raise ValueError(f"Unknown app: {app_name}")
        self._routes[source_name] = app_name

    def initialize(self) -> None:
        """
        Initialize all apps and prepare for execution.
        """
        for name, app in self.apps.items():
            try:
                app.initialize()
                logger.info(f"Initialized app: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize app {name}: {e}")
                raise

        self.bus.open_all()
        self.state = EngineState.READY
        logger.info("Engine ready")

    def tick(self) -> List[TickResult]:
        """
        Execute one tick across all apps.

        Returns:
            List of TickResults for each app that received an observation
        """
        if self.state not in (EngineState.READY, EngineState.RUNNING):
            raise RuntimeError(f"Engine not ready: {self.state}")

        self.state = EngineState.RUNNING
        self._tick += 1
        results = []

        # Read observations from all sources
        observations = self.bus.read_all()

        # Route observations to apps
        for obs in observations:
            app_name = self._routes.get(obs.source, self._default_app)
            if app_name is None:
                continue

            app = self.apps.get(app_name)
            if app is None:
                continue

            try:
                gate_result = app.tick(obs)
                result = TickResult(
                    tick=self._tick,
                    timestamp=time.time(),
                    app_name=app_name,
                    observation=obs,
                    gate_result=gate_result,
                )
            except Exception as e:
                logger.error(f"Error in app {app_name}: {e}")
                result = TickResult(
                    tick=self._tick,
                    timestamp=time.time(),
                    app_name=app_name,
                    observation=obs,
                    gate_result=None,
                    error=str(e),
                )
                self._notify_error(e)

                if self.config.stop_on_error:
                    self.state = EngineState.ERROR
                    raise

            results.append(result)
            self._add_history(result)
            self._notify_tick(result)

        return results

    def run(
        self,
        max_ticks: int = None,
        timeout: float = None,
    ) -> Iterator[TickResult]:
        """
        Run the engine, yielding results.

        Args:
            max_ticks: Maximum ticks to run (overrides config)
            timeout: Maximum time to run in seconds

        Yields:
            TickResult for each tick
        """
        if self.state == EngineState.CREATED:
            self.initialize()

        max_ticks = max_ticks or self.config.max_ticks
        self._start_time = time.time()
        self._running = True

        self._notify_start()

        try:
            while self._running:
                # Check limits
                if max_ticks > 0 and self._tick >= max_ticks:
                    break
                if timeout and (time.time() - self._start_time) > timeout:
                    break

                # Execute tick
                results = self.tick()

                # Yield results
                for result in results:
                    yield result

                # Wait for next tick
                if self.config.tick_interval > 0:
                    time.sleep(self.config.tick_interval)

        finally:
            self._running = False
            self._notify_stop()

    def run_once(self, observation: Observation, app_name: str = None) -> TickResult:
        """
        Run a single observation through an app.

        Useful for testing or manual triggering.

        Args:
            observation: The observation to process
            app_name: App to use (uses default if None)

        Returns:
            TickResult
        """
        if self.state == EngineState.CREATED:
            self.initialize()

        app_name = app_name or self._default_app
        if app_name is None:
            raise ValueError("No app registered")

        app = self.apps.get(app_name)
        if app is None:
            raise ValueError(f"Unknown app: {app_name}")

        self._tick += 1

        try:
            gate_result = app.tick(observation)
            result = TickResult(
                tick=self._tick,
                timestamp=time.time(),
                app_name=app_name,
                observation=observation,
                gate_result=gate_result,
            )
        except Exception as e:
            result = TickResult(
                tick=self._tick,
                timestamp=time.time(),
                app_name=app_name,
                observation=observation,
                gate_result=None,
                error=str(e),
            )

        self._add_history(result)
        return result

    def start(self) -> None:
        """
        Start engine in background thread.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Engine already running")

        self._running = True
        self._thread = threading.Thread(target=self._run_background, daemon=True)
        self._thread.start()

        logger.info("Engine started in background")

    def _run_background(self) -> None:
        """Background execution loop."""
        for _ in self.run():
            if not self._running:
                break

    def stop(self) -> None:
        """Stop the engine."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self.state = EngineState.STOPPED
        self.bus.close_all()
        logger.info("Engine stopped")

    def pause(self) -> None:
        """Pause execution."""
        self.state = EngineState.PAUSED
        for app in self.apps.values():
            app.pause()

    def resume(self) -> None:
        """Resume execution."""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING
            for app in self.apps.values():
                app.resume()

    def reset(self) -> None:
        """Reset engine and all apps."""
        self._tick = 0
        self._history.clear()
        for app in self.apps.values():
            app.reset()
        self.state = EngineState.READY

    # Event hooks

    def on_tick(self, handler: Callable[[TickResult], None]) -> None:
        """Register tick handler."""
        self._on_tick.append(handler)

    def on_error(self, handler: Callable[[Exception], None]) -> None:
        """Register error handler."""
        self._on_error.append(handler)

    def on_start(self, handler: Callable[[], None]) -> None:
        """Register start handler."""
        self._on_start.append(handler)

    def on_stop(self, handler: Callable[[], None]) -> None:
        """Register stop handler."""
        self._on_stop.append(handler)

    def _notify_tick(self, result: TickResult) -> None:
        for handler in self._on_tick:
            try:
                handler(result)
            except Exception as e:
                logger.error(f"Tick handler error: {e}")

    def _notify_error(self, error: Exception) -> None:
        for handler in self._on_error:
            try:
                handler(error)
            except Exception:
                pass

    def _notify_start(self) -> None:
        for handler in self._on_start:
            try:
                handler()
            except Exception as e:
                logger.error(f"Start handler error: {e}")

    def _notify_stop(self) -> None:
        for handler in self._on_stop:
            try:
                handler()
            except Exception as e:
                logger.error(f"Stop handler error: {e}")

    def _add_history(self, result: TickResult) -> None:
        self._history.append(result)
        if len(self._history) > self._history_limit:
            self._history.pop(0)

    # Inspection

    @property
    def current_tick(self) -> int:
        """Current tick count."""
        return self._tick

    @property
    def uptime(self) -> float:
        """Seconds since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_history(self, limit: int = 100) -> List[TickResult]:
        """Get recent tick history."""
        return self._history[-limit:]

    def get_app(self, name: str) -> Optional[WooEdgeApp]:
        """Get an app by name."""
        return self.apps.get(name)

    def get_belief(self, app_name: str = None) -> Optional[BeliefState]:
        """Get belief state of an app."""
        app_name = app_name or self._default_app
        app = self.apps.get(app_name) if app_name else None
        return app.get_belief() if app else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state."""
        return {
            "state": self.state.value,
            "tick": self._tick,
            "uptime": self.uptime,
            "apps": {name: app.to_dict() for name, app in self.apps.items()},
            "sources": list(self.bus.sources.keys()),
            "routes": self._routes,
            "default_app": self._default_app,
        }


def create_engine(
    app: WooEdgeApp,
    source: ObservationSource = None,
    config: EngineConfig = None,
) -> WooEdgeEngine:
    """
    Convenience function to create engine with single app.

    Args:
        app: The WooEdge app
        source: Observation source (optional)
        config: Engine config (optional)

    Returns:
        Configured engine ready to run
    """
    engine = WooEdgeEngine(config)
    engine.register_app(app, "main")
    if source:
        engine.register_source(source)
    return engine


def run_simulation(
    app: WooEdgeApp,
    observations: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[TickResult]:
    """
    Run app on a sequence of observations (simulation mode).

    Args:
        app: The WooEdge app
        observations: List of observation dicts
        verbose: Print results

    Returns:
        List of TickResults
    """
    source = MemorySource("simulation", observations)
    engine = create_engine(app, source)
    engine.initialize()

    results = []
    for result in engine.run(max_ticks=len(observations)):
        results.append(result)
        if verbose and result.gate_result:
            print(f"[{result.tick:3d}] {result.gate_result.final_action:15s} "
                  f"| u={result.gate_result.uncertainty:.2f} "
                  f"| h={result.gate_result.hazard:.2f} "
                  f"| {result.gate_result.reason}")

    return results
