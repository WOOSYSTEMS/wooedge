"""
WooEdge Runtime

Universal runtime for uncertainty-aware applications.
Like JVM abstracts hardware, WooEdge abstracts uncertainty-aware decision making.

Core Components:
- BeliefState: Probability distribution over world states
- EntropyTracker: Uncertainty quantification
- ActionGate: Universal action gating
- ObservationBus: Universal observation ingestion
- WooEdgeApp: Base class for applications
- WooEdgeEngine: Runtime orchestrator

Example:
    from wooedge.runtime import WooEdgeApp, WooEdgeEngine, MemorySource, Observation

    class MyApp(WooEdgeApp[str]):
        world_states = ["good", "bad"]
        action_space = ["act", "wait"]
        observation_schema = {"value": float}

        def compute_likelihoods(self, obs, ctx):
            v = obs.get("value", 0)
            return {"good": 1.0 if v > 0 else 0.1, "bad": 0.1 if v > 0 else 1.0}

        def decide(self, ctx):
            return "act" if ctx.most_likely_state == "good" else "wait"

    # Run
    app = MyApp()
    source = MemorySource("test", [{"value": 1.0}, {"value": -1.0}])
    engine = WooEdgeEngine()
    engine.register_app(app, "main")
    engine.register_source(source)

    for result in engine.run(max_ticks=2):
        print(result.gate_result.final_action)
"""

# Belief management
from .belief import (
    BeliefState,
    LikelihoodModel,
    GaussianLikelihood,
)

# Entropy tracking
from .entropy import (
    EntropyTracker,
    EntropySnapshot,
    UncertaintyLevel,
    compute_entropy,
    compute_max_entropy,
    compute_kl_divergence,
    compute_js_divergence,
)

# Action gating
from .gate import (
    ActionGate,
    GateResult,
    GateDecision,
    GateRule,
    ActionCategory,
    GatePolicy,
    ConservativePolicy,
    AdaptivePolicy,
)

# Observation bus
from .bus import (
    ObservationBus,
    Observation,
    ObservationSource,
    MemorySource,
    CallbackSource,
    TransformSource,
    SourceType,
    SchemaValidator,
)

# Application interface
from .app import (
    WooEdgeApp,
    SimpleApp,
    AppConfig,
    AppContext,
    AppState,
)

# Runtime engine
from .engine import (
    WooEdgeEngine,
    EngineConfig,
    EngineState,
    TickResult,
    create_engine,
    run_simulation,
)


__all__ = [
    # Belief
    "BeliefState",
    "LikelihoodModel",
    "GaussianLikelihood",
    # Entropy
    "EntropyTracker",
    "EntropySnapshot",
    "UncertaintyLevel",
    "compute_entropy",
    "compute_max_entropy",
    "compute_kl_divergence",
    "compute_js_divergence",
    # Gate
    "ActionGate",
    "GateResult",
    "GateDecision",
    "GateRule",
    "ActionCategory",
    "GatePolicy",
    "ConservativePolicy",
    "AdaptivePolicy",
    # Bus
    "ObservationBus",
    "Observation",
    "ObservationSource",
    "MemorySource",
    "CallbackSource",
    "TransformSource",
    "SourceType",
    "SchemaValidator",
    # App
    "WooEdgeApp",
    "SimpleApp",
    "AppConfig",
    "AppContext",
    "AppState",
    # Engine
    "WooEdgeEngine",
    "EngineConfig",
    "EngineState",
    "TickResult",
    "create_engine",
    "run_simulation",
]
