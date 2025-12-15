"""
Tests for WooEdge Runtime.

Tests the core runtime components:
- BeliefState
- EntropyTracker
- ActionGate
- ObservationBus
- WooEdgeApp
- WooEdgeEngine
"""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.runtime import (
    # Belief
    BeliefState,
    GaussianLikelihood,
    # Entropy
    EntropyTracker,
    UncertaintyLevel,
    compute_entropy,
    compute_max_entropy,
    # Gate
    ActionGate,
    GateDecision,
    ActionCategory,
    ConservativePolicy,
    # Bus
    ObservationBus,
    Observation,
    MemorySource,
    CallbackSource,
    SchemaValidator,
    # App
    WooEdgeApp,
    SimpleApp,
    AppConfig,
    AppContext,
    # Engine
    WooEdgeEngine,
    EngineConfig,
    create_engine,
    run_simulation,
)


class TestBeliefState:
    """Tests for BeliefState."""

    def test_uniform_creation(self):
        """Test creating uniform belief."""
        belief = BeliefState.uniform(["a", "b", "c"])
        assert len(belief.states) == 3
        assert belief.prob("a") == pytest.approx(1/3)
        assert belief.prob("b") == pytest.approx(1/3)
        assert belief.prob("c") == pytest.approx(1/3)

    def test_from_priors(self):
        """Test creating belief from priors."""
        belief = BeliefState.from_priors({"a": 2, "b": 1, "c": 1})
        assert belief.prob("a") == pytest.approx(0.5)
        assert belief.prob("b") == pytest.approx(0.25)

    def test_most_likely(self):
        """Test most likely state."""
        belief = BeliefState.from_priors({"good": 0.7, "bad": 0.3})
        assert belief.most_likely == "good"
        assert belief.most_likely_prob == pytest.approx(0.7)

    def test_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        belief = BeliefState.uniform(["a", "b", "c", "d"])
        assert belief.entropy() == pytest.approx(math.log(4))
        assert belief.normalized_entropy() == pytest.approx(1.0)

    def test_entropy_certain(self):
        """Test entropy of certain distribution."""
        belief = BeliefState.from_priors({"a": 1, "b": 0, "c": 0})
        assert belief.entropy() == pytest.approx(0.0)
        assert belief.normalized_entropy() == pytest.approx(0.0)

    def test_bayesian_update(self):
        """Test Bayesian belief update."""
        belief = BeliefState.uniform(["sunny", "cloudy", "rainy"])

        # Observation strongly suggests sunny
        likelihoods = {"sunny": 0.9, "cloudy": 0.1, "rainy": 0.05}
        belief.update_bayesian(likelihoods)

        assert belief.most_likely == "sunny"
        assert belief.prob("sunny") > 0.5

    def test_blend(self):
        """Test blending distributions."""
        belief = BeliefState.from_priors({"a": 0.8, "b": 0.2})
        belief.blend({"a": 0.2, "b": 0.8}, weight=0.5)

        # Should be roughly 50-50 after blending
        assert abs(belief.prob("a") - 0.5) < 0.1

    def test_decay_to_uniform(self):
        """Test decay toward uniform."""
        belief = BeliefState.from_priors({"a": 1.0, "b": 0.0})
        original_prob = belief.prob("a")

        belief.decay_to_uniform(rate=0.1)

        # Should have moved toward 0.5
        assert belief.prob("a") < original_prob
        assert belief.prob("b") > 0

    def test_history(self):
        """Test belief history tracking."""
        belief = BeliefState.uniform(["a", "b"])

        for i in range(5):
            belief.update_bayesian({"a": 0.6, "b": 0.4})

        assert len(belief.history) == 5
        assert belief.update_count == 5


class TestEntropyTracker:
    """Tests for EntropyTracker."""

    def test_track_entropy(self):
        """Test tracking entropy."""
        tracker = EntropyTracker()

        snapshot = tracker.track(entropy=0.5, max_entropy=1.0)

        assert snapshot.normalized == pytest.approx(0.5)
        assert tracker.current is not None

    def test_uncertainty_levels(self):
        """Test categorical uncertainty levels."""
        tracker = EntropyTracker()

        # Certain
        tracker.track(entropy=0.1, max_entropy=1.0)
        assert tracker.current_level == UncertaintyLevel.CERTAIN

        # Very uncertain
        tracker.track(entropy=0.9, max_entropy=1.0)
        assert tracker.current_level == UncertaintyLevel.VERY_UNCERTAIN

    def test_spike_detection(self):
        """Test entropy spike detection."""
        tracker = EntropyTracker(spike_threshold=0.2)

        tracker.track(entropy=0.3, max_entropy=1.0)
        tracker.track(entropy=0.8, max_entropy=1.0)  # Big jump

        assert tracker.is_spike()

    def test_trend_detection(self):
        """Test entropy trend detection."""
        tracker = EntropyTracker()

        # Decreasing entropy (converging)
        for h in [0.9, 0.7, 0.5, 0.3, 0.1]:
            tracker.track(entropy=h, max_entropy=1.0)

        assert tracker.is_converging()


class TestActionGate:
    """Tests for ActionGate."""

    def test_allow_low_uncertainty(self):
        """Test allowing action at low uncertainty."""
        gate = ActionGate(uncertainty_threshold=0.7, hazard_threshold=0.6)

        result = gate.evaluate("buy", uncertainty=0.3, hazard=0.2)

        assert result.allowed
        assert result.decision in (GateDecision.ALLOW, GateDecision.MODIFY)

    def test_block_high_hazard(self):
        """Test blocking action at high hazard."""
        gate = ActionGate(hazard_threshold=0.5)

        result = gate.evaluate("buy", uncertainty=0.3, hazard=0.8)

        assert not result.allowed
        assert result.decision == GateDecision.BLOCK
        assert "HAZARD" in result.reason

    def test_delay_high_uncertainty(self):
        """Test delaying action at high uncertainty."""
        gate = ActionGate(uncertainty_threshold=0.5)

        result = gate.evaluate("buy", uncertainty=0.8, hazard=0.2)

        assert not result.allowed
        assert result.decision == GateDecision.DELAY
        assert "UNCERTAINTY" in result.reason

    def test_observe_always_allowed(self):
        """Test that observe actions are always allowed."""
        gate = ActionGate()
        gate.register_action("scan", ActionCategory.OBSERVE)

        result = gate.evaluate("scan", uncertainty=0.99, hazard=0.99)

        assert result.allowed

    def test_size_scaling(self):
        """Test position size scaling with uncertainty."""
        gate = ActionGate(
            uncertainty_threshold=0.8,
            min_size_multiplier=0.25,
            size_scale_with_confidence=True,
        )

        # Low uncertainty -> high size
        result_low = gate.evaluate("buy", uncertainty=0.1, hazard=0.1)
        # Higher uncertainty -> lower size
        result_high = gate.evaluate("buy", uncertainty=0.6, hazard=0.1)

        assert result_low.modifiers.get("size_multiplier", 1.0) > result_high.modifiers.get("size_multiplier", 1.0)


class TestObservationBus:
    """Tests for ObservationBus."""

    def test_memory_source(self):
        """Test MemorySource."""
        source = MemorySource("test", [{"value": 1}, {"value": 2}])

        obs1 = source.read()
        obs2 = source.read()
        obs3 = source.read()

        assert obs1.get("value") == 1
        assert obs2.get("value") == 2
        assert obs3 is None

    def test_callback_source(self):
        """Test CallbackSource."""
        source = CallbackSource("test")

        source.push({"value": 42})
        obs = source.read()

        assert obs.get("value") == 42

    def test_bus_multiple_sources(self):
        """Test bus with multiple sources."""
        bus = ObservationBus()
        bus.register(MemorySource("s1", [{"a": 1}]))
        bus.register(MemorySource("s2", [{"b": 2}]))

        observations = bus.read_all()

        assert len(observations) == 2

    def test_schema_validator(self):
        """Test schema validation."""
        validator = SchemaValidator({"temp": float, "humid": float})

        obs_valid = Observation(data={"temp": 25.0, "humid": 60.0})
        obs_invalid = Observation(data={"temp": 25.0})  # Missing humid

        assert validator.validate(obs_valid)
        assert not validator.validate(obs_invalid)


class TestWooEdgeApp:
    """Tests for WooEdgeApp."""

    def test_simple_app(self):
        """Test SimpleApp."""
        app = SimpleApp(
            states=["on", "off"],
            actions=["turn_on", "turn_off", "hold"],
            schema={"switch": bool},
            likelihood_fn=lambda obs, ctx: {
                "on": 0.9 if obs.get("switch") else 0.1,
                "off": 0.1 if obs.get("switch") else 0.9,
            },
            decide_fn=lambda ctx: "turn_on" if ctx.most_likely_state == "off" else "hold",
        )
        app.initialize()

        # Switch is off
        obs = Observation(data={"switch": False})
        result = app.tick(obs)

        assert result.final_action == "turn_on"

    def test_app_gating(self):
        """Test that app actions are gated."""
        app = SimpleApp(
            states=["safe", "unsafe"],
            actions=["go", "stop"],
            schema={"danger": float},
            likelihood_fn=lambda obs, ctx: {
                "safe": 0.9 if obs.get("danger", 0) < 0.5 else 0.1,
                "unsafe": 0.1 if obs.get("danger", 0) < 0.5 else 0.9,
            },
            decide_fn=lambda ctx: "go",  # Always wants to go
            hazard_fn=lambda obs, ctx: obs.get("danger", 0),
            config=AppConfig(hazard_threshold=0.5),
        )
        app.initialize()

        # High danger should block
        obs = Observation(data={"danger": 0.8})
        result = app.tick(obs)

        assert not result.allowed
        assert result.final_action != "go"

    def test_app_belief_updates(self):
        """Test that app belief updates correctly."""
        app = SimpleApp(
            states=["good", "bad"],
            actions=["act", "wait"],
            schema={"signal": float},
            likelihood_fn=lambda obs, ctx: {
                "good": 0.9 if obs.get("signal", 0) > 0 else 0.1,
                "bad": 0.1 if obs.get("signal", 0) > 0 else 0.9,
            },
            decide_fn=lambda ctx: "act",
        )
        app.initialize()

        # Initial belief is uniform
        assert app.get_belief().normalized_entropy() == pytest.approx(1.0)

        # Process good signals
        for _ in range(5):
            app.tick(Observation(data={"signal": 1.0}))

        # Belief should converge
        assert app.get_belief().most_likely == "good"
        assert app.get_belief().normalized_entropy() < 0.5


class TestWooEdgeEngine:
    """Tests for WooEdgeEngine."""

    def test_basic_run(self):
        """Test basic engine run."""
        app = SimpleApp(
            states=["a", "b"],
            actions=["x", "y"],
            schema={"v": float},
            likelihood_fn=lambda obs, ctx: {"a": 0.7, "b": 0.3},
            decide_fn=lambda ctx: "x",
        )

        source = MemorySource("test", [{"v": 1}, {"v": 2}, {"v": 3}])
        engine = create_engine(app, source)

        results = list(engine.run(max_ticks=3))

        assert len(results) == 3
        assert all(r.gate_result is not None for r in results)

    def test_run_simulation(self):
        """Test run_simulation helper."""
        app = SimpleApp(
            states=["up", "down"],
            actions=["buy", "sell", "hold"],
            schema={"price": float},
            likelihood_fn=lambda obs, ctx: {
                "up": 0.8 if obs.get("price", 0) > 100 else 0.2,
                "down": 0.2 if obs.get("price", 0) > 100 else 0.8,
            },
            decide_fn=lambda ctx: "buy" if ctx.most_likely_state == "up" else "hold",
        )

        observations = [
            {"price": 105},
            {"price": 110},
            {"price": 95},
        ]

        results = run_simulation(app, observations)

        assert len(results) == 3

    def test_engine_multiple_apps(self):
        """Test engine with multiple apps."""
        app1 = SimpleApp(
            states=["a", "b"],
            actions=["x"],
            schema={"v": float},
            likelihood_fn=lambda obs, ctx: {"a": 0.5, "b": 0.5},
            decide_fn=lambda ctx: "x",
        )
        app2 = SimpleApp(
            states=["c", "d"],
            actions=["y"],
            schema={"v": float},
            likelihood_fn=lambda obs, ctx: {"c": 0.5, "d": 0.5},
            decide_fn=lambda ctx: "y",
        )

        engine = WooEdgeEngine()
        engine.register_app(app1, "app1")
        engine.register_app(app2, "app2")

        s1 = MemorySource("s1", [{"v": 1}])
        s2 = MemorySource("s2", [{"v": 2}])

        engine.register_source(s1, route_to="app1")
        engine.register_source(s2, route_to="app2")

        results = list(engine.run(max_ticks=1))

        assert len(results) == 2
        app_names = {r.app_name for r in results}
        assert "app1" in app_names
        assert "app2" in app_names

    def test_engine_run_once(self):
        """Test single observation processing."""
        app = SimpleApp(
            states=["s"],
            actions=["a"],
            schema={},
            likelihood_fn=lambda obs, ctx: {"s": 1.0},
            decide_fn=lambda ctx: "a",
        )

        engine = WooEdgeEngine()
        engine.register_app(app, "main")

        obs = Observation(data={"test": 123})
        result = engine.run_once(obs)

        assert result.tick == 1
        assert result.gate_result is not None


class TestIntegration:
    """Integration tests for full runtime stack."""

    def test_thermostat_scenario(self):
        """Test a thermostat-like scenario."""
        app = SimpleApp(
            states=["comfortable", "too_cold", "too_hot"],
            actions=["heat_on", "heat_off", "cool_on", "cool_off", "hold"],
            schema={"temperature": float},
            likelihood_fn=lambda obs, ctx: {
                "comfortable": 1.0 if 68 <= obs.get("temperature", 70) <= 74 else 0.1,
                "too_cold": 1.0 if obs.get("temperature", 70) < 68 else 0.05,
                "too_hot": 1.0 if obs.get("temperature", 70) > 74 else 0.05,
            },
            decide_fn=lambda ctx: (
                "heat_on" if ctx.most_likely_state == "too_cold"
                else "cool_on" if ctx.most_likely_state == "too_hot"
                else "hold"
            ),
            config=AppConfig(decay_rate=0.0),  # No decay for predictable behavior
        )

        # Simulate sustained cold readings
        cold_obs = [{"temperature": 65}] * 5
        results_cold = run_simulation(app, cold_obs)

        # Should consistently trigger heat_on
        heat_actions = [r.gate_result.final_action for r in results_cold]
        assert all(a == "heat_on" for a in heat_actions), f"Expected all heat_on, got {heat_actions}"

        # Fresh app for hot scenario
        app2 = SimpleApp(
            states=["comfortable", "too_cold", "too_hot"],
            actions=["heat_on", "heat_off", "cool_on", "cool_off", "hold"],
            schema={"temperature": float},
            likelihood_fn=lambda obs, ctx: {
                "comfortable": 1.0 if 68 <= obs.get("temperature", 70) <= 74 else 0.1,
                "too_cold": 1.0 if obs.get("temperature", 70) < 68 else 0.05,
                "too_hot": 1.0 if obs.get("temperature", 70) > 74 else 0.05,
            },
            decide_fn=lambda ctx: (
                "heat_on" if ctx.most_likely_state == "too_cold"
                else "cool_on" if ctx.most_likely_state == "too_hot"
                else "hold"
            ),
            config=AppConfig(decay_rate=0.0),
        )

        # Simulate sustained hot readings
        hot_obs = [{"temperature": 80}] * 5
        results_hot = run_simulation(app2, hot_obs)

        # Should consistently trigger cool_on
        cool_actions = [r.gate_result.final_action for r in results_hot]
        assert all(a == "cool_on" for a in cool_actions), f"Expected all cool_on, got {cool_actions}"

    def test_uncertainty_gating_scenario(self):
        """Test that high uncertainty gates risky actions."""
        # App that always wants to take action
        app = SimpleApp(
            states=["good", "bad", "unknown"],
            actions=["risky_action", "safe_action", "wait"],
            schema={"signal": float},
            likelihood_fn=lambda obs, ctx: {
                # Ambiguous signal -> high uncertainty
                "good": 0.4 if abs(obs.get("signal", 0)) < 0.1 else 0.8,
                "bad": 0.3 if abs(obs.get("signal", 0)) < 0.1 else 0.1,
                "unknown": 0.3 if abs(obs.get("signal", 0)) < 0.1 else 0.1,
            },
            decide_fn=lambda ctx: "risky_action",  # Always wants risky
            config=AppConfig(uncertainty_threshold=0.5),
        )
        app.action_categories = {"risky_action": ActionCategory.IRREVERSIBLE}

        # Ambiguous signals should gate
        observations = [
            {"signal": 0.05},  # Ambiguous
            {"signal": 0.03},  # Still ambiguous
            {"signal": 1.0},   # Clear signal
            {"signal": 1.0},   # Clear signal
            {"signal": 1.0},   # Clear signal
        ]

        results = run_simulation(app, observations)

        # Early ambiguous signals should be gated
        # After clear signals, action should be allowed
        allowed_count = sum(1 for r in results if r.gate_result.allowed)
        blocked_count = sum(1 for r in results if not r.gate_result.allowed)

        # Some should be blocked due to uncertainty
        assert blocked_count > 0
