#!/usr/bin/env python3
"""Quick test script for WooEdge runtime."""

from wooedge.runtime import SimpleApp, run_simulation, AppConfig

print("WooEdge Runtime Quick Test")
print("=" * 50)

# Test 1: Basic app
print("\n1. Basic app - price tracker")
app = SimpleApp(
    states=["bullish", "bearish"],
    actions=["buy", "sell", "hold"],
    schema={"price": float},
    likelihood_fn=lambda obs, ctx: {
        "bullish": 0.9 if obs.get("price", 0) > 100 else 0.1,
        "bearish": 0.1 if obs.get("price", 0) > 100 else 0.9,
    },
    decide_fn=lambda ctx: "buy" if ctx.most_likely_state == "bullish" else "hold",
)

results = run_simulation(app, [
    {"price": 105},
    {"price": 110},
    {"price": 95},
], verbose=True)

print(f"\nFinal belief: {app.get_belief()}")

# Test 2: Uncertainty gating
print("\n" + "=" * 50)
print("2. Uncertainty gating - ambiguous signals")

app2 = SimpleApp(
    states=["safe", "danger"],
    actions=["go", "stop", "scan"],
    schema={"signal": float},
    likelihood_fn=lambda obs, ctx: {
        # Ambiguous signal near 0.5
        "safe": 0.6 if obs.get("signal", 0.5) > 0.5 else 0.4,
        "danger": 0.4 if obs.get("signal", 0.5) > 0.5 else 0.6,
    },
    decide_fn=lambda ctx: "go",  # Always wants to go
    config=AppConfig(uncertainty_threshold=0.5),
)

results2 = run_simulation(app2, [
    {"signal": 0.51},  # Barely safe
    {"signal": 0.49},  # Barely danger
    {"signal": 0.52},  # Flip-flopping
    {"signal": 0.48},
], verbose=True)

# Test 3: Hazard blocking
print("\n" + "=" * 50)
print("3. Hazard blocking - dangerous conditions")

app3 = SimpleApp(
    states=["normal", "alert"],
    actions=["proceed", "halt"],
    schema={"threat_level": float},
    likelihood_fn=lambda obs, ctx: {
        "normal": 0.9 if obs.get("threat_level", 0) < 0.5 else 0.1,
        "alert": 0.1 if obs.get("threat_level", 0) < 0.5 else 0.9,
    },
    decide_fn=lambda ctx: "proceed",  # Always wants to proceed
    hazard_fn=lambda obs, ctx: obs.get("threat_level", 0),  # Hazard = threat level
    config=AppConfig(hazard_threshold=0.6),
)

results3 = run_simulation(app3, [
    {"threat_level": 0.2},  # Safe
    {"threat_level": 0.4},  # Still safe
    {"threat_level": 0.8},  # DANGER - should block
    {"threat_level": 0.9},  # DANGER - should block
], verbose=True)

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)
