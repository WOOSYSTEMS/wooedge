# WOOEdge

**World-model Online Edge Intelligence** — uncertainty-gated MPC for safe action commitment under partial observability.

## Problem

Autonomous systems must act under uncertainty. When belief state is ambiguous, premature commitment to irreversible actions causes failures:
- Robot enters wrong corridor in symmetric environment
- Assistive device commits to hazardous path before localization
- Agent exploits learned policy in wrong belief mode

Standard approaches either ignore uncertainty (greedy) or require expensive inference (LLMs, large nets).

## Approach

WOOEdge uses **entropy-gated commitment**: delay irreversible actions until belief entropy drops below threshold.

```
predict → act → update → improve
   │        │       │        │
   │        │       │        └── Online Dirichlet transition learning
   │        │       └────────── Particle filter belief update
   │        └────────────────── MPC with uncertainty penalty + commit gate
   └─────────────────────────── Belief propagation through learned model
```

**Key mechanism**: When approaching a fork/commit zone, the planner checks `H(belief) > threshold`. If true, it delays commitment and seeks information (vantage points, scanning). Only commits when sufficiently localized.

## Why Not LLMs

| Aspect | WOOEdge | LLM-based |
|--------|---------|-----------|
| Latency | <1ms/step | 100-1000ms |
| Compute | CPU, ~10MB | GPU, GBs |
| Offline | Yes | Requires API |
| Interpretable | Full transparency | Black box |
| Guarantees | Entropy threshold | None |

WOOEdge is **NumPy-only**, runs on edge devices, and provides formal uncertainty quantification.

## Quick Start

```bash
pip install numpy pytest

# Flagship demo: uncertainty-gated MPC vs baseline
python -m wooedge.cli demo_proof
# → MPC(full)=90% vs MPC(no_uncert)=70%, separation +20pp, <60s

# Assistive navigation: safety gating prevents hazard commitment
python -m wooedge.cli assistive_demo
# → Gated: 100% success, 0% hazard vs Ungated: 0%, 100%

# Run tests
pytest -q  # 92 tests
```

## DecisionSafety API

Reusable module for gating irreversible actions:

```python
from wooedge.safety import DecisionSafety

safety = DecisionSafety(env)
safety.reset(seed=42)
safety.observe(obs)

result = safety.propose(action)
# → {"decision": "ALLOW"|"DELAY"|"ABORT", "entropy": 0.42, "reason": "..."}
```

## Architecture

```
wooedge/
├── env.py        # Partially observable gridworld (symmetric fork-trap maze)
├── belief.py     # Particle filter, entropy computation
├── model.py      # Online transition model (Dirichlet counts)
├── planner.py    # MPC with CEM, commit-gate penalty
├── safety.py     # DecisionSafety module
├── agent.py      # Full agent combining components
└── envs/         # Additional environments (assistive_nav)
```

## Current Limitations

- **Discrete state/action**: Gridworld only, no continuous control
- **Tabular model**: Doesn't scale beyond ~1000 states
- **Known observation model**: Assumes sensor noise distribution known
- **Single agent**: No multi-agent coordination
- **Fixed horizon**: MPC horizon is static, not adaptive

## Targets Verified

| Benchmark | Target | Achieved |
|-----------|--------|----------|
| `demo_proof` | MPC ≥90%, sep ≥+10pp | 90%, +20pp |
| `assistive_demo` | Gated safer | +100pp |
| All tests | Pass | 92/92 |
| Runtime | <60s | ~45s |

## References

- Particle filters: Thrun, Burgard, Fox. *Probabilistic Robotics*. 2005.
- MPC under uncertainty: Rawlings, Mayne. *Model Predictive Control*. 2009.
- Safe RL: Garcıa, Fernández. *A Comprehensive Survey on Safe RL*. 2015.

## License

MIT
