# WOOEdge - World-model Online Edge Intelligence

A predictive world-model agent that learns environment dynamics online and chooses actions to reach goals under uncertainty. **No LLMs, no transformer models, no large pretrained models, no external APIs** - pure algorithmic intelligence.

## Overview

WOOEdge demonstrates "life-like intelligence" through the **predict → act → update → improve** cycle:

1. **Predict**: Maintain belief over position using particle filter
2. **Act**: Plan actions using MPC with learned dynamics model
3. **Update**: Refine belief from observations
4. **Improve**: Learn environment dynamics online

## Proof in One Command

```bash
python -m wooedge.cli demo_proof
```

- **What it demonstrates**: MPC with uncertainty-gating achieves 90% success vs 70% without, on a fork-trap maze where premature commitment is fatal
- **Why it matters**: The agent learns to delay commitment when belief entropy is high, avoiding wrong-fork traps that fool greedy planners
- **Why it's lightweight**: No LLMs, no GPUs, no APIs — runs locally in <60s with pure NumPy

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WOOEdge Agent                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│   │  Particle     │    │  Transition   │    │  MPC          │  │
│   │  Filter       │    │  Model        │    │  Planner      │  │
│   │  (Belief)     │    │  (Learning)   │    │  (Planning)   │  │
│   └───────┬───────┘    └───────┬───────┘    └───────┬───────┘  │
│           │                    │                    │          │
│           └────────────┬───────┴────────────────────┘          │
│                        │                                        │
│                        ▼                                        │
│              ┌─────────────────┐                               │
│              │  Agent Core     │                               │
│              │  predict → act  │                               │
│              │  → update →     │                               │
│              │  improve        │                               │
│              └────────┬────────┘                               │
│                       │                                         │
└───────────────────────┼─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Partially Observable GridWorld                     │
│   • Hidden true state (agent position)                         │
│   • Noisy local 3x3 patch observations                         │
│   • Noisy distance sensors                                      │
│   • Stochastic transitions (action slip)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Environment (`env.py`)

A 12x12 partially observable gridworld:
- **Walls**: Random obstacles + border walls
- **Observations**: Local 3x3 patch + 4-directional distance sensors
- **Noise**: Observation flips + Gaussian sensor noise
- **Transitions**: Actions slip to random direction with probability 0.1

### Belief State (`belief.py`)

Particle filter maintaining probabilistic belief over position:
- **Particles**: 500 weighted samples over possible positions
- **Prediction**: Propagate particles through motion model
- **Update**: Reweight based on observation likelihood
- **Resampling**: Systematic resampling when ESS is low
- **Entropy**: Quantifies uncertainty for planning

### Online Learning (`model.py`)

Learns environment dynamics without prior knowledge:
- **Transition Model**: P(s'|s,a) via Dirichlet-smoothed counts
- **Map Learning**: Bayesian wall belief from observations
- **Visit Counts**: UCB-style exploration bonus
- **Uncertainty**: Inverse count-based uncertainty measure

### Planning (`planner.py`)

Model Predictive Control with uncertainty awareness:
- **Random Shooting**: Sample action sequences
- **CEM Refinement**: Cross-entropy method for elite selection
- **Cost Function**: `J = goal_dist + action_cost - exploration - uncertainty`
- **Horizon**: 10-step lookahead
- **Belief Integration**: Average cost over belief samples

## Installation

```bash
cd wooedge

# Install dependencies (minimal)
pip install numpy

# Optional: matplotlib for visualization
pip install matplotlib

# For running tests
pip install pytest
```

## Usage

### Run Single Episode

```bash
# ASCII visualization (default)
python -m wooedge.cli run --seed 0 --steps 200 --render ascii

# No visualization (fast)
python -m wooedge.cli run --seed 0 --steps 200 --render none --verbose

# With matplotlib (if installed)
python -m wooedge.cli run --seed 0 --steps 200 --render matplotlib

# Different planner
python -m wooedge.cli run --planner greedy --steps 200
python -m wooedge.cli run --planner infogain --steps 200
```

### Run Benchmark

```bash
# Compare planners across multiple seeds
python -m wooedge.cli benchmark --seeds 0 1 2 3 4

# Customize benchmark
python -m wooedge.cli benchmark --seeds 0 1 2 --episodes 10 --steps 300
```

### Quick Demo

```bash
python -m wooedge.cli demo
```

### Detailed Statistics

```bash
python -m wooedge.cli stats --seed 42 --planner mpc
```

## How It Works

### Belief Update (Particle Filter)

The agent doesn't know its true position. Instead, it maintains a probability distribution:

1. **Initialization**: Particles spread uniformly (or near start)
2. **Predict**: Move particles according to action + slip probability
3. **Update**: Weight particles by observation likelihood
   - Distance sensor likelihood: Gaussian around expected reading
   - Patch likelihood: Product of Bernoulli (observation noise model)
4. **Resample**: When effective sample size drops below threshold

```
Belief Entropy = -Σ p(s) log p(s)
```

Higher entropy = more uncertainty about position.

### Online Dynamics Learning

The agent learns the environment as it explores:

1. **Transition Counts**: `counts[s, a, s']` incremented on each observation
2. **Dirichlet Smoothing**: `P(s'|s,a) = (count + α) / (total + n*α)`
3. **Wall Belief**: Bayesian update from observation patches
4. **Exploration Bonus**: `1/√(visits + 1)` for unvisited states

### MPC Planning

Planning trades off multiple objectives:

```
J(action_sequence) = Σ γ^t [
    goal_weight × distance_to_goal
  + action_cost × (1 if move else 0.5)
  - exploration_weight × exploration_bonus
  - uncertainty_weight × model_uncertainty
]
```

Algorithm:
1. Sample starting positions from belief
2. Generate random action sequences
3. Evaluate by rollout using learned model
4. CEM: Keep top-k, update sampling distribution
5. Return first action of best sequence

### What Makes This "Edge Intelligence"

- **No neural networks**: Pure algorithmic computation
- **No pretrained models**: Learns from scratch each episode
- **No external APIs**: Everything runs locally
- **Minimal memory**: ~500 particles, tabular counts
- **Fast**: Runs on any laptop in real-time
- **Interpretable**: Every component is transparent

## Comparison: MPC vs Greedy

| Metric | MPC (full) | MPC (no uncertainty) | Greedy |
|--------|------------|---------------------|--------|
| Success Rate | Higher | Medium | Lower |
| Steps to Goal | Fewer | Medium | More |
| Exploration | Balanced | Low | None |
| Computation | Higher | Medium | Minimal |

The MPC planner with uncertainty terms:
- Explores more efficiently
- Recovers better from wrong beliefs
- Handles noisy observations better

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_env.py -v

# Run with coverage
pytest tests/ --cov=wooedge
```

## Project Structure

```
wooedge/
├── wooedge/
│   ├── __init__.py      # Package initialization
│   ├── __main__.py      # Module entry point
│   ├── env.py           # Gridworld environment
│   ├── belief.py        # Particle filter
│   ├── model.py         # Online transition learning
│   ├── planner.py       # MPC and baseline planners
│   ├── agent.py         # Main agent combining components
│   ├── viz.py           # ASCII and matplotlib visualization
│   └── cli.py           # Command line interface
├── tests/
│   ├── test_env.py
│   ├── test_belief.py
│   ├── test_model.py
│   ├── test_planner.py
│   └── test_agent.py
└── README.md
```

## CLI Options

### `run` command
- `--seed`: Random seed (default: 0)
- `--steps`: Maximum steps (default: 200)
- `--render`: `ascii`, `matplotlib`, or `none`
- `--planner`: `mpc`, `greedy`, or `infogain`
- `--grid-size`: Grid size (default: 12)
- `--wall-density`: Wall density (default: 0.15)
- `--particles`: Number of particles (default: 500)
- `--horizon`: Planning horizon (default: 10)
- `--samples`: Rollout samples (default: 100)
- `--delay`: Animation delay in seconds (default: 0.1)

### `benchmark` command
- `--seeds`: List of seeds to test
- `--episodes`: Episodes per configuration (default: 5)
- `--steps`: Maximum steps (default: 200)

## Extending WOOEdge

### Custom Environment

```python
from wooedge.env import GridWorld, EnvConfig

config = EnvConfig(
    grid_size=20,
    wall_density=0.2,
    obs_noise_prob=0.15,
    slip_prob=0.15,
    seed=123
)
env = GridWorld(config)
```

### Custom Agent Configuration

```python
from wooedge.agent import WOOEdgeAgent, AgentConfig

config = AgentConfig(
    n_particles=1000,
    planner_type="mpc",
    planning_horizon=15,
    uncertainty_weight=0.5,
    exploration_weight=0.3
)
agent = WOOEdgeAgent(config)
```

### Programmatic Usage

```python
from wooedge.env import GridWorld, EnvConfig
from wooedge.agent import WOOEdgeAgent, AgentConfig, run_episode

# Create environment
env = GridWorld(EnvConfig(seed=42))

# Create agent
agent = WOOEdgeAgent(AgentConfig(seed=42))

# Run episode
results = run_episode(env, agent, max_steps=200, verbose=True)

print(f"Success: {results['success']}")
print(f"Steps: {results['steps']}")
print(f"Final entropy: {results['final_entropy']:.4f}")
```

### DecisionSafety API

A reusable module for checking whether actions are safe to commit based on belief uncertainty:

```python
from wooedge.env import GridWorld, EnvConfig
from wooedge.safety import DecisionSafety

env = GridWorld(EnvConfig(maze_type="symmetric_fork_trap", seed=42))
safety = DecisionSafety(env)
safety.reset(seed=42)

obs = env.reset()
safety.observe(obs)

result = safety.propose(action=1)  # DOWN
if result["decision"] == "DELAY":
    print(f"Unsafe: {result['reason']}")
```

Returns `{"decision": "ALLOW"|"DELAY"|"ABORT", "entropy": float, "reason": str, ...}`.

See `examples/safety_demo.py` for a complete example.

## Dependencies

**Required:**
- `numpy` (array operations, probability)

**Optional:**
- `matplotlib` (graphical visualization)
- `pytest` (running tests)

## License

MIT License - Feel free to use, modify, and distribute.

## Contributing

Contributions welcome! Areas for improvement:
- More sophisticated observation models
- Different belief representations (Kalman filter, etc.)
- Additional planning algorithms (MCTS, etc.)
- Multi-agent scenarios
- Continuous state spaces
