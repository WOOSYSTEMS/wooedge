"""
Command Line Interface for WOOEdge

Provides commands for running experiments, benchmarks, and visualizations.
Enhanced benchmark shows separation between uncertainty-aware and greedy planners.
"""

import argparse
import sys
import time
from typing import List, Optional

import numpy as np

from .env import GridWorld, EnvConfig
from .agent import WOOEdgeAgent, AgentConfig, run_episode, compare_planners
from .viz import (
    ASCIIRenderer, MatplotlibRenderer, animate_episode,
    print_statistics, print_benchmark_results
)


def run_single_episode(args) -> None:
    """Run a single episode with visualization."""
    print(f"\n{'='*60}")
    print("WOOEdge - World-model Online Edge Intelligence")
    print(f"{'='*60}")
    print(f"Running episode with seed={args.seed}, steps={args.steps}")
    print(f"Planner: {args.planner}, Render: {args.render}")
    print(f"Maze: {args.maze_type}, Sensor noise: {args.sensor_noise}")
    print(f"{'='*60}\n")

    # Create environment
    env_config = EnvConfig(
        grid_size=args.grid_size,
        maze_type=args.maze_type,
        sensor_noise_prob=args.sensor_noise,
        slip_prob=args.slip_prob,
        trap_corridors=args.trap_corridors,
        seed=args.seed
    )
    env = GridWorld(env_config)

    # Print ambiguity stats
    amb_stats = env.get_ambiguity_stats()
    print(f"Maze ambiguity: {amb_stats['mean_ambiguity']:.1f} positions per reading")
    print(f"Lookout positions: {amb_stats['lookout_count']} / {amb_stats['total_positions']}")
    print()

    # Create agent
    agent_config = AgentConfig(
        n_particles=args.particles,
        planner_type=args.planner,
        planning_horizon=args.horizon,
        n_samples=args.samples,
        goal_weight=args.goal_weight,
        uncertainty_weight=args.uncertainty_weight,
        exploration_weight=args.exploration_weight,
        uniform_init=True,
        seed=args.seed
    )
    agent = WOOEdgeAgent(agent_config)

    if args.render == "none":
        results = run_episode(env, agent, args.steps, verbose=args.verbose)
        print_statistics(results)
    elif args.render == "ascii":
        results = animate_episode(
            env, agent, args.steps,
            delay=args.delay,
            use_matplotlib=False
        )
        print("\n")
        print_statistics(results)
    elif args.render == "matplotlib":
        results = animate_episode(
            env, agent, args.steps,
            delay=args.delay,
            use_matplotlib=True
        )
        print_statistics(results)
    else:
        print(f"Unknown render mode: {args.render}")
        sys.exit(1)

    if results["success"]:
        print(f"\n✓ SUCCESS! Agent reached goal in {results['steps']} steps.")
    else:
        print(f"\n✗ Agent did not reach goal within {args.steps} steps.")

    print(f"Mislocalization events: {results.get('mislocalization_events', 0)}")
    print(f"Average entropy (first 20 steps): {results.get('entropy_first_20', 0):.3f}")


def run_benchmark(args) -> None:
    """
    Run enhanced benchmark comparing planners.

    Shows clear separation between uncertainty-aware MPC and greedy.
    """
    # Apply fast preset
    if args.fast:
        particles = 200
        horizon = 6
        n_samples = 80
        max_steps = 120
        episodes_per_seed = 2
        if args.seeds == [0, 50]:  # Default not overridden
            args.seeds = [0, 20]
    else:
        particles = args.particles
        horizon = args.horizon
        n_samples = args.samples
        max_steps = args.steps
        episodes_per_seed = args.episodes

    print(f"\n{'='*70}")
    print("WOOEdge Benchmark - Uncertainty-Aware vs Greedy Planning")
    print(f"{'='*70}")
    print(f"Mode: {'FAST' if args.fast else 'FULL'}")
    print(f"Seeds: {args.seeds[0]} to {args.seeds[-1]} ({len(args.seeds)} total)")
    print(f"Episodes per seed: {episodes_per_seed}")
    print(f"Max steps: {max_steps}")
    print(f"Maze type: {args.maze_type}")
    print(f"Sensor noise: {args.sensor_noise}, Slip: {args.slip_prob}")
    print(f"Particles: {particles}, Horizon: {horizon}, Samples: {n_samples}")
    print(f"{'='*70}\n")

    # Environment config - designed for ambiguity
    env_config = EnvConfig(
        grid_size=args.grid_size,
        maze_type=args.maze_type,
        sensor_noise_prob=args.sensor_noise,
        slip_prob=args.slip_prob,
        trap_corridors=True,
        seed=args.seeds[0]
    )

    # Show maze ambiguity stats
    test_env = GridWorld(env_config)
    amb_stats = test_env.get_ambiguity_stats()
    print(f"Maze statistics:")
    print(f"  - Grid size: {test_env.grid_size}x{test_env.grid_size}")
    print(f"  - Average positions per sensor reading: {amb_stats['mean_ambiguity']:.1f}")
    print(f"  - Max ambiguity: {amb_stats['max_ambiguity']}")
    print(f"  - Lookout positions: {amb_stats['lookout_count']} / {amb_stats['total_positions']}")
    print()

    # Define planner configurations
    configs = {
        "MPC (full)": AgentConfig(
            planner_type="mpc",
            n_particles=particles,
            planning_horizon=horizon,
            n_samples=n_samples,
            uncertainty_weight=0.5,
            exploration_weight=0.3,
            info_gain_weight=0.4,
            uniform_init=True,
            seed=0
        ),
        "MPC (no uncert)": AgentConfig(
            planner_type="mpc_no_uncert",
            n_particles=particles,
            planning_horizon=horizon,
            n_samples=n_samples,
            uncertainty_weight=0.0,
            exploration_weight=0.0,
            info_gain_weight=0.0,
            uniform_init=True,
            seed=0
        ),
        "Greedy": AgentConfig(
            planner_type="greedy",
            n_particles=particles,
            uniform_init=True,
            seed=0
        ),
    }

    # Run comparison across all seeds
    all_results = {name: {
        "successes": [],
        "steps": [],
        "costs": [],
        "entropy_first_20": [],
        "mislocalization_events": [],
        "mislocalization_rates": [],
        "planning_times": [],
    } for name in configs}

    total_seeds = len(args.seeds)
    benchmark_start = time.time()

    for seed_idx, seed in enumerate(args.seeds):
        if (seed_idx + 1) % 10 == 0 or seed_idx == 0:
            print(f"Processing seed {seed_idx + 1}/{total_seeds}...")

        for episode in range(episodes_per_seed):
            ep_seed = seed * 1000 + episode

            for name, agent_config in configs.items():
                # Create environment
                env = GridWorld(EnvConfig(
                    grid_size=env_config.grid_size if args.maze_type != "symmetric_fork_trap" else 12,
                    maze_type=env_config.maze_type,
                    sensor_noise_prob=env_config.sensor_noise_prob,
                    slip_prob=env_config.slip_prob,
                    trap_corridors=env_config.trap_corridors,
                    seed=ep_seed
                ))

                # Create agent
                agent_dict = {k: v for k, v in agent_config.__dict__.items() if k != 'seed'}
                agent = WOOEdgeAgent(AgentConfig(**agent_dict, seed=ep_seed))

                # Run episode with timing
                ep_start = time.time()
                result = run_episode(env, agent, max_steps)
                ep_time = time.time() - ep_start

                # Calculate per-step planning time
                if result["steps"] > 0:
                    per_step_time = ep_time / result["steps"]
                else:
                    per_step_time = 0

                # Collect results
                all_results[name]["successes"].append(result["success"])
                all_results[name]["steps"].append(result["steps"])
                all_results[name]["costs"].append(result["total_cost"])
                all_results[name]["entropy_first_20"].append(result["entropy_first_20"])
                all_results[name]["mislocalization_events"].append(result["mislocalization_events"])
                all_results[name]["mislocalization_rates"].append(result["mislocalization_rate"])
                all_results[name]["planning_times"].append(per_step_time)

    benchmark_time = time.time() - benchmark_start

    # Compute final statistics
    final_results = {}
    for name, data in all_results.items():
        final_results[name] = {
            "success_rate": np.mean(data["successes"]),
            "mean_steps": np.mean(data["steps"]),
            "std_steps": np.std(data["steps"]),
            "mean_cost": np.mean(data["costs"]),
            "mean_entropy_first_20": np.mean(data["entropy_first_20"]),
            "total_mislocalization": sum(data["mislocalization_events"]),
            "mean_mislocalization_rate": np.mean(data["mislocalization_rates"]),
            "mean_planning_time_ms": np.mean(data["planning_times"]) * 1000,
        }

    # Print results
    print("\n" + "="*90)
    print("BENCHMARK RESULTS")
    print("="*90)

    print(f"\n{'Planner':<18} {'Success%':>10} {'Avg Steps':>10} {'Entropy20':>10} "
          f"{'Misloc':>8} {'ms/step':>10}")
    print("-"*90)

    for name in ["MPC (full)", "MPC (no uncert)", "Greedy"]:
        r = final_results[name]
        print(f"{name:<18} "
              f"{r['success_rate']*100:>9.1f}% "
              f"{r['mean_steps']:>10.1f} "
              f"{r['mean_entropy_first_20']:>10.3f} "
              f"{r['total_mislocalization']:>8d} "
              f"{r['mean_planning_time_ms']:>9.1f}")

    print("-"*90)

    # Timing summary
    print(f"\nBenchmark runtime: {benchmark_time:.1f}s")
    total_episodes = total_seeds * episodes_per_seed * len(configs)
    print(f"Total episodes: {total_episodes}")
    print(f"Average time per episode: {benchmark_time/total_episodes*1000:.0f}ms")

    # Analysis
    print("\n" + "="*90)
    print("ANALYSIS")
    print("="*90)

    mpc_success = final_results["MPC (full)"]["success_rate"]
    mpc_no_uncert_success = final_results["MPC (no uncert)"]["success_rate"]
    greedy_success = final_results["Greedy"]["success_rate"]

    print(f"\nSuccess rate comparison:")
    print(f"  MPC (full):      {mpc_success*100:.1f}%")
    print(f"  MPC (no uncert): {mpc_no_uncert_success*100:.1f}%")
    print(f"  Greedy:          {greedy_success*100:.1f}%")

    if mpc_success > greedy_success + 0.1:
        print(f"\n✓ MPC outperforms Greedy by {(mpc_success - greedy_success)*100:.1f} percentage points")
    else:
        print(f"\n⚠ MPC advantage over Greedy: {(mpc_success - greedy_success)*100:.1f} pp")

    if mpc_success > mpc_no_uncert_success + 0.05:
        print(f"✓ Uncertainty awareness improves MPC by {(mpc_success - mpc_no_uncert_success)*100:.1f} pp")
    else:
        print(f"⚠ Uncertainty awareness effect: {(mpc_success - mpc_no_uncert_success)*100:.1f} pp")

    print(f"\nMislocalization events:")
    print(f"  MPC (full):      {final_results['MPC (full)']['total_mislocalization']}")
    print(f"  MPC (no uncert): {final_results['MPC (no uncert)']['total_mislocalization']}")
    print(f"  Greedy:          {final_results['Greedy']['total_mislocalization']}")

    print(f"\nEntropy in first 20 steps (higher = more uncertainty maintained):")
    print(f"  MPC (full):      {final_results['MPC (full)']['mean_entropy_first_20']:.3f}")
    print(f"  MPC (no uncert): {final_results['MPC (no uncert)']['mean_entropy_first_20']:.3f}")
    print(f"  Greedy:          {final_results['Greedy']['mean_entropy_first_20']:.3f}")

    # Target check
    print("\n" + "="*90)
    print("TARGET CHECK")
    print("="*90)
    targets_met = 0
    if greedy_success <= 0.70:
        print(f"✓ Greedy ≤ 70%: {greedy_success*100:.1f}%")
        targets_met += 1
    else:
        print(f"✗ Greedy ≤ 70%: {greedy_success*100:.1f}% (too high)")

    if mpc_success >= 0.90:
        print(f"✓ MPC (full) ≥ 90%: {mpc_success*100:.1f}%")
        targets_met += 1
    else:
        print(f"✗ MPC (full) ≥ 90%: {mpc_success*100:.1f}% (too low)")

    if mpc_success > mpc_no_uncert_success:
        print(f"✓ MPC (full) > MPC (no uncert): {mpc_success*100:.1f}% > {mpc_no_uncert_success*100:.1f}%")
        targets_met += 1
    else:
        print(f"✗ MPC (full) > MPC (no uncert): {mpc_success*100:.1f}% vs {mpc_no_uncert_success*100:.1f}%")

    print(f"\nTargets met: {targets_met}/3")
    print("="*90)


def run_demo(args) -> None:
    """Run a quick demo of the system."""
    print(f"\n{'='*60}")
    print("WOOEdge Demo - Edge Intelligence Without LLMs")
    print(f"{'='*60}\n")

    print("This demo shows a predictive world-model agent that:")
    print("  1. Maintains belief over its position (particle filter)")
    print("  2. Learns environment dynamics online")
    print("  3. Plans actions using MPC with uncertainty awareness")
    print()
    print("The environment has AMBIGUOUS sensors - multiple positions")
    print("produce identical readings. The agent must explore to localize!")
    print()

    input("Press Enter to start the demo...")

    # Run episode with challenging setup
    env_config = EnvConfig(
        grid_size=14,
        maze_type="symmetric",
        sensor_noise_prob=0.25,
        slip_prob=0.15,
        trap_corridors=True,
        seed=42
    )
    env = GridWorld(env_config)

    agent_config = AgentConfig(
        n_particles=800,
        planner_type="mpc",
        planning_horizon=10,
        n_samples=100,
        uncertainty_weight=0.5,
        exploration_weight=0.3,
        uniform_init=True,
        seed=42
    )
    agent = WOOEdgeAgent(agent_config)

    print("\nStarting episode...")
    print("Watch the agent (A) navigate toward the goal (G)")
    print("Belief distribution shown as intensity (░▒▓)")
    print("Starting with UNIFORM belief (max uncertainty)\n")

    results = animate_episode(env, agent, 150, delay=0.15, use_matplotlib=False)

    print("\n" + "="*60)
    if results["success"]:
        print(f"SUCCESS! Agent reached the goal in {results['steps']} steps.")
    else:
        print("Agent did not reach the goal in time.")
    print(f"Total cost: {results['total_cost']:.2f}")
    print("="*60)


def verify_separation(args) -> None:
    """
    Fast verification of planner separation (<60s).

    Single run comparing MPC(full) vs MPC(no_uncert).
    MPC(full) uses commit gate with rollout penalty (only for fork mazes).
    """
    import time

    # Hard-capped parameters for <60s runtime
    PARTICLES = 100
    HORIZON = 4
    N_SAMPLES = 30
    N_BELIEF = 10
    MAX_STEPS = args.steps
    seeds = list(range(args.seeds[0], args.seeds[1]))

    # Fixed commit gate parameters (only used for fork mazes)
    H_THRESH = 0.16
    COMMIT_PENALTY = 40.0

    # Check if this maze type has forks
    test_env = GridWorld(EnvConfig(maze_type=args.maze_type, seed=0))
    has_fork = getattr(test_env, 'has_fork', False)

    print(f"\n{'='*70}")
    print("WOOEdge Fast Verification")
    print(f"{'='*70}")
    print(f"Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} total), Steps: {MAX_STEPS}")
    print(f"Params: particles={PARTICLES}, horizon={HORIZON}, samples={N_SAMPLES}")
    print(f"Maze: {args.maze_type}, Has fork: {has_fork}")
    if has_fork:
        print(f"Commit gate: MPC(full) only, H_thresh={H_THRESH}, penalty={COMMIT_PENALTY}")
    else:
        print(f"Commit gate: DISABLED (no fork structure)")
    print(f"{'='*70}\n")

    # MPC(full) - commit gate only enabled for fork mazes
    mpc_full_config = AgentConfig(
        planner_type="mpc",
        n_particles=PARTICLES,
        planning_horizon=HORIZON,
        n_samples=N_SAMPLES,
        n_belief_samples=N_BELIEF,
        uncertainty_weight=0.5,
        exploration_weight=0.3,
        info_gain_weight=0.4,
        mirror_invariant=True,
        commit_gate_enabled=has_fork,  # Only enable for fork mazes
        commit_gate_entropy_threshold=H_THRESH,
        commit_gate_penalty=COMMIT_PENALTY,
    )

    # MPC(no_uncert) - no commit gate
    mpc_no_config = AgentConfig(
        planner_type="mpc_no_uncert",
        n_particles=PARTICLES,
        planning_horizon=HORIZON,
        n_samples=N_SAMPLES,
        n_belief_samples=N_BELIEF,
        mirror_invariant=True,
        commit_gate_enabled=False,
    )

    # Track results
    results = {
        "MPC (full)": {"successes": [], "wrong_commit": [], "entropy_first_15": []},
        "MPC (no uncert)": {"successes": [], "wrong_commit": [], "entropy_first_15": []},
    }

    start_time = time.time()

    for seed in seeds:
        for name, config in [("MPC (full)", mpc_full_config), ("MPC (no uncert)", mpc_no_config)]:
            env = GridWorld(EnvConfig(
                maze_type=args.maze_type,
                sensor_noise_prob=0.3,
                slip_prob=0.15,
                mirror_invariant=True,
                seed=seed
            ))

            agent_dict = {k: v for k, v in config.__dict__.items() if k != 'seed'}
            agent = WOOEdgeAgent(AgentConfig(**agent_dict, seed=seed))
            result = run_episode(env, agent, MAX_STEPS)

            results[name]["successes"].append(result["success"])
            # Only track wrong_commit for fork mazes
            if has_fork:
                results[name]["wrong_commit"].append(result.get("wrong_commit", False))
            results[name]["entropy_first_15"].append(result.get("entropy_first_15", 0))

        if (seed - seeds[0] + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Seed {seed - seeds[0] + 1}/{len(seeds)} ({elapsed:.1f}s)")

    runtime = time.time() - start_time

    # Compute metrics
    mpc_full = np.mean(results["MPC (full)"]["successes"]) * 100
    mpc_no = np.mean(results["MPC (no uncert)"]["successes"]) * 100
    separation = mpc_full - mpc_no

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    if has_fork:
        wrong_full = np.mean(results["MPC (full)"]["wrong_commit"]) * 100
        wrong_no = np.mean(results["MPC (no uncert)"]["wrong_commit"]) * 100
        print(f"{'Planner':<18} {'Success%':>10} {'WrongCommit%':>14}")
        print("-"*70)
        print(f"{'MPC (full)':<18} {mpc_full:>9.1f}% {wrong_full:>13.1f}%")
        print(f"{'MPC (no uncert)':<18} {mpc_no:>9.1f}% {wrong_no:>13.1f}%")
    else:
        print(f"{'Planner':<18} {'Success%':>10} {'WrongCommit%':>14}")
        print("-"*70)
        print(f"{'MPC (full)':<18} {mpc_full:>9.1f}% {'N/A':>13}")
        print(f"{'MPC (no uncert)':<18} {mpc_no:>9.1f}% {'N/A':>13}")
    print("-"*70)
    print(f"Runtime: {runtime:.1f}s")

    print(f"\nTargets:")
    print(f"  MPC(full) >= 90%: {mpc_full:.1f}% " + ("OK" if mpc_full >= 90 else "FAIL"))
    print(f"  Separation >= 10pp: {separation:+.1f}pp " + ("OK" if separation >= 10 else "FAIL"))
    print(f"  Runtime < 60s: {runtime:.1f}s " + ("OK" if runtime < 60 else "FAIL"))
    print(f"{'='*70}")


def debug_separation(args) -> None:
    """
    Debug mode: per-step logging of commit gate behavior.

    Shows entropy, commit zone flags, and chosen actions.
    Only shows commit zone columns for fork mazes.
    """
    PARTICLES = 100
    HORIZON = 4
    N_SAMPLES = 30
    N_BELIEF = 10
    H_THRESH = 0.16
    COMMIT_PENALTY = 40.0

    env = GridWorld(EnvConfig(
        maze_type=args.maze_type,
        sensor_noise_prob=0.3,
        slip_prob=0.15,
        mirror_invariant=True,
        seed=args.seed
    ))

    has_fork = getattr(env, 'has_fork', False)

    print(f"\n{'='*80}")
    print("DEBUG: Per-step analysis")
    print(f"{'='*80}")
    print(f"Seed: {args.seed}, Steps: {args.steps}, Maze: {args.maze_type}")
    print(f"Has fork: {has_fork}")
    if has_fork:
        print(f"H_thresh={H_THRESH}, commit_penalty={COMMIT_PENALTY}")
        print(f"Commit zones: goal={len(env.commit_zone_goal)}, trap={len(env.commit_zone_trap)}")
        if env.fork_split_y is not None:
            print(f"Fork split at y={env.fork_split_y}")
    else:
        print(f"Commit gate: DISABLED (no fork structure)")
    print(f"{'='*80}\n")

    agent_config = AgentConfig(
        planner_type="mpc",
        n_particles=PARTICLES,
        planning_horizon=HORIZON,
        n_samples=N_SAMPLES,
        n_belief_samples=N_BELIEF,
        uncertainty_weight=0.5,
        exploration_weight=0.3,
        info_gain_weight=0.4,
        mirror_invariant=True,
        commit_gate_enabled=has_fork,  # Only enable for fork mazes
        commit_gate_entropy_threshold=H_THRESH,
        commit_gate_penalty=COMMIT_PENALTY,
        seed=args.seed,
    )

    agent = WOOEdgeAgent(agent_config)
    obs = env.reset()
    agent.setup(env, know_start=False)

    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    from .env import ACTION_DELTAS, Action

    if has_fork:
        print(f"{'Step':<5} {'Pos':<8} {'Entropy':>8} {'InGoal':>7} {'InTrap':>7} {'Action':<6} {'ToTrap':>7}")
    else:
        print(f"{'Step':<5} {'Pos':<8} {'Entropy':>8} {'Action':<6}")
    print("-" * 80)

    done = False
    for step in range(args.steps):
        true_pos = env.agent_pos
        entropy = agent.belief.get_entropy()
        norm_entropy = entropy / (np.log(PARTICLES) + 1e-6)

        action = agent.act(obs)
        action_name = ACTION_NAMES[action]

        if has_fork:
            in_goal = true_pos in env.commit_zone_goal
            in_trap = true_pos in env.commit_zone_trap

            # Check if action would move into trap zone
            dy, dx = ACTION_DELTAS[Action(action)]
            next_pos = (true_pos[0] + dy, true_pos[1] + dx)
            would_enter_trap = (true_pos not in env.commit_zone_trap and
                               next_pos in env.commit_zone_trap)

            print(f"{step:<5} {str(true_pos):<8} {norm_entropy:>8.3f} "
                  f"{'Y' if in_goal else 'N':>7} {'Y' if in_trap else 'N':>7} "
                  f"{action_name:<6} {'Y' if would_enter_trap else 'N':>7}")
        else:
            print(f"{step:<5} {str(true_pos):<8} {norm_entropy:>8.3f} {action_name:<6}")

        obs, cost, done, info = env.step(action)
        agent.record_step(info["true_pos"], cost)

        if done:
            print(f"\n>>> GOAL REACHED at step {step + 1} <<<")
            break

    if not done:
        print(f"\n>>> Did not reach goal in {args.steps} steps <<<")

    print(f"\nFinal position: {env.agent_pos}")
    if has_fork:
        print(f"Wrong commit: {agent.state.wrong_commit}")
    else:
        print(f"Wrong commit: N/A (no fork)")
    print(f"{'='*80}")


def sweep_commit_gate(args) -> None:
    """
    Sweep commit_penalty and H_thresh values.

    Tests 2D grid of H_thresh and commit_penalty combinations.
    Only valid for fork mazes (e.g., symmetric_fork_trap).
    """
    import time

    # Check if maze has fork structure
    test_env = GridWorld(EnvConfig(maze_type=args.maze_type, seed=0))
    has_fork = getattr(test_env, 'has_fork', False)

    if not has_fork:
        print(f"\nERROR: sweep_commit_gate requires a fork maze.")
        print(f"Maze type '{args.maze_type}' has no fork structure.")
        print(f"Use --maze-type symmetric_fork_trap instead.")
        return

    PARTICLES = 100
    HORIZON = 4
    N_SAMPLES = 30
    N_BELIEF = 10
    MAX_STEPS = args.steps
    seeds = list(range(args.seeds[0], args.seeds[1]))

    H_THRESH_VALUES = [0.2, 0.3, 0.4]
    PENALTY_VALUES = [100, 200, 500]

    print(f"\n{'='*80}")
    print("Commit Gate Parameter Sweep (2D)")
    print(f"{'='*80}")
    print(f"Maze: {args.maze_type} (has_fork=True)")
    print(f"Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} total), Steps: {MAX_STEPS}")
    print(f"H_thresh sweep: {H_THRESH_VALUES}")
    print(f"commit_penalty sweep: {PENALTY_VALUES}")
    print(f"{'='*80}\n")

    all_results = []

    for H_THRESH in H_THRESH_VALUES:
      for penalty in PENALTY_VALUES:
        print(f"\n--- H_thresh={H_THRESH}, commit_penalty={penalty} ---")

        mpc_full_config = AgentConfig(
            planner_type="mpc",
            n_particles=PARTICLES,
            planning_horizon=HORIZON,
            n_samples=N_SAMPLES,
            n_belief_samples=N_BELIEF,
            uncertainty_weight=0.5,
            exploration_weight=0.3,
            info_gain_weight=0.4,
            mirror_invariant=True,
            commit_gate_enabled=True,
            commit_gate_entropy_threshold=H_THRESH,
            commit_gate_penalty=penalty,
        )

        mpc_no_config = AgentConfig(
            planner_type="mpc_no_uncert",
            n_particles=PARTICLES,
            planning_horizon=HORIZON,
            n_samples=N_SAMPLES,
            n_belief_samples=N_BELIEF,
            mirror_invariant=True,
            commit_gate_enabled=False,
        )

        results = {"full": [], "no": [], "wrong_full": [], "wrong_no": []}
        start_time = time.time()

        for seed in seeds:
            for name, config in [("full", mpc_full_config), ("no", mpc_no_config)]:
                env = GridWorld(EnvConfig(
                    maze_type=args.maze_type,
                    sensor_noise_prob=0.3,
                    slip_prob=0.15,
                    mirror_invariant=True,
                    seed=seed
                ))

                agent_dict = {k: v for k, v in config.__dict__.items() if k != 'seed'}
                agent = WOOEdgeAgent(AgentConfig(**agent_dict, seed=seed))
                result = run_episode(env, agent, MAX_STEPS)

                results[name].append(result["success"])
                results[f"wrong_{name}"].append(result.get("wrong_commit", False))

        runtime = time.time() - start_time

        mpc_full = np.mean(results["full"]) * 100
        mpc_no = np.mean(results["no"]) * 100
        sep = mpc_full - mpc_no
        wrong_full = np.mean(results["wrong_full"]) * 100
        wrong_no = np.mean(results["wrong_no"]) * 100

        print(f"  MPC(full): {mpc_full:5.1f}%, WrongCommit: {wrong_full:5.1f}%")
        print(f"  MPC(no):   {mpc_no:5.1f}%, WrongCommit: {wrong_no:5.1f}%")
        print(f"  Separation: {sep:+.1f}pp, Runtime: {runtime:.1f}s")

        meets = mpc_full >= 90 and sep >= 10 and runtime < 60
        if meets:
            print(f"  >>> MEETS TARGETS <<<")

        all_results.append({
            "h_thresh": H_THRESH,
            "penalty": penalty,
            "mpc_full": mpc_full,
            "mpc_no": mpc_no,
            "sep": sep,
            "wrong_full": wrong_full,
            "wrong_no": wrong_no,
            "runtime": runtime,
            "meets": meets,
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'H_thresh':<10} {'Penalty':<10} {'MPC(full)':<12} {'MPC(no)':<10} {'Sep':<10} {'WrongFull':<12}")
    print("-" * 64)
    for r in all_results:
        print(f"{r['h_thresh']:<10} {r['penalty']:<10} {r['mpc_full']:>5.1f}%       {r['mpc_no']:>5.1f}%     {r['sep']:>+5.1f}pp   {r['wrong_full']:>5.1f}%")

    # Best config
    valid = [r for r in all_results if r["meets"]]
    if valid:
        best = max(valid, key=lambda x: x["sep"])
        print(f"\nBEST CONFIG: H_thresh={best['h_thresh']}, penalty={best['penalty']}")
        print(f"  MPC(full)={best['mpc_full']:.1f}%, Sep={best['sep']:+.1f}pp")
    else:
        best = max(all_results, key=lambda x: x["sep"])
        print(f"\nNO CONFIG MEETS TARGETS. Best by separation:")
        print(f"  H_thresh={best['h_thresh']}, penalty={best['penalty']}")
        print(f"  MPC(full)={best['mpc_full']:.1f}%, Sep={best['sep']:+.1f}pp")
    print(f"{'='*80}")


def run_stats(args) -> None:
    """Run and print detailed statistics."""
    print(f"\n{'='*60}")
    print("WOOEdge Statistics Run")
    print(f"{'='*60}\n")

    env_config = EnvConfig(
        grid_size=args.grid_size,
        maze_type=args.maze_type,
        sensor_noise_prob=args.sensor_noise,
        slip_prob=args.slip_prob,
        seed=args.seed
    )
    env = GridWorld(env_config)

    # Show maze stats
    amb_stats = env.get_ambiguity_stats()
    print("Maze Statistics:")
    print(f"  Unique sensor readings: {amb_stats['unique_readings']}")
    print(f"  Average ambiguity: {amb_stats['mean_ambiguity']:.2f}")
    print(f"  Max ambiguity: {amb_stats['max_ambiguity']}")
    print(f"  Lookout positions: {amb_stats['lookout_count']}")
    print()

    agent_config = AgentConfig(
        planner_type=args.planner,
        uniform_init=True,
        seed=args.seed
    )
    agent = WOOEdgeAgent(agent_config)

    results = run_episode(env, agent, args.steps, verbose=True)

    print("\n" + "="*60)
    print("DETAILED STATISTICS")
    print("="*60)
    print_statistics(results)

    # Print agent statistics
    agent_stats = agent.get_statistics()
    print("\nAgent Internal Statistics:")
    print_statistics(agent_stats, prefix="  ")

    # Entropy analysis
    if agent.state.entropy_history:
        print("\nEntropy Analysis:")
        ent = agent.state.entropy_history
        print(f"  Initial entropy: {ent[0]:.4f}")
        print(f"  Final entropy: {ent[-1]:.4f}")
        print(f"  Min entropy: {min(ent):.4f}")
        print(f"  Max entropy: {max(ent):.4f}")
        print(f"  Mean entropy: {np.mean(ent):.4f}")
        if len(ent) >= 20:
            print(f"  Entropy (first 20): {np.mean(ent[:20]):.4f}")
        print(f"  Trend: {agent._compute_entropy_trend()}")

    # Mislocalization analysis
    if agent.state.distance_error_history:
        print("\nMislocalization Analysis:")
        print(f"  Mislocalization events: {agent.state.mislocalization_events}")
        print(f"  Mislocalization rate: {agent.get_mislocalization_rate()*100:.1f}%")
        print(f"  Mean distance error: {np.mean(agent.state.distance_error_history):.2f}")
        print(f"  Max distance error: {max(agent.state.distance_error_history)}")


def sweep_conservatism(args) -> None:
    """
    Deterministic parameter sweep for MPC conservatism tuning.

    Evaluates a grid of parameter settings and selects the best one based on:
    - Maximize separation (MPC full - MPC no_uncert)
    - Subject to: success_full >= 90% AND runtime < 60s
    """
    import time
    import itertools

    # Fixed compute parameters
    PARTICLES = 100
    HORIZON = 5
    N_SAMPLES = 30
    N_BELIEF = 10
    MAX_STEPS = args.steps
    seeds = list(range(args.seeds[0], args.seeds[1]))

    # Parameter grid (6-10 settings)
    param_grid = [
        # (uncertainty_weight, info_gain_weight, risk_threshold, risk_penalty, label)
        (0.4, 0.3, 0.5, 5.0, "baseline"),
        (0.5, 0.4, 0.5, 5.0, "med_uncert"),
        (0.6, 0.4, 0.5, 5.0, "high_uncert"),
        (0.5, 0.4, 0.4, 6.0, "strict_gate"),
        (0.5, 0.4, 0.6, 4.0, "loose_gate"),
        (0.5, 0.5, 0.5, 5.0, "high_info"),
        (0.4, 0.5, 0.4, 6.0, "info+strict"),
        (0.6, 0.3, 0.4, 7.0, "aggressive"),
    ]

    print(f"\n{'='*90}")
    print("WOOEdge Conservatism Parameter Sweep")
    print(f"{'='*90}")
    print(f"Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} total), Steps: {MAX_STEPS}")
    print(f"Maze: {args.maze_type}, Compute: particles={PARTICLES}, horizon={HORIZON}, samples={N_SAMPLES}")
    print(f"Evaluating {len(param_grid)} parameter configurations...")
    print(f"{'='*90}\n")

    # Results storage
    all_results = []

    for idx, (uw, ig, rt, rp, label) in enumerate(param_grid):
        config_start = time.time()

        # Create MPC(full) config with current parameters
        mpc_full_config = AgentConfig(
            planner_type="mpc",
            n_particles=PARTICLES,
            planning_horizon=HORIZON,
            n_samples=N_SAMPLES,
            n_belief_samples=N_BELIEF,
            uncertainty_weight=uw,
            exploration_weight=0.3,
            info_gain_weight=ig,
            mirror_invariant=True,
            risk_gate_enabled=True,
            risk_gate_entropy_threshold=rt,
            risk_gate_penalty=rp,
        )

        # MPC(no_uncert) config - same compute, no uncertainty awareness
        mpc_no_config = AgentConfig(
            planner_type="mpc_no_uncert",
            n_particles=PARTICLES,
            planning_horizon=HORIZON,
            n_samples=N_SAMPLES,
            n_belief_samples=N_BELIEF,
            mirror_invariant=True,
            risk_gate_enabled=False,
        )

        # Track results for this config
        full_successes = []
        no_successes = []
        full_steps = []
        full_entropy = []

        for seed in seeds:
            env = GridWorld(EnvConfig(
                maze_type=args.maze_type,
                sensor_noise_prob=0.3,
                slip_prob=0.15,
                mirror_invariant=True,
                seed=seed
            ))

            # Run MPC(full)
            agent_dict = {k: v for k, v in mpc_full_config.__dict__.items() if k != 'seed'}
            agent = WOOEdgeAgent(AgentConfig(**agent_dict, seed=seed))
            result_full = run_episode(env, agent, MAX_STEPS)
            full_successes.append(result_full["success"])
            full_steps.append(result_full.get("steps", MAX_STEPS))
            full_entropy.append(result_full.get("entropy_first_15", 0))

            # Reset env and run MPC(no_uncert)
            env = GridWorld(EnvConfig(
                maze_type=args.maze_type,
                sensor_noise_prob=0.3,
                slip_prob=0.15,
                mirror_invariant=True,
                seed=seed
            ))
            agent_dict = {k: v for k, v in mpc_no_config.__dict__.items() if k != 'seed'}
            agent = WOOEdgeAgent(AgentConfig(**agent_dict, seed=seed))
            result_no = run_episode(env, agent, MAX_STEPS)
            no_successes.append(result_no["success"])

        config_time = time.time() - config_start

        success_full = np.mean(full_successes) * 100
        success_no = np.mean(no_successes) * 100
        separation = success_full - success_no
        avg_steps = np.mean(full_steps)
        avg_entropy = np.mean(full_entropy)

        all_results.append({
            "label": label,
            "uw": uw,
            "ig": ig,
            "rt": rt,
            "rp": rp,
            "success_full": success_full,
            "success_no": success_no,
            "separation": separation,
            "avg_steps": avg_steps,
            "avg_entropy": avg_entropy,
            "runtime": config_time,
        })

        print(f"[{idx+1}/{len(param_grid)}] {label}: full={success_full:.0f}%, no={success_no:.0f}%, "
              f"sep={separation:+.0f}pp, time={config_time:.1f}s")

    # Print results table
    print(f"\n{'='*110}")
    print("SWEEP RESULTS")
    print(f"{'='*110}")
    print(f"{'Config':<12} {'uw':>5} {'ig':>5} {'rt':>5} {'rp':>5} | "
          f"{'Full%':>6} {'No%':>6} {'Sep':>6} | {'Steps':>6} {'Ent15':>7} {'Time':>6}")
    print("-"*110)

    for r in all_results:
        # Mark if meets constraints
        meets = "OK" if r["success_full"] >= 90 and r["runtime"] < 60 else ""
        print(f"{r['label']:<12} {r['uw']:>5.2f} {r['ig']:>5.2f} {r['rt']:>5.2f} {r['rp']:>5.1f} | "
              f"{r['success_full']:>5.0f}% {r['success_no']:>5.0f}% {r['separation']:>+5.0f}pp | "
              f"{r['avg_steps']:>6.1f} {r['avg_entropy']:>7.3f} {r['runtime']:>5.1f}s {meets}")

    print("-"*110)

    # Select best configuration
    valid_configs = [r for r in all_results if r["success_full"] >= 90 and r["runtime"] < 60]

    if valid_configs:
        # Maximize separation among valid configs
        best = max(valid_configs, key=lambda x: x["separation"])
        print(f"\nBEST CONFIG (max separation with success>=90%, runtime<60s):")
        print(f"  Label: {best['label']}")
        print(f"  Parameters: uncertainty_weight={best['uw']}, info_gain_weight={best['ig']}, "
              f"risk_threshold={best['rt']}, risk_penalty={best['rp']}")
        print(f"  Results: success_full={best['success_full']:.0f}%, separation={best['separation']:+.0f}pp, "
              f"runtime={best['runtime']:.1f}s")

        print(f"\n  To lock these as defaults, update verify_separation with:")
        print(f"    uncertainty_weight={best['uw']},")
        print(f"    info_gain_weight={best['ig']},")
        print(f"    risk_gate_entropy_threshold={best['rt']},")
        print(f"    risk_gate_penalty={best['rp']},")
    else:
        # Fallback: pick config with best separation that has >= 85% success
        relaxed = [r for r in all_results if r["success_full"] >= 85]
        if relaxed:
            best = max(relaxed, key=lambda x: x["separation"])
            print(f"\nNO CONFIG meets all constraints. Best relaxed (success>=85%):")
            print(f"  Label: {best['label']}")
            print(f"  Results: success_full={best['success_full']:.0f}%, separation={best['separation']:+.0f}pp")
        else:
            best = max(all_results, key=lambda x: x["separation"])
            print(f"\nNO VALID CONFIG. Best by separation:")
            print(f"  Label: {best['label']}")
            print(f"  Results: success_full={best['success_full']:.0f}%, separation={best['separation']:+.0f}pp")

    print(f"{'='*110}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="WOOEdge - World-model Online Edge Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m wooedge.cli run --seed 0 --steps 200 --render ascii
  python -m wooedge.cli benchmark --seeds 0 50
  python -m wooedge.cli demo
  python -m wooedge.cli stats --seed 42 --planner mpc
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single episode")
    run_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    run_parser.add_argument("--steps", type=int, default=300, help="Max steps")
    run_parser.add_argument("--render", choices=["ascii", "matplotlib", "none"],
                           default="ascii", help="Render mode")
    run_parser.add_argument("--planner", choices=["mpc", "mpc_no_uncert", "greedy", "infogain"],
                           default="mpc", help="Planner type")
    run_parser.add_argument("--grid-size", type=int, default=16, help="Grid size")
    run_parser.add_argument("--maze-type", choices=["symmetric", "symmetric_fork_trap", "corridor", "random"],
                           default="symmetric", help="Maze type")
    run_parser.add_argument("--sensor-noise", type=float, default=0.3,
                           help="Sensor noise probability")
    run_parser.add_argument("--slip-prob", type=float, default=0.15,
                           help="Action slip probability")
    run_parser.add_argument("--trap-corridors", type=bool, default=True,
                           help="Include trap corridors")
    run_parser.add_argument("--particles", type=int, default=1000,
                           help="Number of particles")
    run_parser.add_argument("--horizon", type=int, default=12,
                           help="Planning horizon")
    run_parser.add_argument("--samples", type=int, default=150,
                           help="Number of rollout samples")
    run_parser.add_argument("--goal-weight", type=float, default=1.0,
                           help="Goal distance weight")
    run_parser.add_argument("--uncertainty-weight", type=float, default=0.5,
                           help="Uncertainty weight")
    run_parser.add_argument("--exploration-weight", type=float, default=0.3,
                           help="Exploration weight")
    run_parser.add_argument("--delay", type=float, default=0.1,
                           help="Animation delay (seconds)")
    run_parser.add_argument("--verbose", action="store_true",
                           help="Print step details")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--fast", action="store_true",
                             help="Fast preset: particles=200, horizon=6, samples=80, steps=120, episodes=2, seeds=20")
    bench_parser.add_argument("--seeds", type=int, nargs=2, default=[0, 50],
                             help="Seed range (start end)")
    bench_parser.add_argument("--steps", type=int, default=300, help="Max steps")
    bench_parser.add_argument("--episodes", type=int, default=1,
                             help="Episodes per seed")
    bench_parser.add_argument("--particles", type=int, default=1000,
                             help="Number of particles")
    bench_parser.add_argument("--horizon", type=int, default=12,
                             help="Planning horizon")
    bench_parser.add_argument("--samples", type=int, default=150,
                             help="Number of rollout samples")
    bench_parser.add_argument("--grid-size", type=int, default=16, help="Grid size")
    bench_parser.add_argument("--maze-type", choices=["symmetric", "symmetric_fork_trap", "corridor", "random"],
                             default="symmetric", help="Maze type")
    bench_parser.add_argument("--sensor-noise", type=float, default=0.3,
                             help="Sensor noise probability")
    bench_parser.add_argument("--slip-prob", type=float, default=0.15,
                             help="Slip probability")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a quick demo")

    # Verify separation command (fast)
    verify_parser = subparsers.add_parser("verify_separation",
                                          help="Fast verification (<60s)")
    verify_parser.add_argument("--seeds", type=int, nargs=2, default=[0, 20],
                              help="Seed range (start end)")
    verify_parser.add_argument("--steps", type=int, default=120, help="Max steps")
    verify_parser.add_argument("--maze-type",
                              choices=["symmetric", "symmetric_fork_trap", "corridor", "random"],
                              default="symmetric", help="Maze type")

    # Sweep conservatism command
    sweep_parser = subparsers.add_parser("sweep_conservatism",
                                         help="Parameter sweep for MPC conservatism tuning")
    sweep_parser.add_argument("--seeds", type=int, nargs=2, default=[0, 20],
                             help="Seed range (start end)")
    sweep_parser.add_argument("--steps", type=int, default=120, help="Max steps")
    sweep_parser.add_argument("--maze-type",
                             choices=["symmetric", "symmetric_fork_trap", "corridor", "random"],
                             default="symmetric", help="Maze type")

    # Debug separation command
    debug_parser = subparsers.add_parser("debug_separation",
                                         help="Debug per-step commit gate behavior")
    debug_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    debug_parser.add_argument("--steps", type=int, default=25, help="Max steps")
    debug_parser.add_argument("--maze-type",
                             choices=["symmetric", "symmetric_fork_trap", "corridor", "random"],
                             default="symmetric", help="Maze type")

    # Sweep commit gate command
    sweep_commit_parser = subparsers.add_parser("sweep_commit_gate",
                                                help="Sweep commit_penalty values")
    sweep_commit_parser.add_argument("--seeds", type=int, nargs=2, default=[0, 20],
                                    help="Seed range (start end)")
    sweep_commit_parser.add_argument("--steps", type=int, default=120, help="Max steps")
    sweep_commit_parser.add_argument("--maze-type",
                                    choices=["symmetric", "symmetric_fork_trap", "corridor", "random"],
                                    default="symmetric", help="Maze type")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Run with detailed stats")
    stats_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    stats_parser.add_argument("--steps", type=int, default=200, help="Max steps")
    stats_parser.add_argument("--planner", choices=["mpc", "mpc_no_uncert", "greedy", "infogain"],
                             default="mpc", help="Planner type")
    stats_parser.add_argument("--grid-size", type=int, default=16, help="Grid size")
    stats_parser.add_argument("--maze-type", choices=["symmetric", "symmetric_fork_trap", "corridor", "random"],
                             default="symmetric", help="Maze type")
    stats_parser.add_argument("--sensor-noise", type=float, default=0.3,
                             help="Sensor noise")
    stats_parser.add_argument("--slip-prob", type=float, default=0.15,
                             help="Slip probability")

    args = parser.parse_args()

    if args.command == "run":
        run_single_episode(args)
    elif args.command == "benchmark":
        # Convert seed range to list
        args.seeds = list(range(args.seeds[0], args.seeds[1]))
        run_benchmark(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "verify_separation":
        verify_separation(args)
    elif args.command == "sweep_conservatism":
        sweep_conservatism(args)
    elif args.command == "debug_separation":
        debug_separation(args)
    elif args.command == "sweep_commit_gate":
        sweep_commit_gate(args)
    elif args.command == "stats":
        run_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
