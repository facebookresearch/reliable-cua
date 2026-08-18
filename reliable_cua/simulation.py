# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Synthetic benchmark generation and calibration experiments.

Generates benchmark data with known ground truth for validating
the hierarchical bootstrap's coverage and CI width.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as sp_stats

from .bootstrap import BootstrapConfig, auto_config, bootstrap_suite_score
from .metrics import suite_score, trimmed_mean
from .types import (
    AppResult,
    ConfigurationResult,
    EvalSuite,
    ScenarioResult,
)


@dataclass
class SimulationConfig:
    """Configuration for generating synthetic benchmark data."""

    n_apps: int = 15
    scenarios_per_app: int = 20
    instances_per_scenario: int = 5
    n_profiles: int = 5
    n_themes: int = 5
    n_ui_states: int = 5
    n_rollouts: int = 3
    app_success_probs: Optional[np.ndarray] = None
    within_app_noise_scenario: float = 0.0
    within_app_noise_config: float = 0.0
    active_axes: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"instance", "profile", "theme", "ui_state"})
    )
    seed: int = 42

    def __post_init__(self):
        if self.app_success_probs is None:
            self.app_success_probs = np.linspace(0.05, 0.95, self.n_apps)


def simulate_suite(
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[EvalSuite, float]:
    """Generate a synthetic EvalSuite with known ground truth.

    Args:
        config: Simulation configuration.
        rng: Random number generator. If None, creates from config.seed.

    Returns:
        Tuple of (EvalSuite, true_suite_score).
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    n_p = config.n_profiles if "profile" in config.active_axes else 1
    n_t = config.n_themes if "theme" in config.active_axes else 1
    n_u = config.n_ui_states if "ui_state" in config.active_axes else 1
    n_i = config.instances_per_scenario if "instance" in config.active_axes else 1

    apps = []
    for a in range(config.n_apps):
        base_prob = config.app_success_probs[a]
        scenarios = []
        for s in range(config.scenarios_per_app):
            # Scenario-level noise
            p_s = np.clip(
                base_prob + rng.normal(0, config.within_app_noise_scenario),
                0.01, 0.99,
            )
            configs = []
            for i in range(n_i):
                for d in range(n_p):
                    for t in range(n_t):
                        for u in range(n_u):
                            # Config-level noise
                            p_c = np.clip(
                                p_s + rng.normal(0, config.within_app_noise_config),
                                0.01, 0.99,
                            )
                            rollouts = [
                                bool(x)
                                for x in rng.binomial(1, p_c, size=config.n_rollouts)
                            ]
                            instance = f"instance_{i}" if "instance" in config.active_axes else None
                            profile = f"profile_{d}" if "profile" in config.active_axes else None
                            theme = f"theme_{t}" if "theme" in config.active_axes else None
                            ui_state = f"ui_{u}" if "ui_state" in config.active_axes else None
                            configs.append(
                                ConfigurationResult(
                                    instance=instance,
                                    profile=profile,
                                    theme=theme,
                                    ui_state=ui_state,
                                    rollouts=rollouts,
                                )
                            )
            scenarios.append(ScenarioResult(scenario_name=f"scenario_{s}", configurations=configs))
        apps.append(AppResult(app_name=f"app_{a}", scenarios=scenarios))

    suite = EvalSuite(
        name="simulation",
        apps=apps,
        active_axes=config.active_axes,
    )

    true_score = trimmed_mean(config.app_success_probs, 0.1)
    return suite, true_score


def _single_experiment(args):
    """Run one experiment (used by multiprocessing)."""
    sim_config, boot_config, trim_proportion, exp_seed = args

    rng = np.random.default_rng(exp_seed)
    suite, _ = simulate_suite(sim_config, rng=rng)
    true_score = trimmed_mean(sim_config.app_success_probs, trim_proportion)

    if boot_config is None:
        cfg = auto_config(suite, seed=int(rng.integers(0, 2**31)))
    else:
        cfg = BootstrapConfig(
            n_replicates=boot_config.n_replicates,
            confidence_level=boot_config.confidence_level,
            resample_scenarios=boot_config.resample_scenarios,
            resample_instances=boot_config.resample_instances,
            resample_profiles=boot_config.resample_profiles,
            resample_themes=boot_config.resample_themes,
            resample_ui_states=boot_config.resample_ui_states,
            resample_rollouts=boot_config.resample_rollouts,
            seed=int(rng.integers(0, 2**31)),
        )

    result = bootstrap_suite_score(suite, trim_proportion, cfg)
    covered = bool(result.ci.lower <= true_score <= result.ci.upper)
    return covered, float(result.ci.width), float(result.point_estimate - true_score)


def run_coverage_experiment(
    sim_config: SimulationConfig,
    boot_config: Optional[BootstrapConfig] = None,
    n_experiments: int = 100,
    trim_proportion: float = 0.1,
    n_jobs: Optional[int] = None,
) -> Dict[str, float]:
    """Run coverage experiment: generate data, bootstrap, check coverage.

    Args:
        sim_config: Simulation configuration.
        boot_config: Bootstrap configuration. If None, auto-configured.
        n_experiments: Number of independent experiments.
        trim_proportion: Trimmed mean proportion.
        n_jobs: Number of parallel workers. None = all CPUs.

    Returns:
        Dict with keys: coverage, mean_width, mean_bias, rmse.
    """
    rng = np.random.default_rng(sim_config.seed)
    seeds = [int(s) for s in rng.integers(0, 2**31, size=n_experiments)]
    args = [(sim_config, boot_config, trim_proportion, seed) for seed in seeds]

    import multiprocessing as mp
    actual_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
    if actual_jobs > 1:
        try:
            with mp.Pool(actual_jobs) as pool:
                results = pool.map(_single_experiment, args)
        except (PermissionError, OSError):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=actual_jobs) as executor:
                results = list(executor.map(_single_experiment, args))
    else:
        results = [_single_experiment(a) for a in args]

    covers = sum(r[0] for r in results)
    widths = [r[1] for r in results]
    biases_arr = np.array([r[2] for r in results])

    return {
        "coverage": covers / n_experiments,
        "mean_width": float(np.mean(widths)),
        "mean_bias": float(np.mean(biases_arr)),
        "rmse": float(np.sqrt(np.mean(biases_arr**2))),
    }


def run_level_ablation(
    sim_config: SimulationConfig,
    n_experiments: int = 100,
    n_replicates: int = 300,
) -> Dict[str, Dict[str, float]]:
    """Test coverage for different resampling level combinations.

    Returns:
        Dict mapping ablation label -> {coverage, mean_width}.
    """
    has_configs = bool(sim_config.active_axes & {"instance", "profile", "theme", "ui_state"})

    ablations = [
        ("Scen+Config+Roll", True, has_configs, True),
        ("Scen+Config", True, has_configs, False),
        ("Scen+Roll", True, False, True),
        ("Config+Roll", False, has_configs, True),
        ("Scen only", True, False, False),
        ("Config only", False, has_configs, False),
        ("Roll only", False, False, True),
    ]

    results = {}
    for label, rs, rc, rr in ablations:
        cfg = BootstrapConfig(
            n_replicates=n_replicates,
            resample_scenarios=rs,
            resample_instances=rc,
            resample_profiles=rc,
            resample_themes=rc,
            resample_ui_states=rc,
            resample_rollouts=rr,
        )
        result = run_coverage_experiment(sim_config, cfg, n_experiments)
        results[label] = result

    return results


def _single_wald_experiment(args):
    """Run one Wald comparison experiment (used by multiprocessing)."""
    from scipy.stats import norm

    sim_config, confidence_level, exp_seed = args
    rng = np.random.default_rng(exp_seed)
    true_score = trimmed_mean(sim_config.app_success_probs, 0.1)
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    suite, _ = simulate_suite(sim_config, rng=rng)
    cfg = auto_config(suite, n_replicates=300, seed=int(rng.integers(0, 2**31)))

    result = bootstrap_suite_score(suite, 0.1, cfg)
    h_covered = bool(result.ci.lower <= true_score <= result.ci.upper)

    all_outcomes = []
    for app in suite.apps:
        for scenario in app.scenarios:
            for config in scenario.configurations:
                all_outcomes.extend(config.rollouts)
    p_hat = sum(all_outcomes) / len(all_outcomes) if all_outcomes else 0.5
    n = len(all_outcomes)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n)
    w_covered = bool((p_hat - margin) <= true_score <= (p_hat + margin))

    return h_covered, float(result.ci.width), w_covered, float(2 * margin)


def run_wald_comparison(
    sim_config: SimulationConfig,
    n_experiments: int = 100,
    confidence_level: float = 0.95,
    n_jobs: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare hierarchical bootstrap CI with naive Wald CI."""
    rng = np.random.default_rng(sim_config.seed)
    seeds = [int(s) for s in rng.integers(0, 2**31, size=n_experiments)]
    args = [(sim_config, confidence_level, seed) for seed in seeds]

    import multiprocessing as mp
    actual_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
    if actual_jobs > 1:
        try:
            with mp.Pool(actual_jobs) as pool:
                results = pool.map(_single_wald_experiment, args)
        except (PermissionError, OSError):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=actual_jobs) as executor:
                results = list(executor.map(_single_wald_experiment, args))
    else:
        results = [_single_wald_experiment(a) for a in args]

    h_covers = sum(r[0] for r in results)
    h_widths = [r[1] for r in results]
    w_covers = sum(r[2] for r in results)
    w_widths = [r[3] for r in results]

    return {
        "hierarchical": {
            "coverage": h_covers / n_experiments,
            "mean_width": float(np.mean(h_widths)),
        },
        "wald": {
            "coverage": w_covers / n_experiments,
            "mean_width": float(np.mean(w_widths)),
        },
    }


def run_all_simulations(verbose: bool = True) -> Dict[str, Any]:
    """Run all calibration simulations and return results.

    All independent experiment tasks are collected upfront and dispatched
    to a single process pool for maximum parallelism.

    Structure: 15 apps (fixed), 10 scenarios/app, 5×5×5×5 configs (for
    combinatorial sims) or no configs (deterministic sim).
    """
    import multiprocessing as mp

    N_EXP = 100
    N_BOOT = 300
    N_SCEN = 10
    CI_LEVELS = [0.95, 0.80]
    TRIM = 0.1

    # ── Collect all tasks ──────────────────────────────────────────
    # Each task: (key, sim_config, boot_config, trim, seed)
    # key identifies where to store the result
    rng = np.random.default_rng(42)

    all_tasks = []  # list of (key, _single_experiment args)

    # SIM 1: Coverage vs rollouts
    for cl in CI_LEVELS:
        for n_roll in [2, 3, 5, 10]:
            cfg = SimulationConfig(n_rollouts=n_roll, scenarios_per_app=N_SCEN, seed=42)
            boot = BootstrapConfig(n_replicates=N_BOOT, confidence_level=cl)
            seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
            for seed in seeds:
                all_tasks.append(("sim1", cl, n_roll, (cfg, boot, TRIM, seed)))

    # SIM 2: Level ablation
    ablations = [
        ("Scen+Cfg+Roll (full)", True, True, True),
        ("Scen+Cfg", True, True, False),
        ("Scen+Roll", True, False, True),
        ("Cfg+Roll", False, True, True),
        ("Scen only", True, False, False),
        ("Cfg only", False, True, False),
        ("Roll only", False, False, True),
    ]
    for cl in CI_LEVELS:
        for label, rs, rc, rr in ablations:
            boot = BootstrapConfig(
                n_replicates=N_BOOT, confidence_level=cl,
                resample_scenarios=rs, resample_instances=rc,
                resample_profiles=rc, resample_themes=rc,
                resample_ui_states=rc, resample_rollouts=rr,
            )
            cfg = SimulationConfig(n_rollouts=3, scenarios_per_app=N_SCEN, seed=42)
            seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
            for seed in seeds:
                all_tasks.append(("sim2", cl, label, (cfg, boot, TRIM, seed)))

    # SIM 3: Heterogeneous
    for cl in CI_LEVELS:
        for label, rs, rc, rr in [("Full", True, True, True), ("Roll only", False, False, True)]:
            cfg = SimulationConfig(
                n_rollouts=3, scenarios_per_app=N_SCEN,
                within_app_noise_scenario=0.08, within_app_noise_config=0.03, seed=42,
            )
            boot = BootstrapConfig(
                n_replicates=N_BOOT, confidence_level=cl,
                resample_scenarios=rs, resample_instances=rc,
                resample_profiles=rc, resample_themes=rc,
                resample_ui_states=rc, resample_rollouts=rr,
            )
            seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
            for seed in seeds:
                all_tasks.append(("sim3", cl, label, (cfg, boot, TRIM, seed)))

    # SIM 4: Deterministic
    for cl in CI_LEVELS:
        cfg = SimulationConfig(
            n_rollouts=3, scenarios_per_app=20, active_axes=frozenset(), seed=42,
        )
        boot = BootstrapConfig(n_replicates=N_BOOT, confidence_level=cl)
        seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
        for seed in seeds:
            all_tasks.append(("sim4", cl, None, (cfg, boot, TRIM, seed)))

    # SIM 5: Wald comparison (homogeneous)
    cfg_wald = SimulationConfig(n_rollouts=3, scenarios_per_app=N_SCEN, seed=42)
    seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
    for seed in seeds:
        all_tasks.append(("sim5", 0.95, "homogeneous", (cfg_wald, 0.95, seed)))

    # SIM 6: Wald comparison (heterogeneous)
    cfg_wald_het = SimulationConfig(
        n_rollouts=3, scenarios_per_app=N_SCEN,
        within_app_noise_scenario=0.08, within_app_noise_config=0.03, seed=42,
    )
    seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
    for seed in seeds:
        all_tasks.append(("sim6", 0.95, "heterogeneous", (cfg_wald_het, 0.95, seed)))

    if verbose:
        print(f"Dispatching {len(all_tasks)} tasks across {mp.cpu_count()} workers...")

    # ── Run all tasks in parallel ──────────────────────────────────
    def _run_task(task):
        sim_id, cl, sub_key, args = task
        if sim_id in ("sim5", "sim6"):
            return (sim_id, cl, sub_key, _single_wald_experiment(args))
        return (sim_id, cl, sub_key, _single_experiment(args))

    try:
        with mp.Pool(mp.cpu_count()) as pool:
            raw_results = pool.map(_run_task, all_tasks)
    except (PermissionError, OSError):
        # Fallback: use threads (numpy releases GIL for array ops)
        from concurrent.futures import ThreadPoolExecutor
        n_workers = mp.cpu_count()
        if verbose:
            print(f"  (multiprocessing unavailable, using {n_workers} threads)")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            raw_results = list(executor.map(_run_task, all_tasks))

    # ── Aggregate results ──────────────────────────────────────────
    from collections import defaultdict
    buckets = defaultdict(list)
    for sim_id, cl, sub_key, result in raw_results:
        buckets[(sim_id, cl, sub_key)].append(result)

    def _agg(entries):
        covers = sum(r[0] for r in entries)
        widths = [r[1] for r in entries]
        biases = np.array([r[2] for r in entries])
        n = len(entries)
        return {
            "coverage": covers / n,
            "mean_width": float(np.mean(widths)),
            "mean_bias": float(np.mean(biases)),
            "rmse": float(np.sqrt(np.mean(biases**2))),
        }

    results = {}

    # SIM 1
    sim1 = {}
    for cl in CI_LEVELS:
        sim1[cl] = {}
        for n_roll in [2, 3, 5, 10]:
            sim1[cl][n_roll] = _agg(buckets[("sim1", cl, n_roll)])
    results["coverage_vs_rollouts"] = sim1

    # SIM 2
    sim2 = {}
    for cl in CI_LEVELS:
        sim2[cl] = {}
        for label, _, _, _ in ablations:
            sim2[cl][label] = _agg(buckets[("sim2", cl, label)])
    results["level_ablation"] = sim2

    # SIM 3
    sim3 = {}
    for cl in CI_LEVELS:
        sim3[cl] = {}
        for label in ["Full", "Roll only"]:
            sim3[cl][label] = _agg(buckets[("sim3", cl, label)])
    results["heterogeneous"] = sim3

    # SIM 4
    sim4 = {}
    for cl in CI_LEVELS:
        sim4[cl] = _agg(buckets[("sim4", cl, None)])
    results["deterministic"] = sim4

    # SIM 5 & 6: Wald (different result format)
    def _agg_wald(entries):
        n = len(entries)
        return {
            "hierarchical": {
                "coverage": sum(r[0] for r in entries) / n,
                "mean_width": float(np.mean([r[1] for r in entries])),
            },
            "wald": {
                "coverage": sum(r[2] for r in entries) / n,
                "mean_width": float(np.mean([r[3] for r in entries])),
            },
        }
    results["wald_homogeneous"] = _agg_wald(buckets[("sim5", 0.95, "homogeneous")])
    results["wald_heterogeneous"] = _agg_wald(buckets[("sim6", 0.95, "heterogeneous")])

    # ── Print results ──────────────────────────────────────────────
    if verbose:
        print()
        print("=" * 60)
        print(f"SIM 1: Coverage vs. rollouts (15 apps, {N_SCEN} scen, 5x5x5x5 configs)")
        for cl in CI_LEVELS:
            print(f"\n  {int(cl*100)}% CI:")
            print(f"  {'R':<6} {'Coverage':>9} {'Width':>8} {'Bias':>9} {'RMSE':>8}")
            print(f"  {'-'*43}")
            for n_roll in [2, 3, 5, 10]:
                res = sim1[cl][n_roll]
                print(f"  {n_roll:<6} {res['coverage']:>9.3f} {res['mean_width']:>8.3f} "
                      f"{res['mean_bias']:>+9.4f} {res['rmse']:>8.4f}")

        print()
        print("=" * 60)
        print(f"SIM 2: Level ablation (R=3, {N_SCEN} scen)")
        for cl in CI_LEVELS:
            print(f"\n  {int(cl*100)}% CI:")
            print(f"  {'Config':<22} {'Coverage':>9} {'Width':>8}")
            print(f"  {'-'*42}")
            for label, _, _, _ in ablations:
                res = sim2[cl][label]
                print(f"  {label:<22} {res['coverage']:>9.3f} {res['mean_width']:>8.3f}")

        print()
        print("=" * 60)
        print(f"SIM 3: Heterogeneous within-app (R=3, {N_SCEN} scen)")
        for cl in CI_LEVELS:
            print(f"\n  {int(cl*100)}% CI:")
            print(f"  {'Config':<22} {'Coverage':>9} {'Width':>8}")
            print(f"  {'-'*42}")
            for label in ["Full", "Roll only"]:
                res = sim3[cl][label]
                print(f"  {label:<22} {res['coverage']:>9.3f} {res['mean_width']:>8.3f}")

        print()
        print("=" * 60)
        print("SIM 4: Deterministic benchmark (no config axes, 20 scen, R=3)")
        for cl in CI_LEVELS:
            res = sim4[cl]
            print(f"  {int(cl*100)}% CI: coverage={res['coverage']:.3f}, width={res['mean_width']:.3f}")

        print()
        print("=" * 60)
        print(f"SIM 5: Hierarchical vs. Wald — homogeneous ({N_SCEN} scen)")
        print(f"  {'Method':<22} {'Coverage':>9} {'Width':>8}")
        print(f"  {'-'*42}")
        for method, res in results["wald_homogeneous"].items():
            print(f"  {method:<22} {res['coverage']:>9.3f} {res['mean_width']:>8.3f}")

        print()
        print("=" * 60)
        print(f"SIM 6: Hierarchical vs. Wald — heterogeneous ({N_SCEN} scen)")
        print(f"  {'Method':<22} {'Coverage':>9} {'Width':>8}")
        print(f"  {'-'*42}")
        for method, res in results["wald_heterogeneous"].items():
            print(f"  {method:<22} {res['coverage']:>9.3f} {res['mean_width']:>8.3f}")

    return results


if __name__ == "__main__":
    t0 = time.time()
    run_all_simulations(verbose=True)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
