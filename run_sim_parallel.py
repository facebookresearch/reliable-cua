# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Run all simulations via subprocess-based parallelism.

Splits the 2900 tasks into N worker subprocesses (bypasses GIL and
sandbox restrictions on multiprocessing.Pool semaphores).
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

N_WORKERS = os.cpu_count() or 4


def main():
    t0 = time.time()

    # Step 1: Generate all tasks in the parent process
    print("Generating tasks...")
    gen_script = """
import json, sys
import numpy as np
from reliable_cua.simulation import SimulationConfig
from reliable_cua.bootstrap import BootstrapConfig

N_EXP = 100
N_BOOT = 300
N_SCEN = 10
TRIM = 0.1
CI_LEVELS = [0.95, 0.80]

rng = np.random.default_rng(42)
tasks = []

# SIM 1
for cl in CI_LEVELS:
    for n_roll in [2, 3, 5, 10]:
        seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
        for seed in seeds:
            tasks.append({"sim": "sim1", "cl": cl, "key": n_roll,
                          "sim_cfg": {"n_rollouts": n_roll, "scenarios_per_app": N_SCEN, "seed": 42},
                          "boot_cfg": {"n_replicates": N_BOOT, "confidence_level": cl},
                          "trim": TRIM, "seed": seed, "type": "coverage"})

# SIM 2
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
        seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
        for seed in seeds:
            tasks.append({"sim": "sim2", "cl": cl, "key": label,
                          "sim_cfg": {"n_rollouts": 3, "scenarios_per_app": N_SCEN, "seed": 42},
                          "boot_cfg": {"n_replicates": N_BOOT, "confidence_level": cl,
                                       "resample_scenarios": rs, "resample_instances": rc,
                                       "resample_profiles": rc, "resample_themes": rc,
                                       "resample_ui_states": rc, "resample_rollouts": rr},
                          "trim": TRIM, "seed": seed, "type": "coverage"})

# SIM 3
for cl in CI_LEVELS:
    for label, rs, rc, rr in [("Full", True, True, True), ("Roll only", False, False, True)]:
        seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
        for seed in seeds:
            tasks.append({"sim": "sim3", "cl": cl, "key": label,
                          "sim_cfg": {"n_rollouts": 3, "scenarios_per_app": N_SCEN,
                                      "within_app_noise_scenario": 0.08, "within_app_noise_config": 0.03, "seed": 42},
                          "boot_cfg": {"n_replicates": N_BOOT, "confidence_level": cl,
                                       "resample_scenarios": rs, "resample_instances": rc,
                                       "resample_profiles": rc, "resample_themes": rc,
                                       "resample_ui_states": rc, "resample_rollouts": rr},
                          "trim": TRIM, "seed": seed, "type": "coverage"})

# SIM 4
for cl in CI_LEVELS:
    seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
    for seed in seeds:
        tasks.append({"sim": "sim4", "cl": cl, "key": None,
                      "sim_cfg": {"n_rollouts": 3, "scenarios_per_app": 20, "active_axes": [], "seed": 42},
                      "boot_cfg": {"n_replicates": N_BOOT, "confidence_level": cl},
                      "trim": TRIM, "seed": seed, "type": "coverage"})

# SIM 5: Wald homogeneous
seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
for seed in seeds:
    tasks.append({"sim": "sim5", "cl": 0.95, "key": "homogeneous",
                  "sim_cfg": {"n_rollouts": 3, "scenarios_per_app": N_SCEN, "seed": 42},
                  "seed": seed, "type": "wald"})

# SIM 6: Wald heterogeneous
seeds = [int(s) for s in rng.integers(0, 2**31, size=N_EXP)]
for seed in seeds:
    tasks.append({"sim": "sim6", "cl": 0.95, "key": "heterogeneous",
                  "sim_cfg": {"n_rollouts": 3, "scenarios_per_app": N_SCEN,
                              "within_app_noise_scenario": 0.08, "within_app_noise_config": 0.03, "seed": 42},
                  "seed": seed, "type": "wald"})

json.dump(tasks, sys.stdout)
"""
    result = subprocess.run(
        [sys.executable, "-c", gen_script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"Task generation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    tasks = json.loads(result.stdout)
    print(f"Generated {len(tasks)} tasks, dispatching to {N_WORKERS} workers...")

    # Step 2: Write task chunks to temp files
    chunk_size = (len(tasks) + N_WORKERS - 1) // N_WORKERS
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    worker_script = '''
import json, sys
import numpy as np
from reliable_cua.simulation import SimulationConfig, _single_experiment, _single_wald_experiment
from reliable_cua.bootstrap import BootstrapConfig

tasks = json.load(sys.stdin)
results = []
for t in tasks:
    sc = t["sim_cfg"]
    active = frozenset(sc.pop("active_axes")) if "active_axes" in sc else None
    sim_cfg = SimulationConfig(**sc) if active is None else SimulationConfig(**sc, active_axes=active)

    if t["type"] == "wald":
        r = _single_wald_experiment((sim_cfg, t["cl"], t["seed"]))
        results.append({"sim": t["sim"], "cl": t["cl"], "key": t["key"], "result": list(r)})
    else:
        boot_cfg = BootstrapConfig(**t["boot_cfg"])
        r = _single_experiment((sim_cfg, boot_cfg, t["trim"], t["seed"]))
        results.append({"sim": t["sim"], "cl": t["cl"], "key": t["key"], "result": list(r)})

json.dump(results, sys.stdout)
'''

    # Step 3: Launch workers
    procs = []
    inputs = []
    for i, chunk in enumerate(chunks):
        p = subprocess.Popen(
            [sys.executable, "-c", worker_script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        procs.append(p)
        inputs.append(json.dumps(chunk))
        print(f"  Worker {i}: {len(chunk)} tasks")

    # Step 4: Send input and collect results
    import concurrent.futures
    def _run_worker(args):
        i, p, inp = args
        stdout, stderr = p.communicate(input=inp, timeout=1800)
        return i, p.returncode, stdout, stderr

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(procs)) as ex:
        futures = [ex.submit(_run_worker, (i, p, inp)) for i, (p, inp) in enumerate(zip(procs, inputs))]
        for f in concurrent.futures.as_completed(futures):
            i, rc, stdout, stderr = f.result()
            if rc != 0:
                print(f"Worker {i} failed:\n{stderr}", file=sys.stderr)
                sys.exit(1)
            all_results.extend(json.loads(stdout))
            print(f"  Worker {i} done")

    elapsed = time.time() - t0
    print(f"\nAll workers done in {elapsed:.1f}s")

    # Step 5: Aggregate and print
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in all_results:
        buckets[(r["sim"], r["cl"], str(r["key"]))].append(tuple(r["result"]))

    def _agg(entries):
        n = len(entries)
        covers = sum(r[0] for r in entries)
        widths = [r[1] for r in entries]
        biases = [r[2] for r in entries]
        return {
            "coverage": covers / n,
            "mean_width": sum(widths) / n,
            "mean_bias": sum(biases) / n,
        }

    N_SCEN = 10
    CI_LEVELS = [0.95, 0.80]

    print()
    print("=" * 60)
    print(f"SIM 1: Coverage vs. rollouts (15 apps, {N_SCEN} scen, 5x5x5x5 configs)")
    for cl in CI_LEVELS:
        print(f"\n  {int(cl*100)}% CI:")
        print(f"  {'R':<6} {'Coverage':>9} {'Width':>8} {'Bias':>9}")
        print(f"  {'-'*35}")
        for n_roll in [2, 3, 5, 10]:
            res = _agg(buckets[("sim1", cl, str(n_roll))])
            print(f"  {n_roll:<6} {res['coverage']:>9.3f} {res['mean_width']:>8.3f} {res['mean_bias']:>+9.4f}")

    print()
    print("=" * 60)
    print(f"SIM 2: Level ablation (R=3, {N_SCEN} scen)")
    ablation_labels = ["Scen+Cfg+Roll (full)", "Scen+Cfg", "Scen+Roll", "Cfg+Roll", "Scen only", "Cfg only", "Roll only"]
    for cl in CI_LEVELS:
        print(f"\n  {int(cl*100)}% CI:")
        print(f"  {'Config':<22} {'Coverage':>9} {'Width':>8}")
        print(f"  {'-'*42}")
        for label in ablation_labels:
            res = _agg(buckets[("sim2", cl, label)])
            print(f"  {label:<22} {res['coverage']:>9.3f} {res['mean_width']:>8.3f}")

    print()
    print("=" * 60)
    print(f"SIM 3: Heterogeneous within-app (R=3, {N_SCEN} scen)")
    for cl in CI_LEVELS:
        print(f"\n  {int(cl*100)}% CI:")
        print(f"  {'Config':<22} {'Coverage':>9} {'Width':>8}")
        print(f"  {'-'*42}")
        for label in ["Full", "Roll only"]:
            res = _agg(buckets[("sim3", cl, label)])
            print(f"  {label:<22} {res['coverage']:>9.3f} {res['mean_width']:>8.3f}")

    print()
    print("=" * 60)
    print("SIM 4: Deterministic benchmark (no config axes, 20 scen, R=3)")
    for cl in CI_LEVELS:
        res = _agg(buckets[("sim4", cl, "None")])
        print(f"  {int(cl*100)}% CI: coverage={res['coverage']:.3f}, width={res['mean_width']:.3f}")

    def _print_wald(label, bucket_key):
        print()
        print("=" * 60)
        print(f"{label} ({N_SCEN} scen)")
        print(f"  {'Method':<22} {'Coverage':>9} {'Width':>8}")
        print(f"  {'-'*42}")
        entries = buckets[bucket_key]
        n = len(entries)
        h_cov = sum(r[0] for r in entries) / n
        h_w = sum(r[1] for r in entries) / n
        w_cov = sum(r[2] for r in entries) / n
        w_w = sum(r[3] for r in entries) / n
        print(f"  {'hierarchical':<22} {h_cov:>9.3f} {h_w:>8.3f}")
        print(f"  {'wald':<22} {w_cov:>9.3f} {w_w:>8.3f}")

    _print_wald("SIM 5: Hierarchical vs. Wald — homogeneous", ("sim5", 0.95, "homogeneous"))
    _print_wald("SIM 6: Hierarchical vs. Wald — heterogeneous", ("sim6", 0.95, "heterogeneous"))

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
