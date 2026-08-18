# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Basic usage example for reliable-cua.

Demonstrates:
1. Creating an EvalSuite from synthetic data
2. Computing suite scores with bootstrap CIs
3. Per-app analysis
4. Wilson score intervals
5. Performance profiles
6. Variability decomposition
"""

import numpy as np
import reliable_cua as rcua

# ── 1. Generate synthetic evaluation data ────────────────────────────
print("=" * 60)
print("1. Generating synthetic benchmark data")
print("=" * 60)

from reliable_cua.simulation import SimulationConfig, simulate_suite

config = SimulationConfig(
    n_apps=15,
    scenarios_per_app=7,
    n_profiles=3,
    n_themes=3,
    n_ui_states=3,
    n_rollouts=3,
    app_success_probs=np.array([
        0.92, 0.85, 0.78, 0.72, 0.65,
        0.58, 0.52, 0.45, 0.38, 0.32,
        0.25, 0.18, 0.12, 0.08, 0.03,
    ]),
    active_axes=frozenset({"profile", "theme", "ui_state"}),
    seed=42,
)

suite, true_score = simulate_suite(config)
total_rollouts = sum(
    c.num_rollouts
    for a in suite.apps for s in a.scenarios
    for c in s.configurations
)
print(f"  Apps: {suite.num_apps}")
print(f"  Active axes: {suite.active_axes}")
print(f"  Total rollouts: {total_rollouts:,}")
print(f"  True suite score: {true_score:.3f}")

# ── 2. Suite-level score with bootstrap CI ───────────────────────────
print()
print("=" * 60)
print("2. Suite score with hierarchical bootstrap CI")
print("=" * 60)

boot_config = rcua.BootstrapConfig(n_replicates=1000, seed=42)
result = rcua.bootstrap_suite_score(suite, trim_proportion=0.1, config=boot_config)

print(f"  Point estimate: {result.point_estimate:.3f}")
print(f"  95% CI: [{result.ci.lower:.3f}, {result.ci.upper:.3f}]")
print(f"  CI width: {result.ci.width:.3f}")
print(f"  True score in CI: {result.ci.lower <= true_score <= result.ci.upper}")

# ── 3. Per-app analysis ──────────────────────────────────────────────
print()
print("=" * 60)
print("3. Per-app scores with bootstrap CIs")
print("=" * 60)

app_results = rcua.bootstrap_per_app(suite, config=boot_config)
for name, br in sorted(app_results.items(), key=lambda x: x[1].point_estimate, reverse=True):
    print(f"  {name:>8}: {br.point_estimate:.3f}  [{br.ci.lower:.3f}, {br.ci.upper:.3f}]")

# ── 4. Wilson score intervals ────────────────────────────────────────
print()
print("=" * 60)
print("4. Wilson score intervals for individual configurations")
print("=" * 60)

examples = [(0, 3), (1, 3), (2, 3), (3, 3), (5, 10), (50, 100)]
for k, n in examples:
    ci = rcua.wilson_score_interval(k, n)
    print(f"  {k}/{n} successes: center={ci.point:.3f}  [{ci.lower:.3f}, {ci.upper:.3f}]")

# ── 5. Performance profiles ─────────────────────────────────────────
print()
print("=" * 60)
print("5. Performance profile")
print("=" * 60)

tau_list = np.linspace(0, 1, 11)
profile = rcua.performance_profile(suite, tau_list)
for tau, f in zip(tau_list, profile):
    bar = "#" * int(f * 40)
    print(f"  tau={tau:.1f}: F={f:.2f}  {bar}")

# ── 6. Variability decomposition ────────────────────────────────────
print()
print("=" * 60)
print("6. Variability decomposition by axis")
print("=" * 60)

decomps = rcua.decompose_all_axes(suite, threshold=0.10)
for axis, d in decomps.items():
    print(f"  {axis:>10}: MAD={d.mad:.3f}  ATE={d.ate:+.3f}  "
          f"P(deg>{d.threshold})={d.p_degradation:.3f}  ({d.n_pairs} pairs)")

print()
print("Done!")
