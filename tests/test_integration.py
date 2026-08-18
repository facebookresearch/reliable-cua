# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Integration tests: end-to-end coverage validation."""

import numpy as np
import pytest

from reliable_cua.bootstrap import BootstrapConfig, auto_config, bootstrap_suite_score
from reliable_cua.metrics import trimmed_mean
from reliable_cua.simulation import SimulationConfig, simulate_suite


class TestCoverageCalibration:
    """Small-scale coverage test to verify the bootstrap is not obviously broken.

    Uses fewer experiments/replicates than the full simulation for speed.
    """

    def test_homogeneous_coverage(self):
        """Full hierarchical bootstrap should achieve >= 85% coverage (conservative)."""
        rng = np.random.default_rng(42)
        sim_cfg = SimulationConfig(
            n_apps=10,
            scenarios_per_app=10,
            n_profiles=2,
            n_themes=2,
            n_ui_states=2,
            n_rollouts=3,
            app_success_probs=np.linspace(0.1, 0.9, 10),
            seed=42,
        )
        true_score = trimmed_mean(sim_cfg.app_success_probs, 0.1)

        n_experiments = 50
        covers = 0
        for _ in range(n_experiments):
            suite, _ = simulate_suite(sim_cfg, rng=rng)
            cfg = auto_config(suite, n_replicates=100, seed=int(rng.integers(0, 2**31)))
            result = bootstrap_suite_score(suite, 0.1, cfg)
            if result.ci.lower <= true_score <= result.ci.upper:
                covers += 1

        coverage = covers / n_experiments
        assert coverage >= 0.85, f"Coverage {coverage:.2f} < 0.85"

    def test_deterministic_coverage(self):
        """Deterministic benchmark (no config axes) should also cover."""
        rng = np.random.default_rng(123)
        sim_cfg = SimulationConfig(
            n_apps=10,
            scenarios_per_app=10,
            n_rollouts=5,
            active_axes=frozenset(),
            app_success_probs=np.linspace(0.1, 0.9, 10),
            seed=123,
        )
        true_score = trimmed_mean(sim_cfg.app_success_probs, 0.1)

        n_experiments = 50
        covers = 0
        for _ in range(n_experiments):
            suite, _ = simulate_suite(sim_cfg, rng=rng)
            cfg = auto_config(suite, n_replicates=100, seed=int(rng.integers(0, 2**31)))
            result = bootstrap_suite_score(suite, 0.1, cfg)
            if result.ci.lower <= true_score <= result.ci.upper:
                covers += 1

        coverage = covers / n_experiments
        assert coverage >= 0.85, f"Coverage {coverage:.2f} < 0.85"
