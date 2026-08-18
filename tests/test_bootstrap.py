# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Tests for the hierarchical bootstrap."""

import numpy as np
import pytest

from reliable_cua.bootstrap import (
    BootstrapConfig,
    auto_config,
    bootstrap_per_app,
    bootstrap_suite_score,
    hierarchical_bootstrap,
)
from reliable_cua.metrics import trimmed_mean
from reliable_cua.types import (
    AppResult,
    ConfigurationResult,
    EvalSuite,
    ScenarioResult,
)


def _make_suite(app_probs, n_scenarios=5, n_rollouts=10, active_axes=frozenset()):
    """Helper: create a suite with given per-app success probabilities."""
    rng = np.random.default_rng(42)
    apps = []
    for i, p in enumerate(app_probs):
        scenarios = []
        for s in range(n_scenarios):
            rollouts = [bool(x) for x in rng.binomial(1, p, size=n_rollouts)]
            config = ConfigurationResult(rollouts=rollouts)
            scenarios.append(ScenarioResult(scenario_name=f"s{s}", configurations=[config]))
        apps.append(AppResult(app_name=f"app_{i}", scenarios=scenarios))
    return EvalSuite(name="test", apps=apps, active_axes=active_axes)


class TestAutoConfig:
    def test_no_axes(self):
        suite = EvalSuite(name="t", active_axes=frozenset())
        cfg = auto_config(suite)
        assert cfg.resample_scenarios is True
        assert cfg.resample_rollouts is True
        assert cfg.resample_instances is False
        assert cfg.resample_profiles is False
        assert cfg.resample_themes is False
        assert cfg.resample_ui_states is False

    def test_all_axes(self):
        suite = EvalSuite(
            name="t",
            active_axes=frozenset({"instance", "profile", "theme", "ui_state"}),
        )
        cfg = auto_config(suite)
        assert cfg.resample_instances is True
        assert cfg.resample_profiles is True
        assert cfg.resample_themes is True
        assert cfg.resample_ui_states is True

    def test_overrides(self):
        suite = EvalSuite(name="t", active_axes=frozenset())
        cfg = auto_config(suite, n_replicates=500, resample_scenarios=False)
        assert cfg.n_replicates == 500
        assert cfg.resample_scenarios is False


class TestHierarchicalBootstrap:
    def test_deterministic_with_seed(self):
        suite = _make_suite([0.5, 0.7, 0.3])
        cfg = BootstrapConfig(n_replicates=100, seed=42)
        r1 = bootstrap_suite_score(suite, config=cfg)
        cfg2 = BootstrapConfig(n_replicates=100, seed=42)
        r2 = bootstrap_suite_score(suite, config=cfg2)
        assert r1.point_estimate == r2.point_estimate
        np.testing.assert_array_equal(r1.bootstrap_distribution, r2.bootstrap_distribution)

    def test_ci_brackets_point(self):
        suite = _make_suite([0.5, 0.7, 0.3, 0.8, 0.2])
        result = bootstrap_suite_score(suite, config=BootstrapConfig(n_replicates=500, seed=42))
        assert result.ci.lower <= result.point_estimate <= result.ci.upper

    def test_all_success(self):
        """All apps 100% success -> tight CI around 1.0."""
        suite = _make_suite([1.0, 1.0, 1.0])
        result = bootstrap_suite_score(suite, config=BootstrapConfig(n_replicates=200, seed=42))
        assert result.point_estimate == pytest.approx(1.0)
        assert result.ci.width < 0.01

    def test_single_app(self):
        """Works with a single app."""
        suite = _make_suite([0.5])
        result = bootstrap_suite_score(suite, config=BootstrapConfig(n_replicates=200, seed=42))
        assert 0 < result.point_estimate < 1

    def test_custom_statistic(self):
        suite = _make_suite([0.5, 0.7, 0.3])
        stat = lambda x: float(np.median(x))
        result = hierarchical_bootstrap(
            suite, stat, BootstrapConfig(n_replicates=200, seed=42)
        )
        assert result.ci.lower <= result.point_estimate <= result.ci.upper


class TestBootstrapPerApp:
    def test_returns_all_apps(self):
        suite = _make_suite([0.5, 0.7, 0.3])
        results = bootstrap_per_app(suite, BootstrapConfig(n_replicates=100, seed=42))
        assert set(results.keys()) == {"app_0", "app_1", "app_2"}
        for name, br in results.items():
            assert br.ci.lower <= br.point_estimate <= br.ci.upper
