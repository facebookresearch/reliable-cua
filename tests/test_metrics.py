# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Tests for aggregation metrics."""

import numpy as np
import pytest

from reliable_cua.metrics import (
    _app_success_rate,
    aggregate_mean,
    aggregate_median,
    per_app_scores,
    per_scenario_scores,
    suite_score,
    trimmed_mean,
    wald_suite_ci,
)
from reliable_cua.types import (
    AppResult,
    ConfigurationResult,
    EvalSuite,
    ScenarioResult,
)


def _make_app(name, rollouts_list):
    """Helper: create an AppResult with one scenario per rollout list."""
    scenarios = []
    for i, rollouts in enumerate(rollouts_list):
        config = ConfigurationResult(rollouts=rollouts)
        scenarios.append(ScenarioResult(scenario_name=f"s{i}", configurations=[config]))
    return AppResult(app_name=name, scenarios=scenarios)


class TestTrimmedMean:
    def test_basic(self):
        # [1,2,3,4,5] with trim=0.2 -> discard 1 from each tail -> mean(2,3,4)=3
        assert trimmed_mean(np.array([1, 2, 3, 4, 5]), 0.2) == 3.0

    def test_no_trim(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert trimmed_mean(arr, 0.0) == pytest.approx(2.0)

    def test_ten_percent(self):
        # 15 elements, trim 10% = 1.5 from each side -> discard ~1 from each
        arr = np.linspace(0.05, 0.95, 15)
        result = trimmed_mean(arr, 0.1)
        # Should be close to 0.5 (symmetric distribution)
        assert abs(result - 0.5) < 0.01


class TestAppSuccessRate:
    def test_simple(self):
        app = _make_app("a", [[True, True, False]])
        assert _app_success_rate(app) == pytest.approx(2 / 3)

    def test_multiple_scenarios(self):
        app = _make_app("a", [[True, True], [False, False]])
        assert _app_success_rate(app) == pytest.approx(0.5)

    def test_empty(self):
        app = _make_app("a", [[]])
        assert np.isnan(_app_success_rate(app))


class TestSuiteScore:
    def test_basic(self):
        apps = [
            _make_app("a", [[True, True, True]]),  # 1.0
            _make_app("b", [[False, False, False]]),  # 0.0
            _make_app("c", [[True, False]]),  # 0.5
        ]
        suite = EvalSuite(name="test", apps=apps)
        score = suite_score(suite, trim_proportion=0.0)
        assert score == pytest.approx(0.5)

    def test_per_app(self):
        apps = [
            _make_app("a", [[True, True]]),
            _make_app("b", [[True, False]]),
        ]
        suite = EvalSuite(name="test", apps=apps)
        scores = per_app_scores(suite)
        assert scores["a"] == pytest.approx(1.0)
        assert scores["b"] == pytest.approx(0.5)

    def test_per_scenario(self):
        apps = [
            _make_app("a", [[True, True], [False, False]]),
        ]
        suite = EvalSuite(name="test", apps=apps)
        scores = per_scenario_scores(suite)
        assert scores[("a", "s0")] == pytest.approx(1.0)
        assert scores[("a", "s1")] == pytest.approx(0.0)


class TestWaldSuiteCi:
    def test_basic(self):
        # 2 apps: app_a has 8/10 successes, app_b has 2/10 successes
        # Flat: 10/20 = 0.5
        apps = [
            _make_app("a", [[True] * 8 + [False] * 2]),
            _make_app("b", [[True] * 2 + [False] * 8]),
        ]
        suite = EvalSuite(name="test", apps=apps)
        result = wald_suite_ci(suite, confidence_level=0.95)
        assert result.point == pytest.approx(0.5)
        # Wald SE = sqrt(0.5 * 0.5 / 20) ≈ 0.1118
        # 95% CI: 0.5 ± 1.96 * 0.1118 ≈ [0.281, 0.719]
        assert result.lower == pytest.approx(0.281, abs=0.01)
        assert result.upper == pytest.approx(0.719, abs=0.01)
        assert result.confidence_level == 0.95

    def test_ignores_hierarchy(self):
        # Same total successes/rollouts, different hierarchy -> same CI
        apps_flat = [
            _make_app("a", [[True] * 10 + [False] * 10]),
        ]
        apps_split = [
            _make_app("a", [[True] * 10]),
            _make_app("b", [[False] * 10]),
        ]
        r1 = wald_suite_ci(EvalSuite(name="t1", apps=apps_flat))
        r2 = wald_suite_ci(EvalSuite(name="t2", apps=apps_split))
        assert r1.point == pytest.approx(r2.point)
        assert r1.width == pytest.approx(r2.width)

    def test_extreme_rates_clamped(self):
        # All successes -> CI upper clamped to 1.0
        apps = [_make_app("a", [[True] * 5])]
        result = wald_suite_ci(EvalSuite(name="t", apps=apps))
        assert result.point == pytest.approx(1.0)
        assert result.upper <= 1.0
        assert result.lower >= 0.0

    def test_empty_suite(self):
        suite = EvalSuite(name="empty", apps=[])
        result = wald_suite_ci(suite)
        assert np.isnan(result.point)

    def test_80_percent_ci_narrower(self):
        apps = [
            _make_app("a", [[True] * 5 + [False] * 5]),
            _make_app("b", [[True] * 3 + [False] * 7]),
        ]
        suite = EvalSuite(name="test", apps=apps)
        ci_95 = wald_suite_ci(suite, confidence_level=0.95)
        ci_80 = wald_suite_ci(suite, confidence_level=0.80)
        assert ci_80.width < ci_95.width
