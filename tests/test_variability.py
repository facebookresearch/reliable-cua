# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Tests for variability decomposition."""

import numpy as np
import pytest

from reliable_cua.types import (
    AppResult,
    ConfigurationResult,
    EvalSuite,
    MatchedPair,
    ScenarioResult,
)
from reliable_cua.variability import compute_variability, extract_matched_pairs


def _make_suite_with_configs():
    """Create a suite with instance, profile, and theme configs for testing."""
    configs = [
        ConfigurationResult(
            instance="inst1", profile="p1", theme="dark", ui_state="home",
            rollouts=[True, True, True],  # 1.0
        ),
        ConfigurationResult(
            instance="inst1", profile="p1", theme="light", ui_state="home",
            rollouts=[True, False, False],  # 0.333
        ),
        ConfigurationResult(
            instance="inst1", profile="p2", theme="dark", ui_state="home",
            rollouts=[False, False, False],  # 0.0
        ),
        ConfigurationResult(
            instance="inst1", profile="p2", theme="light", ui_state="home",
            rollouts=[True, True, False],  # 0.667
        ),
    ]
    scenario = ScenarioResult(scenario_name="s1", configurations=configs)
    app = AppResult(app_name="app1", scenarios=[scenario])
    return EvalSuite(
        name="test", apps=[app],
        active_axes=frozenset({"instance", "profile", "theme", "ui_state"}),
    )


class TestExtractMatchedPairs:
    def test_theme_pairs(self):
        suite = _make_suite_with_configs()
        pairs = extract_matched_pairs(suite, "theme")
        # Should find 2 pairs: (inst1,p1,home) dark vs light, (inst1,p2,home) dark vs light
        assert len(pairs) == 2
        for p in pairs:
            assert p.axis == "theme"
            assert {p.value_a, p.value_b} == {"dark", "light"}

    def test_profile_pairs(self):
        suite = _make_suite_with_configs()
        pairs = extract_matched_pairs(suite, "profile")
        # Should find 2 pairs: (inst1,dark,home) p1 vs p2, (inst1,light,home) p1 vs p2
        assert len(pairs) == 2
        for p in pairs:
            assert {p.value_a, p.value_b} == {"p1", "p2"}

    def test_no_pairs_single_value(self):
        suite = _make_suite_with_configs()
        pairs = extract_matched_pairs(suite, "ui_state")
        # All configs have ui_state="home", so no pairs
        assert len(pairs) == 0

    def test_instance_pairs_single_value(self):
        suite = _make_suite_with_configs()
        pairs = extract_matched_pairs(suite, "instance")
        # All configs have instance="inst1", so no pairs
        assert len(pairs) == 0

    def test_instance_pairs_multiple_values(self):
        """Instance decomposition works when multiple instances exist."""
        configs = [
            ConfigurationResult(
                instance="inst1", profile="p1", theme="dark", ui_state="home",
                rollouts=[True, True, True],  # 1.0
            ),
            ConfigurationResult(
                instance="inst2", profile="p1", theme="dark", ui_state="home",
                rollouts=[False, False, False],  # 0.0
            ),
        ]
        scenario = ScenarioResult(scenario_name="s1", configurations=configs)
        app = AppResult(app_name="app1", scenarios=[scenario])
        suite = EvalSuite(
            name="test", apps=[app],
            active_axes=frozenset({"instance", "profile", "theme", "ui_state"}),
        )
        pairs = extract_matched_pairs(suite, "instance")
        assert len(pairs) == 1
        assert {pairs[0].value_a, pairs[0].value_b} == {"inst1", "inst2"}

    def test_invalid_axis(self):
        suite = _make_suite_with_configs()
        with pytest.raises(ValueError):
            extract_matched_pairs(suite, "invalid")


class TestComputeVariability:
    def test_basic(self):
        pairs = [
            MatchedPair("theme", "a", "b", 0.8, 0.6, "app", "s", "i"),
            MatchedPair("theme", "a", "b", 0.5, 0.5, "app", "s", "i"),
        ]
        result = compute_variability(pairs, threshold=0.10)
        assert result.axis == "theme"
        # deltas = [0.2, 0.0], abs = [0.2, 0.0]
        assert result.mad == pytest.approx(0.1)
        assert result.ate == pytest.approx(0.1)
        # 0.2 > 0.10 = True, 0.0 > 0.10 = False -> 0.5
        assert result.p_degradation == pytest.approx(0.5)
        assert result.n_pairs == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_variability([], threshold=0.10)
