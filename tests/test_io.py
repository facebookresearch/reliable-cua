# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Tests for data loading utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from reliable_cua.io import from_arrays, from_dict, load_jsonl, parse_task_id


class TestParseTaskId:
    def test_basic(self):
        app, scen, inst = parse_task_id("banking__view_transaction__deposit_1")
        assert app == "banking"
        assert scen == "view_transaction"
        assert inst == "deposit_1"

    def test_instance_with_separator(self):
        """Instance ID may contain the separator."""
        app, scen, inst = parse_task_id("email__send__personal__fjohnson__4")
        assert app == "email"
        assert scen == "send"
        assert inst == "personal__fjohnson__4"

    def test_too_few_parts(self):
        with pytest.raises(ValueError):
            parse_task_id("banking__only_two")


class TestFromDict:
    def test_round_trip(self):
        data = {
            "banking": {
                "transfer": {
                    "inst_1|default|None|None": [True, True, False],
                }
            }
        }
        suite = from_dict(data, name="test")
        assert suite.num_apps == 1
        app = suite.apps[0]
        assert app.app_name == "banking"
        assert len(app.scenarios) == 1
        cfg = app.scenarios[0].configurations[0]
        assert cfg.rollouts == [True, True, False]
        assert cfg.instance == "inst_1"
        assert cfg.profile == "default"

    def test_multi_config(self):
        data = {
            "app": {
                "scen": {
                    "i1|p1|dark|home": [True, False],
                    "i1|p1|light|home": [True, True],
                }
            }
        }
        suite = from_dict(
            data, active_axes=frozenset({"instance", "profile", "theme", "ui_state"})
        )
        configs = suite.apps[0].scenarios[0].configurations
        assert len(configs) == 2


class TestFromArrays:
    def test_basic(self):
        import numpy as np

        suite = from_arrays(
            app_names=["a", "b"],
            scenario_names_per_app={
                "a": ["s0", "s1"],
                "b": ["s0"],
            },
            outcomes_per_app={
                "a": np.array([[True, False], [True, True]]),
                "b": np.array([[False, False]]),
            },
        )
        assert suite.num_apps == 2
        assert suite.active_axes == frozenset()


class TestLoadJsonl:
    def test_flat_format(self, tmp_path):
        p = tmp_path / "results.jsonl"
        lines = [
            {"task_id": "app1__scen1__inst1", "pass": True},
            {"task_id": "app1__scen1__inst1", "pass": False},
            {"task_id": "app1__scen2__inst1", "pass": True},
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines))
        suite = load_jsonl(p)
        assert suite.num_apps == 1
        s0 = suite.apps[0].scenarios[0]
        assert s0.configurations[0].num_rollouts == 2

    def test_amaia_format(self, tmp_path):
        p = tmp_path / "all_metrics.jsonl"
        lines = [
            {
                "task_name": "env:reward:path;app1__scen1__inst1",
                "metrics": [{"terminal_metrics": {"pass": True}}],
            },
            {
                "task_name": "env:reward:path;app1__scen1__inst1",
                "metrics": [{"terminal_metrics": {"pass": False}}],
            },
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines))
        suite = load_jsonl(p)
        assert suite.num_apps == 1
        cfg = suite.apps[0].scenarios[0].configurations[0]
        assert cfg.rollouts == [True, False]

    def test_amaia_format_with_variation_fields(self, tmp_path):
        """Variation fields (profile, theme, ui_state) are parsed from terminal_metrics."""
        p = tmp_path / "all_metrics.jsonl"
        lines = [
            {
                "task_name": "env:reward:path;banking__check_balance__high_balance_0",
                "metrics": [{"terminal_metrics": {
                    "pass": True,
                    "profile": "test-profile-1",
                    "theme": "midnight",
                    "ui_state": "contacts_var0",
                }}],
            },
            {
                "task_name": "env:reward:path;banking__check_balance__high_balance_0",
                "metrics": [{"terminal_metrics": {
                    "pass": False,
                    "profile": "test-profile-1",
                    "theme": "midnight",
                    "ui_state": "contacts_var0",
                }}],
            },
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines))
        suite = load_jsonl(p)
        assert suite.active_axes == frozenset({"profile", "theme", "ui_state"})
        assert suite.num_apps == 1
        cfg = suite.apps[0].scenarios[0].configurations[0]
        assert cfg.instance == "high_balance_0"
        assert cfg.profile == "test-profile-1"
        assert cfg.theme == "midnight"
        assert cfg.ui_state == "contacts_var0"
        assert cfg.rollouts == [True, False]

    def test_amaia_format_without_variation_fields(self, tmp_path):
        """Backward compat: amaia-collab entries without variation fields still work."""
        p = tmp_path / "all_metrics.jsonl"
        lines = [
            {
                "task_name": "env:reward:path;app1__scen1__inst1",
                "metrics": [{"terminal_metrics": {"pass": True}}],
            },
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines))
        suite = load_jsonl(p)
        assert suite.active_axes == frozenset()
        cfg = suite.apps[0].scenarios[0].configurations[0]
        assert cfg.instance == "inst1"
        assert cfg.profile is None
        assert cfg.theme is None
        assert cfg.ui_state is None
        assert cfg.rollouts == [True]

    def test_instance_axis_auto_detected(self, tmp_path):
        """Instance axis is auto-detected when multiple instances per scenario exist."""
        p = tmp_path / "results.jsonl"
        lines = [
            {"task_id": "app1__scen1__inst1", "pass": True},
            {"task_id": "app1__scen1__inst2", "pass": False},
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines))
        suite = load_jsonl(p)
        assert "instance" in suite.active_axes
