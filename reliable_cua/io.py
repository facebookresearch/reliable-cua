# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Data loading utilities for CUA evaluation results.

Supports loading from:
- amaia-collab JSONL output (all_metrics.jsonl)
- Flat JSONL with explicit fields
- Nested Python dicts
- Simple numpy arrays (rollouts-only mode)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np

from .types import (
    AppResult,
    ConfigurationResult,
    EvalSuite,
    ScenarioResult,
)


def parse_task_id(task_id: str, separator: str = "__") -> Tuple[str, str, str]:
    """Parse a task_id into (app_name, scenario_name, instance_id).

    Args:
        task_id: e.g. "banking__view_transaction_details__deposit_1"
            or "banking__view_transaction_details" (no instance).
        separator: Delimiter between components (default "__").

    Returns:
        Tuple of (app_name, scenario_name, instance_id).
        instance_id is None if not present.

    Raises:
        ValueError: If task_id does not contain at least 2 components.
    """
    parts = task_id.split(separator)
    if len(parts) < 2:
        raise ValueError(
            f"task_id must have at least 2 '{separator}'-separated parts, "
            f"got {len(parts)}: {task_id!r}"
        )
    app_name = parts[0]
    scenario_name = parts[1]
    # If more than 2 parts, the instance_id may contain the separator
    instance_id = separator.join(parts[2:]) if len(parts) > 2 else None
    return app_name, scenario_name, instance_id


def _extract_task_id_from_task_name(task_name: str) -> str:
    """Extract task_id from amaia-collab task_name format.

    Format: "env_config:reward_fn:path;task_id"
    """
    if ";" in task_name:
        return task_name.rsplit(";", 1)[-1]
    return task_name


def load_jsonl(
    path: Union[str, Path],
    *,
    active_axes: Optional[FrozenSet[str]] = None,
    task_id_field: str = "task_id",
    success_field: str = "pass",
    name: Optional[str] = None,
) -> EvalSuite:
    """Load evaluation results from a JSONL file.

    Supports two formats:

    **amaia-collab format** (auto-detected by presence of ``task_name``)::

        {"task_name": "env:reward:path;banking__view__dep_1",
         "metrics": [{"terminal_metrics": {"pass": true}}]}

    **Flat format**::

        {"task_id": "banking__view__dep_1", "pass": true,
         "profile": "default", "theme": "dark", "ui_state": "home"}

    Args:
        path: Path to the JSONL file.
        active_axes: Which environmental axes are active. If None, inferred
            from the presence of instance/profile/theme/ui_state fields.
        task_id_field: Field name for task ID in flat format.
        success_field: Field name for success outcome.
        name: Name for the EvalSuite. Defaults to filename.

    Returns:
        Populated EvalSuite.
    """
    path = Path(path)
    if name is None:
        name = path.stem

    # Parse all records
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Detect format
            if "task_name" in obj and "metrics" in obj:
                # amaia-collab format
                task_id = _extract_task_id_from_task_name(obj["task_name"])
                metrics = obj["metrics"]
                if isinstance(metrics, list) and len(metrics) > 0:
                    tm = metrics[0].get("terminal_metrics", {})
                else:
                    tm = {}
                success = bool(tm.get(success_field, False))
                records.append({
                    "task_id": task_id,
                    "success": success,
                    "instance": tm.get("instance"),
                    "profile": tm.get("profile"),
                    "theme": tm.get("theme"),
                    "ui_state": tm.get("ui_state"),
                })
            else:
                # Flat format
                task_id = obj.get(task_id_field, "")
                success = bool(obj.get(success_field, False))
                record = {
                    "task_id": task_id,
                    "success": success,
                    "instance": obj.get("instance"),
                    "profile": obj.get("profile"),
                    "theme": obj.get("theme"),
                    "ui_state": obj.get("ui_state"),
                }
                records.append(record)

    return _build_suite_from_records(records, active_axes=active_axes, name=name)


def _build_suite_from_records(
    records: List[Dict[str, Any]],
    active_axes: Optional[FrozenSet[str]],
    name: str,
) -> EvalSuite:
    """Build an EvalSuite from a list of parsed records."""
    # Infer active axes if not provided
    if active_axes is None:
        has_profile = any(r.get("profile") is not None for r in records)
        has_theme = any(r.get("theme") is not None for r in records)
        has_ui_state = any(r.get("ui_state") is not None for r in records)
        axes = set()
        if has_profile:
            axes.add("profile")
        if has_theme:
            axes.add("theme")
        if has_ui_state:
            axes.add("ui_state")
        # Infer instance axis: check if any scenario has multiple instances
        instances_per_scenario: Dict[Tuple[str, str], set] = defaultdict(set)
        for record in records:
            try:
                app, scenario, instance = parse_task_id(record["task_id"])
                # Prefer instance from record over task_id parse
                if record.get("instance") is not None:
                    instance = record["instance"]
                if instance is not None:
                    instances_per_scenario[(app, scenario)].add(instance)
            except ValueError:
                continue
        if any(len(insts) > 1 for insts in instances_per_scenario.values()):
            axes.add("instance")
        active_axes = frozenset(axes)

    # Group: app -> scenario -> config_key -> [outcomes]
    tree: Dict[str, Dict[str, Dict[str, List[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for record in records:
        task_id = record["task_id"]
        try:
            app, scenario, instance = parse_task_id(task_id)
        except ValueError:
            continue  # Skip malformed task_ids

        # Prefer instance from record (terminal_metrics) over task_id parse
        if record.get("instance") is not None:
            instance = record["instance"]

        profile = record.get("profile")
        theme = record.get("theme")
        ui_state = record.get("ui_state")
        config_key = f"{instance}|{profile}|{theme}|{ui_state}"

        tree[app][scenario][config_key].append(record["success"])

    return _tree_to_suite(tree, active_axes, name)


def _tree_to_suite(
    tree: Dict[str, Dict[str, Dict[str, List[bool]]]],
    active_axes: FrozenSet[str],
    name: str,
) -> EvalSuite:
    """Convert a nested dict tree to an EvalSuite."""
    apps = []
    for app_name in sorted(tree):
        scenarios = []
        for scenario_name in sorted(tree[app_name]):
            configs = []
            for config_key, rollouts in tree[app_name][scenario_name].items():
                parts = config_key.split("|", 3)
                instance = parts[0] if parts[0] != "None" else None
                profile = parts[1] if len(parts) > 1 and parts[1] != "None" else None
                theme = parts[2] if len(parts) > 2 and parts[2] != "None" else None
                ui_state = parts[3] if len(parts) > 3 and parts[3] != "None" else None
                configs.append(
                    ConfigurationResult(
                        instance=instance,
                        profile=profile,
                        theme=theme,
                        ui_state=ui_state,
                        rollouts=rollouts,
                    )
                )
            scenarios.append(ScenarioResult(scenario_name=scenario_name, configurations=configs))
        apps.append(AppResult(app_name=app_name, scenarios=scenarios))

    return EvalSuite(name=name, apps=apps, active_axes=active_axes)


def from_dict(
    data: Dict[str, Dict[str, Dict[str, List[bool]]]],
    *,
    active_axes: Optional[FrozenSet[str]] = None,
    name: str = "eval",
) -> EvalSuite:
    """Create EvalSuite from a nested dict.

    Structure::

        {app: {scenario: {config_key: [bool, ...]}}}

    ``config_key`` format: ``"instance|profile|theme|ui_state"`` (use
    ``"None"`` for inactive axes).

    Args:
        data: Nested dict of evaluation results.
        active_axes: Which axes are active. If None, defaults to empty.
        name: Suite name.

    Returns:
        Populated EvalSuite.
    """
    if active_axes is None:
        active_axes = frozenset()
    return _tree_to_suite(data, active_axes, name)


def from_arrays(
    app_names: List[str],
    scenario_names_per_app: Dict[str, List[str]],
    outcomes_per_app: Dict[str, np.ndarray],
    *,
    name: str = "eval",
) -> EvalSuite:
    """Create EvalSuite from arrays (simplest case: no configs).

    For the deterministic benchmark case where there are only
    apps -> scenarios -> rollouts, with no environmental axes.

    Args:
        app_names: List of app names.
        scenario_names_per_app: Dict mapping app_name -> list of scenario names.
        outcomes_per_app: Dict mapping app_name -> 2-D bool array of shape
            (n_scenarios, n_rollouts).
        name: Suite name.

    Returns:
        EvalSuite with active_axes=frozenset() (no environmental axes).
    """
    apps = []
    for app_name in app_names:
        scenarios_list = scenario_names_per_app[app_name]
        outcomes = outcomes_per_app[app_name]
        scenarios = []
        for s_idx, s_name in enumerate(scenarios_list):
            rollouts = [bool(x) for x in outcomes[s_idx]]
            config = ConfigurationResult(rollouts=rollouts)
            scenarios.append(ScenarioResult(scenario_name=s_name, configurations=[config]))
        apps.append(AppResult(app_name=app_name, scenarios=scenarios))

    return EvalSuite(name=name, apps=apps, active_axes=frozenset())
