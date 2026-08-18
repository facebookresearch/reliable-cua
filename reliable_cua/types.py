# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Core data structures for hierarchical CUA evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np


@dataclass
class ConfigurationResult:
    """Results for a single (instance, profile, theme, ui_state) configuration.

    When an environmental axis is inactive, its field is None.
    """

    instance: Optional[str] = None
    profile: Optional[str] = None
    theme: Optional[str] = None
    ui_state: Optional[str] = None
    rollouts: List[bool] = field(default_factory=list)

    @property
    def num_rollouts(self) -> int:
        return len(self.rollouts)

    @property
    def num_successes(self) -> int:
        return sum(self.rollouts)

    @property
    def success_rate(self) -> float:
        if not self.rollouts:
            return float("nan")
        return self.num_successes / self.num_rollouts


@dataclass
class ScenarioResult:
    """Results for a single scenario (task template)."""

    scenario_name: str
    configurations: List[ConfigurationResult] = field(default_factory=list)


@dataclass
class AppResult:
    """Results for a single application."""

    app_name: str
    scenarios: List[ScenarioResult] = field(default_factory=list)


@dataclass
class EvalSuite:
    """Complete evaluation results for a benchmark suite.

    The hierarchical tree: apps -> scenarios -> configurations -> rollouts.
    ``active_axes`` records which environmental variability axes were enabled.
    Valid values: {"instance", "profile", "theme", "ui_state"}.
    Rollouts (agent stochasticity) are always present.
    """

    name: str
    apps: List[AppResult] = field(default_factory=list)
    active_axes: FrozenSet[str] = field(default_factory=frozenset)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def app_names(self) -> List[str]:
        return [app.app_name for app in self.apps]

    @property
    def num_apps(self) -> int:
        return len(self.apps)


@dataclass(frozen=True)
class IntervalEstimate:
    """A point estimate with a confidence interval."""

    point: float
    lower: float
    upper: float
    confidence_level: float = 0.95

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass
class BootstrapResult:
    """Full result of a bootstrap procedure."""

    point_estimate: float
    ci: IntervalEstimate
    bootstrap_distribution: np.ndarray
    n_replicates: int


@dataclass
class MatchedPair:
    """A pair of configurations differing on exactly one axis."""

    axis: str
    value_a: str
    value_b: str
    success_rate_a: float
    success_rate_b: float
    app_name: str
    scenario_name: str
    instance_id: str


@dataclass
class VariabilityDecomposition:
    """Result of variability decomposition along one axis."""

    axis: str
    mad: float
    ate: float
    p_degradation: float
    threshold: float
    n_pairs: int
    mad_ci: Optional[IntervalEstimate] = None
    ate_ci: Optional[IntervalEstimate] = None
    p_degradation_ci: Optional[IntervalEstimate] = None
