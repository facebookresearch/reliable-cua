# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Fixed-app hierarchical bootstrap for CUA evaluation.

Implements the hierarchical bootstrap described in the PRISM framework:
apps are treated as a fixed population (never resampled), while scenarios,
configurations, and rollouts are resampled within each app.

Performance: the tree is flattened into a 6D numpy array once (where
possible), and all B replicates are generated via batch index operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .metrics import trimmed_mean
from .types import (
    AppResult,
    BootstrapResult,
    EvalSuite,
    IntervalEstimate,
)


@dataclass
class BootstrapConfig:
    """Configuration for the hierarchical bootstrap."""

    n_replicates: int = 1000
    confidence_level: float = 0.95
    resample_scenarios: bool = True
    resample_instances: bool = True
    resample_profiles: bool = True
    resample_themes: bool = True
    resample_ui_states: bool = True
    resample_rollouts: bool = True
    seed: Optional[int] = None


def auto_config(suite: EvalSuite, **overrides) -> BootstrapConfig:
    """Auto-configure bootstrap based on which axes are active in the suite."""
    cfg = BootstrapConfig(
        resample_scenarios=True,
        resample_instances="instance" in suite.active_axes,
        resample_profiles="profile" in suite.active_axes,
        resample_themes="theme" in suite.active_axes,
        resample_ui_states="ui_state" in suite.active_axes,
        resample_rollouts=True,
    )
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown BootstrapConfig field: {k}")
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Vectorized bootstrap on regular (padded) arrays
# ---------------------------------------------------------------------------

def _try_regularize_app(app: AppResult) -> Optional[np.ndarray]:
    """Try to pack an app into a regular 6D array.

    Returns array of shape (S, I, P, T, U, R) if the app has regular
    structure (same config axes per scenario, same rollouts per config).
    Returns None if irregular.

    Axes: [scenario, instance, profile, theme, ui_state, rollout]
    Dimensions are 1 when those axes are inactive (single default config).
    """
    if not app.scenarios:
        return None

    # Check all scenarios have same config structure
    n_configs = set()
    n_rollouts = set()
    for s in app.scenarios:
        n_configs.add(len(s.configurations))
        for c in s.configurations:
            n_rollouts.add(len(c.rollouts))

    if len(n_configs) != 1 or len(n_rollouts) != 1:
        return None
    n_cfg = n_configs.pop()
    n_roll = n_rollouts.pop()
    if n_cfg == 0 or n_roll == 0:
        return None

    # Determine axis dimensions from config keys
    instances = set()
    profiles = set()
    themes = set()
    ui_states = set()
    for s in app.scenarios:
        for c in s.configurations:
            instances.add(c.instance)
            profiles.add(c.profile)
            themes.add(c.theme)
            ui_states.add(c.ui_state)

    inst_list = sorted(instances, key=lambda x: (x is None, x))
    prof_list = sorted(profiles, key=lambda x: (x is None, x))
    theme_list = sorted(themes, key=lambda x: (x is None, x))
    ui_list = sorted(ui_states, key=lambda x: (x is None, x))
    n_i, n_p, n_t, n_u = len(inst_list), len(prof_list), len(theme_list), len(ui_list)

    if n_i * n_p * n_t * n_u != n_cfg:
        return None  # Not a full Cartesian product

    inst_idx = {v: i for i, v in enumerate(inst_list)}
    prof_idx = {v: i for i, v in enumerate(prof_list)}
    theme_idx = {v: i for i, v in enumerate(theme_list)}
    ui_idx = {v: i for i, v in enumerate(ui_list)}

    n_s = len(app.scenarios)
    arr = np.empty((n_s, n_i, n_p, n_t, n_u, n_roll), dtype=np.float64)

    for si, s in enumerate(app.scenarios):
        for c in s.configurations:
            ii = inst_idx[c.instance]
            pi = prof_idx[c.profile]
            ti = theme_idx[c.theme]
            ui = ui_idx[c.ui_state]
            arr[si, ii, pi, ti, ui, :] = [float(r) for r in c.rollouts]

    return arr


def _bootstrap_regular_app(
    arr: np.ndarray,
    rng: np.random.Generator,
    config: BootstrapConfig,
    B: int,
) -> np.ndarray:
    """Vectorized bootstrap for a regular 6D app array.

    arr shape: (S, I, P, T, U, R)
    Returns: (B,) array of resampled app means.
    """
    ns, ni, np_, nt, nu, nr = arr.shape
    d = arr  # start with full array

    # Level 1: Scenario resampling — (B, S) indices into dim 0
    if config.resample_scenarios and ns > 1:
        si = rng.integers(0, ns, size=(B, ns))
        d = d[si]
    else:
        d = np.broadcast_to(d[np.newaxis], (B,) + d.shape).copy()

    # Level 2a: Instance resampling — resample along dim 2
    if config.resample_instances and ni > 1:
        ii_full = rng.integers(0, ni, size=(B, ns, ni))
        b_idx = np.arange(B)[:, None, None]
        s_idx = np.arange(ns)[None, :, None]
        d = d[b_idx, s_idx, ii_full]

    # Level 2b: Profile resampling — resample along dim 3
    if config.resample_profiles and np_ > 1:
        pi = rng.integers(0, np_, size=(B, np_))
        b_idx = np.arange(B)[:, None, None, None]
        s_idx = np.arange(d.shape[1])[None, :, None, None]
        i_idx = np.arange(d.shape[2])[None, None, :, None]
        d = d[b_idx, s_idx, i_idx, pi[:, None, None, :]]

    # Level 2c: Theme resampling — resample along dim 4
    if config.resample_themes and nt > 1:
        ti = rng.integers(0, nt, size=(B, nt))
        b_idx = np.arange(B)[:, None, None, None, None]
        s_idx = np.arange(d.shape[1])[None, :, None, None, None]
        i_idx = np.arange(d.shape[2])[None, None, :, None, None]
        p_idx = np.arange(d.shape[3])[None, None, None, :, None]
        d = d[b_idx, s_idx, i_idx, p_idx, ti[:, None, None, None, :]]

    # Level 2d: UI state resampling — resample along dim 5
    if config.resample_ui_states and nu > 1:
        ui = rng.integers(0, nu, size=(B, nu))
        b_idx = np.arange(B)[:, None, None, None, None, None]
        s_idx = np.arange(d.shape[1])[None, :, None, None, None, None]
        i_idx = np.arange(d.shape[2])[None, None, :, None, None, None]
        p_idx = np.arange(d.shape[3])[None, None, None, :, None, None]
        t_idx = np.arange(d.shape[4])[None, None, None, None, :, None]
        d = d[b_idx, s_idx, i_idx, p_idx, t_idx, ui[:, None, None, None, None, :]]

    # Level 3: Rollout resampling — resample along last dim
    if config.resample_rollouts and nr > 1:
        ri = rng.integers(0, nr, size=(B, nr))
        ndim = d.ndim
        idx = [np.arange(d.shape[dim]).reshape(
            (1,) * dim + (-1,) + (1,) * (ndim - dim - 1)
        ) for dim in range(ndim - 1)]
        idx.append(ri.reshape((B,) + (1,) * (ndim - 2) + (nr,)))
        d = d[tuple(idx)]

    # Mean over all axes except B (axis 0)
    return d.reshape(B, -1).mean(axis=1)


# ---------------------------------------------------------------------------
# Fallback: tree-based bootstrap for irregular apps
# ---------------------------------------------------------------------------

class _GroupedApp:
    """Pre-grouped representation for apps with irregular structure."""

    def __init__(self, app: AppResult):
        outcomes_list: List[float] = []
        self._scenario_groups: List[_ScenarioGroup] = []

        offset = 0
        for scenario in app.scenarios:
            sg = _ScenarioGroup()
            for config in scenario.configurations:
                key = (config.instance, config.profile, config.theme, config.ui_state)
                n = len(config.rollouts)
                indices = np.arange(offset, offset + n, dtype=np.int64)
                outcomes_list.extend(float(r) for r in config.rollouts)
                sg.add_config(key, indices)
                offset += n
            self._scenario_groups.append(sg)

        self.outcomes = np.array(outcomes_list, dtype=np.float64)

    def resample(self, rng: np.random.Generator, config: BootstrapConfig) -> float:
        scenarios = self._scenario_groups
        n_scen = len(scenarios)

        if config.resample_scenarios and n_scen > 1:
            scen_indices = rng.integers(0, n_scen, size=n_scen)
        else:
            scen_indices = range(n_scen)

        all_idx: List[np.ndarray] = []
        for si in scen_indices:
            sg = scenarios[si]
            all_idx.append(sg.get_indices(rng, config))

        if not all_idx:
            return float("nan")
        combined = np.concatenate(all_idx)
        if config.resample_rollouts and len(combined) > 1:
            combined = combined[rng.integers(0, len(combined), size=len(combined))]
        return float(self.outcomes[combined].mean())


class _ScenarioGroup:
    def __init__(self):
        self._config_indices: Dict[tuple, np.ndarray] = {}
        self._axis_groups: Optional[Dict[str, Dict]] = None
        self._all_indices: Optional[np.ndarray] = None

    def add_config(self, key: tuple, indices: np.ndarray):
        self._config_indices[key] = indices

    def _build(self):
        groups = {"instance": {}, "profile": {}, "theme": {}, "ui_state": {}}
        all_idx = []
        for (i, p, t, u), indices in self._config_indices.items():
            all_idx.append(indices)
            groups["instance"].setdefault(i, []).append(indices)
            groups["profile"].setdefault(p, []).append(indices)
            groups["theme"].setdefault(t, []).append(indices)
            groups["ui_state"].setdefault(u, []).append(indices)
        self._axis_groups = {
            ax: {k: np.concatenate(v) for k, v in grp.items()}
            for ax, grp in groups.items()
        }
        self._all_indices = np.concatenate(all_idx) if all_idx else np.array([], dtype=np.int64)

    def get_indices(self, rng: np.random.Generator, config: BootstrapConfig) -> np.ndarray:
        if self._axis_groups is None:
            self._build()

        needs_resample = any([
            config.resample_instances and len(self._axis_groups["instance"]) > 1,
            config.resample_profiles and len(self._axis_groups["profile"]) > 1,
            config.resample_themes and len(self._axis_groups["theme"]) > 1,
            config.resample_ui_states and len(self._axis_groups["ui_state"]) > 1,
        ])
        if not needs_resample:
            return self._all_indices

        def _resample_keys(ax_name, do_resample):
            grp = self._axis_groups[ax_name]
            keys = list(grp.keys())
            if do_resample and len(keys) > 1:
                return [keys[i] for i in rng.integers(0, len(keys), size=len(keys))]
            return keys

        inst_keys = _resample_keys("instance", config.resample_instances)
        prof_keys = _resample_keys("profile", config.resample_profiles)
        theme_keys = _resample_keys("theme", config.resample_themes)
        ui_keys = _resample_keys("ui_state", config.resample_ui_states)

        pieces = []
        for i in inst_keys:
            for p in prof_keys:
                for t in theme_keys:
                    for u in ui_keys:
                        idx = self._config_indices.get((i, p, t, u))
                        if idx is not None:
                            pieces.append(idx)
        return np.concatenate(pieces) if pieces else self._all_indices


# ---------------------------------------------------------------------------
# Compiled app: chooses fast path or fallback
# ---------------------------------------------------------------------------

class _CompiledApp:
    """Wraps either a regular 6D array or a grouped fallback."""

    def __init__(self, app: AppResult):
        self._regular = _try_regularize_app(app)
        self._grouped = None if self._regular is not None else _GroupedApp(app)

    def bootstrap_batch(
        self, rng: np.random.Generator, config: BootstrapConfig, B: int
    ) -> np.ndarray:
        """Return (B,) array of resampled app means."""
        if self._regular is not None:
            return _bootstrap_regular_app(self._regular, rng, config, B)
        # Fallback: loop
        return np.array([self._grouped.resample(rng, config) for _ in range(B)])


def _compile_app(app: AppResult) -> _CompiledApp:
    return _CompiledApp(app)


def _resample_compiled_app(
    capp: _CompiledApp, rng: np.random.Generator, config: BootstrapConfig
) -> float:
    return capp.bootstrap_batch(rng, config, 1)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hierarchical_bootstrap(
    suite: EvalSuite,
    statistic: Callable[[np.ndarray], float],
    config: Optional[BootstrapConfig] = None,
) -> BootstrapResult:
    """Run the fixed-app hierarchical bootstrap.

    Apps are never resampled. Within each app, scenarios, configuration
    axes, and rollouts are resampled according to ``config``.
    """
    if config is None:
        config = auto_config(suite)

    rng = np.random.default_rng(config.seed)
    B = config.n_replicates

    # Compile all apps
    compiled = [_CompiledApp(app) for app in suite.apps]
    n_apps = len(compiled)

    # Point estimate
    from .metrics import _app_success_rate
    app_scores = np.array([_app_success_rate(app) for app in suite.apps])
    point_estimate = statistic(app_scores)

    # Bootstrap: generate all B resampled scores per app in batch
    all_app_boots = np.empty((n_apps, B))
    for a in range(n_apps):
        all_app_boots[a] = compiled[a].bootstrap_batch(rng, config, B)

    # Apply statistic across apps for each replicate
    boot_dist = np.array([statistic(all_app_boots[:, b]) for b in range(B)])

    # Percentile CI
    alpha = 1 - config.confidence_level
    lower = float(np.percentile(boot_dist, 100 * alpha / 2))
    upper = float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))

    ci = IntervalEstimate(
        point=point_estimate, lower=lower, upper=upper,
        confidence_level=config.confidence_level,
    )
    return BootstrapResult(
        point_estimate=point_estimate, ci=ci,
        bootstrap_distribution=boot_dist, n_replicates=B,
    )


def bootstrap_suite_score(
    suite: EvalSuite,
    trim_proportion: float = 0.1,
    config: Optional[BootstrapConfig] = None,
) -> BootstrapResult:
    """Bootstrap the suite-level trimmed mean score."""
    stat = lambda x: trimmed_mean(x, trim_proportion)
    return hierarchical_bootstrap(suite, stat, config)


def bootstrap_per_app(
    suite: EvalSuite,
    config: Optional[BootstrapConfig] = None,
) -> Dict[str, BootstrapResult]:
    """Bootstrap per-app success rates independently."""
    if config is None:
        config = auto_config(suite)

    rng = np.random.default_rng(config.seed)
    from .metrics import _app_success_rate

    results = {}
    for app in suite.apps:
        point = _app_success_rate(app)
        capp = _CompiledApp(app)
        boot_dist = capp.bootstrap_batch(rng, config, config.n_replicates)

        alpha = 1 - config.confidence_level
        lower = float(np.percentile(boot_dist, 100 * alpha / 2))
        upper = float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))

        ci = IntervalEstimate(
            point=point, lower=lower, upper=upper,
            confidence_level=config.confidence_level,
        )
        results[app.app_name] = BootstrapResult(
            point_estimate=point, ci=ci,
            bootstrap_distribution=boot_dist, n_replicates=config.n_replicates,
        )

    return results
