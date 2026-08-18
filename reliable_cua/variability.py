# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Variability decomposition by environmental axis.

Implements matched-pair analysis to isolate the effect of each
variability axis (instance, profile, theme, UI state) on agent performance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bootstrap import BootstrapConfig
from .types import (
    EvalSuite,
    IntervalEstimate,
    MatchedPair,
    VariabilityDecomposition,
)

# The four config-level axes that can be decomposed
_CONFIG_AXES = ("instance", "profile", "theme", "ui_state")


def extract_matched_pairs(
    suite: EvalSuite,
    axis: str,
    min_rollouts: int = 1,
) -> List[MatchedPair]:
    """Extract configuration pairs differing on exactly one axis.

    For each (app, scenario), groups configurations by their values on
    all axes OTHER than ``axis``, then enumerates all pairs of distinct
    ``axis`` values within each group.

    Args:
        suite: The evaluation suite.
        axis: The axis to decompose ("instance", "profile", "theme",
            or "ui_state").
        min_rollouts: Minimum rollouts per configuration to include.

    Returns:
        List of MatchedPair objects.
    """
    if axis not in _CONFIG_AXES:
        raise ValueError(f"axis must be one of {_CONFIG_AXES}, got {axis!r}")

    other_axes = [a for a in _CONFIG_AXES if a != axis]
    pairs = []

    for app in suite.apps:
        for scenario in app.scenarios:
            # Group configs by their values on the OTHER axes
            groups: Dict[Tuple, List] = defaultdict(list)
            for config in scenario.configurations:
                if config.num_rollouts < min_rollouts:
                    continue
                axis_val = getattr(config, axis)
                if axis_val is None:
                    continue
                other_key = tuple(getattr(config, a) for a in other_axes)
                groups[other_key].append((axis_val, config.success_rate))

            # Enumerate pairs within each group
            for entries in groups.values():
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        val_a, rate_a = entries[i]
                        val_b, rate_b = entries[j]
                        pairs.append(
                            MatchedPair(
                                axis=axis,
                                value_a=val_a,
                                value_b=val_b,
                                success_rate_a=rate_a,
                                success_rate_b=rate_b,
                                app_name=app.app_name,
                                scenario_name=scenario.scenario_name,
                                instance_id=getattr(config, "instance", None),
                            )
                        )

    return pairs


def compute_variability(
    pairs: List[MatchedPair],
    threshold: float = 0.10,
) -> VariabilityDecomposition:
    """Compute MAD, ATE, and P(degradation) from matched pairs.

    Args:
        pairs: List of matched pairs (from ``extract_matched_pairs``).
        threshold: Degradation threshold for P(degradation).

    Returns:
        VariabilityDecomposition with computed metrics.

    Raises:
        ValueError: If pairs list is empty.
    """
    if not pairs:
        raise ValueError("No matched pairs to compute variability from")

    deltas = np.array([p.success_rate_a - p.success_rate_b for p in pairs])
    abs_deltas = np.abs(deltas)

    return VariabilityDecomposition(
        axis=pairs[0].axis,
        mad=float(np.mean(abs_deltas)),
        ate=float(np.mean(deltas)),
        p_degradation=float(np.mean(abs_deltas > threshold)),
        threshold=threshold,
        n_pairs=len(pairs),
    )


def decompose_all_axes(
    suite: EvalSuite,
    threshold: float = 0.10,
    min_rollouts: int = 1,
    bootstrap_config: Optional[BootstrapConfig] = None,
) -> Dict[str, VariabilityDecomposition]:
    """Decompose variability across all active config-level axes.

    Args:
        suite: The evaluation suite.
        threshold: Degradation threshold.
        min_rollouts: Minimum rollouts per configuration.
        bootstrap_config: If provided, compute bootstrap CIs for each metric.

    Returns:
        Dict mapping axis name -> VariabilityDecomposition.
        Only includes axes that have matched pairs.
    """
    results = {}
    for axis in _CONFIG_AXES:
        if axis not in suite.active_axes:
            continue
        pairs = extract_matched_pairs(suite, axis, min_rollouts)
        if not pairs:
            continue
        decomp = compute_variability(pairs, threshold)

        if bootstrap_config is not None:
            decomp = _bootstrap_variability(pairs, threshold, bootstrap_config)

        results[axis] = decomp

    return results


def _bootstrap_variability(
    pairs: List[MatchedPair],
    threshold: float,
    config: BootstrapConfig,
) -> VariabilityDecomposition:
    """Bootstrap CIs for variability metrics by resampling matched pairs."""
    rng = np.random.default_rng(config.seed)
    n = len(pairs)
    deltas = np.array([p.success_rate_a - p.success_rate_b for p in pairs])
    abs_deltas = np.abs(deltas)

    mad_boot = np.empty(config.n_replicates)
    ate_boot = np.empty(config.n_replicates)
    pdeg_boot = np.empty(config.n_replicates)

    for b in range(config.n_replicates):
        idx = rng.integers(0, n, size=n)
        mad_boot[b] = np.mean(abs_deltas[idx])
        ate_boot[b] = np.mean(deltas[idx])
        pdeg_boot[b] = np.mean(abs_deltas[idx] > threshold)

    alpha = 1 - config.confidence_level

    def _ci(boot, point):
        return IntervalEstimate(
            point=point,
            lower=float(np.percentile(boot, 100 * alpha / 2)),
            upper=float(np.percentile(boot, 100 * (1 - alpha / 2))),
            confidence_level=config.confidence_level,
        )

    point_mad = float(np.mean(abs_deltas))
    point_ate = float(np.mean(deltas))
    point_pdeg = float(np.mean(abs_deltas > threshold))

    return VariabilityDecomposition(
        axis=pairs[0].axis,
        mad=point_mad,
        ate=point_ate,
        p_degradation=point_pdeg,
        threshold=threshold,
        n_pairs=n,
        mad_ci=_ci(mad_boot, point_mad),
        ate_ci=_ci(ate_boot, point_ate),
        p_degradation_ci=_ci(pdeg_boot, point_pdeg),
    )
