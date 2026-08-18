# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Performance profiles for CUA evaluation.

A performance profile F(tau) gives the fraction of apps on which a
model achieves at least success rate tau. The area under the profile
corresponds to the suite-level mean score.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .bootstrap import BootstrapConfig, _compile_app, _resample_compiled_app, auto_config
from .metrics import _app_success_rate
from .types import EvalSuite, IntervalEstimate


def performance_profile(
    suite: EvalSuite,
    tau_list: np.ndarray,
) -> np.ndarray:
    """Compute the performance profile F(tau).

    F(tau) = fraction of apps with success rate >= tau.

    Args:
        suite: The evaluation suite.
        tau_list: Array of threshold values in [0, 1].

    Returns:
        Array of same length as tau_list with profile values.
    """
    app_scores = np.array([_app_success_rate(app) for app in suite.apps])
    tau_list = np.asarray(tau_list)
    return np.array([np.mean(app_scores >= tau) for tau in tau_list])


def performance_profiles_with_ci(
    suites: Dict[str, EvalSuite],
    tau_list: np.ndarray,
    config: Optional[BootstrapConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute performance profiles with bootstrap CIs for multiple methods.

    Args:
        suites: Dict mapping method name -> EvalSuite.
        tau_list: Array of threshold values.
        config: Bootstrap configuration. If None, auto-configured per suite.

    Returns:
        Tuple of (profiles_dict, cis_dict).
        profiles_dict maps method -> 1-D array of profile values.
        cis_dict maps method -> 2-D array of shape (2, len(tau_list)),
        where row 0 is lower bounds and row 1 is upper bounds.
    """
    tau_list = np.asarray(tau_list)
    profiles = {}
    cis = {}

    for method_name, suite in suites.items():
        cfg = config if config is not None else auto_config(suite)
        rng = np.random.default_rng(cfg.seed)

        app_scores = np.array([_app_success_rate(app) for app in suite.apps])
        profile = np.array([np.mean(app_scores >= tau) for tau in tau_list])
        profiles[method_name] = profile

        # Bootstrap
        compiled = [_compile_app(app) for app in suite.apps]
        boot_profiles = np.empty((cfg.n_replicates, len(tau_list)))
        for b in range(cfg.n_replicates):
            resampled = np.array(
                [_resample_compiled_app(c, rng, cfg) for c in compiled]
            )
            boot_profiles[b] = [np.mean(resampled >= tau) for tau in tau_list]

        alpha = 1 - cfg.confidence_level
        lower = np.percentile(boot_profiles, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_profiles, 100 * (1 - alpha / 2), axis=0)
        cis[method_name] = np.stack([lower, upper], axis=0)

    return profiles, cis
