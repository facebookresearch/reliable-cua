# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Aggregation metrics for hierarchical CUA evaluation."""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from .types import AppResult, EvalSuite, IntervalEstimate


def trimmed_mean(scores: np.ndarray, proportion: float = 0.1) -> float:
    """Compute the trimmed mean, discarding a fraction from each tail.

    With proportion=0.1 and 15 apps, this discards the top and bottom
    ~1-2 app scores. This is less aggressive than rliable's IQM (25%),
    which would discard ~7 of 15 apps.

    Args:
        scores: 1-D array of scores.
        proportion: Fraction to trim from each tail (default 0.1).

    Returns:
        Trimmed mean as a float.
    """
    return float(sp_stats.trim_mean(scores, proportion))


def _app_success_rate(app: AppResult) -> float:
    """Compute the mean success rate for a single app.

    Averages over all rollouts across all scenarios and configurations
    within this app.
    """
    total = 0
    successes = 0
    for scenario in app.scenarios:
        for config in scenario.configurations:
            total += config.num_rollouts
            successes += config.num_successes
    if total == 0:
        return float("nan")
    return successes / total


def per_app_scores(suite: EvalSuite) -> Dict[str, float]:
    """Compute per-app success rates.

    Returns:
        Dict mapping app_name -> mean success rate.
    """
    return {app.app_name: _app_success_rate(app) for app in suite.apps}


def suite_score(suite: EvalSuite, trim_proportion: float = 0.1) -> float:
    """Compute the suite-level score: trimmed mean of per-app success rates.

    Args:
        suite: The evaluation suite.
        trim_proportion: Fraction to trim from each tail (default 0.1).

    Returns:
        Suite-level score as a float.
    """
    scores = np.array([_app_success_rate(app) for app in suite.apps])
    valid = scores[~np.isnan(scores)]
    if len(valid) == 0:
        return float("nan")
    return trimmed_mean(valid, trim_proportion)


def per_scenario_scores(suite: EvalSuite) -> Dict[Tuple[str, str], float]:
    """Compute per-scenario success rates.

    Returns:
        Dict mapping (app_name, scenario_name) -> mean success rate.
    """
    result = {}
    for app in suite.apps:
        for scenario in app.scenarios:
            total = 0
            successes = 0
            for config in scenario.configurations:
                total += config.num_rollouts
                successes += config.num_successes
            rate = successes / total if total > 0 else float("nan")
            result[(app.app_name, scenario.scenario_name)] = rate
    return result


def wald_suite_ci(
    suite: EvalSuite,
    confidence_level: float = 0.95,
) -> IntervalEstimate:
    """Compute a naive (non-hierarchical) Wald CI for the suite success rate.

    Flattens all rollouts across all apps, scenarios, and configurations
    into a single bag, then computes a Wald normal-approximation interval.
    This deliberately ignores the hierarchical data structure and treats
    every rollout as an independent coin flip — the standard (incorrect)
    practice in most CUA benchmarks.

    Useful as a baseline to contrast against the hierarchical bootstrap CI,
    demonstrating how much the naive approach underestimates uncertainty.

    Args:
        suite: The evaluation suite.
        confidence_level: Confidence level (default 0.95).

    Returns:
        IntervalEstimate with the flat success rate and Wald CI.
    """
    total = 0
    successes = 0
    for app in suite.apps:
        for scenario in app.scenarios:
            for config in scenario.configurations:
                total += config.num_rollouts
                successes += config.num_successes
    if total == 0:
        return IntervalEstimate(
            point=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            confidence_level=confidence_level,
        )
    p_hat = successes / total
    z = sp_stats.norm.ppf(1 - (1 - confidence_level) / 2)
    se = np.sqrt(p_hat * (1 - p_hat) / total)
    lower = max(0.0, p_hat - z * se)
    upper = min(1.0, p_hat + z * se)
    return IntervalEstimate(
        point=p_hat,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def aggregate_mean(scores: np.ndarray) -> float:
    """Plain mean of scores (for compatibility)."""
    return float(np.nanmean(scores))


def aggregate_median(scores: np.ndarray) -> float:
    """Median of scores (for compatibility)."""
    return float(np.nanmedian(scores))
