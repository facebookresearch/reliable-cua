# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Wilson score intervals for binary outcomes.

Implements the Wilson score interval (Wilson 1927), which provides
superior coverage to the Wald interval at extreme success rates and
small sample sizes. See Brown, Cai & DasGupta (2001) for comparison.
"""

import math
from typing import Tuple

import numpy as np
from scipy import stats

from .types import IntervalEstimate


def wilson_score_interval(
    n_successes: int,
    n_trials: int,
    confidence_level: float = 0.95,
) -> IntervalEstimate:
    """Compute the Wilson score confidence interval for a binomial proportion.

    Args:
        n_successes: Number of successes (0 <= n_successes <= n_trials).
        n_trials: Number of trials (>= 1).
        confidence_level: Confidence level (default 0.95).

    Returns:
        IntervalEstimate with Wilson center as point estimate.

    Raises:
        ValueError: If inputs are invalid.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not (0 <= n_successes <= n_trials):
        raise ValueError(
            f"n_successes must be in [0, n_trials], got {n_successes} with n_trials={n_trials}"
        )

    p_hat = n_successes / n_trials
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    z2 = z * z
    R = n_trials

    denominator = 1 + z2 / R
    center = (p_hat + z2 / (2 * R)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / R + z2 / (4 * R * R))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return IntervalEstimate(
        point=center,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def wilson_score_batch(
    successes: np.ndarray,
    trials: np.ndarray,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Wilson score intervals.

    Args:
        successes: Array of success counts.
        trials: Array of trial counts (must be >= 1).
        confidence_level: Confidence level.

    Returns:
        Tuple of (centers, lowers, uppers) arrays.
    """
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    z2 = z * z

    p_hat = successes / trials
    denom = 1 + z2 / trials
    centers = (p_hat + z2 / (2 * trials)) / denom
    margins = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / trials + z2 / (4 * trials * trials))

    lowers = np.clip(centers - margins, 0.0, 1.0)
    uppers = np.clip(centers + margins, 0.0, 1.0)

    return centers, lowers, uppers
