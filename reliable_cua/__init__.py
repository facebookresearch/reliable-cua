# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""reliable-cua: Statistical evaluation for PRISM-compliant CUA benchmarks."""

from .types import (
    AppResult,
    BootstrapResult,
    ConfigurationResult,
    EvalSuite,
    IntervalEstimate,
    MatchedPair,
    ScenarioResult,
    VariabilityDecomposition,
)
from .wilson import wilson_score_batch, wilson_score_interval
from .metrics import (
    aggregate_mean,
    aggregate_median,
    per_app_scores,
    per_scenario_scores,
    suite_score,
    trimmed_mean,
    wald_suite_ci,
)
from .bootstrap import (
    BootstrapConfig,
    auto_config,
    bootstrap_per_app,
    bootstrap_suite_score,
    hierarchical_bootstrap,
)
from .io import from_arrays, from_dict, load_jsonl, parse_task_id
from .variability import (
    compute_variability,
    decompose_all_axes,
    extract_matched_pairs,
)
from .profiles import performance_profile, performance_profiles_with_ci
from . import plot_utils

__version__ = "0.1.0"
