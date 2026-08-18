# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Plotting utilities for CUA evaluation results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .types import BootstrapResult, EvalSuite, MatchedPair, VariabilityDecomposition
from .variability import extract_matched_pairs


def _setup_style():
    """Apply consistent plot style."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.size": 12,
    })


def plot_interval_estimates(
    results: Dict[str, Dict[str, BootstrapResult]],
    metric_names: List[str],
    *,
    methods: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    xlabel: str = "Score",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot horizontal bars with bootstrap CIs for multiple metrics/methods.

    Args:
        results: Nested dict: {method_name: {metric_name: BootstrapResult}}.
        metric_names: Which metrics to plot (one subplot each).
        methods: Method ordering. If None, uses dict order.
        colors: Optional method -> color mapping.
        figsize: Figure size.
        xlabel: X-axis label.

    Returns:
        (Figure, array of Axes).
    """
    _setup_style()
    if methods is None:
        methods = list(results.keys())
    n_metrics = len(metric_names)
    if figsize is None:
        figsize = (3.5 * n_metrics, 0.5 * len(methods) + 1)

    palette = colors or dict(zip(methods, sns.color_palette("colorblind", len(methods))))

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=True)
    if n_metrics == 1:
        axes = np.array([axes])

    for ax, metric in zip(axes, metric_names):
        for i, method in enumerate(methods):
            br = results[method][metric]
            color = palette.get(method, "steelblue")
            ax.barh(i, br.point_estimate, height=0.5, color=color, alpha=0.7)
            ax.plot(
                [br.ci.lower, br.ci.upper], [i, i],
                color="black", linewidth=2, zorder=5,
            )
            ax.plot(br.point_estimate, i, "o", color="black", markersize=4, zorder=6)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlabel(xlabel)
        ax.set_title(metric)
        ax.invert_yaxis()

    fig.tight_layout()
    return fig, axes


def plot_per_app_scores(
    results: Dict[str, BootstrapResult],
    *,
    sort_by_score: bool = True,
    color: str = "steelblue",
    figsize: Optional[Tuple[float, float]] = None,
    xlabel: str = "Success Rate",
    title: str = "Per-App Performance",
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-app dot plot with bootstrap CIs.

    Args:
        results: Dict mapping app_name -> BootstrapResult.
        sort_by_score: Whether to sort apps by score (descending).
        color: Bar/dot color.
        figsize: Figure size.
        xlabel: X-axis label.
        title: Plot title.

    Returns:
        (Figure, Axes).
    """
    _setup_style()
    items = list(results.items())
    if sort_by_score:
        items.sort(key=lambda x: x[1].point_estimate, reverse=True)

    names = [name for name, _ in items]
    points = [br.point_estimate for _, br in items]
    lowers = [br.ci.lower for _, br in items]
    uppers = [br.ci.upper for _, br in items]

    if figsize is None:
        figsize = (8, max(3, 0.4 * len(names)))

    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(names))
    ax.barh(y, points, height=0.5, color=color, alpha=0.6)
    for i in y:
        ax.plot([lowers[i], uppers[i]], [i, i], color="black", linewidth=1.5)
        ax.plot(points[i], i, "o", color="black", markersize=4)

    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def plot_performance_profiles(
    profiles: Dict[str, np.ndarray],
    tau_list: np.ndarray,
    *,
    profile_cis: Optional[Dict[str, np.ndarray]] = None,
    colors: Optional[Dict[str, str]] = None,
    alpha: float = 0.15,
    figsize: Tuple[float, float] = (7, 5),
    xlabel: str = r"Success Rate Threshold ($\tau$)",
    ylabel: str = r"Fraction of Apps ($F(\tau)$)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot performance profile curves with optional CI bands.

    Args:
        profiles: Dict mapping method -> 1-D profile array.
        tau_list: Threshold values.
        profile_cis: Optional CIs: method -> (2, len(tau)) array.
        colors: Method -> color mapping.
        alpha: CI band transparency.
        figsize: Figure size.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        ax: Existing axes to plot on.

    Returns:
        The Axes object.
    """
    _setup_style()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    methods = list(profiles.keys())
    palette = colors or dict(zip(methods, sns.color_palette("colorblind", len(methods))))

    for method in methods:
        color = palette.get(method, None)
        ax.plot(tau_list, profiles[method], label=method, color=color, linewidth=2)
        if profile_cis is not None and method in profile_cis:
            ci = profile_cis[method]
            ax.fill_between(tau_list, ci[0], ci[1], alpha=alpha, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return ax


def plot_variability_decomposition(
    decompositions: Dict[str, VariabilityDecomposition],
    metric: str = "mad",
    *,
    color: str = "steelblue",
    figsize: Tuple[float, float] = (6, 3),
    xlabel: Optional[str] = None,
    title: str = "Variability Decomposition",
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart of variability metrics by axis.

    Args:
        decompositions: Dict mapping axis -> VariabilityDecomposition.
        metric: Which metric to plot ("mad", "ate", or "p_degradation").
        color: Bar color.
        figsize: Figure size.
        xlabel: X-axis label (auto-generated if None).
        title: Plot title.

    Returns:
        (Figure, Axes).
    """
    _setup_style()
    if xlabel is None:
        labels = {"mad": "Mean Absolute Deviation", "ate": "Average Treatment Effect",
                  "p_degradation": "P(Degradation)"}
        xlabel = labels.get(metric, metric)

    axes_names = list(decompositions.keys())
    values = [getattr(decompositions[a], metric) for a in axes_names]

    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(axes_names))
    ax.barh(y, values, height=0.5, color=color, alpha=0.7)

    # Add CI whiskers if available
    ci_attr = f"{metric}_ci"
    for i, axis_name in enumerate(axes_names):
        ci = getattr(decompositions[axis_name], ci_attr, None)
        if ci is not None:
            ax.plot([ci.lower, ci.upper], [i, i], color="black", linewidth=2)
            ax.plot(values[i], i, "o", color="black", markersize=4)

    ax.set_yticks(list(y))
    ax.set_yticklabels([a.replace("_", " ").title() for a in axes_names])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def plot_matched_pair_distributions(
    suites: Dict[str, EvalSuite],
    axes: Optional[List[str]] = None,
    *,
    colors: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    threshold: float = 0.10,
) -> Tuple[plt.Figure, np.ndarray]:
    """Violin + strip plot of matched-pair |delta| distributions per axis per model.

    For each environmental axis, shows the full distribution of absolute
    performance differences between matched configuration pairs. This reveals
    whether instability is uniform (all configs shift a little) or heavy-tailed
    (most are fine but some are catastrophic).

    Args:
        suites: Dict mapping model_name -> EvalSuite.
        axes: Which axes to plot. If None, uses all active axes found.
        colors: Model -> color mapping.
        figsize: Figure size.
        threshold: Degradation threshold line.

    Returns:
        (Figure, array of Axes).
    """
    _setup_style()
    import pandas as pd

    model_names = list(suites.keys())

    # Discover axes
    if axes is None:
        all_axes: set = set()
        for suite in suites.values():
            all_axes.update(suite.active_axes)
        axes = sorted(all_axes)

    # Build dataframe of scenario-level |deltas|.
    # For each (model, axis, app, scenario), group configs by the axis value,
    # compute per-group success rates, then take pairwise |Δ|.
    # This averages out the binary noise from individual rollouts.
    from collections import defaultdict

    records = []
    for model_name, suite in suites.items():
        for axis in axes:
            if axis not in suite.active_axes:
                continue
            for app in suite.apps:
                for scenario in app.scenarios:
                    # Group configs by their value on this axis
                    groups: Dict[str, List[float]] = defaultdict(list)
                    for config in scenario.configurations:
                        val = getattr(config, axis)
                        if val is None:
                            continue
                        groups[val].append(config.success_rate)

                    if len(groups) < 2:
                        continue

                    # Compute mean success rate per group
                    group_means = {
                        v: float(np.mean(rates)) for v, rates in groups.items()
                    }

                    # Pairwise |Δ| between group means
                    vals = list(group_means.values())
                    for i in range(len(vals)):
                        for j in range(i + 1, len(vals)):
                            records.append({
                                "Model": model_name,
                                "Axis": axis.replace("_", " ").title(),
                                "|Δ|": abs(vals[i] - vals[j]),
                            })

    if not records:
        raise ValueError("No matched pairs found for any axis")

    df = pd.DataFrame(records)

    n_axes = len(axes)
    if figsize is None:
        figsize = (4 * n_axes, 4)

    palette = colors or dict(
        zip(model_names, sns.color_palette("colorblind", len(model_names)))
    )

    fig, ax_arr = plt.subplots(1, n_axes, figsize=figsize, sharey=True)
    if n_axes == 1:
        ax_arr = np.array([ax_arr])

    axis_labels = [a.replace("_", " ").title() for a in axes]

    for i, (axis, axis_label) in enumerate(zip(axes, axis_labels)):
        ax = ax_arr[i]
        subset = df[df["Axis"] == axis_label]
        if subset.empty:
            ax.set_title(axis_label)
            continue

        sns.violinplot(
            data=subset, x="Model", y="|Δ|", ax=ax,
            palette=palette, inner=None, alpha=0.3, cut=0,
        )
        sns.stripplot(
            data=subset, x="Model", y="|Δ|", ax=ax,
            palette=palette, size=1.5, alpha=0.15, jitter=True,
        )
        # Overlay box for quartiles
        sns.boxplot(
            data=subset, x="Model", y="|Δ|", ax=ax,
            palette=palette, width=0.15, fliersize=0,
            boxprops=dict(alpha=0.7), whiskerprops=dict(alpha=0.7),
            medianprops=dict(color="black", linewidth=1.5),
            capprops=dict(alpha=0.7),
        )

        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(axis_label)
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Absolute Performance Shift |Δ|")
        else:
            ax.set_ylabel("")

    fig.tight_layout()
    return fig, ax_arr
