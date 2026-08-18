# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Generate README figures from simulation results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

os.makedirs("images", exist_ok=True)

# ── Colors ──────────────────────────────────────────────────────────────
C_FULL = "#4C72B0"
C_ROLL = "#C44E52"
C_WALD = "#DD8452"
C_80 = "#8172B3"
C_95 = "#4C72B0"
C_OK = "#55A868"
C_BAD = "#C44E52"


# =====================================================================
# Figure 1: Coverage & CI Width vs Rollouts (95% and 80%)
# =====================================================================
def fig1_coverage_vs_rollouts():
    rollouts = [2, 3, 5, 10]

    # 95% CI results
    cov_95 = [1.0, 1.0, 1.0, 1.0]
    wid_95 = [0.020, 0.017, 0.014, 0.010]

    # 80% CI results
    cov_80 = [1.0, 1.0, 1.0, 1.0]
    wid_80 = [0.013, 0.011, 0.009, 0.007]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: CI Width
    ax1.plot(rollouts, wid_95, "o-", color=C_95, linewidth=2, markersize=8, label="95% CI")
    ax1.plot(rollouts, wid_80, "s--", color=C_80, linewidth=2, markersize=8, label="80% CI")
    ax1.set_xlabel("Rollouts per Configuration (R)")
    ax1.set_ylabel("Mean CI Width")
    ax1.set_title("CI Width Narrows with More Rollouts")
    ax1.set_xticks(rollouts)
    ax1.legend()
    ax1.set_ylim(0, 0.03)

    # Right: Coverage (both are 100%, so show as horizontal reference)
    ax2.bar(
        [r - 0.15 for r in rollouts], cov_95, width=0.3,
        color=C_95, alpha=0.8, label="95% CI"
    )
    ax2.bar(
        [r + 0.15 for r in rollouts], cov_80, width=0.3,
        color=C_80, alpha=0.8, label="80% CI"
    )
    ax2.axhline(y=0.95, color=C_95, linestyle=":", alpha=0.6, label="95% target")
    ax2.axhline(y=0.80, color=C_80, linestyle=":", alpha=0.6, label="80% target")
    ax2.set_xlabel("Rollouts per Configuration (R)")
    ax2.set_ylabel("Empirical Coverage")
    ax2.set_title("Coverage Meets or Exceeds Target")
    ax2.set_xticks(rollouts)
    ax2.set_ylim(0.5, 1.05)
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig("images/coverage_vs_rollouts.png", bbox_inches="tight")
    plt.close()
    print("  -> images/coverage_vs_rollouts.png")


# =====================================================================
# Figure 2: Level Ablation — which resampling levels matter?
# =====================================================================
def fig2_level_ablation():
    labels = [
        "Scen+Config+Roll\n(full)",
        "Scen+Config",
        "Scen+Roll",
        "Config+Roll",
        "Scen only",
        "Config only",
        "Roll only",
    ]
    cov_95 = [1.0, 1.0, 0.97, 1.0, 0.96, 1.0, 0.86]
    cov_80 = [1.0, 1.0, 0.98, 1.0, 0.77, 1.0, 0.59]
    wid_95 = [0.017, 0.013, 0.005, 0.014, 0.003, 0.011, 0.003]
    wid_80 = [0.011, 0.009, 0.003, 0.009, 0.002, 0.007, 0.002]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    w = 0.35

    # Left: Coverage
    bars1 = ax1.bar(x - w/2, cov_95, w, label="95% CI", color=C_95, alpha=0.8)
    bars2 = ax1.bar(x + w/2, cov_80, w, label="80% CI", color=C_80, alpha=0.8)
    ax1.axhline(y=0.95, color=C_95, linestyle=":", alpha=0.5)
    ax1.axhline(y=0.80, color=C_80, linestyle=":", alpha=0.5)
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title("Coverage by Resampling Level")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, ha="center")
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(fontsize=9)

    # Highlight under-covering configs
    for i, (c95, c80) in enumerate(zip(cov_95, cov_80)):
        if c95 < 0.95:
            bars1[i].set_edgecolor(C_BAD)
            bars1[i].set_linewidth(2)
        if c80 < 0.80:
            bars2[i].set_edgecolor(C_BAD)
            bars2[i].set_linewidth(2)

    # Right: CI Width
    ax2.bar(x - w/2, wid_95, w, label="95% CI", color=C_95, alpha=0.8)
    ax2.bar(x + w/2, wid_80, w, label="80% CI", color=C_80, alpha=0.8)
    ax2.set_ylabel("Mean CI Width")
    ax2.set_title("CI Width by Resampling Level")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, ha="center")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("images/level_ablation.png", bbox_inches="tight")
    plt.close()
    print("  -> images/level_ablation.png")


# =====================================================================
# Figure 3: Heterogeneous noise — full vs roll-only
# =====================================================================
def fig3_heterogeneous():
    methods = ["Full\nhierarchical", "Roll only"]
    cov_95 = [0.95, 0.15]
    cov_80 = [0.89, 0.08]
    wid_95 = [0.030, 0.003]
    wid_80 = [0.020, 0.002]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    x = np.arange(len(methods))
    w = 0.35

    # Coverage
    ax1.bar(x - w/2, cov_95, w, label="95% CI", color=C_FULL, alpha=0.8)
    ax1.bar(x + w/2, cov_80, w, label="80% CI", color=C_80, alpha=0.8)
    ax1.axhline(y=0.95, color=C_FULL, linestyle=":", alpha=0.5)
    ax1.axhline(y=0.80, color=C_80, linestyle=":", alpha=0.5)
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title("Coverage Under Heterogeneous Noise")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)

    # Annotate the failures
    ax1.annotate("15%", (1 - w/2, 0.15), ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=C_BAD)
    ax1.annotate("8%", (1 + w/2, 0.08), ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=C_BAD)

    # CI Width
    ax2.bar(x - w/2, wid_95, w, label="95% CI", color=C_FULL, alpha=0.8)
    ax2.bar(x + w/2, wid_80, w, label="80% CI", color=C_80, alpha=0.8)
    ax2.set_ylabel("Mean CI Width")
    ax2.set_title("CI Width")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend(fontsize=9)

    # Annotate the widths
    for i in range(2):
        ax2.text(i - w/2, wid_95[i] + 0.002, f"{wid_95[i]:.3f}",
                 ha="center", va="bottom", fontsize=9)
        ax2.text(i + w/2, wid_80[i] + 0.002, f"{wid_80[i]:.3f}",
                 ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig("images/heterogeneous.png", bbox_inches="tight")
    plt.close()
    print("  -> images/heterogeneous.png")


# =====================================================================
# Figure 4: Hierarchical vs Wald
# =====================================================================
def fig4_wald_comparison():
    fig, ax = plt.subplots(figsize=(6, 4))

    methods = ["Hierarchical\nBootstrap", "Naive Wald"]
    cov = [1.0, 0.96]
    wid = [0.017, 0.004]
    colors = [C_FULL, C_WALD]

    x = np.arange(len(methods))

    # Paired bars: coverage and width on dual axes
    ax.bar(x - 0.18, cov, 0.35, label="Coverage", color=colors, alpha=0.8)
    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95% target")
    ax.set_ylabel("Empirical Coverage")
    ax.set_ylim(0.85, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title("Homogeneous: Both Cover, But...")

    ax2 = ax.twinx()
    bars = ax2.bar(x + 0.18, wid, 0.35, color=colors, alpha=0.4, hatch="//",
                   label="CI Width")
    ax2.set_ylabel("Mean CI Width")
    ax2.set_ylim(0, 0.025)

    # Add text annotations
    ax.text(0 - 0.18, 1.005, "100%", ha="center", fontsize=10, color=C_FULL)
    ax.text(1 - 0.18, 0.965, "96%", ha="center", fontsize=10, color=C_WALD)
    ax2.text(0 + 0.18, 0.020, "0.017", ha="center", fontsize=9)
    ax2.text(1 + 0.18, 0.007, "0.004", ha="center", fontsize=9)

    # Add warning note
    ax.text(0.5, 0.87, "Wald CIs are 4.3x narrower but collapse\nunder heterogeneous noise (see Sim 3)",
            transform=ax.transData, ha="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD", edgecolor="#FFCC02", alpha=0.9))

    fig.tight_layout()
    fig.savefig("images/wald_comparison.png", bbox_inches="tight")
    plt.close()
    print("  -> images/wald_comparison.png")


# =====================================================================
# Figure 5: Library usage showcase — synthetic evaluation example
# =====================================================================
def fig5_showcase():
    """Generate a showcase figure using the library itself."""
    from reliable_cua.simulation import SimulationConfig, simulate_suite
    from reliable_cua.bootstrap import BootstrapConfig, bootstrap_suite_score, bootstrap_per_app
    from reliable_cua.wilson import wilson_score_interval
    from reliable_cua.profiles import performance_profile

    # Generate two "models" with different strengths
    probs_a = np.array([0.92, 0.85, 0.80, 0.75, 0.70, 0.65, 0.58, 0.52,
                        0.45, 0.38, 0.30, 0.25, 0.18, 0.12, 0.05])
    probs_b = np.array([0.78, 0.75, 0.72, 0.68, 0.65, 0.62, 0.60, 0.55,
                        0.50, 0.45, 0.42, 0.38, 0.32, 0.28, 0.22])

    cfg_a = SimulationConfig(n_apps=15, scenarios_per_app=7, n_profiles=3,
                             n_themes=3, n_ui_states=3, n_rollouts=3,
                             app_success_probs=probs_a, seed=100)
    cfg_b = SimulationConfig(n_apps=15, scenarios_per_app=7, n_profiles=3,
                             n_themes=3, n_ui_states=3, n_rollouts=3,
                             app_success_probs=probs_b, seed=200)

    suite_a, _ = simulate_suite(cfg_a)
    suite_b, _ = simulate_suite(cfg_b)
    suite_a.name = "Model A"
    suite_b.name = "Model B"

    boot_cfg = BootstrapConfig(n_replicates=1000, seed=42)

    # Suite scores
    res_a = bootstrap_suite_score(suite_a, config=boot_cfg)
    res_b = bootstrap_suite_score(suite_b, config=boot_cfg)

    # Per-app
    apps_a = bootstrap_per_app(suite_a, config=boot_cfg)
    apps_b = bootstrap_per_app(suite_b, config=boot_cfg)

    # Performance profiles
    tau = np.linspace(0, 1, 51)
    prof_a = performance_profile(suite_a, tau)
    prof_b = performance_profile(suite_b, tau)

    # Wilson scores for one app
    app0 = suite_a.apps[0]
    wilson_rates = []
    wilson_cis = []
    for scen in app0.scenarios:
        for cfg in scen.configurations:
                wi = wilson_score_interval(cfg.num_successes, cfg.num_rollouts)
                wilson_rates.append(wi.point)
                wilson_cis.append((wi.lower, wi.upper))

    # ── Create figure ──
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Suite scores comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = ["Model A", "Model B"]
    points = [res_a.point_estimate, res_b.point_estimate]
    lowers = [res_a.ci.lower, res_b.ci.lower]
    uppers = [res_a.ci.upper, res_b.ci.upper]
    colors = [C_FULL, C_WALD]

    ax1.barh([0, 1], points, height=0.5, color=colors, alpha=0.7)
    for i in range(2):
        ax1.plot([lowers[i], uppers[i]], [i, i], color="black", linewidth=2.5)
        ax1.plot(points[i], i, "o", color="black", markersize=6)
        ax1.text(uppers[i] + 0.01, i,
                 f"{points[i]:.3f} [{lowers[i]:.3f}, {uppers[i]:.3f}]",
                 va="center", fontsize=9)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(models)
    ax1.set_xlabel("Suite Score (10% Trimmed Mean)")
    ax1.set_title("(a) Suite-Level Comparison with 95% CI")
    ax1.set_xlim(0, 0.85)
    ax1.invert_yaxis()

    # Panel B: Per-app scores
    ax2 = fig.add_subplot(gs[0, 1])
    app_names = sorted(apps_a.keys())
    y = np.arange(len(app_names))
    for i, name in enumerate(app_names):
        ra = apps_a[name]
        rb = apps_b[name]
        ax2.plot([ra.ci.lower, ra.ci.upper], [i - 0.12, i - 0.12],
                 color=C_FULL, linewidth=1.5, alpha=0.7)
        ax2.plot(ra.point_estimate, i - 0.12, "o", color=C_FULL, markersize=4)
        ax2.plot([rb.ci.lower, rb.ci.upper], [i + 0.12, i + 0.12],
                 color=C_WALD, linewidth=1.5, alpha=0.7)
        ax2.plot(rb.point_estimate, i + 0.12, "s", color=C_WALD, markersize=4)
    ax2.set_yticks(y)
    ax2.set_yticklabels([n.replace("app_", "App ") for n in app_names], fontsize=8)
    ax2.set_xlabel("Success Rate")
    ax2.set_title("(b) Per-App Scores with 95% CI")
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.plot([], [], "o-", color=C_FULL, label="Model A")
    ax2.plot([], [], "s-", color=C_WALD, label="Model B")
    ax2.legend(fontsize=9, loc="lower right")

    # Panel C: Performance profiles
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(tau, prof_a, color=C_FULL, linewidth=2, label="Model A")
    ax3.plot(tau, prof_b, color=C_WALD, linewidth=2, label="Model B")
    ax3.fill_between(tau, prof_a, alpha=0.1, color=C_FULL)
    ax3.fill_between(tau, prof_b, alpha=0.1, color=C_WALD)
    ax3.set_xlabel(r"Success Rate Threshold ($\tau$)")
    ax3.set_ylabel(r"Fraction of Apps with Score $\geq \tau$")
    ax3.set_title("(c) Performance Profiles")
    ax3.legend(fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.05)

    # Panel D: Wilson score intervals for one app
    ax4 = fig.add_subplot(gs[1, 1])
    n_show = min(40, len(wilson_rates))
    sorted_idx = np.argsort(wilson_rates)[::-1][:n_show]
    for plot_i, orig_i in enumerate(sorted_idx):
        lo, hi = wilson_cis[orig_i]
        pt = wilson_rates[orig_i]
        color = C_OK if pt > 0.5 else C_BAD
        ax4.plot([lo, hi], [plot_i, plot_i], color=color, linewidth=1, alpha=0.6)
        ax4.plot(pt, plot_i, "o", color=color, markersize=3)
    ax4.set_xlabel("Wilson Score (Success Rate)")
    ax4.set_ylabel("Configuration (sorted)")
    ax4.set_title(f"(d) Wilson Intervals for {app0.app_name.replace('app_', 'App ')}")
    ax4.set_xlim(0, 1)
    ax4.invert_yaxis()
    ax4.set_yticks([])

    fig.savefig("images/showcase.png", bbox_inches="tight")
    plt.close()
    print("  -> images/showcase.png")


# =====================================================================
# Figure 6: Bootstrap distribution example
# =====================================================================
def fig6_bootstrap_distribution():
    from reliable_cua.simulation import SimulationConfig, simulate_suite
    from reliable_cua.bootstrap import BootstrapConfig, bootstrap_suite_score

    cfg = SimulationConfig(n_apps=15, scenarios_per_app=10, n_rollouts=3, seed=42)
    suite, true_score = simulate_suite(cfg)
    boot_cfg = BootstrapConfig(n_replicates=2000, seed=42)
    result = bootstrap_suite_score(suite, config=boot_cfg)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(result.bootstrap_distribution, bins=50, color=C_FULL, alpha=0.6,
            edgecolor="white", density=True, label="Bootstrap distribution")
    ax.axvline(result.point_estimate, color="black", linewidth=2,
               linestyle="-", label=f"Point estimate: {result.point_estimate:.3f}")
    ax.axvline(result.ci.lower, color=C_BAD, linewidth=2, linestyle="--",
               label=f"95% CI: [{result.ci.lower:.3f}, {result.ci.upper:.3f}]")
    ax.axvline(result.ci.upper, color=C_BAD, linewidth=2, linestyle="--")
    ax.axvline(true_score, color=C_OK, linewidth=2, linestyle=":",
               label=f"True score: {true_score:.3f}")

    ax.fill_betweenx([0, ax.get_ylim()[1] * 1.1],
                     result.ci.lower, result.ci.upper,
                     alpha=0.08, color=C_BAD)
    ax.set_xlabel("Suite Score (10% Trimmed Mean)")
    ax.set_ylabel("Density")
    ax.set_title("Hierarchical Bootstrap Distribution (B=2000)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("images/bootstrap_distribution.png", bbox_inches="tight")
    plt.close()
    print("  -> images/bootstrap_distribution.png")


def fig7_wald_heterogeneous():
    fig, ax = plt.subplots(figsize=(6, 4))

    methods = ["Hierarchical\nBootstrap", "Naive Wald"]
    cov = [0.99, 0.20]
    wid = [0.030, 0.004]
    colors = [C_FULL, C_WALD]

    x = np.arange(len(methods))

    ax.bar(x - 0.18, cov, 0.35, label="Coverage", color=colors, alpha=0.8)
    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95% target")
    ax.set_ylabel("Empirical Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title("Heterogeneous: Wald Collapses")

    ax2 = ax.twinx()
    bars = ax2.bar(x + 0.18, wid, 0.35, color=colors, alpha=0.4, hatch="//",
                   label="CI Width")
    ax2.set_ylabel("Mean CI Width")
    ax2.set_ylim(0, 0.04)

    # Add text annotations
    ax.text(0 - 0.18, 0.995 + 0.01, "99%", ha="center", fontsize=10, color=C_FULL)
    ax.text(1 - 0.18, 0.20 + 0.02, "20%", ha="center", fontsize=10,
            fontweight="bold", color=C_BAD)
    ax2.text(0 + 0.18, 0.031, "0.030", ha="center", fontsize=9)
    ax2.text(1 + 0.18, 0.005, "0.004", ha="center", fontsize=9)

    ax.text(0.5, 0.55, "Wald misses the true value\n4 out of 5 times",
            transform=ax.transData, ha="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8D7DA", edgecolor=C_BAD, alpha=0.9))

    fig.tight_layout()
    fig.savefig("images/wald_heterogeneous.png", bbox_inches="tight")
    plt.close()
    print("  -> images/wald_heterogeneous.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_coverage_vs_rollouts()
    fig2_level_ablation()
    fig3_heterogeneous()
    fig4_wald_comparison()
    fig5_showcase()
    fig6_bootstrap_distribution()
    fig7_wald_heterogeneous()
    print("Done!")
