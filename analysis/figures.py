"""
Publication-Quality Figure Generation Script for AMFS 2.0 (Stage 1, Section 18).

Generates all 12 required scientific figures from actual Stage 1 Pilot and Confirmatory CSV outputs:
1. fig01_false_certification_vs_K.png: False certification rate vs K (with Clopper-Pearson / bootstrap uncertainty intervals, Pilot vs Confirmatory).
2. fig02_false_certification_vs_B.png: False certification rate vs B across certification methods.
3. fig03_certification_rate_vs_K.png: Certification rate (power) vs K across B=256 and B=1024.
4. fig04_selection_bias_vs_K.png: Selection bias (in normalized loss units) vs K for In-Sample, Fresh Remeasurement, and Independent Opt.
5. fig05_standardized_selection_bias_vs_K.png: Standardized selection bias (Z_sel in SE units) vs K across Pilot and Confirmatory splits.
6. fig06_bound_slack_vs_B.png: Bound slack vs B (in normalized loss and standard-error units) vs selection bias magnitude.
7. fig07_coverage_vs_nominal.png: One-sided coverage vs nominal confidence level (1 - delta in {0.80, 0.90, 0.95, 0.99}).
8. fig08_graph_level_variability.png: Graph-level variability of exact loss, observed loss, and certification margin across n=12 and n=14 graphs.
9. fig09_false_certification_vs_target_margin.png: False certification rate across target threshold / margin grid.
10. fig10_method_comparison.png: Multi-metric comparison of Naive EB, Bonferroni EB, Hoeffding, and Reference Student-t bounds (QAOA & Synthetic Control).
11. fig11_insample_vs_fresh_remeasurement.png: Paired trajectory comparison of In-Sample vs Fresh Remeasurement (Condition A vs Condition C).
12. fig12_shot_overhead_vs_reliability.png: Total shot overhead vs certification reliability & statistical power frontier.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certification.decisions import evaluate_certification_method


# Set clean publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = {
    "naive_eb": "#1b4f72",       # Deep Navy
    "bonferroni_eb": "#117864",  # Deep Teal
    "hoeffding": "#7d6608",      # Ochre / Olive
    "reference_t": "#b03a2e",    # Crimson Red
    "in_sample": "#c0392b",      # Red-Orange
    "fresh": "#1f618d",          # Steel Blue
    "ind_opt": "#148f77",        # Emerald
    "non_adap": "#5d6d7e",       # Slate Gray
}

METHOD_LABELS = {
    "naive_eb": "Naive Empirical-Bernstein (Method 1)",
    "bonferroni_eb": "Bonferroni EB (Method 2)",
    "hoeffding": "Hoeffding Bound (Method 3)",
    "reference_t": "Student-t Reference (Method 4, Non-Guaranteed)",
}


def generate_all_stage1_figures(
    base_dir: str = "/tmp/AMFS_2.0",
    persist_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    intermediate_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"),
):
    proc_dir = os.path.join(base_dir, "results_processed")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    agg_df = pd.read_csv(os.path.join(proc_dir, "aggregate_results.csv"))
    trial_df = pd.read_csv(os.path.join(proc_dir, "trial_summary.csv"))
    cert_df = pd.read_csv(os.path.join(proc_dir, "certification_results.csv"))
    remeas_df = pd.read_csv(os.path.join(proc_dir, "remeasurement_results.csv"))
    syn_df = pd.read_csv(os.path.join(proc_dir, "synthetic_control_results.csv"))
    exp_grid_df = pd.read_csv(os.path.join(proc_dir, "exploratory_threshold_grid.csv"))

    pooled_p2 = agg_df[(agg_df["qaoa_depth"] == 2) & (agg_df["graph_size_n"] == "pooled_12_14")].copy()

    # ---------------------------------------------------------
    # Figure 1: False Certification Rate vs K
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax_idx, split_name in enumerate(["pilot", "confirmatory"]):
        ax = axes[ax_idx]
        sub = pooled_p2[
            (pooled_p2["experiment_split"] == split_name)
            & (pooled_p2["evaluation_mode"] == "in_sample")
            & (pooled_p2["shot_budget"] == 1024)
        ]
        for m_idx, m in enumerate(["naive_eb", "bonferroni_eb", "hoeffding", "reference_t"]):
            m_df = sub[sub["certification_method"] == m].sort_values("candidate_prefix_K")
            ks = m_df["candidate_prefix_K"].to_numpy() + (m_idx - 1.5) * 1.2
            rates = m_df["false_certification_rate"].to_numpy()
            yerr_low = np.maximum(0.0, rates - m_df["false_certification_rate_ci_lower"].to_numpy())
            yerr_high = np.maximum(0.0, m_df["false_certification_rate_ci_upper"].to_numpy() - rates)
            ax.errorbar(
                ks, rates, yerr=[yerr_low, yerr_high],
                marker="o", capsize=3.5, linewidth=1.8,
                color=PALETTE[m], label=METHOD_LABELS[m]
            )
        ax.axhline(0.05, color="#333333", linestyle=":", linewidth=1.2, label="Nominal alpha = 0.05")
        ax.set_title(f"{split_name.capitalize()} Split (p=2, B=1024, Pooled n=12,14)")
        ax.set_xlabel("Candidate Prefix K")
        ax.set_xticks([25, 50, 100])
        ax.set_ylim(-0.02, 0.36)
        if ax_idx == 0:
            ax.set_ylabel("False Certification Rate P_FC (with 95% Upper Limit)")
            ax.legend(loc="upper left", frameon=True)
    fig.suptitle("Figure 1: False Certification Rate vs. Candidate Prefix K (In-Sample Selection)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig01_false_certification_vs_K.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 2: False Certification Rate vs B
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax_idx, split_name in enumerate(["pilot", "confirmatory"]):
        ax = axes[ax_idx]
        sub = pooled_p2[
            (pooled_p2["experiment_split"] == split_name)
            & (pooled_p2["evaluation_mode"] == "in_sample")
            & (pooled_p2["candidate_prefix_K"] == 100)
        ]
        x_pos = np.arange(2)
        width = 0.18
        for m_idx, m in enumerate(["naive_eb", "bonferroni_eb", "hoeffding", "reference_t"]):
            m_df = sub[sub["certification_method"] == m].sort_values("shot_budget")
            rates = m_df["false_certification_rate"].to_numpy()
            yerr_low = np.maximum(0.0, rates - m_df["false_certification_rate_ci_lower"].to_numpy())
            yerr_high = np.maximum(0.0, m_df["false_certification_rate_ci_upper"].to_numpy() - rates)
            ax.bar(
                x_pos + (m_idx - 1.5) * width, rates, width=width,
                yerr=[yerr_low, yerr_high], capsize=3.5,
                color=PALETTE[m], alpha=0.88, label=METHOD_LABELS[m]
            )
        ax.axhline(0.05, color="#333333", linestyle=":", linewidth=1.2)
        ax.set_title(f"{split_name.capitalize()} Split (K=100, p=2, Pooled n=12,14)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["B = 256 shots", "B = 1024 shots"])
        ax.set_ylim(0.0, 0.35)
        if ax_idx == 0:
            ax.set_ylabel("False Certification Rate P_FC")
            ax.legend(loc="upper left", frameon=True)
    fig.suptitle("Figure 2: False Certification Rate vs. Shot Budget B at K=100", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig02_false_certification_vs_B.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 3: Certification Rate (Power) vs K
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax_idx, B_val in enumerate([256, 1024]):
        ax = axes[ax_idx]
        for split_name, ls, marker in [("pilot", "--", "s"), ("confirmatory", "-", "o")]:
            sub = pooled_p2[
                (pooled_p2["experiment_split"] == split_name)
                & (pooled_p2["evaluation_mode"] == "in_sample")
                & (pooled_p2["shot_budget"] == B_val)
            ]
            for m in ["naive_eb", "bonferroni_eb", "hoeffding", "reference_t"]:
                m_df = sub[sub["certification_method"] == m].sort_values("candidate_prefix_K")
                lbl = f"{m} ({split_name[:4]})" if ax_idx == 0 else None
                ax.plot(
                    m_df["candidate_prefix_K"], m_df["certification_rate"],
                    linestyle=ls, marker=marker, linewidth=1.8, color=PALETTE[m], label=lbl
                )
        ax.set_title(f"Shot Budget B = {B_val} (p=2, In-Sample)")
        ax.set_xlabel("Candidate Prefix K")
        ax.set_xticks([25, 50, 100])
        ax.set_ylim(-0.03, 0.88)
        if ax_idx == 0:
            ax.set_ylabel("Certification Rate P(U <= tau)")
            ax.legend(ncol=2, loc="upper left", frameon=True, fontsize=7.8)
    fig.suptitle("Figure 3: Certification Rate vs. Candidate Prefix K Across Shot Budgets", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig03_certification_rate_vs_K.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 4: Selection Bias vs K in Normalized Loss Units
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax_idx, split_name in enumerate(["pilot", "confirmatory"]):
        ax = axes[ax_idx]
        sub = pooled_p2[
            (pooled_p2["experiment_split"] == split_name)
            & (pooled_p2["certification_method"] == "naive_eb")
        ]
        for B_val, ls in [(256, "-"), (1024, "--")]:
            for mode, col, m_lbl in [
                ("in_sample", PALETTE["in_sample"], "Condition A: In-Sample"),
                ("fresh_remeasurement", PALETTE["fresh"], "Condition C: Fresh Remeas."),
                ("independent_optimization", PALETTE["ind_opt"], "Condition D: Indep. Opt."),
            ]:
                m_df = sub[(sub["shot_budget"] == B_val) & (sub["evaluation_mode"] == mode)].sort_values("candidate_prefix_K")
                if len(m_df) == 0:
                    continue
                ks = m_df["candidate_prefix_K"].to_numpy()
                means = m_df["mean_selection_bias"].to_numpy()
                y_low = np.maximum(0.0, means - m_df["mean_selection_bias_ci_lower"].to_numpy())
                y_high = np.maximum(0.0, m_df["mean_selection_bias_ci_upper"].to_numpy() - means)
                ax.errorbar(
                    ks, means, yerr=[y_low, y_high],
                    linestyle=ls, marker="o" if B_val == 256 else "^",
                    capsize=3, linewidth=1.7, color=col,
                    label=f"{m_lbl} (B={B_val})"
                )
        ax.axhline(0.0, color="#222222", linestyle=":", linewidth=1.2)
        ax.set_title(f"{split_name.capitalize()} Split (p=2, Pooled n=12,14)")
        ax.set_xlabel("Candidate Prefix K")
        ax.set_xticks([25, 50, 100])
        if ax_idx == 0:
            ax.set_ylabel("Mean Selection Bias E[mu_hat - mu] (Loss Units)")
            ax.legend(loc="lower left", frameon=True, fontsize=7.5)
    fig.suptitle("Figure 4: Selection Bias vs. K in Normalized Loss Units (Negative = Optimism)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig04_selection_bias_vs_K.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 5: Standardized Selection Bias (Z_sel) vs K
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax_idx, split_name in enumerate(["pilot", "confirmatory"]):
        ax = axes[ax_idx]
        sub = pooled_p2[
            (pooled_p2["experiment_split"] == split_name)
            & (pooled_p2["certification_method"] == "naive_eb")
        ]
        for B_val, ls in [(256, "-"), (1024, "--")]:
            for mode, col, m_lbl in [
                ("in_sample", PALETTE["in_sample"], "Condition A: In-Sample"),
                ("fresh_remeasurement", PALETTE["fresh"], "Condition C: Fresh Remeas."),
            ]:
                m_df = sub[(sub["shot_budget"] == B_val) & (sub["evaluation_mode"] == mode)].sort_values("candidate_prefix_K")
                ks = m_df["candidate_prefix_K"].to_numpy()
                zmeans = m_df["mean_standardized_selection_bias"].to_numpy()
                y_low = np.maximum(0.0, zmeans - m_df["mean_standardized_selection_bias_ci_lower"].to_numpy())
                y_high = np.maximum(0.0, m_df["mean_standardized_selection_bias_ci_upper"].to_numpy() - zmeans)
                ax.errorbar(
                    ks, zmeans, yerr=[y_low, y_high],
                    linestyle=ls, marker="o" if B_val == 256 else "s",
                    capsize=3.5, linewidth=1.8, color=col,
                    label=f"{m_lbl} (B={B_val})"
                )
        ax.axhline(0.0, color="#222222", linestyle=":", linewidth=1.2)
        ax.axhline(-1.645, color="#b03a2e", linestyle="-.", linewidth=1.1, label="-1.645 SE (95% z-threshold)")
        ax.set_title(f"{split_name.capitalize()} Split (p=2, Pooled n=12,14)")
        ax.set_xlabel("Candidate Prefix K")
        ax.set_xticks([25, 50, 100])
        if ax_idx == 0:
            ax.set_ylabel("Standardized Selection Bias Z_sel (SE Units)")
            ax.legend(loc="lower left", frameon=True, fontsize=8)
    fig.suptitle("Figure 5: Standardized Selection Bias Z_sel vs. Candidate Prefix K", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig05_standardized_selection_bias_vs_K.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 6: Bound Slack vs Shot Budget B (and vs |Selection Bias|)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sub_b = pooled_p2[
        (pooled_p2["experiment_split"] == "confirmatory")
        & (pooled_p2["evaluation_mode"] == "in_sample")
        & (pooled_p2["candidate_prefix_K"] == 100)
    ]
    x_pos = np.arange(2)
    width = 0.18
    for m_idx, m in enumerate(["naive_eb", "bonferroni_eb", "hoeffding", "reference_t"]):
        m_df = sub_b[sub_b["certification_method"] == m].sort_values("shot_budget")
        axes[0].bar(
            x_pos + (m_idx - 1.5) * width, m_df["mean_bound_slack"], width=width,
            color=PALETTE[m], alpha=0.88, label=METHOD_LABELS[m]
        )
        axes[1].bar(
            x_pos + (m_idx - 1.5) * width, m_df["mean_bound_slack_standard_errors"], width=width,
            color=PALETTE[m], alpha=0.88, label=METHOD_LABELS[m]
        )
    # Overlay |mean_selection_bias| to show why EB covers while Reference undercovers!
    bias_mag = np.abs(sub_b[sub_b["certification_method"] == "naive_eb"].sort_values("shot_budget")["mean_selection_bias"].to_numpy())
    zbias_mag = np.abs(sub_b[sub_b["certification_method"] == "naive_eb"].sort_values("shot_budget")["mean_standardized_selection_bias"].to_numpy())
    axes[0].plot(x_pos, bias_mag, "k--D", linewidth=2.0, markersize=7, label="|In-Sample Selection Bias|")
    axes[1].plot(x_pos, zbias_mag, "k--D", linewidth=2.0, markersize=7, label="|Standardized Bias Z_sel|")

    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(["B = 256", "B = 1024"])
    axes[0].set_ylabel("Mean Bound Slack (U - mu_hat) [Loss Units]")
    axes[0].set_title("A. Bound Slack in Normalized Loss Units (K=100)")
    axes[0].legend(loc="upper right", frameon=True, fontsize=7.8)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["B = 256", "B = 1024"])
    axes[1].set_ylabel("Mean Bound Slack [(U - mu_hat) / SE]")
    axes[1].set_title("B. Bound Slack in Standard-Error Units (K=100)")
    fig.suptitle("Figure 6: Bound Slack vs. Shot Budget B Compared Against Selection Bias Magnitude", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig06_bound_slack_vs_B.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 7: One-Sided Coverage vs Nominal Confidence Level (1 - delta)
    # ---------------------------------------------------------
    deltas_grid = [0.20, 0.10, 0.05, 0.01]
    nom_levels = [1.0 - d for d in deltas_grid]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    # Panel A: QAOA Confirmatory (K=100, B=1024, In-Sample)
    # Panel B: Synthetic Control Bernoulli(0.30) (K=100, B=1024, In-Sample)
    qaoa_k100 = cert_df[
        (cert_df["experiment_split"] == "confirmatory")
        & (cert_df["qaoa_depth"] == 2)
        & (cert_df["shot_budget"] == 1024)
        & (cert_df["candidate_prefix_K"] == 100)
        & (cert_df["evaluation_mode"] == "in_sample")
        & (cert_df["certification_method"] == "naive_eb")
    ]
    for m in ["naive_eb", "bonferroni_eb", "hoeffding", "reference_t"]:
        cov_qaoa = []
        for d_val in deltas_grid:
            covered_flags = []
            for _, r in qaoa_k100.iterrows():
                c_res = evaluate_certification_method(
                    m, r["observed_mean"], r["observed_variance"], 1024, d_val, 100, r["target_threshold"]
                )
                covered_flags.append(r["exact_expected_loss"] <= c_res["upper_confidence_bound"] + 1e-15)
            cov_qaoa.append(np.mean(covered_flags))
        axes[0].plot(nom_levels, cov_qaoa, marker="o", linewidth=2.0, color=PALETTE[m], label=METHOD_LABELS[m])

    syn_b1024 = syn_df[
        (syn_df["synthetic_regime"] == "standard_bernoulli_0_1")
        & (syn_df["shots_per_candidate"] == 1024)
        & (syn_df["candidate_count_K"] == 100)
    ]
    for m in ["naive_eb", "bonferroni_eb", "hoeffding", "reference_t"]:
        cov_syn = []
        for d_val in deltas_grid:
            covered_flags = []
            for _, r in syn_b1024.iterrows():
                c_res = evaluate_certification_method(
                    m, r["selected_observed_mean"], r["selected_observed_variance"], 1024, d_val, 100, 0.30
                )
                covered_flags.append(0.30 <= c_res["upper_confidence_bound"] + 1e-15)
            cov_syn.append(np.mean(covered_flags))
        axes[1].plot(nom_levels, cov_syn, marker="s", linewidth=2.0, color=PALETTE[m], label=METHOD_LABELS[m])

    for ax_idx, title_str in enumerate([
        "A. QAOA MaxCut Confirmatory (K=100, B=1024)",
        "B. Synthetic Bernoulli(0.30) Control (K=100, B=1024)",
    ]):
        axes[ax_idx].plot(nom_levels, nom_levels, "k:", linewidth=1.5, label="Ideal Nominal Coverage (y = 1 - delta)")
        axes[ax_idx].set_title(title_str)
        axes[ax_idx].set_xlabel("Nominal Confidence Level (1 - delta)")
        axes[ax_idx].set_xticks(nom_levels)
        axes[ax_idx].set_ylim(-0.05, 1.05)
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("Empirical One-Sided Coverage P(mu <= U)")
            axes[ax_idx].legend(loc="center left", frameon=True, fontsize=8)
    fig.suptitle("Figure 7: Empirical One-Sided Coverage vs. Nominal Confidence Level at K=100", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig07_coverage_vs_nominal.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 8: Graph-Level Variability Across n=12 and n=14
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax_idx, n_val in enumerate([12, 14]):
        ax = axes[ax_idx]
        sub_g = trial_df[
            (trial_df["qaoa_depth"] == 2)
            & (trial_df["shot_budget"] == 1024)
            & (trial_df["candidate_prefix_K"] == 100)
            & (trial_df["evaluation_mode"] == "in_sample")
            & (trial_df["certification_method"] == "naive_eb")
            & (trial_df["graph_size_n"] == n_val)
        ].sort_values(["experiment_split", "selected_exact_loss"])

        x_idx = np.arange(len(sub_g))
        tau_val = sub_g["target_threshold"].iloc[0]
        pilot_mask = sub_g["experiment_split"] == "pilot"

        ax.scatter(
            x_idx[pilot_mask], sub_g[pilot_mask]["selected_exact_loss"],
            color="#1b4f72", marker="o", s=45, label="Exact Ground-Truth Loss mu(theta_hat) [Pilot]"
        )
        ax.scatter(
            x_idx[~pilot_mask], sub_g[~pilot_mask]["selected_exact_loss"],
            color="#117864", marker="D", s=45, label="Exact Ground-Truth Loss mu(theta_hat) [Confirmatory]"
        )
        ax.scatter(
            x_idx, sub_g["selected_observed_loss"],
            color="#c0392b", marker="v", s=35, alpha=0.8, label="In-Sample Observed Loss mu_hat"
        )
        ax.errorbar(
            x_idx, sub_g["selected_observed_loss"],
            yerr=[np.zeros(len(sub_g)), sub_g["bound_slack"].to_numpy()],
            fmt="none", ecolor="#7f8c8d", elinewidth=1.2, capsize=2.5, alpha=0.7, label="Naive EB Upper Bound U_EB"
        )
        ax.axhline(tau_val, color="#222222", linestyle="--", linewidth=1.5, label=f"Frozen Target Threshold tau_{n_val} = {tau_val:.4f}")
        ax.set_title(f"3-Regular Graphs: n = {n_val} (10 Pilot + 10 Confirmatory)")
        ax.set_xlabel("Graph Index (Ordered by Split & Exact Loss)")
        if ax_idx == 0:
            ax.set_ylabel("Normalized MaxCut Loss L(x)")
            ax.legend(loc="upper left", frameon=True, fontsize=7.5)
    fig.suptitle("Figure 8: Graph-Level Heterogeneity of Exact Loss, Observed Loss, and Naive EB Bounds (p=2, B=1024, K=100)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig08_graph_level_variability.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 9: False Certification vs Target Threshold / Target Margin
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax_idx, B_val in enumerate([256, 1024]):
        ax = axes[ax_idx]
        sub_e = exp_grid_df[
            (exp_grid_df["experiment_split"] == "confirmatory")
            & (exp_grid_df["shot_budget"] == B_val)
            & (exp_grid_df["candidate_prefix_K"] == 100)
        ]
        for m in ["naive_eb", "bonferroni_eb", "hoeffding", "reference_t"]:
            m_grp = sub_e[sub_e["certification_method"] == m].groupby("exploratory_tau")[
                ["in_sample_false_certification", "fresh_false_certification"]
            ].mean().reset_index()
            ax.plot(
                m_grp["exploratory_tau"], m_grp["in_sample_false_certification"],
                linestyle="-", marker="o", markersize=4, linewidth=1.8, color=PALETTE[m],
                label=f"{METHOD_LABELS[m]} (In-Sample)"
            )
            if m == "reference_t":
                ax.plot(
                    m_grp["exploratory_tau"], m_grp["fresh_false_certification"],
                    linestyle="--", marker="^", markersize=4, linewidth=1.5, color="#1f618d",
                    label="Student-t Reference (Fresh Remeas.)"
                )
        ax.axvline(0.2619, color="#555555", linestyle=":", linewidth=1.2, label="Frozen tau_12 = 0.2619")
        ax.axvline(0.2687, color="#222222", linestyle="-.", linewidth=1.2, label="Frozen tau_14 = 0.2687")
        ax.set_title(f"Confirmatory Split: B = {B_val}, K = 100 (p=2)")
        ax.set_xlabel("Target Threshold tau (Exploratory Sensitivity Grid)")
        ax.set_ylim(-0.02, 0.45)
        if ax_idx == 0:
            ax.set_ylabel("False Certification Rate P(U <= tau and mu > tau)")
            ax.legend(loc="upper left", frameon=True, fontsize=7.5)
    fig.suptitle("Figure 9: False Certification Rate Across Target Threshold / Margin Regimes", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig09_false_certification_vs_target_margin.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 10: Comparison of Naive EB, Bonferroni EB, Hoeffding, and Reference Bounds
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    # Panel A: QAOA Confirmatory K=100, B=1024 (Cert Rate, Correct Cert Rate, False Cert Rate, One-Sided Undercoverage)
    sub_m = pooled_p2[
        (pooled_p2["experiment_split"] == "confirmatory")
        & (pooled_p2["shot_budget"] == 1024)
        & (pooled_p2["candidate_prefix_K"] == 100)
        & (pooled_p2["evaluation_mode"] == "in_sample")
    ].set_index("certification_method").loc[["naive_eb", "bonferroni_eb", "hoeffding", "reference_t"]]

    metrics_to_plot = [
        ("certification_rate", "Total Cert. Rate", "#2874a6"),
        ("correct_certification_rate", "Correct Cert. Rate", "#1e8449"),
        ("false_certification_rate", "False Cert. Rate (P_FC)", "#cb4335"),
    ]
    x_pos = np.arange(4)
    width = 0.22
    for idx_m, (col_name, lbl, col_hex) in enumerate(metrics_to_plot):
        axes[0].bar(x_pos + (idx_m - 1) * width, sub_m[col_name], width=width, color=col_hex, label=lbl)
    axes[0].plot(x_pos, 1.0 - sub_m["one_sided_coverage"], "k--o", linewidth=1.8, label="1 - One-Sided Coverage")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(["Naive EB", "Bonferroni EB", "Hoeffding", "Student-t Ref"])
    axes[0].set_ylabel("Probability / Rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("A. QAOA Confirmatory (p=2, B=1024, K=100, In-Sample)")
    axes[0].legend(loc="upper left", frameon=True, fontsize=8)

    # Panel B: Synthetic Control Standard Bernoulli(0.30) Undercoverage / False Cert vs K at B=1024
    syn_sub = syn_df[
        (syn_df["synthetic_regime"] == "standard_bernoulli_0_1")
        & (syn_df["shots_per_candidate"] == 1024)
    ].groupby("candidate_count_K")[
        ["false_certification", "bonferroni_false_certification", "hoeffding_false_certification", "reference_false_certification", "fresh_false_certification"]
    ].mean().reset_index()

    axes[1].plot(syn_sub["candidate_count_K"], syn_sub["false_certification"], "-o", color=PALETTE["naive_eb"], linewidth=2.0, label="Naive EB (In-Sample)")
    axes[1].plot(syn_sub["candidate_count_K"], syn_sub["bonferroni_false_certification"], "-s", color=PALETTE["bonferroni_eb"], linewidth=2.0, label="Bonferroni EB (In-Sample)")
    axes[1].plot(syn_sub["candidate_count_K"], syn_sub["hoeffding_false_certification"], "-^", color=PALETTE["hoeffding"], linewidth=2.0, label="Hoeffding (In-Sample)")
    axes[1].plot(syn_sub["candidate_count_K"], syn_sub["reference_false_certification"], "-D", color=PALETTE["reference_t"], linewidth=2.0, label="Student-t Ref (In-Sample)")
    axes[1].plot(syn_sub["candidate_count_K"], syn_sub["fresh_false_certification"], "--o", color="#5d6d7e", linewidth=1.8, label="Naive EB (Fresh Remeas.)")
    axes[1].axhline(0.05, color="#222222", linestyle=":", linewidth=1.2, label="Nominal delta = 0.05")
    axes[1].set_xlabel("Candidate Count K")
    axes[1].set_ylabel("False Certification / Undercoverage Rate")
    axes[1].set_title("B. Condition E Synthetic Control (Bernoulli 0.30, B=1024)")
    axes[1].legend(loc="center right", frameon=True, fontsize=7.8)

    fig.suptitle("Figure 10: Direct Comparison of Naive EB, Bonferroni EB, Hoeffding, and Reference Bounds", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig10_method_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 11: In-Sample vs Fresh-Remeasurement Certification (Condition A vs Condition C)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    # Merge in-sample and fresh_remeasurement for p=2, K=100, B=1024
    in_df = trial_df[
        (trial_df["qaoa_depth"] == 2)
        & (trial_df["shot_budget"] == 1024)
        & (trial_df["candidate_prefix_K"] == 100)
        & (trial_df["evaluation_mode"] == "in_sample")
        & (trial_df["certification_method"] == "naive_eb")
    ].copy()
    fr_df = trial_df[
        (trial_df["qaoa_depth"] == 2)
        & (trial_df["shot_budget"] == 1024)
        & (trial_df["candidate_prefix_K"] == 100)
        & (trial_df["evaluation_mode"] == "fresh_remeasurement")
        & (trial_df["certification_method"] == "naive_eb")
    ].copy()
    paired = pd.merge(
        in_df, fr_df,
        on=["experiment_split", "graph_id", "trial_id", "selected_candidate_id", "selected_exact_loss"],
        suffixes=("_insample", "_fresh")
    )

    axes[0].scatter(
        paired["selected_exact_loss"], paired["selected_observed_loss_insample"],
        color=PALETTE["in_sample"], alpha=0.85, s=42, marker="v", label="Condition A: In-Sample Mean (mu_hat)"
    )
    axes[0].scatter(
        paired["selected_exact_loss"], paired["selected_observed_loss_fresh"],
        color=PALETTE["fresh"], alpha=0.85, s=42, marker="o", label="Condition C: Fresh Remeasured Mean"
    )
    lims = [0.225, 0.295]
    axes[0].plot(lims, lims, "k--", linewidth=1.3, label="Unbiased Identity Line (y = mu)")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("Exact Ground-Truth Loss mu(theta_hat_100)")
    axes[0].set_ylabel("Observed Finite-Shot Loss Estimate")
    axes[0].set_title("A. Paired Selected Candidates (p=2, B=1024, K=100)")
    axes[0].legend(loc="upper left", frameon=True, fontsize=8)

    # Panel B: Distribution of Estimation Error (Observed - Exact) for In-Sample vs Fresh Remeasurement
    bins = np.linspace(-0.025, 0.015, 22)
    axes[1].hist(
        paired["selection_bias_insample"], bins=bins, color=PALETTE["in_sample"],
        alpha=0.65, edgecolor="black", label=f"In-Sample Error (Mean = {paired['selection_bias_insample'].mean():.4f})"
    )
    axes[1].hist(
        paired["selection_bias_fresh"], bins=bins, color=PALETTE["fresh"],
        alpha=0.65, edgecolor="black", label=f"Fresh Remeas. Error (Mean = {paired['selection_bias_fresh'].mean():.4f})"
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.4)
    axes[1].set_xlabel("Estimation Error (mu_hat - mu) [Normalized Loss Units]")
    axes[1].set_ylabel("Number of Trajectories (Pilot + Confirmatory)")
    axes[1].set_title("B. Elimination of Winner's Curse via Fresh Remeasurement")
    axes[1].legend(loc="upper left", frameon=True, fontsize=8)

    fig.suptitle("Figure 11: In-Sample Selection vs. Fresh Independent Remeasurement (Condition A vs. Condition C)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig11_insample_vs_fresh_remeasurement.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 12: Shot Overhead vs Certification Reliability & Power
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sub_cost = pooled_p2[pooled_p2["experiment_split"] == "confirmatory"].copy()

    for mode, marker, m_lbl in [
        ("non_adaptive_k1", "P", "Cond B: Non-Adaptive K=1"),
        ("in_sample", "o", "Cond A: In-Sample (K=25,50,100)"),
        ("fresh_remeasurement", "s", "Cond C: Fresh Remeas. (+B shots)"),
    ]:
        for m in ["naive_eb", "bonferroni_eb", "reference_t"]:
            df_c = sub_cost[(sub_cost["evaluation_mode"] == mode) & (sub_cost["certification_method"] == m)].sort_values("mean_total_shots")
            if len(df_c) == 0:
                continue
            lbl = f"{m} | {m_lbl}" if mode != "non_adaptive_k1" or m == "naive_eb" else None
            axes[0].plot(
                df_c["mean_total_shots"] / 1000.0, df_c["one_sided_coverage"],
                marker=marker, linewidth=1.6, color=PALETTE[m],
                linestyle="-" if mode == "in_sample" else "--", label=lbl
            )
            axes[1].plot(
                df_c["mean_total_shots"] / 1000.0, df_c["correct_certification_rate"],
                marker=marker, linewidth=1.6, color=PALETTE[m],
                linestyle="-" if mode == "in_sample" else "--", label=lbl
            )

    axes[0].axhline(0.95, color="black", linestyle=":", linewidth=1.3, label="95% Nominal Coverage Floor")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Total Measurement Shot Cost (Thousands of Shots, Log Scale)")
    axes[0].set_ylabel("One-Sided Upper-Bound Coverage P(mu <= U)")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_title("A. Total Shot Cost vs. Statistical Coverage Reliability")
    axes[0].legend(loc="lower left", frameon=True, fontsize=7.2)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Total Measurement Shot Cost (Thousands of Shots, Log Scale)")
    axes[1].set_ylabel("Correct Certification Rate P(U <= tau and mu <= tau)")
    axes[1].set_ylim(-0.02, 0.65)
    axes[1].set_title("B. Total Shot Cost vs. Verified True-Positive Power")

    fig.suptitle("Figure 12: Total Measurement Shot Cost vs. Certification Reliability and Statistical Power", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig12_shot_overhead_vs_reliability.png"), bbox_inches="tight")
    plt.close(fig)

    # Copy all 12 figures to persistent AMFS_2.0/figures and intermediate folder for PDF compilation
    persist_fig_dir = os.path.join(persist_dir, "figures")
    os.makedirs(persist_fig_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    generated_files = sorted(os.listdir(fig_dir))
    for fname in generated_files:
        src = os.path.join(fig_dir, fname)
        shutil.copy2(src, os.path.join(persist_fig_dir, fname))
        shutil.copy2(src, os.path.join(intermediate_dir, fname))
    print(f"Generated and synced {len(generated_files)} figures:", generated_files)


if __name__ == "__main__":
    generate_all_stage1_figures()
