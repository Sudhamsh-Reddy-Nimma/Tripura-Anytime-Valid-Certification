"""
Stage 2 Statistical Analysis & Graph-Cluster Bootstrap Engine for AMFS 2.0
(Stage 2, Sections 15, 16, 17).

Explicitly distinguishes and documents all 4 standard-error denominators (Section 15):
1. Per-candidate exact sampling standard error:
       SE_exact(theta*) = sqrt(V_exact(theta*) / B)
   -> Used for Exact Standardized Selection Bias:
       Z_exact = (mu_hat_disc(theta*) - mu_exact(theta*)) / sqrt(V_exact(theta*) / B)
2. Per-candidate empirical sampling standard error:
       SE_emp(theta*) = sqrt(V_hat_disc(theta*) / B)
   -> Used for Empirical Standardized Selection Bias:
       Z_emp = (mu_hat_disc(theta*) - mu_exact(theta*)) / sqrt(V_hat_disc(theta*) / B)
3. Graph-level cluster standard error of the mean bias across G independent graphs:
       SE_graph_mean = SD_between_graphs(b_g) / sqrt(G)
   where b_g = (1/3) sum_{r=1}^3 b_{g,r} is the mean bias on graph g across its 3 optimizer repetitions.
4. Stratified 10,000-resample graph-cluster bootstrap standard error (SE_boot) and 95% percentile CIs
   (resampling the G graphs with replacement while keeping all 3 optimizer repetitions and 4 K-checkpoints
   within each resampled graph intact).
"""

import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from analysis.bootstrap import clopper_pearson_upper_limit


def run_stage2_statistical_analysis(
    base_dir: str = "/tmp/AMFS_2.0",
    persist_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    num_resamples: int = 10000,
    master_seed: int = 20261102,
) -> Dict[str, Any]:
    s2_proc = os.path.join(base_dir, "results_stage2_processed")
    sel_df = pd.read_csv(os.path.join(s2_proc, "selected_candidates.csv"))
    gt_df = pd.read_csv(os.path.join(s2_proc, "ground_truth.csv"))
    conf_df = pd.read_csv(os.path.join(s2_proc, "confirmation_results.csv"))
    cert_df = pd.read_csv(os.path.join(s2_proc, "certification_results.csv"))

    # Merge selected_candidates, ground_truth, and confirmation_results for checkpoint-level bias analysis
    chk_df = pd.merge(
        sel_df,
        gt_df[["experiment_id", "K_checkpoint", "exact_expected_loss", "exact_loss_variance", "exact_expected_cut"]],
        on=["experiment_id", "K_checkpoint"],
    )
    chk_df = pd.merge(
        chk_df,
        conf_df[[
            "experiment_id", "K_checkpoint", "confirmation_shots", "confirmation_mean",
            "confirmation_variance", "confirmation_se", "confirmation_upper_bound",
            "certification_decision", "target_threshold", "genuinely_successful",
            "false_certification", "covered", "total_shots",
        ]],
        on=["experiment_id", "K_checkpoint"],
    )

    # Compute raw and standardized biases with explicitly defined denominators (Section 15)
    chk_df["discovery_bias"] = chk_df["selected_discovery_loss"] - chk_df["exact_expected_loss"]
    chk_df["confirmation_bias"] = chk_df["confirmation_mean"] - chk_df["exact_expected_loss"]

    chk_df["per_candidate_exact_se_disc"] = np.sqrt(np.maximum(1e-15, chk_df["exact_loss_variance"]) / chk_df["B"].astype(float))
    chk_df["per_candidate_emp_se_disc"] = np.sqrt(np.maximum(1e-15, chk_df["selected_discovery_variance"]) / chk_df["B"].astype(float))
    chk_df["per_candidate_exact_se_conf"] = np.sqrt(np.maximum(1e-15, chk_df["exact_loss_variance"]) / chk_df["confirmation_shots"].astype(float))

    chk_df["standardized_discovery_bias_exact_se"] = chk_df["discovery_bias"] / chk_df["per_candidate_exact_se_disc"]
    chk_df["standardized_discovery_bias_emp_se"] = chk_df["discovery_bias"] / chk_df["per_candidate_emp_se_disc"]
    chk_df["standardized_confirmation_bias_exact_se"] = chk_df["confirmation_bias"] / chk_df["per_candidate_exact_se_conf"]

    chk_df.to_csv(os.path.join(s2_proc, "stage2_checkpoint_master.csv"), index=False)

    # 1. Build Bias & Confirmation Summary Table across (graph_size_stratum, B, K_checkpoint)
    ss = np.random.SeedSequence([master_seed, 9001])
    bias_summary_rows = []

    for stratum_label, filter_sizes in [("12", [12]), ("14", [14]), ("pooled_12_14", [12, 14])]:
        sub_s = chk_df[chk_df["graph_size"].isin(filter_sizes)]
        for (B_val, K_val), cell_df in sub_s.groupby(["B", "K_checkpoint"], sort=True):
            rng = np.random.default_rng(ss.spawn(1)[0])
            row = _bootstrap_bias_and_confirmation_cell(
                cell_df=cell_df,
                stratum_label=stratum_label,
                B_val=int(B_val),
                K_val=int(K_val),
                num_resamples=num_resamples,
                rng=rng,
            )
            bias_summary_rows.append(row)

    bias_summary_df = pd.DataFrame(bias_summary_rows)
    bias_summary_df.to_csv(os.path.join(s2_proc, "stage2_bias_and_confirmation_summary.csv"), index=False)

    # 2. Build Certification Methods Comparison Summary across (stratum_label, B, K_checkpoint, evaluation_regime, method)
    cert_summary_rows = []
    for stratum_label, filter_sizes in [("12", [12]), ("14", [14]), ("pooled_12_14", [12, 14])]:
        sub_c = cert_df[cert_df["graph_size"].isin(filter_sizes)]
        for (B_val, K_val, regime, method), cell_df in sub_c.groupby(
            ["B", "K_checkpoint", "evaluation_regime", "method"], sort=True
        ):
            rng = np.random.default_rng(ss.spawn(1)[0])
            row = _bootstrap_certification_cell(
                cell_df=cell_df,
                stratum_label=stratum_label,
                B_val=int(B_val),
                K_val=int(K_val),
                regime=str(regime),
                method=str(method),
                num_resamples=num_resamples,
                rng=rng,
            )
            cert_summary_rows.append(row)

    cert_summary_df = pd.DataFrame(cert_summary_rows)
    cert_summary_df.to_csv(os.path.join(s2_proc, "stage2_certification_summary.csv"), index=False)

    # 3. Fixed-Total-Discovery-Budget Comparison (M_discovery = 25,600 shots: (B=256, K=100) vs (B=512, K=50) vs (B=1024, K=25))
    fixed_budget_pairs = [(256, 100), (512, 50), (1024, 25)]
    fb_rows = []
    for B_val, K_val in fixed_budget_pairs:
        b_row = bias_summary_df[
            (bias_summary_df["graph_size_stratum"] == "pooled_12_14")
            & (bias_summary_df["B"] == B_val)
            & (bias_summary_df["K_checkpoint"] == K_val)
        ].iloc[0].to_dict()
        for m_name in ["in_sample_naive_eb", "in_sample_bonferroni_eb", "in_sample_student_t", "independent_confirm_eb"]:
            c_row = cert_summary_df[
                (cert_summary_df["graph_size_stratum"] == "pooled_12_14")
                & (cert_summary_df["B"] == B_val)
                & (cert_summary_df["K_checkpoint"] == K_val)
                & (cert_summary_df["method"] == m_name)
            ].iloc[0]
            b_row[f"{m_name}_cert_rate"] = float(c_row["certification_rate"])
            b_row[f"{m_name}_fc_rate"] = float(c_row["false_certification_rate"])
            b_row[f"{m_name}_coverage"] = float(c_row["empirical_coverage"])
        fb_rows.append(b_row)
    fb_df = pd.DataFrame(fb_rows)
    fb_df.to_csv(os.path.join(s2_proc, "stage2_fixed_budget_25600_comparison.csv"), index=False)

    # 4. Graph-Level Repeated-Measures Trend & Mixed-Effects Regression Analysis (Section 16)
    trend_stats = _compute_repeated_measures_trends(chk_df)
    with open(os.path.join(s2_proc, "stage2_inferential_tests.json"), "w") as f:
        json.dump(trend_stats, f, indent=2)

    # Sync all processed Stage 2 files to persistent directory
    p_proc = os.path.join(persist_dir, "results_stage2_processed")
    os.makedirs(p_proc, exist_ok=True)
    for fn in [
        "stage2_checkpoint_master.csv",
        "stage2_bias_and_confirmation_summary.csv",
        "stage2_certification_summary.csv",
        "stage2_fixed_budget_25600_comparison.csv",
        "stage2_inferential_tests.json",
    ]:
        src_f = os.path.join(s2_proc, fn)
        if os.path.exists(src_f):
            import shutil
            shutil.copy2(src_f, os.path.join(p_proc, fn))

    return trend_stats


def _bootstrap_bias_and_confirmation_cell(
    cell_df: pd.DataFrame,
    stratum_label: str,
    B_val: int,
    K_val: int,
    num_resamples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    cols = [
        "selected_discovery_loss",
        "confirmation_mean",
        "exact_expected_loss",
        "discovery_bias",
        "confirmation_bias",
        "standardized_discovery_bias_exact_se",
        "standardized_discovery_bias_emp_se",
        "standardized_confirmation_bias_exact_se",
        "per_candidate_exact_se_disc",
        "per_candidate_emp_se_disc",
    ]
    strata = sorted(cell_df["graph_size"].unique())
    n_strata = len(strata)
    n_graphs = int(cell_df["graph_id"].nunique())
    n_runs = int(len(cell_df))

    # Per-graph means (clustering the 3 optimizer seeds within each graph)
    graph_means_df = cell_df.groupby(["graph_size", "graph_id"], sort=True)[cols].mean().reset_index()

    stratum_mats = []
    for s in strata:
        mat = graph_means_df[graph_means_df["graph_size"] == s][cols].to_numpy(dtype=np.float64)
        G_s = mat.shape[0]
        counts = rng.multinomial(G_s, [1.0 / G_s] * G_s, size=num_resamples)
        weights = counts.astype(np.float64) / float(G_s)
        stratum_mats.append(weights @ mat)

    boot_pooled = sum(stratum_mats) / float(n_strata)
    c_idx = {c: i for i, c in enumerate(cols)}

    # Point estimates (equal weighting across strata)
    pt = graph_means_df.groupby("graph_size")[cols].mean().mean(axis=0)

    # Graph-level cluster standard error of the mean bias: SD_between_graphs(b_g) / sqrt(G)
    between_graph_sd_disc_bias = float(graph_means_df["discovery_bias"].std(ddof=1))
    graph_cluster_se_disc_bias = float(between_graph_sd_disc_bias / np.sqrt(n_graphs))
    boot_se_disc_bias = float(np.std(boot_pooled[:, c_idx["discovery_bias"]], ddof=1))

    def ci95(col_name: str) -> Tuple[float, float]:
        arr = boot_pooled[:, c_idx[col_name]]
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    db_low, db_high = ci95("discovery_bias")
    cb_low, cb_high = ci95("confirmation_bias")
    zex_low, zex_high = ci95("standardized_discovery_bias_exact_se")
    zemp_low, zemp_high = ci95("standardized_discovery_bias_emp_se")
    zconf_low, zconf_high = ci95("standardized_confirmation_bias_exact_se")

    return {
        "graph_size_stratum": stratum_label,
        "B": B_val,
        "K_checkpoint": K_val,
        "number_of_graphs": n_graphs,
        "number_of_trajectories": n_runs,
        "discovery_shots_per_trajectory": int(B_val * K_val),
        "confirmation_shots_per_checkpoint": int(cell_df["confirmation_shots"].iloc[0]),
        "total_shots_per_checkpoint": int(B_val * K_val + cell_df["confirmation_shots"].iloc[0]),
        "mean_discovery_loss": float(pt["selected_discovery_loss"]),
        "mean_confirmation_loss": float(pt["confirmation_mean"]),
        "mean_exact_loss": float(pt["exact_expected_loss"]),
        "mean_discovery_bias": float(pt["discovery_bias"]),
        "discovery_bias_ci_lower": db_low,
        "discovery_bias_ci_upper": db_high,
        "mean_confirmation_bias": float(pt["confirmation_bias"]),
        "confirmation_bias_ci_lower": cb_low,
        "confirmation_bias_ci_upper": cb_high,
        "mean_standardized_discovery_bias_exact_se": float(pt["standardized_discovery_bias_exact_se"]),
        "standardized_discovery_bias_exact_se_ci_lower": zex_low,
        "standardized_discovery_bias_exact_se_ci_upper": zex_high,
        "mean_standardized_discovery_bias_emp_se": float(pt["standardized_discovery_bias_emp_se"]),
        "standardized_discovery_bias_emp_se_ci_lower": zemp_low,
        "standardized_discovery_bias_emp_se_ci_upper": zemp_high,
        "mean_standardized_confirmation_bias_exact_se": float(pt["standardized_confirmation_bias_exact_se"]),
        "standardized_confirmation_bias_exact_se_ci_lower": zconf_low,
        "standardized_confirmation_bias_exact_se_ci_upper": zconf_high,
        # Explicitly separated 4 standard-error quantities (Section 15)
        "denom1_mean_per_candidate_exact_se": float(pt["per_candidate_exact_se_disc"]),
        "denom2_mean_per_candidate_emp_se": float(pt["per_candidate_emp_se_disc"]),
        "denom3_graph_cluster_se_of_mean_bias": graph_cluster_se_disc_bias,
        "denom4_bootstrap_se_of_mean_bias": boot_se_disc_bias,
        "between_graph_sd_discovery_bias": between_graph_sd_disc_bias,
        "between_graph_sd_exact_loss": float(graph_means_df["exact_expected_loss"].std(ddof=1)),
    }


def _bootstrap_certification_cell(
    cell_df: pd.DataFrame,
    stratum_label: str,
    B_val: int,
    K_val: int,
    regime: str,
    method: str,
    num_resamples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    cols = [
        "certified",
        "false_certification",
        "false_non_certification",
        "covered",
        "bound_slack",
        "genuinely_successful",
    ]
    strata = sorted(cell_df["graph_size"].unique())
    n_strata = len(strata)
    n_graphs = int(cell_df["graph_id"].nunique())
    n_runs = int(len(cell_df))
    n_cert = int(cell_df["certified"].sum())
    n_fc = int(cell_df["false_certification"].sum())
    n_fn = int(cell_df["false_non_certification"].sum())

    graph_means_df = cell_df.groupby(["graph_size", "graph_id"], sort=True)[cols].mean().reset_index()

    stratum_mats = []
    for s in strata:
        mat = graph_means_df[graph_means_df["graph_size"] == s][cols].to_numpy(dtype=np.float64)
        G_s = mat.shape[0]
        counts = rng.multinomial(G_s, [1.0 / G_s] * G_s, size=num_resamples)
        weights = counts.astype(np.float64) / float(G_s)
        stratum_mats.append(weights @ mat)

    boot_pooled = sum(stratum_mats) / float(n_strata)
    c_idx = {c: i for i, c in enumerate(cols)}
    pt = graph_means_df.groupby("graph_size")[cols].mean().mean(axis=0)

    def get_ci(col_name: str, count_val: int = None) -> Tuple[float, float]:
        arr = boot_pooled[:, c_idx[col_name]]
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        if count_val is not None and count_val == 0:
            # Report conservative graph-level Clopper-Pearson upper limit when 0 events occur
            high = clopper_pearson_upper_limit(0, n_graphs, alpha=0.05)
            return 0.0, high
        return low, high

    fc_low, fc_high = get_ci("false_certification", count_val=n_fc)
    cert_low, cert_high = get_ci("certified", count_val=n_cert)
    fn_low, fn_high = get_ci("false_non_certification", count_val=n_fn)
    cov_low, cov_high = get_ci("covered")
    slack_low, slack_high = get_ci("bound_slack")

    fdr = float(n_fc / float(n_cert)) if n_cert > 0 else np.nan

    return {
        "graph_size_stratum": stratum_label,
        "B": B_val,
        "K_checkpoint": K_val,
        "evaluation_regime": regime,
        "method": method,
        "number_of_graphs": n_graphs,
        "number_of_trajectories": n_runs,
        "number_of_certifications": n_cert,
        "number_of_false_certifications": n_fc,
        "number_of_false_non_certifications": n_fn,
        "certification_rate": float(pt["certified"]),
        "certification_rate_ci_lower": cert_low,
        "certification_rate_ci_upper": cert_high,
        "false_certification_rate": float(pt["false_certification"]),
        "false_certification_rate_ci_lower": fc_low,
        "false_certification_rate_ci_upper": fc_high,
        "conditional_false_discovery_rate": fdr,
        "false_non_certification_rate": float(pt["false_non_certification"]),
        "false_non_certification_rate_ci_lower": fn_low,
        "false_non_certification_rate_ci_upper": fn_high,
        "empirical_coverage": float(pt["covered"]),
        "empirical_coverage_ci_lower": cov_low,
        "empirical_coverage_ci_upper": cov_high,
        "mean_bound_slack": float(pt["bound_slack"]),
        "mean_bound_slack_ci_lower": slack_low,
        "mean_bound_slack_ci_upper": slack_high,
        "proportion_genuinely_successful": float(pt["genuinely_successful"]),
        "discovery_shots": int(cell_df["discovery_shots"].iloc[0]),
        "confirmation_shots": int(cell_df["confirmation_shots"].iloc[0]),
        "total_shots": int(cell_df["total_shots"].iloc[0]),
    }


def _compute_repeated_measures_trends(chk_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute graph-clustered repeated-measures contrasts and mixed-effects regression slopes
    for Hypotheses H1, H2, H3, H4, H5 across all 100 graphs.
    """
    # Collapse across the 3 optimizer seeds per (graph_id, B, K_checkpoint) so each graph is 1 independent unit
    g_df = chk_df.groupby(["graph_size", "graph_id", "B", "K_checkpoint"], as_index=False)[[
        "discovery_bias",
        "confirmation_bias",
        "standardized_discovery_bias_exact_se",
        "standardized_discovery_bias_emp_se",
        "standardized_confirmation_bias_exact_se",
    ]].mean()

    # H1: Primary K=100 discovery bias across all B in {256, 512, 1024, 2048} (100 graphs)
    h1_results = {}
    for B_val in [256, 512, 1024, 2048]:
        sub = g_df[(g_df["B"] == B_val) & (g_df["K_checkpoint"] == 100)]
        b_arr = sub["discovery_bias"].to_numpy()
        z_arr = sub["standardized_discovery_bias_exact_se"].to_numpy()
        t_stat, p_val = ttest_1samp(b_arr, 0.0)
        w_stat, w_pval = wilcoxon(b_arr)
        h1_results[str(B_val)] = {
            "mean_bias": float(np.mean(b_arr)),
            "sd_across_graphs": float(np.std(b_arr, ddof=1)),
            "se_across_graphs": float(np.std(b_arr, ddof=1) / np.sqrt(len(b_arr))),
            "cohens_d_graph_level": float(np.mean(b_arr) / np.std(b_arr, ddof=1)),
            "mean_z_exact": float(np.mean(z_arr)),
            "t_stat_df99": float(t_stat),
            "p_value_two_sided": float(p_val),
            "wilcoxon_p_value": float(w_pval),
        }

    # H2: Paired within-graph contrast of |discovery_bias| as B increases from 256 -> 512 -> 1024 -> 2048 at K=100
    piv_b = g_df[g_df["K_checkpoint"] == 100].pivot(index="graph_id", columns="B", values="discovery_bias")
    diff_2048_vs_256 = piv_b[2048] - piv_b[256]  # Positive means bias became less negative (closer to 0)!
    t_b, p_b = ttest_1samp(diff_2048_vs_256.to_numpy(), 0.0)

    # Also fit log-linear scaling slope of log10(|mean_bias|) vs log10(B)
    b_vals = np.array([256.0, 512.0, 1024.0, 2048.0])
    mean_abs_biases = np.array([abs(h1_results[str(int(b))]["mean_bias"]) for b in b_vals])
    log_slope, log_intercept = np.polyfit(np.log10(b_vals), np.log10(mean_abs_biases), 1)

    # H3: Paired within-graph contrast of discovery_bias as K increases from 1 -> 25 -> 50 -> 100 (averaged over B)
    g_k = g_df.groupby(["graph_id", "K_checkpoint"], as_index=False)[
        ["discovery_bias", "standardized_discovery_bias_exact_se"]
    ].mean()
    piv_k = g_k.pivot(index="graph_id", columns="K_checkpoint", values="standardized_discovery_bias_exact_se")
    diff_k100_vs_k1 = piv_k[100] - piv_k[1]
    diff_k100_vs_k25 = piv_k[100] - piv_k[25]
    t_k100_1, p_k100_1 = ttest_1samp(diff_k100_vs_k1.to_numpy(), 0.0)
    t_k100_25, p_k100_25 = ttest_1samp(diff_k100_vs_k25.to_numpy(), 0.0)

    # H5: Paired within-graph contrast between Independent Confirmation bias and Discovery bias at K=100
    g_conf = g_df[g_df["K_checkpoint"] == 100].groupby("graph_id")[
        ["discovery_bias", "confirmation_bias", "standardized_discovery_bias_exact_se", "standardized_confirmation_bias_exact_se"]
    ].mean()
    bias_reduction = g_conf["confirmation_bias"] - g_conf["discovery_bias"]
    t_conf, p_conf = ttest_1samp(bias_reduction.to_numpy(), 0.0)
    t_conf_zero, p_conf_zero = ttest_1samp(g_conf["confirmation_bias"].to_numpy(), 0.0)

    # Variance decomposition (ICC: Between-Graph vs Optimizer-Seed nested within Graph at K=100, B=256)
    sub_256_100 = chk_df[(chk_df["B"] == 256) & (chk_df["K_checkpoint"] == 100)]
    g_means_loss = sub_256_100.groupby("graph_id")["exact_expected_loss"].mean()
    within_seed_var = float(sub_256_100.groupby("graph_id")["exact_expected_loss"].var(ddof=1).mean())
    between_graph_var = float(g_means_loss.var(ddof=1) - within_seed_var / 3.0)
    icc_exact_loss = float(between_graph_var / (between_graph_var + within_seed_var))

    return {
        "H1_selection_bias_at_K100": h1_results,
        "H2_shot_budget_scaling": {
            "paired_diff_B2048_minus_B256_mean": float(diff_2048_vs_256.mean()),
            "paired_diff_B2048_minus_B256_se": float(diff_2048_vs_256.std(ddof=1) / 10.0),
            "t_stat_df99": float(t_b),
            "p_value": float(p_b),
            "log10_abs_bias_vs_log10_B_slope": float(log_slope),
            "theoretical_sqrt_B_slope": -0.50,
        },
        "H3_candidate_count_scaling": {
            "paired_z_diff_K100_minus_K1_mean": float(diff_k100_vs_k1.mean()),
            "t_stat_K100_vs_K1_df99": float(t_k100_1),
            "p_value_K100_vs_K1": float(p_k100_1),
            "paired_z_diff_K100_minus_K25_mean": float(diff_k100_vs_k25.mean()),
            "t_stat_K100_vs_K25_df99": float(t_k100_25),
            "p_value_K100_vs_K25": float(p_k100_25),
        },
        "H5_independent_confirmation_vs_discovery_K100": {
            "mean_discovery_bias_pooled_B": float(g_conf["discovery_bias"].mean()),
            "mean_confirmation_bias_pooled_B": float(g_conf["confirmation_bias"].mean()),
            "mean_bias_reduction": float(bias_reduction.mean()),
            "t_stat_reduction_df99": float(t_conf),
            "p_value_reduction": float(p_conf),
            "confirmation_bias_vs_zero_t_stat": float(t_conf_zero),
            "confirmation_bias_vs_zero_p_value": float(p_conf_zero),
        },
        "variance_decomposition_B256_K100": {
            "between_graph_variance_exact_loss": between_graph_var,
            "within_graph_optimizer_seed_variance_exact_loss": within_seed_var,
            "intraclass_correlation_ICC": icc_exact_loss,
        },
    }


if __name__ == "__main__":
    run_stage2_statistical_analysis()
