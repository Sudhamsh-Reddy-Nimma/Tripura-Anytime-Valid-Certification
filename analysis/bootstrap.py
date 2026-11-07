"""
Stratified Graph-Cluster Bootstrap Resampling and Zero-Event Exact Upper Limits
for AMFS 2.0 (Stage 1, Section 14).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def clopper_pearson_upper_limit(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return np.nan
    if k >= n:
        return 1.0
    if k == 0:
        return float(1.0 - (alpha ** (1.0 / float(n))))
    return float(beta_dist.ppf(1.0 - alpha, k + 1, n - k))


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k >= n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def graph_cluster_bootstrap_cell(
    df_cell: pd.DataFrame,
    num_resamples: int = 10000,
    confidence: float = 0.95,
    rng: np.random.Generator = None,
) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng(20261101)

    alpha = 1.0 - confidence
    q_low = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)

    metric_cols = [
        "certified",
        "false_certification",
        "correct_certification",
        "false_non_certification",
        "upper_bound_covered",
        "two_sided_covered",
        "selection_bias",
        "standardized_selection_bias",
        "bound_slack",
        "bound_slack_standard_errors",
    ]

    strata = sorted(df_cell["graph_size_n"].unique())
    n_strata = len(strata)

    stratum_graph_means = []
    for s in strata:
        df_s = df_cell[df_cell["graph_size_n"] == s]
        g_grouped = df_s.groupby("graph_id", sort=True)[metric_cols].mean()
        stratum_graph_means.append(g_grouped.to_numpy(dtype=np.float64))

    boot_means_per_stratum = []
    for g_mat in stratum_graph_means:
        G_s = g_mat.shape[0]
        if G_s == 1:
            boot_s = np.repeat(g_mat, num_resamples, axis=0)
        else:
            counts = rng.multinomial(G_s, [1.0 / G_s] * G_s, size=num_resamples)
            weights = counts.astype(np.float64) / float(G_s)
            boot_s = weights @ g_mat
        boot_means_per_stratum.append(boot_s)

    boot_pooled = sum(boot_means_per_stratum) / float(n_strata)

    col_idx = {name: i for i, name in enumerate(metric_cols)}
    cert_boot = boot_pooled[:, col_idx["certified"]]
    fc_boot = boot_pooled[:, col_idx["false_certification"]]

    zero_cert_mask = cert_boot <= 1e-12
    zero_cert_resample_freq = float(np.mean(zero_cert_mask))
    valid_fdr_samples = np.where(~zero_cert_mask, fc_boot / np.maximum(cert_boot, 1e-12), np.nan)
    valid_fdr_clean = valid_fdr_samples[~np.isnan(valid_fdr_samples)]

    if valid_fdr_clean.size > 0:
        fdr_ci_low = float(np.percentile(valid_fdr_clean, q_low))
        fdr_ci_high = float(np.percentile(valid_fdr_clean, q_high))
    else:
        fdr_ci_low = np.nan
        fdr_ci_high = np.nan

    n_trials_total = len(df_cell)
    n_graphs_total = df_cell["graph_id"].nunique()
    n_fc = int(df_cell["false_certification"].sum())
    n_cert = int(df_cell["certified"].sum())

    def get_ci(col_name: str, is_non_zero_binary: bool = False, count_val: int = 0) -> Tuple[float, float]:
        samples = boot_pooled[:, col_idx[col_name]]
        finite_s = samples[np.isfinite(samples)]
        if finite_s.size == 0:
            return np.nan, np.nan
        b_low = float(np.percentile(finite_s, q_low))
        b_high = float(np.percentile(finite_s, q_high))
        if is_non_zero_binary and count_val == 0:
            cp_high = clopper_pearson_upper_limit(0, n_trials_total, alpha=alpha)
            return 0.0, cp_high
        return b_low, b_high

    fc_low, fc_high = get_ci("false_certification", is_non_zero_binary=True, count_val=n_fc)
    cert_low, cert_high = get_ci("certified", is_non_zero_binary=True, count_val=n_cert)
    cov1_low, cov1_high = get_ci("upper_bound_covered")
    cov2_low, cov2_high = get_ci("two_sided_covered")
    bias_low, bias_high = get_ci("selection_bias")
    zbias_low, zbias_high = get_ci("standardized_selection_bias")
    slack_low, slack_high = get_ci("bound_slack")

    per_graph_df = df_cell.groupby("graph_id").agg(
        graph_mean_exact_loss=("selected_exact_loss", "mean"),
        graph_mean_obs_loss=("selected_observed_loss", "mean"),
        graph_mean_bias=("selection_bias", "mean"),
        graph_cert_rate=("certified", "mean"),
        graph_fc_rate=("false_certification", "mean"),
    )
    between_graph_sd_exact_loss = float(per_graph_df["graph_mean_exact_loss"].std(ddof=1)) if n_graphs_total > 1 else 0.0
    between_graph_sd_bias = float(per_graph_df["graph_mean_bias"].std(ddof=1)) if n_graphs_total > 1 else 0.0
    between_graph_sd_fc_rate = float(per_graph_df["graph_fc_rate"].std(ddof=1)) if n_graphs_total > 1 else 0.0

    return {
        "false_certification_rate_ci_lower": fc_low,
        "false_certification_rate_ci_upper": fc_high,
        "false_discovery_rate_ci_lower": fdr_ci_low,
        "false_discovery_rate_ci_upper": fdr_ci_high,
        "zero_certification_bootstrap_frequency": zero_cert_resample_freq,
        "certification_rate_ci_lower": cert_low,
        "certification_rate_ci_upper": cert_high,
        "one_sided_coverage_ci_lower": cov1_low,
        "one_sided_coverage_ci_upper": cov1_high,
        "two_sided_coverage_ci_lower": cov2_low,
        "two_sided_coverage_ci_upper": cov2_high,
        "mean_selection_bias_ci_lower": bias_low,
        "mean_selection_bias_ci_upper": bias_high,
        "mean_standardized_selection_bias_ci_lower": zbias_low,
        "mean_standardized_selection_bias_ci_upper": zbias_high,
        "mean_bound_slack_ci_lower": slack_low,
        "mean_bound_slack_ci_upper": slack_high,
        "between_graph_sd_exact_loss": between_graph_sd_exact_loss,
        "between_graph_sd_selection_bias": between_graph_sd_bias,
        "between_graph_sd_fc_rate": between_graph_sd_fc_rate,
    }
