"""
Aggregate Results Computation for `aggregate_results.csv` (Section 12.J, Section 13, Section 14).
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from analysis.bootstrap import graph_cluster_bootstrap_cell


def build_aggregate_results(
    trial_summary_df: pd.DataFrame,
    num_bootstrap_resamples: int = 10000,
    master_seed: int = 20261101,
) -> pd.DataFrame:
    ss = np.random.SeedSequence([master_seed, 8001])
    rows: List[Dict[str, Any]] = []

    group_cols = [
        "experiment_split",
        "graph_size_n",
        "qaoa_depth",
        "shot_budget",
        "candidate_prefix_K",
        "evaluation_mode",
        "certification_method",
    ]

    for keys, df_cell in trial_summary_df.groupby(group_cols, sort=True):
        split_name, n_val, depth_p, B_val, K_val, eval_mode, cert_method = keys
        rng = np.random.default_rng(ss.spawn(1)[0])
        row = _summarize_single_cell(
            df_cell=df_cell,
            split_name=str(split_name),
            graph_size_label=str(n_val),
            qaoa_depth=int(depth_p),
            shot_budget=int(B_val),
            candidate_prefix_K=int(K_val),
            evaluation_mode=str(eval_mode),
            certification_method=str(cert_method),
            num_resamples=num_bootstrap_resamples,
            rng=rng,
        )
        rows.append(row)

    pooled_group_cols = [
        "experiment_split",
        "qaoa_depth",
        "shot_budget",
        "candidate_prefix_K",
        "evaluation_mode",
        "certification_method",
    ]
    for keys, df_cell in trial_summary_df.groupby(pooled_group_cols, sort=True):
        if df_cell["graph_size_n"].nunique() < 2:
            continue
        split_name, depth_p, B_val, K_val, eval_mode, cert_method = keys
        rng = np.random.default_rng(ss.spawn(1)[0])
        row = _summarize_single_cell(
            df_cell=df_cell,
            split_name=str(split_name),
            graph_size_label="pooled_12_14",
            qaoa_depth=int(depth_p),
            shot_budget=int(B_val),
            candidate_prefix_K=int(K_val),
            evaluation_mode=str(eval_mode),
            certification_method=str(cert_method),
            num_resamples=num_bootstrap_resamples,
            rng=rng,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _summarize_single_cell(
    df_cell: pd.DataFrame,
    split_name: str,
    graph_size_label: str,
    qaoa_depth: int,
    shot_budget: int,
    candidate_prefix_K: int,
    evaluation_mode: str,
    certification_method: str,
    num_resamples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n_graphs = int(df_cell["graph_id"].nunique())
    n_trials = int(len(df_cell))
    n_cert = int(df_cell["certified"].sum())
    n_fc = int(df_cell["false_certification"].sum())

    strata = sorted(df_cell["graph_size_n"].unique())
    if len(strata) > 1:
        stratum_means = df_cell.groupby("graph_size_n")[[
            "target_threshold",
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
            "target_margin",
            "total_shots",
            "optimization_runtime_seconds",
        ]].mean().mean(axis=0)
        tau_val = float(stratum_means["target_threshold"])
        fc_rate = float(stratum_means["false_certification"])
        cert_rate = float(stratum_means["certified"])
        correct_cert_rate = float(stratum_means["correct_certification"])
        false_non_cert_rate = float(stratum_means["false_non_certification"])
        one_sided_cov = float(stratum_means["upper_bound_covered"])
        two_sided_cov = float(stratum_means["two_sided_covered"])
        mean_bias = float(stratum_means["selection_bias"])
        mean_zbias = float(stratum_means["standardized_selection_bias"])
        mean_slack = float(stratum_means["bound_slack"])
        mean_slack_se = float(stratum_means["bound_slack_standard_errors"])
        mean_margin = float(stratum_means["target_margin"])
        mean_shots = float(stratum_means["total_shots"])
        mean_runtime = float(stratum_means["optimization_runtime_seconds"])
    else:
        tau_val = float(df_cell["target_threshold"].iloc[0])
        fc_rate = float(n_fc / float(n_trials))
        cert_rate = float(n_cert / float(n_trials))
        correct_cert_rate = float(df_cell["correct_certification"].mean())
        false_non_cert_rate = float(df_cell["false_non_certification"].mean())
        one_sided_cov = float(df_cell["upper_bound_covered"].mean())
        two_sided_cov = float(df_cell["two_sided_covered"].mean())
        mean_bias = float(df_cell["selection_bias"].mean())
        mean_zbias = float(df_cell["standardized_selection_bias"].mean())
        mean_slack = float(df_cell["bound_slack"].mean())
        mean_slack_se = float(df_cell["bound_slack_standard_errors"].mean())
        mean_margin = float(df_cell["target_margin"].mean())
        mean_shots = float(df_cell["total_shots"].mean())
        mean_runtime = float(df_cell["optimization_runtime_seconds"].mean())

    if n_cert > 0:
        fdr = float(n_fc / float(n_cert))
    else:
        fdr = np.nan

    if fc_rate > (1.0 - one_sided_cov) + 1e-12:
        raise AssertionError(
            f"Consistency check violated: P_FC ({fc_rate}) > 1 - one_sided_coverage ({1.0 - one_sided_cov})"
        )

    boot_ci = graph_cluster_bootstrap_cell(
        df_cell=df_cell,
        num_resamples=num_resamples,
        confidence=0.95,
        rng=rng,
    )

    row = {
        "experiment_split": split_name,
        "graph_size_n": graph_size_label,
        "qaoa_depth": qaoa_depth,
        "shot_budget": shot_budget,
        "candidate_prefix_K": candidate_prefix_K,
        "target_threshold": tau_val,
        "evaluation_mode": evaluation_mode,
        "certification_method": certification_method,
        "number_of_graphs": n_graphs,
        "number_of_trials": n_trials,
        "number_of_certifications": n_cert,
        "number_of_false_certifications": n_fc,
        "false_certification_rate": fc_rate,
        "false_discovery_rate": fdr,
        "certification_rate": cert_rate,
        "correct_certification_rate": correct_cert_rate,
        "false_non_certification_rate": false_non_cert_rate,
        "one_sided_coverage": one_sided_cov,
        "two_sided_coverage": two_sided_cov,
        "mean_selection_bias": mean_bias,
        "mean_standardized_selection_bias": mean_zbias,
        "mean_bound_slack": mean_slack,
        "mean_bound_slack_standard_errors": mean_slack_se,
        "mean_target_margin": mean_margin,
        "mean_total_shots": mean_shots,
        "mean_runtime_seconds": mean_runtime,
        "structural_consistency_verified": True,
    }
    row.update(boot_ci)
    return row
