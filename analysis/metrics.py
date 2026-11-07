"""
Primary Statistical Endpoints, Threshold Calibration, and Trial/Certification Table Builders
for AMFS 2.0 (Stage 1, Sections 11, 12, 13, 16).
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def calibrate_pilot_thresholds(
    pilot_selected_losses_by_size: Dict[int, List[float]],
) -> Dict[int, Dict[str, Any]]:
    calibrated: Dict[int, Dict[str, Any]] = {}
    for n, losses in sorted(pilot_selected_losses_by_size.items()):
        arr = np.asarray(losses, dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"No pilot losses available to calibrate threshold for graph size n={n}.")
        p10 = float(np.percentile(arr, 10))
        p25 = float(np.percentile(arr, 25))
        p50 = float(np.percentile(arr, 50))
        p75 = float(np.percentile(arr, 75))
        p90 = float(np.percentile(arr, 90))
        primary_tau = float(np.median([p25, p50, p75]))
        calibrated[int(n)] = {
            "graph_size_n": int(n),
            "num_pooled_pilot_selections": int(arr.size),
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "primary_threshold_tau": primary_tau,
            "exploratory_grid": [round(p10, 6), round(p25, 6), round(p50, 6), round(p75, 6), round(p90, 6)],
        }
    return calibrated


def compute_certification_and_trial_metrics(
    run_id: str,
    graph_id: str,
    trial_id: str,
    graph_size_n: int,
    experiment_split: str,
    qaoa_depth: int,
    shot_budget: int,
    candidate_id: str,
    candidate_prefix_K: int,
    evaluation_mode: str,
    cert_dict: Dict[str, Any],
    exact_expected_loss: float,
    exact_loss_variance: float,
    total_discovery_shots: int,
    total_confirmation_shots: int,
    optimization_runtime_seconds: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tau = float(cert_dict["target_threshold"])
    obs_mean = float(cert_dict["observed_mean"])
    obs_var = float(cert_dict["observed_variance"])
    ub = float(cert_dict["upper_confidence_bound"])
    lb_2s = float(cert_dict["lower_confidence_bound"])
    ub_2s = float(cert_dict["two_sided_upper_bound"])
    certified = bool(cert_dict["certified"])

    mu_true = float(exact_expected_loss)
    var_true = float(max(0.0, exact_loss_variance))

    genuinely_successful = bool(mu_true <= tau)
    false_certification = bool(certified and (not genuinely_successful))
    correct_certification = bool(certified and genuinely_successful)
    false_non_certification = bool((not certified) and genuinely_successful)

    target_margin = float(mu_true - tau)
    bound_slack = float(cert_dict["bound_slack"])
    bound_slack_se = float(cert_dict["bound_slack_standard_errors"])

    selection_bias = float(obs_mean - mu_true)
    true_se = np.sqrt(var_true / float(shot_budget)) if var_true > 1e-15 else np.nan
    standardized_selection_bias = float(selection_bias / true_se) if np.isfinite(true_se) else np.nan

    upper_bound_covered = bool(mu_true <= ub + 1e-15)
    two_sided_covered = bool((lb_2s - 1e-15 <= mu_true) and (mu_true <= ub_2s + 1e-15))

    cert_row = {
        "run_id": run_id,
        "graph_id": graph_id,
        "trial_id": trial_id,
        "graph_size_n": int(graph_size_n),
        "experiment_split": experiment_split,
        "candidate_id": candidate_id,
        "candidate_prefix_K": int(candidate_prefix_K),
        "qaoa_depth": int(qaoa_depth),
        "shot_budget": int(shot_budget),
        "evaluation_mode": evaluation_mode,
        "certification_method": cert_dict["certification_method"],
        "delta": float(cert_dict["delta"]),
        "observed_mean": obs_mean,
        "observed_variance": obs_var,
        "upper_confidence_bound": ub,
        "lower_confidence_bound": lb_2s,
        "two_sided_upper_bound": ub_2s,
        "target_threshold": tau,
        "certified": certified,
        "exact_expected_loss": mu_true,
        "genuinely_successful": genuinely_successful,
        "false_certification": false_certification,
        "correct_certification": correct_certification,
        "false_non_certification": false_non_certification,
        "upper_bound_covered": upper_bound_covered,
        "two_sided_covered": two_sided_covered,
        "target_margin": target_margin,
        "bound_slack": bound_slack,
        "bound_slack_standard_errors": bound_slack_se,
        "selection_bias": selection_bias,
        "standardized_selection_bias": standardized_selection_bias,
    }

    trial_row = {
        "run_id": run_id,
        "graph_id": graph_id,
        "trial_id": trial_id,
        "graph_size_n": int(graph_size_n),
        "experiment_split": experiment_split,
        "qaoa_depth": int(qaoa_depth),
        "shot_budget": int(shot_budget),
        "candidate_prefix_K": int(candidate_prefix_K),
        "evaluation_mode": evaluation_mode,
        "target_threshold": tau,
        "selected_candidate_id": candidate_id,
        "selected_observed_loss": obs_mean,
        "selected_exact_loss": mu_true,
        "selection_bias": selection_bias,
        "standardized_selection_bias": standardized_selection_bias,
        "certification_method": cert_dict["certification_method"],
        "delta": float(cert_dict["delta"]),
        "upper_confidence_bound": ub,
        "lower_confidence_bound": lb_2s,
        "two_sided_upper_bound": ub_2s,
        "certified": certified,
        "genuinely_successful": genuinely_successful,
        "false_certification": false_certification,
        "correct_certification": correct_certification,
        "false_non_certification": false_non_certification,
        "upper_bound_covered": upper_bound_covered,
        "two_sided_covered": two_sided_covered,
        "bound_slack": bound_slack,
        "bound_slack_standard_errors": bound_slack_se,
        "target_margin": target_margin,
        "certification_rate_indicator": int(certified),
        "total_discovery_shots": int(total_discovery_shots),
        "total_confirmation_shots": int(total_confirmation_shots),
        "total_shots": int(total_discovery_shots + total_confirmation_shots),
        "optimization_runtime_seconds": float(optimization_runtime_seconds),
    }

    return cert_row, trial_row
