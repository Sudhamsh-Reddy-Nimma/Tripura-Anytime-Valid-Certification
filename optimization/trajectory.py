"""
Trajectory Data Structures and Prefix Selection Logic for AMFS 2.0 (Stage 1 & Stage 2).

STRICT GROUND-TRUTH SEPARATION (Stage 2 Section 14):
Trajectory records store ONLY finite-shot candidate evaluations and measurement statistics.
Zero ground-truth fields are ever stored on OptimizationTrajectory.
"""

from typing import List, Dict, Any
import json
import numpy as np


def format_float_list_json(values) -> str:
    arr = [round(float(x), 8) for x in values]
    return json.dumps(arr, separators=(",", ":"))


class OptimizationTrajectory:
    def __init__(
        self,
        run_id: str,
        graph_id: str,
        trial_id: str,
        qaoa_depth: int,
        shot_budget: int,
    ):
        self.run_id = run_id
        self.graph_id = graph_id
        self.trial_id = trial_id
        self.qaoa_depth = int(qaoa_depth)
        self.shot_budget = int(shot_budget)
        self.evaluations: List[Dict[str, Any]] = []
        self.shot_histograms: List[Dict[str, Any]] = []
        self.shot_observations: List[Dict[str, Any]] = []

    def record_evaluation(
        self,
        theta: np.ndarray,
        measurement_result: Dict[str, Any],
        runtime_seconds: float,
        eval_seed: int,
        store_full_shots: bool = False,
        store_histogram: bool = False,
        graph_edges_m: int = None,
        graph_n: int = None,
    ) -> Dict[str, Any]:
        eval_idx = len(self.evaluations) + 1
        candidate_id = f"{self.run_id}_C{eval_idx:03d}"
        p = self.qaoa_depth
        gammas = theta[:p]
        betas = theta[p:]

        record = {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "trial_id": self.trial_id,
            "candidate_id": candidate_id,
            "evaluation_index": eval_idx,
            "qaoa_depth": p,
            "shot_budget": self.shot_budget,
            "parameter_vector": format_float_list_json(theta),
            "gamma_parameters": format_float_list_json(gammas),
            "beta_parameters": format_float_list_json(betas),
            "theta_array": np.asarray(theta, dtype=np.float64).copy(),
            "observed_mean_cut": float(measurement_result["observed_mean_cut"]),
            "observed_mean_loss": float(measurement_result["observed_mean_loss"]),
            "observed_variance_loss": float(measurement_result["observed_variance_loss"]),
            "observed_standard_error": float(measurement_result["observed_standard_error"]),
            "minimum_observed_loss": float(measurement_result["minimum_observed_loss"]),
            "maximum_observed_loss": float(measurement_result["maximum_observed_loss"]),
            "unique_bitstring_count": int(measurement_result["unique_bitstring_count"]),
            "evaluation_runtime_seconds": float(runtime_seconds),
            "random_seed": str(eval_seed),
        }
        self.evaluations.append(record)

        if store_histogram and measurement_result["histogram"]:
            for h in measurement_result["histogram"]:
                self.shot_histograms.append({
                    "run_id": self.run_id,
                    "graph_id": self.graph_id,
                    "trial_id": self.trial_id,
                    "candidate_id": candidate_id,
                    "evaluation_index": eval_idx,
                    "qaoa_depth": p,
                    "bitstring": h["bitstring"],
                    "count": int(h["count"]),
                    "shot_budget": int(h["shot_budget"]),
                    "observed_frequency": float(h["observed_frequency"]),
                    "cut_value": float(h["cut_value"]),
                    "normalized_loss": float(h["normalized_loss"]),
                })

        if store_full_shots:
            sampled_indices = measurement_result["sampled_indices"]
            shot_cuts = measurement_result["shot_cuts"]
            shot_losses = measurement_result["shot_losses"]
            for s_idx in range(len(sampled_indices)):
                bstr = format(int(sampled_indices[s_idx]), f"0{graph_n}b")
                self.shot_observations.append({
                    "run_id": self.run_id,
                    "graph_id": self.graph_id,
                    "trial_id": self.trial_id,
                    "candidate_id": candidate_id,
                    "evaluation_index": eval_idx,
                    "qaoa_depth": p,
                    "shot_index": s_idx + 1,
                    "bitstring": bstr,
                    "cut_value": float(shot_cuts[s_idx]),
                    "normalized_loss": float(shot_losses[s_idx]),
                })

        return record

    def select_best_in_prefix(self, K: int) -> Dict[str, Any]:
        if len(self.evaluations) < K:
            raise ValueError(
                f"Requested prefix K={K}, but trajectory {self.run_id} only has {len(self.evaluations)} evaluations."
            )
        prefix_evals = self.evaluations[: int(K)]
        best_eval = min(prefix_evals, key=lambda r: (r["observed_mean_loss"], r["evaluation_index"]))
        return best_eval
