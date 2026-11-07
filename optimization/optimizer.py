"""
SciPy COBYLA Optimizer Implementation for AMFS 2.0 (Stage 1, Section 6).
"""

import time
import json
from typing import Dict, Any
import numpy as np
from scipy.optimize import minimize
from optimization.initialization import generate_initial_parameters
from optimization.trajectory import OptimizationTrajectory, format_float_list_json
from optimization.objective import FiniteShotQAOAObjective


def run_qaoa_optimization(
    run_id: str,
    graph: Dict[str, Any],
    trial_id: str,
    qaoa_depth: int,
    shot_budget: int,
    init_seed_seq: np.random.SeedSequence,
    shot_seed_seq: np.random.SeedSequence,
    requested_max_evaluations: int = 100,
    rhobeg: float = 0.2,
    tol: float = 1e-6,
    maxiter: int = 100,
    initialization_method: str = "wurtz_love_3regular_jitter",
    jitter_half_width: float = 0.05,
    enforce_prefix_completion: bool = True,
    store_full_shots: bool = False,
    store_histogram: bool = False,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    init_rng = np.random.default_rng(init_seed_seq)
    run_seed_int = int(init_seed_seq.generate_state(1, dtype=np.uint64)[0])

    theta_0 = generate_initial_parameters(
        p=qaoa_depth,
        rng=init_rng,
        jitter_half_width=jitter_half_width,
        method=initialization_method,
    )

    trajectory = OptimizationTrajectory(
        run_id=run_id,
        graph_id=graph["graph_id"],
        trial_id=trial_id,
        qaoa_depth=qaoa_depth,
        shot_budget=shot_budget,
    )

    objective = FiniteShotQAOAObjective(
        graph=graph,
        qaoa_depth=qaoa_depth,
        shot_budget=shot_budget,
        shot_seed_seq=shot_seed_seq,
        trajectory=trajectory,
        max_evaluations=requested_max_evaluations + 50,
        store_full_shots=store_full_shots,
        store_histogram=store_histogram,
    )

    if requested_max_evaluations == 1:
        objective.evaluate_single(theta_0)
        cobyla_native_evals = 1
        cobyla_success = True
        termination_reason = "Non-adaptive K=1 baseline evaluation completed"
        final_returned_theta = theta_0.copy()
    else:
        res = minimize(
            objective,
            theta_0.copy(),
            method="COBYLA",
            options={
                "rhobeg": float(rhobeg),
                "tol": float(tol),
                "maxiter": int(requested_max_evaluations),
                "disp": False,
            },
        )
        cobyla_native_evals = len(trajectory.evaluations)
        cobyla_success = bool(res.success)
        termination_reason = str(res.message)
        final_returned_theta = np.asarray(res.x, dtype=np.float64).copy()

        restart_count = 0
        while enforce_prefix_completion and len(trajectory.evaluations) < requested_max_evaluations:
            restart_count += 1
            best_so_far = min(
                trajectory.evaluations,
                key=lambda r: (r["observed_mean_loss"], r["evaluation_index"]),
            )
            center_theta = best_so_far["theta_array"].copy()
            step_scale = max(0.04, float(rhobeg) * (0.65 ** restart_count))
            perturbed_theta = center_theta + init_rng.normal(0.0, step_scale * 0.25, size=center_theta.shape)
            remaining = requested_max_evaluations - len(trajectory.evaluations)

            if remaining < (2 * qaoa_depth + 2):
                for _ in range(remaining):
                    probe_theta = center_theta + init_rng.normal(0.0, step_scale * 0.15, size=center_theta.shape)
                    objective.evaluate_single(probe_theta)
                break

            res_cont = minimize(
                objective,
                perturbed_theta,
                method="COBYLA",
                options={
                    "rhobeg": step_scale,
                    "tol": float(tol),
                    "maxiter": int(remaining),
                    "disp": False,
                },
            )
            final_returned_theta = np.asarray(res_cont.x, dtype=np.float64).copy()

        if len(trajectory.evaluations) > requested_max_evaluations:
            valid_cids = {r["candidate_id"] for r in trajectory.evaluations[:requested_max_evaluations]}
            trajectory.evaluations = trajectory.evaluations[:requested_max_evaluations]
            trajectory.shot_histograms = [
                h for h in trajectory.shot_histograms if h["candidate_id"] in valid_cids
            ]
            trajectory.shot_observations = [
                s for s in trajectory.shot_observations if s["candidate_id"] in valid_cids
            ]

        if restart_count > 0:
            termination_reason = (
                f"Primary COBYLA converged at {cobyla_native_evals} evals ({termination_reason}); "
                f"completed {len(trajectory.evaluations)} evals via {restart_count} trust-region continuation(s)"
            )

    total_runtime = time.perf_counter() - t_start
    actual_evals = len(trajectory.evaluations)

    prefixes = [1] if requested_max_evaluations == 1 else [25, 50, 100]
    selected_by_prefix: Dict[int, str] = {}
    for K in prefixes:
        if actual_evals >= K:
            best_k = trajectory.select_best_in_prefix(K)
            selected_by_prefix[K] = best_k["candidate_id"]
        else:
            selected_by_prefix[K] = ""

    for rec in trajectory.evaluations:
        idx = rec["evaluation_index"]
        cid = rec["candidate_id"]
        rec["in_prefix_K25"] = bool(idx <= 25)
        rec["in_prefix_K50"] = bool(idx <= 50)
        rec["in_prefix_K100"] = bool(idx <= 100)
        rec["is_selected_K25"] = bool(cid == selected_by_prefix.get(25, ""))
        rec["is_selected_K50"] = bool(cid == selected_by_prefix.get(50, ""))
        rec["is_selected_K100"] = bool(cid == selected_by_prefix.get(100, ""))

    run_metadata = {
        "run_id": run_id,
        "graph_id": graph["graph_id"],
        "trial_id": trial_id,
        "qaoa_depth": int(qaoa_depth),
        "shot_budget": int(shot_budget),
        "initialization_method": initialization_method,
        "initial_parameter_vector": format_float_list_json(theta_0),
        "optimizer_name": "SciPy_COBYLA",
        "optimizer_settings": json.dumps(
            {
                "rhobeg": rhobeg,
                "tol": tol,
                "maxiter": maxiter,
                "enforce_prefix_completion": enforce_prefix_completion,
                "cobyla_native_evaluations": cobyla_native_evals,
            },
            separators=(",", ":"),
        ),
        "requested_max_evaluations": int(requested_max_evaluations),
        "actual_evaluations": int(actual_evals),
        "cobyla_native_evaluations": int(cobyla_native_evals),
        "prefix_K25_completed": bool(actual_evals >= 25),
        "prefix_K50_completed": bool(actual_evals >= 50),
        "prefix_K100_completed": bool(actual_evals >= 100),
        "termination_reason": termination_reason,
        "optimizer_success": bool(cobyla_success),
        "final_returned_parameter_vector": format_float_list_json(final_returned_theta),
        "total_runtime_seconds": float(total_runtime),
        "random_seed": str(run_seed_int),
    }

    return {
        "run_metadata": run_metadata,
        "trajectory": trajectory,
        "selected_by_prefix": selected_by_prefix,
    }
