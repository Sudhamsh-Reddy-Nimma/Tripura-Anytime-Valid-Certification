"""
Stage 2 Full Experimental Execution Engine with Incremental Checkpointing & Resumption
for AMFS 2.0 (Stage 2, Sections 5-14, 18, 19).

Design:
- 100 unique connected non-isomorphic random 3-regular graphs (50 n=12, 50 n=14), strictly disjoint from Stage 1.
- Primary QAOA depth p = 2, Wurtz-Love 3-regular initialization + jitter.
- 4 discovery shot budgets B in {256, 512, 1024, 2048}.
- 3 independent optimizer seeds per (graph, B) -> 100 * 4 * 3 = 1,200 trajectories.
- Each trajectory runs up to 100 sequential evaluations and saves checkpoints K in {1, 25, 50, 100}.
- Independent confirmation: B_confirm = 4096 fresh shots per selected checkpoint from a dedicated confirmation seed stream.
- Offline ground truth: Computed strictly in a separate post-discovery phase after all candidates are frozen.
- Supports batch checkpointing and automatic resumption without duplicating or corrupting existing trials.
"""

import os
import time
import json
import hashlib
import platform
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Set, Tuple
import numpy as np
import scipy
import pandas as pd

from graphs.generation import generate_graph_ensemble
from graphs.serialization import encode_graph_adjacency
from optimization.optimizer import run_qaoa_optimization
from simulator.measurement import sample_loss_counts
from certification.decisions import evaluate_certification_method
from analysis.ground_truth import exact_expectation


FROZEN_THRESHOLDS = {
    12: 0.2618910742981325,
    14: 0.26866083912936534,
}
STAGE2_CHECKPOINTS_K = [1, 25, 50, 100]
STAGE2_SHOT_BUDGETS_B = [256, 512, 1024, 2048]
STAGE2_SEEDS_PER_CELL = 3
STAGE2_B_CONFIRM = 4096
STAGE2_ALPHA = 0.05


def _execute_single_graph_trajectories(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function that runs all 4 (shot budgets B) x 3 (optimizer seeds) = 12 trajectories
    and their independent confirmation evaluations (B_confirm = 4096) for a single graph.
    Note: Does NOT compute or access exact ground truth (strictly separated per Section 14).
    """
    graph = task["graph"]
    graph_id = graph["graph_id"]
    n = int(graph["graph_size_n"])
    seed_triplets = task["seed_triplets"]  # Dict[(B, rep_idx)] -> (opt_ss, meas_ss, conf_ss)
    config_hash = task["config_hash"]
    sw_version = task["software_version"]

    meta_rows = []
    cand_eval_rows = []
    selected_cand_rows = []
    confirm_rows = []

    for B in STAGE2_SHOT_BUDGETS_B:
        for rep_idx in range(1, STAGE2_SEEDS_PER_CELL + 1):
            exp_id = f"S2_{graph_id}_B{B}_R{rep_idx:02d}"
            opt_ss, meas_ss, conf_ss = seed_triplets[(B, rep_idx)]

            opt_seed_str = str(int(opt_ss.generate_state(1, dtype=np.uint64)[0]))
            meas_seed_str = str(int(meas_ss.generate_state(1, dtype=np.uint64)[0]))
            conf_seed_str = str(int(conf_ss.generate_state(1, dtype=np.uint64)[0]))

            opt_res = run_qaoa_optimization(
                run_id=exp_id,
                graph=graph,
                trial_id=f"R{rep_idx:02d}",
                qaoa_depth=2,
                shot_budget=B,
                init_seed_seq=opt_ss,
                shot_seed_seq=meas_ss,
                requested_max_evaluations=100,
                enforce_prefix_completion=True,
                store_full_shots=False,
                store_histogram=False,
            )
            rmeta = opt_res["run_metadata"]
            traj = opt_res["trajectory"]

            meta_rows.append({
                "experiment_id": exp_id,
                "graph_id": graph_id,
                "graph_size": n,
                "B": int(B),
                "K_max": 100,
                "optimizer_rep": int(rep_idx),
                "optimizer_seed": opt_seed_str,
                "measurement_seed": meas_seed_str,
                "confirmation_seed": conf_seed_str,
                "initial_parameter_vector": rmeta["initial_parameter_vector"],
                "actual_evaluations": int(rmeta["actual_evaluations"]),
                "cobyla_native_evaluations": int(rmeta["cobyla_native_evaluations"]),
                "optimizer_status": rmeta["termination_reason"],
                "optimizer_success": bool(rmeta["optimizer_success"]),
                "runtime_seconds": float(rmeta["total_runtime_seconds"]),
                "software_version": sw_version,
                "configuration_hash": config_hash,
            })

            for ev in traj.evaluations:
                cand_eval_rows.append({
                    "experiment_id": exp_id,
                    "graph_id": graph_id,
                    "graph_size": n,
                    "B": int(B),
                    "evaluation_index": int(ev["evaluation_index"]),
                    "parameter_vector": ev["parameter_vector"],
                    "observed_loss": float(ev["observed_mean_loss"]),
                    "sample_variance": float(ev["observed_variance_loss"]),
                    "standard_error": float(ev["observed_standard_error"]),
                    "shot_count": int(ev["shot_budget"]),
                    "shot_seed": ev["random_seed"],
                    "optimizer_status": "active" if ev["evaluation_index"] <= rmeta["cobyla_native_evaluations"] else "trust_region_continuation",
                })

            # Extract checkpoints K in {1, 25, 50, 100} and perform independent confirmation (B_confirm = 4096)
            for K in STAGE2_CHECKPOINTS_K:
                sel = traj.select_best_in_prefix(K)
                theta_star = sel["theta_array"]

                selected_cand_rows.append({
                    "experiment_id": exp_id,
                    "graph_id": graph_id,
                    "graph_size": n,
                    "B": int(B),
                    "optimizer_rep": int(rep_idx),
                    "K_checkpoint": int(K),
                    "selected_parameters": sel["parameter_vector"],
                    "selected_evaluation_index": int(sel["evaluation_index"]),
                    "selected_discovery_loss": float(sel["observed_mean_loss"]),
                    "selected_discovery_variance": float(sel["observed_variance_loss"]),
                    "selected_discovery_se": float(sel["observed_standard_error"]),
                    "evaluations_completed_at_checkpoint": int(K),
                    "discovery_shots": int(B * K),
                    "optimizer_status": rmeta["termination_reason"],
                })

                # Independent confirmation measurement batch of B_confirm = 4096 shots
                conf_child_ss = conf_ss.spawn(1)[0]
                conf_rng = np.random.default_rng(conf_child_ss)
                conf_meas = sample_loss_counts(
                    graph=graph,
                    theta=theta_star,
                    B=STAGE2_B_CONFIRM,
                    rng=conf_rng,
                    p=2,
                    build_histogram=False,
                )
                tau_n = FROZEN_THRESHOLDS[n]
                # Primary independent confirmation bound: one-sided Empirical Bernstein at alpha = 0.05
                c_eb = evaluate_certification_method(
                    "naive_eb",
                    conf_meas["observed_mean_loss"],
                    conf_meas["observed_variance_loss"],
                    STAGE2_B_CONFIRM,
                    STAGE2_ALPHA,
                    K=1,  # Single frozen candidate evaluated on fresh confirmation data!
                    target_threshold=tau_n,
                )
                c_hoeff = evaluate_certification_method(
                    "hoeffding",
                    conf_meas["observed_mean_loss"],
                    conf_meas["observed_variance_loss"],
                    STAGE2_B_CONFIRM,
                    STAGE2_ALPHA,
                    K=1,
                    target_threshold=tau_n,
                )
                c_t = evaluate_certification_method(
                    "student_t_ref",
                    conf_meas["observed_mean_loss"],
                    conf_meas["observed_variance_loss"],
                    STAGE2_B_CONFIRM,
                    STAGE2_ALPHA,
                    K=1,
                    target_threshold=tau_n,
                )

                confirm_rows.append({
                    "experiment_id": exp_id,
                    "graph_id": graph_id,
                    "graph_size": n,
                    "B": int(B),
                    "optimizer_rep": int(rep_idx),
                    "K_checkpoint": int(K),
                    "confirmation_shots": STAGE2_B_CONFIRM,
                    "confirmation_mean": float(conf_meas["observed_mean_loss"]),
                    "confirmation_variance": float(conf_meas["observed_variance_loss"]),
                    "confirmation_se": float(conf_meas["observed_standard_error"]),
                    "confirmation_upper_bound": float(c_eb["upper_confidence_bound"]),
                    "confirmation_bound_slack": float(c_eb["bound_slack"]),
                    "certification_decision": bool(c_eb["certified"]),
                    "confirmation_hoeffding_upper_bound": float(c_hoeff["upper_confidence_bound"]),
                    "confirmation_hoeffding_certified": bool(c_hoeff["certified"]),
                    "confirmation_student_t_upper_bound": float(c_t["upper_confidence_bound"]),
                    "confirmation_student_t_certified": bool(c_t["certified"]),
                    "discovery_shots": int(B * K),
                    "total_shots": int(B * K + STAGE2_B_CONFIRM),
                })

    return {
        "graph_id": graph_id,
        "meta_rows": meta_rows,
        "cand_eval_rows": cand_eval_rows,
        "selected_cand_rows": selected_cand_rows,
        "confirm_rows": confirm_rows,
    }


def generate_or_load_stage2_graphs(
    stage2_raw_dir: str,
    stage1_graphs_csv: str,
    graph_ss: np.random.SeedSequence,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Generate 100 unique connected non-isomorphic 3-regular graphs (50 n=12, 50 n=14)
    strictly disjoint from Stage 1 graphs, or load from checkpoint if already generated.
    """
    s1_df = pd.read_csv(stage1_graphs_csv)
    s1_iso_12 = set(s1_df[s1_df["graph_size_n"] == 12]["graph_isomorphism_class_id"])
    s1_iso_14 = set(s1_df[s1_df["graph_size_n"] == 14]["graph_isomorphism_class_id"])

    g12 = generate_graph_ensemble(
        n=12, count=50, split_name="stage2", seed_seq=graph_ss.spawn(1)[0],
        existing_isomorphism_ids=s1_iso_12, id_prefix="S2"
    )
    g14 = generate_graph_ensemble(
        n=14, count=50, split_name="stage2", seed_seq=graph_ss.spawn(1)[0],
        existing_isomorphism_ids=s1_iso_14, id_prefix="S2"
    )
    all_graphs = g12 + g14

    rows = []
    for g in all_graphs:
        rows.append({
            "graph_id": g["graph_id"],
            "graph_size": int(g["graph_size_n"]),
            "edge_list": encode_graph_adjacency(g["edges"]),
            "graph_seed": str(g["graph_generation_seed"]),
            "graph_generation_metadata": json.dumps({
                "degree": 3,
                "edge_count_m": g["edge_count_m"],
                "is_connected": g["graph_is_connected"],
                "isomorphism_class_id": g["graph_isomorphism_class_id"],
                "graph_hash": g["graph_hash"],
                "sampling_method": g["graph_sampling_method"],
                "split": "stage2_confirmatory_100",
            }, separators=(",", ":")),
        })
    df_graphs = pd.DataFrame(rows)
    os.makedirs(stage2_raw_dir, exist_ok=True)
    df_graphs.to_csv(os.path.join(stage2_raw_dir, "graphs.csv"), index=False)
    return all_graphs, df_graphs


def run_stage2_batch(
    base_dir: str = "/tmp/AMFS_2.0",
    master_seed: int = 20261102,
    start_graph_idx: int = 0,
    end_graph_idx: int = 100,
    finalize_offline_ground_truth: bool = True,
):
    """
    Execute Stage 2 trajectories for graphs in slice [start_graph_idx : end_graph_idx]
    using 2 parallel worker processes and checkpoint results to disk.
    When `finalize_offline_ground_truth=True`, runs the strictly separated offline ground-truth
    and certification assembly across all 100 graphs.
    """
    t0 = time.perf_counter()
    s2_raw = os.path.join(base_dir, "results_stage2_raw")
    s2_proc = os.path.join(base_dir, "results_stage2_processed")
    ckpt_dir = os.path.join(s2_raw, "checkpoints")
    os.makedirs(s2_raw, exist_ok=True)
    os.makedirs(s2_proc, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    root_ss = np.random.SeedSequence(master_seed)
    graph_ss, traj_ss = root_ss.spawn(2)

    s1_graphs_csv = os.path.join(base_dir, "results_raw", "graphs.csv")
    all_graphs, _ = generate_or_load_stage2_graphs(s2_raw, s1_graphs_csv, graph_ss)

    config_payload = json.dumps({
        "protocol_version": "2.0.0-stage2-frozen",
        "master_seed": master_seed,
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "B_grid": STAGE2_SHOT_BUDGETS_B,
        "K_checkpoints": STAGE2_CHECKPOINTS_K,
        "B_confirm": STAGE2_B_CONFIRM,
    }, sort_keys=True)
    config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()[:16]
    sw_version = f"Python_{platform.python_version()}_NumPy_{np.__version__}_SciPy_{scipy.__version__}"

    # Pre-spawn deterministic seed streams for all 100 graphs so batch slices are 100% invariant!
    graph_traj_streams = traj_ss.spawn(len(all_graphs))
    tasks_to_run = []
    for idx in range(start_graph_idx, min(end_graph_idx, len(all_graphs))):
        g = all_graphs[idx]
        g_id = g["graph_id"]
        ckpt_file = os.path.join(ckpt_dir, f"{g_id}.json")
        if os.path.exists(ckpt_file):
            continue
        g_ss = graph_traj_streams[idx]
        seed_triplets = {}
        for B in STAGE2_SHOT_BUDGETS_B:
            for rep_idx in range(1, STAGE2_SEEDS_PER_CELL + 1):
                opt_s, meas_s, conf_s = g_ss.spawn(3)
                seed_triplets[(B, rep_idx)] = (opt_s, meas_s, conf_s)
        tasks_to_run.append({
            "graph": g,
            "seed_triplets": seed_triplets,
            "config_hash": config_hash,
            "software_version": sw_version,
        })

    if tasks_to_run:
        print(f"[Stage 2 Execution] Running {len(tasks_to_run)} graphs (indices {start_graph_idx}..{end_graph_idx-1}) across 2 workers...", flush=True)
        with ProcessPoolExecutor(max_workers=2) as pool:
            for res in pool.map(_execute_single_graph_trajectories, tasks_to_run):
                g_id = res["graph_id"]
                ckpt_file = os.path.join(ckpt_dir, f"{g_id}.json")
                with open(ckpt_file, "w") as f:
                    json.dump(res, f)
        print(f"  -> Completed batch [{start_graph_idx}:{end_graph_idx}] in {time.perf_counter() - t0:.2f}s.", flush=True)

    if not finalize_offline_ground_truth:
        return

    # Verify all 100 graph checkpoints exist before assembling final CSVs and offline ground truth
    completed_ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".json"))
    print(f"[Stage 2 Assembly & Offline Ground Truth] Assembling {len(completed_ckpts)}/100 graph checkpoints...", flush=True)
    if len(completed_ckpts) < len(all_graphs):
        raise RuntimeError(f"Only {len(completed_ckpts)}/{len(all_graphs)} graphs completed.")

    meta_rows = []
    cand_eval_rows = []
    selected_cand_rows = []
    confirm_rows = []

    for g in all_graphs:
        g_id = g["graph_id"]
        with open(os.path.join(ckpt_dir, f"{g_id}.json"), "r") as f:
            data = json.load(f)
        meta_rows.extend(data["meta_rows"])
        cand_eval_rows.extend(data["cand_eval_rows"])
        selected_cand_rows.extend(data["selected_cand_rows"])
        confirm_rows.extend(data["confirm_rows"])

    meta_df = pd.DataFrame(meta_rows)
    cand_df = pd.DataFrame(cand_eval_rows)
    sel_df = pd.DataFrame(selected_cand_rows)
    conf_df = pd.DataFrame(confirm_rows)

    meta_df.to_csv(os.path.join(s2_raw, "experiment_metadata.csv"), index=False)
    cand_df.to_csv(os.path.join(s2_raw, "candidate_evaluations.csv"), index=False)
    sel_df.to_csv(os.path.join(s2_proc, "selected_candidates.csv"), index=False)

    # STRICTLY SEPARATED OFFLINE GROUND-TRUTH PHASE (Section 14):
    # Now that all optimization trajectories and candidate selections are frozen on disk,
    # compute exact expected loss mu_exact(theta*) for each selected checkpoint candidate!
    t_gt0 = time.perf_counter()
    graph_lookup = {g["graph_id"]: g for g in all_graphs}
    gt_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    gt_rows = []

    for _, srow in sel_df.iterrows():
        exp_id = srow["experiment_id"]
        g_id = srow["graph_id"]
        n = int(srow["graph_size"])
        K = int(srow["K_checkpoint"])
        param_str = srow["selected_parameters"]
        cache_key = (g_id, param_str)
        if cache_key not in gt_cache:
            theta_vec = json.loads(param_str)
            gt_cache[cache_key] = exact_expectation(
                graph_lookup[g_id], theta_vec, p=2, target_threshold=FROZEN_THRESHOLDS[n]
            )
        gt_val = gt_cache[cache_key]
        gt_rows.append({
            "experiment_id": exp_id,
            "graph_id": g_id,
            "graph_size": n,
            "B": int(srow["B"]),
            "optimizer_rep": int(srow["optimizer_rep"]),
            "K_checkpoint": K,
            "exact_expected_loss": float(gt_val["exact_expected_loss"]),
            "exact_loss_variance": float(gt_val["exact_loss_variance"]),
            "exact_expected_cut": float(gt_val["exact_expected_cut"]),
        })

    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(os.path.join(s2_proc, "ground_truth.csv"), index=False)
    print(f"  -> Computed offline ground_truth.csv ({len(gt_df)} rows, {len(gt_cache)} unique circuits) in {time.perf_counter() - t_gt0:.2f}s.", flush=True)

    # Merge exact_expected_loss into confirmation_results.csv and build certification_results.csv (Section 19)
    merged_sel_gt = pd.merge(
        sel_df,
        gt_df[["experiment_id", "K_checkpoint", "exact_expected_loss", "exact_loss_variance"]],
        on=["experiment_id", "K_checkpoint"],
    )
    merged_conf_gt = pd.merge(
        conf_df,
        gt_df[["experiment_id", "K_checkpoint", "exact_expected_loss", "exact_loss_variance"]],
        on=["experiment_id", "K_checkpoint"],
    )

    # Annotate confirmation_results.csv with ground-truth outcomes
    tau_vec = merged_conf_gt["graph_size"].map(FROZEN_THRESHOLDS)
    merged_conf_gt["target_threshold"] = tau_vec
    merged_conf_gt["genuinely_successful"] = merged_conf_gt["exact_expected_loss"] <= tau_vec
    # Section 12 rule: A false certification occurs when U(theta) <= tau while mu(theta) > tau strictly!
    merged_conf_gt["false_certification"] = (
        merged_conf_gt["certification_decision"] & (merged_conf_gt["exact_expected_loss"] > tau_vec)
    )
    merged_conf_gt["covered"] = (
        merged_conf_gt["exact_expected_loss"] <= merged_conf_gt["confirmation_upper_bound"] + 1e-15
    )
    merged_conf_gt["confirmation_bias"] = (
        merged_conf_gt["confirmation_mean"] - merged_conf_gt["exact_expected_loss"]
    )
    merged_conf_gt.to_csv(os.path.join(s2_proc, "confirmation_results.csv"), index=False)

    # Build certification_results.csv across all 4 in-sample methods AND independent confirmation methods
    cert_rows = []
    in_sample_methods = [
        ("naive_eb", "in_sample_naive_eb"),
        ("bonferroni_eb", "in_sample_bonferroni_eb"),
        ("hoeffding", "in_sample_hoeffding"),
        ("student_t_ref", "in_sample_student_t"),
    ]
    for idx_row, srow in merged_sel_gt.iterrows():
        exp_id = srow["experiment_id"]
        g_id = srow["graph_id"]
        n = int(srow["graph_size"])
        B = int(srow["B"])
        rep_idx = int(srow["optimizer_rep"])
        K = int(srow["K_checkpoint"])
        obs_mean = float(srow["selected_discovery_loss"])
        obs_var = float(srow["selected_discovery_variance"])
        mu_exact = float(srow["exact_expected_loss"])
        var_exact = float(srow["exact_loss_variance"])
        tau_n = FROZEN_THRESHOLDS[n]

        for m_code, m_label in in_sample_methods:
            c_res = evaluate_certification_method(
                m_code, obs_mean, obs_var, B, STAGE2_ALPHA, K=K, target_threshold=tau_n
            )
            ub = float(c_res["upper_confidence_bound"])
            cert_flag = bool(c_res["certified"])
            # Section 12: False certification occurs when U <= tau and mu > tau
            fc_flag = bool(cert_flag and (mu_exact > tau_n))
            cov_flag = bool(mu_exact <= ub + 1e-15)
            fn_flag = bool((not cert_flag) and (mu_exact <= tau_n))
            sig_level = STAGE2_ALPHA / float(K) if m_code == "bonferroni_eb" else STAGE2_ALPHA

            cert_rows.append({
                "experiment_id": exp_id,
                "graph_id": g_id,
                "graph_size": n,
                "B": B,
                "optimizer_rep": rep_idx,
                "K_checkpoint": K,
                "evaluation_regime": "in_sample",
                "method": m_label,
                "significance_level": float(sig_level),
                "observed_loss": obs_mean,
                "sample_variance": obs_var,
                "upper_bound": ub,
                "bound_slack": float(ub - obs_mean),
                "target_threshold": tau_n,
                "certified": cert_flag,
                "exact_expected_loss": mu_exact,
                "exact_loss_variance": var_exact,
                "genuinely_successful": bool(mu_exact <= tau_n),
                "false_certification": fc_flag,
                "false_non_certification": fn_flag,
                "covered": cov_flag,
                "discovery_shots": int(B * K),
                "confirmation_shots": 0,
                "total_shots": int(B * K),
            })

    # Also append independent confirmation rows into certification_results.csv for unified comparison
    for _, crow in merged_conf_gt.iterrows():
        exp_id = crow["experiment_id"]
        g_id = crow["graph_id"]
        n = int(crow["graph_size"])
        B = int(crow["B"])
        rep_idx = int(crow["optimizer_rep"])
        K = int(crow["K_checkpoint"])
        c_mean = float(crow["confirmation_mean"])
        c_var = float(crow["confirmation_variance"])
        mu_exact = float(crow["exact_expected_loss"])
        var_exact = float(crow["exact_loss_variance"])
        tau_n = FROZEN_THRESHOLDS[n]

        for m_code, m_label in [
            ("naive_eb", "independent_confirm_eb"),
            ("hoeffding", "independent_confirm_hoeffding"),
            ("student_t_ref", "independent_confirm_student_t"),
        ]:
            c_res = evaluate_certification_method(
                m_code, c_mean, c_var, STAGE2_B_CONFIRM, STAGE2_ALPHA, K=1, target_threshold=tau_n
            )
            ub = float(c_res["upper_confidence_bound"])
            cert_flag = bool(c_res["certified"])
            fc_flag = bool(cert_flag and (mu_exact > tau_n))
            cov_flag = bool(mu_exact <= ub + 1e-15)
            fn_flag = bool((not cert_flag) and (mu_exact <= tau_n))

            cert_rows.append({
                "experiment_id": exp_id,
                "graph_id": g_id,
                "graph_size": n,
                "B": B,
                "optimizer_rep": rep_idx,
                "K_checkpoint": K,
                "evaluation_regime": "independent_confirmation",
                "method": m_label,
                "significance_level": float(STAGE2_ALPHA),
                "observed_loss": c_mean,
                "sample_variance": c_var,
                "upper_bound": ub,
                "bound_slack": float(ub - c_mean),
                "target_threshold": tau_n,
                "certified": cert_flag,
                "exact_expected_loss": mu_exact,
                "exact_loss_variance": var_exact,
                "genuinely_successful": bool(mu_exact <= tau_n),
                "false_certification": fc_flag,
                "false_non_certification": fn_flag,
                "covered": cov_flag,
                "discovery_shots": int(B * K),
                "confirmation_shots": int(STAGE2_B_CONFIRM),
                "total_shots": int(B * K + STAGE2_B_CONFIRM),
            })

    cert_df = pd.DataFrame(cert_rows)
    cert_df.to_csv(os.path.join(s2_proc, "certification_results.csv"), index=False)
    print(f"  -> Saved certification_results.csv ({len(cert_df)} rows). Total assembly time: {time.perf_counter() - t0:.2f}s.", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        s_idx = int(sys.argv[1])
        e_idx = int(sys.argv[2])
        fin = bool(int(sys.argv[3]))
        run_stage2_batch(start_graph_idx=s_idx, end_graph_idx=e_idx, finalize_offline_ground_truth=fin)
    else:
        run_stage2_batch(start_graph_idx=0, end_graph_idx=100, finalize_offline_ground_truth=True)
