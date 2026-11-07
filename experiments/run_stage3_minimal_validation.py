"""
Minimal Statistical Validation & Repair Script for AMFS 2.0 (50 Stratified Benchmark Graphs).

Compares on 50 stratified graphs (20 n=12 + 30 n=14) from the frozen 500-graph benchmark:
  - A. Existing Implementation (`A_existing_full_coupled_exploratory`):
       Reuses common uniform random sequence (`offset=cur_arm_shots`) across all candidates +
       exploratory Holm step-down & adaptive-leader pairwise betting.
  - B. Fresh-Randomness Adaptive Candidate Implementation (`B_fresh_adaptive_simplex_exploratory`):
       Preserves common random numbers ONLY for the predetermined initial simplex vertices (0..2p),
       while all adaptively generated COBYLA candidates receive fresh independent random streams.
  - C1. Corrected Sequential Certification Procedure (`C1_corrected_guaranteed_simplex_coupled`):
       Fresh independent random streams for all adaptive candidates + strictly guaranteed simultaneous
       family-wise error control (pre-allocated alpha_arm = alpha_discovery / K_max, zero Holm/pairwise
       adaptive-leader shortcuts) + independent-seed two-sided symmetric Phase C sequential confirmation.
  - C2. Corrected Pure Uncoupled Procedure (`C2_corrected_guaranteed_pure_uncoupled`):
       Same strict mathematical guarantees as C1 with 100% uncoupled independent streams even on the initial simplex.
  - D & E. Literature Baselines (`D_hao_2025_seq_conf_baseline`, `E_uniform_2560_seq_conf_baseline`).

Also executes a Sequence-Length Sensitivity Experiment across M_disc in {1280, 2560, 3840, 5120}
to test convergence between A (Existing) and B/C (Corrected) as shot sequence length increases.

All outputs are written to `results_stage3_corrected/minimal_validation_50graphs/` without modifying
any existing 500-graph dataset.
"""

import os
import json
import yaml
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from graphs.generation import generate_graph_ensemble
from simulator.statevector import get_cut_diagonal
from optimization.adaptive_cobyla import CachedQuantumSampler, run_cobyla_stage3_discovery
from certification.confidence_sequences import run_sequential_confirmation_from_losses
from analysis.ground_truth import exact_expectation
from analysis.stage3_analysis import holm_bonferroni_adjust, clopper_pearson_one_sided_upper


def _eval_single_validation_graph(task: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    g_idx = int(task["g_idx"])
    graph = task["graph"]
    cfg = task["cfg"]
    cfg_hash = str(task["cfg_hash"])
    master_seed = int(cfg["master_seed"])
    conf_offset = int(cfg["confirmatory_seed_offset"])
    taus = {int(k): float(v) for k, v in cfg["benchmark"]["frozen_thresholds"].items()}

    n = int(graph["n"])
    m_edges = len(graph["edges"])
    tau = taus[n]
    cuts_diag = get_cut_diagonal(graph)
    max_cut_opt = float(np.max(cuts_diag))
    min_loss_opt = float(1.0 - max_cut_opt / float(m_edges))
    shared_cdf_cache: Dict[bytes, np.ndarray] = {}
    gt_cache: Dict[bytes, Dict[str, float]] = {}

    def _get_gt(theta_vec: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(theta_vec, dtype=np.float64)
        k = arr.tobytes()
        if k not in gt_cache:
            gt = exact_expectation(graph, arr, p=2)
            gt_cache[k] = {
                "exact_loss": float(gt["exact_expected_loss"]),
                "exact_var": float(gt["exact_loss_variance"]),
                "exact_cut": float(gt["exact_expected_cut"]),
            }
        return gt_cache[k]

    def _run_method(
        method_id: str,
        cobyla_mode: str,
        disc_budget: int,
        max_cands: int,
        simp_shots: int,
        pilot_shots: int,
        coupling_mode: str,
        enable_holm: bool,
        enable_pairwise: bool,
        cs_method_disc: str,
        cs_method_conf: str,
        alpha_u_frac_conf: float,
        seed_tag: int = 0,
    ) -> Dict[str, Any]:
        s_init, s_shot = np.random.SeedSequence([master_seed, g_idx, seed_tag]).spawn(2)
        disc = run_cobyla_stage3_discovery(
            graph=graph,
            qaoa_depth=2,
            target_threshold=tau,
            init_seed_seq=s_init,
            shot_seed_seq=s_shot,
            cobyla_mode=cobyla_mode,
            total_discovery_budget=disc_budget,
            max_candidates=max_cands,
            pilot_shots=pilot_shots,
            simplex_anchor_shots=simp_shots,
            batch_shots=32,
            allocation_method="top_two_challenger" if cobyla_mode not in ("hao_style_cobyla", "standard_fixed_cobyla") else "uniform",
            elimination_mode="combined" if cobyla_mode not in ("hao_style_cobyla", "standard_fixed_cobyla") else "none",
            epsilon=0.008,
            alpha_discovery=0.05,
            cs_method=cs_method_disc,
            rhobeg=0.10,
            tol=1e-4,
            initialization_method="wurtz_love_3regular_jitter",
            jitter_half_width=0.07,
            enable_early_stopping=(cobyla_mode not in ("hao_style_cobyla", "standard_fixed_cobyla")),
            enable_target_promising_stop=(cobyla_mode not in ("hao_style_cobyla", "standard_fixed_cobyla")),
            enable_holm_step_down=enable_holm,
            enable_pairwise_betting=enable_pairwise,
            coupling_mode=coupling_mode,
            shared_cdf_cache=shared_cdf_cache,
        )

        arm_gts = [_get_gt(a.theta)["exact_loss"] for a in disc["arms"]]
        phase_a_family_covered = all(
            (a.sim_lower_bound - 1e-9 <= arm_gts[idx_a] <= a.sim_upper_bound + 1e-9)
            for idx_a, a in enumerate(disc["arms"])
        )
        pool_best = float(min(arm_gts))
        inc_elim = sum(
            1 for idx_a, a in enumerate(disc["arms"])
            if a.eliminated and (
                (a.elimination_reason == "best_candidate_eliminated" and arm_gts[idx_a] <= pool_best + 1e-9)
                or (a.elimination_reason == "target_rejected" and arm_gts[idx_a] <= tau)
            )
        )

        m_hash = int(hashlib.sha256(f"{method_id}_{disc_budget}".encode("utf-8")).hexdigest()[:8], 16)
        c_ss = np.random.SeedSequence([master_seed + conf_offset, g_idx, m_hash])
        c_sampler = CachedQuantumSampler(graph, p=2, shot_seed_seq=c_ss, shared_cdf_cache=shared_cdf_cache)
        conf_losses, _ = c_sampler.draw_shots(disc["frozen_theta"], 4096, return_indices=False)

        c_res = run_sequential_confirmation_from_losses(
            conf_losses,
            target_threshold=tau,
            alpha=0.05,
            batch_size=256,
            max_shots=4096,
            method=cs_method_conf,
            m_intrinsic=5.0,
            alpha_upper_fraction=alpha_u_frac_conf,
        )

        gt_star = _get_gt(disc["frozen_theta"])
        true_mu = float(gt_star["exact_loss"])
        true_cut = float(gt_star["exact_cut"])
        approx_ratio = float(true_cut / max_cut_opt)
        certified = bool(c_res["certified"])
        false_cert = bool(certified and (true_mu > tau))
        phase_c_covered = bool(c_res["L_t"] - 1e-9 <= true_mu <= c_res["U_t"] + 1e-9)

        return {
            "graph_id": graph["graph_id"],
            "graph_index": g_idx,
            "n_vertices": n,
            "method_id": method_id,
            "statistical_guarantee_status": disc["statistical_guarantee_status"],
            "coupling_mode": disc["coupling_mode"],
            "holm_step_down": disc["holm_step_down"],
            "pairwise_betting": disc["pairwise_betting"],
            "cs_method_discovery": disc["cs_method"],
            "cs_method_confirmation": cs_method_conf,
            "alpha_upper_fraction_conf": alpha_u_frac_conf,
            "config_hash": cfg_hash,
            "discovery_budget_cap": disc_budget,
            "discovery_shots": int(disc["total_discovery_shots"]),
            "confirmation_shots": int(c_res["stopping_shots"]),
            "total_shots": int(disc["total_discovery_shots"] + c_res["stopping_shots"]),
            "candidates_evaluated": int(disc["num_candidates_evaluated"]),
            "candidates_eliminated": int(disc["num_candidates_eliminated"]),
            "pct_candidates_eliminated": float(disc["pct_candidates_eliminated"]),
            "incorrect_eliminations": int(inc_elim),
            "phase_a_family_cs_covered": bool(phase_a_family_covered),
            "phase_c_cs_covered": bool(phase_c_covered),
            "certified": certified,
            "target_rejected": bool(c_res["target_rejected"]),
            "inconclusive": bool(c_res["inconclusive"]),
            "false_certification": false_cert,
            "true_expected_loss": true_mu,
            "approximation_ratio": approx_ratio,
            "optimality_gap": float(true_mu - min_loss_opt),
            "simple_regret_in_pool": float(true_mu - pool_best),
            "true_feasible": bool(true_mu <= tau),
            "empirical_discovery_loss": float(disc["selected_empirical_mean"]),
            "empirical_confirmation_loss": float(c_res["confirmation_mean"]),
            "L_conf": float(c_res["L_t"]),
            "U_conf": float(c_res["U_t"]),
        }

    m_specs = [
        ("A_existing_full_coupled_exploratory", "proposed_sota_full_coupled", 2560, 10, 256, 128, "full", True, True, "wsr_agrapa_betting", "wsr_agrapa_betting", 0.90),
        ("B_fresh_adaptive_simplex_exploratory", "proposed_sota_hybrid_simplex", 2560, 10, 256, 128, "simplex_only", True, True, "wsr_agrapa_betting", "wsr_agrapa_betting", 0.90),
        ("C1_corrected_guaranteed_simplex_coupled", "proposed_certified_corrected", 2560, 10, 256, 128, "simplex_only", False, False, "wsr_agrapa_betting", "wsr_agrapa_betting", 0.50),
        ("C2_corrected_guaranteed_pure_uncoupled", "proposed_certified_corrected", 2560, 10, 256, 128, "none", False, False, "wsr_agrapa_betting", "wsr_agrapa_betting", 0.50),
        ("D_hao_2025_seq_conf_baseline", "hao_style_cobyla", 2560, 8, 320, 320, "none", False, False, "howard_gamma_exponential_eb", "howard_gamma_exponential_eb", 0.50),
        ("E_uniform_2560_seq_conf_baseline", "standard_fixed_cobyla", 2560, 16, 160, 160, "none", False, False, "howard_gamma_exponential_eb", "howard_gamma_exponential_eb", 0.50),
    ]
    t_rows = [_run_method(*spec) for spec in m_specs]

    s_rows = []
    for b_disc, b_simp, b_pil in zip(
        cfg["sequence_length_sensitivity"]["discovery_budgets"],
        cfg["sequence_length_sensitivity"]["simplex_anchor_shots"],
        cfg["sequence_length_sensitivity"]["trial_pilot_shots"],
    ):
        s_rows.append(_run_method("A_existing_full_coupled", "proposed_sota_full_coupled", b_disc, 10, b_simp, b_pil, "full", True, True, "wsr_agrapa_betting", "wsr_agrapa_betting", 0.90, seed_tag=b_disc))
        s_rows.append(_run_method("B_fresh_adaptive_simplex", "proposed_sota_hybrid_simplex", b_disc, 10, b_simp, b_pil, "simplex_only", True, True, "wsr_agrapa_betting", "wsr_agrapa_betting", 0.90, seed_tag=b_disc))
        s_rows.append(_run_method("C1_corrected_guaranteed", "proposed_certified_corrected", b_disc, 10, b_simp, b_pil, "simplex_only", False, False, "wsr_agrapa_betting", "wsr_agrapa_betting", 0.50, seed_tag=b_disc))

    return {"trials": t_rows, "sens": s_rows}


def run_minimal_validation_50graphs(base_dir: str = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg_path = os.path.join(base_dir, "configs", "stage3_minimal_validation.yaml")
    with open(cfg_path, "rb") as f:
        cfg_bytes = f.read()
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
    cfg = yaml.safe_load(cfg_bytes.decode("utf-8"))

    out_dir = os.path.join(base_dir, "results_stage3_corrected", "minimal_validation_50graphs")
    os.makedirs(out_dir, exist_ok=True)

    g12_all = generate_graph_ensemble(
        n=12, count=80, split_name="stage3_n12",
        seed_seq=np.random.SeedSequence(20263012), id_prefix="G_S3_N12"
    )
    g14_all = generate_graph_ensemble(
        n=14, count=420, split_name="stage3_n14",
        seed_seq=np.random.SeedSequence(20263014), id_prefix="G_S3_N14"
    )
    val_tasks = [
        {"g_idx": i, "graph": g12_all[i], "cfg": cfg, "cfg_hash": cfg_hash} for i in range(20)
    ] + [
        {"g_idx": 80 + j, "graph": g14_all[j], "cfg": cfg, "cfg_hash": cfg_hash} for j in range(30)
    ]

    trial_rows, seq_sens_rows = [], []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        for res in ex.map(_eval_single_validation_graph, val_tasks, chunksize=5):
            trial_rows.extend(res["trials"])
            seq_sens_rows.extend(res["sens"])

    df_trials = pd.DataFrame(trial_rows)
    df_sens = pd.DataFrame(seq_sens_rows)

    df_trials.to_csv(os.path.join(out_dir, "minimal_validation_50graphs_trials.csv"), index=False)
    df_sens.to_csv(os.path.join(out_dir, "minimal_validation_sequence_length_sensitivity.csv"), index=False)

    sum_rows = []
    for m_id, grp in df_trials.groupby("method_id", sort=False):
        G = len(grp)
        k_fc = int(grp["false_certification"].sum())
        sum_rows.append({
            "method_id": m_id,
            "statistical_guarantee_status": str(grp["statistical_guarantee_status"].iloc[0]),
            "coupling_mode": str(grp["coupling_mode"].iloc[0]),
            "graphs_G": G,
            "mean_true_expected_loss": float(grp["true_expected_loss"].mean()),
            "sd_true_expected_loss": float(grp["true_expected_loss"].std(ddof=1)),
            "mean_approximation_ratio": float(grp["approximation_ratio"].mean()),
            "mean_discovery_shots": float(grp["discovery_shots"].mean()),
            "mean_confirmation_shots": float(grp["confirmation_shots"].mean()),
            "mean_total_shots": float(grp["total_shots"].mean()),
            "pct_candidates_eliminated": float(grp["pct_candidates_eliminated"].mean()),
            "incorrect_eliminations": int(grp["incorrect_eliminations"].sum()),
            "phase_a_family_cs_coverage": float(grp["phase_a_family_cs_covered"].mean()),
            "phase_c_cs_coverage": float(grp["phase_c_cs_covered"].mean()),
            "certification_rate": float(grp["certified"].mean()),
            "false_certifications": k_fc,
            "false_certification_rate": float(k_fc / float(G)),
            "clopper_pearson_95_ub": clopper_pearson_one_sided_upper(k_fc, G, 0.95),
        })
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(out_dir, "minimal_validation_50graphs_summary.csv"), index=False)

    sens_sum = (
        df_sens.groupby(["discovery_budget_cap", "method_id", "statistical_guarantee_status"], as_index=False)
        .agg(
            graphs_G=("graph_id", "nunique"),
            mean_true_expected_loss=("true_expected_loss", "mean"),
            mean_approximation_ratio=("approximation_ratio", "mean"),
            mean_discovery_shots=("discovery_shots", "mean"),
            mean_confirmation_shots=("confirmation_shots", "mean"),
            mean_total_shots=("total_shots", "mean"),
            pct_candidates_eliminated=("pct_candidates_eliminated", "mean"),
            phase_a_family_cs_coverage=("phase_a_family_cs_covered", "mean"),
            phase_c_cs_coverage=("phase_c_cs_covered", "mean"),
            certification_rate=("certified", "mean"),
            false_certifications=("false_certification", "sum"),
        )
    )
    piv_s_loss = df_sens.pivot(index=["discovery_budget_cap", "graph_id"], columns="method_id", values="true_expected_loss")
    gap_rows = []
    for b_cap in cfg["sequence_length_sensitivity"]["discovery_budgets"]:
        sub_p = piv_s_loss.loc[b_cap]
        diff_AB = sub_p["A_existing_full_coupled"] - sub_p["B_fresh_adaptive_simplex"]
        diff_AC = sub_p["A_existing_full_coupled"] - sub_p["C1_corrected_guaranteed"]
        diff_BC = sub_p["B_fresh_adaptive_simplex"] - sub_p["C1_corrected_guaranteed"]
        gap_rows.append({
            "discovery_budget_cap": int(b_cap),
            "mean_loss_A_existing": float(sub_p["A_existing_full_coupled"].mean()),
            "mean_loss_B_fresh_adaptive": float(sub_p["B_fresh_adaptive_simplex"].mean()),
            "mean_loss_C1_corrected_guaranteed": float(sub_p["C1_corrected_guaranteed"].mean()),
            "paired_diff_A_minus_B": float(diff_AB.mean()),
            "paired_diff_A_minus_C1": float(diff_AC.mean()),
            "paired_diff_B_minus_C1": float(diff_BC.mean()),
            "abs_diff_B_vs_C1": float(np.abs(diff_BC).mean()),
        })
    df_sens_gaps = pd.DataFrame(gap_rows)
    sens_sum.to_csv(os.path.join(out_dir, "minimal_validation_sequence_sensitivity_summary.csv"), index=False)
    df_sens_gaps.to_csv(os.path.join(out_dir, "minimal_validation_convergence_gaps.csv"), index=False)

    piv_loss = df_trials.pivot(index="graph_id", columns="method_id", values="true_expected_loss")
    piv_shots = df_trials.pivot(index="graph_id", columns="method_id", values="total_shots")

    pairs = [
        ("B_fresh_adaptive_simplex_exploratory", "A_existing_full_coupled_exploratory", "true_expected_loss", "solution_quality"),
        ("C1_corrected_guaranteed_simplex_coupled", "A_existing_full_coupled_exploratory", "true_expected_loss", "solution_quality"),
        ("C1_corrected_guaranteed_simplex_coupled", "B_fresh_adaptive_simplex_exploratory", "true_expected_loss", "solution_quality"),
        ("C1_corrected_guaranteed_simplex_coupled", "D_hao_2025_seq_conf_baseline", "true_expected_loss", "solution_quality"),
        ("C1_corrected_guaranteed_simplex_coupled", "E_uniform_2560_seq_conf_baseline", "true_expected_loss", "solution_quality"),
        ("C2_corrected_guaranteed_pure_uncoupled", "D_hao_2025_seq_conf_baseline", "true_expected_loss", "solution_quality"),
        ("C2_corrected_guaranteed_pure_uncoupled", "E_uniform_2560_seq_conf_baseline", "true_expected_loss", "solution_quality"),
        ("C1_corrected_guaranteed_simplex_coupled", "B_fresh_adaptive_simplex_exploratory", "total_shots", "shot_cost"),
        ("C1_corrected_guaranteed_simplex_coupled", "D_hao_2025_seq_conf_baseline", "total_shots", "shot_cost"),
        ("C1_corrected_guaranteed_simplex_coupled", "E_uniform_2560_seq_conf_baseline", "total_shots", "shot_cost"),
        ("B_fresh_adaptive_simplex_exploratory", "D_hao_2025_seq_conf_baseline", "total_shots", "shot_cost"),
    ]
    test_rows = []
    for ma, mb, met, fam in pairs:
        piv = piv_loss if met == "true_expected_loss" else piv_shots
        xa = piv[ma].to_numpy(dtype=np.float64)
        xb = piv[mb].to_numpy(dtype=np.float64)
        diff = xa - xb
        md = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        se = float(sd / np.sqrt(len(diff)))
        if sd <= 1e-15:
            t_s, p_u, p_w = 0.0, 1.0, 1.0
        else:
            t_s, p_u = ttest_rel(xa, xb)
            try:
                _, p_w = wilcoxon(diff)
            except Exception:
                p_w = 1.0
        test_rows.append({
            "hypothesis_family": fam,
            "method_a": ma,
            "method_b": mb,
            "metric": met,
            "graphs_G": int(len(diff)),
            "mean_a": float(np.mean(xa)),
            "mean_b": float(np.mean(xb)),
            "paired_mean_diff": md,
            "ci95_lower": float(md - 1.96 * se),
            "ci95_upper": float(md + 1.96 * se),
            "paired_t_stat": float(t_s),
            "p_value_unadjusted": float(p_u),
            "p_value_wilcoxon": float(p_w),
        })

    for fam_name in ["solution_quality", "shot_cost"]:
        f_idx = [i for i, r in enumerate(test_rows) if r["hypothesis_family"] == fam_name]
        f_adj = holm_bonferroni_adjust([test_rows[i]["p_value_unadjusted"] for i in f_idx])
        for i_pos, ph in zip(f_idx, f_adj):
            test_rows[i_pos]["p_value_holm_family"] = float(ph)
            md = test_rows[i_pos]["paired_mean_diff"]
            if ph < 0.05:
                test_rows[i_pos]["significance_classification"] = (
                    "significant_improvement_a_over_b" if md < 0.0 else "significant_degradation_a_vs_b"
                )
            else:
                test_rows[i_pos]["significance_classification"] = "statistically_inconclusive_at_0.05"

    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv(os.path.join(out_dir, "minimal_validation_50graphs_paired_tests.csv"), index=False)

    print(f"Completed 50-graph minimal validation in {time.perf_counter()-t0:.1f}s.")
    print("\n=== 50-GRAPH TARGETED VALIDATION SUMMARY (M_disc = 2560) ===")
    print(df_sum.to_string(index=False))
    print("\n=== SEQUENCE-LENGTH SENSITIVITY CONVERGENCE GAPS ===")
    print(df_sens_gaps.to_string(index=False))
    print("\n=== PAIRED STATISTICAL TESTS (50 GRAPHS) ===")
    print(df_tests[["hypothesis_family", "method_a", "method_b", "metric", "paired_mean_diff", "ci95_lower", "ci95_upper", "p_value_unadjusted", "p_value_holm_family", "significance_classification"]].to_string(index=False))
    return {"summary": df_sum, "sens_gaps": df_sens_gaps, "tests": df_tests}


if __name__ == "__main__":
    run_minimal_validation_50graphs()
