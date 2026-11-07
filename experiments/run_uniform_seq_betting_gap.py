"""
Closing the Confirmation-Engine Attribution Gap on the 500-Graph QAOA Benchmark
===============================================================================
Evaluates the matched 2,560-shot Uniform COBYLA discovery trajectory
(`uniform_disc_2560`, 16 candidates x 160 shots) followed by Phase B freezing
and Phase C sequential confirmation (`b_conf = 256`, `B_conf = 4096`, `alpha = 0.05`)
using the exact same Waudby-Smith & Ramdas (2024) aGRAPA hedged-betting confidence
sequence (`wsr_agrapa_betting`, `alpha_upper_fraction = 0.90`) deployed in
Tripura Modes U, H, and C.

Outputs:
  - results_stage3_corrected/processed/uniform_disc_2560_seq_conf_betting_500graphs.csv
  - Console summary of mean +/- SE discovery, confirmation, and total shots,
    certification rates, false certifications, and paired t-test / Wilcoxon
    signed-rank comparisons against Uniform + sequential (EB), Mode U, and Mode H.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from certification.confidence_sequences import run_sequential_confirmation_from_losses
from experiments.run_stage3 import load_canonical_stage3_config
from optimization.adaptive_cobyla import CachedQuantumSampler, run_cobyla_stage3_discovery


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) == "experiments":
    BASE_DIR = os.path.dirname(BASE_DIR)

CFG, CFG_HASH, _ = load_canonical_stage3_config(BASE_DIR)
GRAPHS_DF = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/raw/graphs.csv"))
E2E_DF = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/raw/stage3_corrected_e2e_runs.csv"))
MASTER_SEED = int(CFG["master_seed"])
TAUS = {int(k): float(v) for k, v in CFG["benchmark"]["frozen_thresholds"].items()}


def eval_single_graph(g_idx: int) -> Dict[str, Any]:
    row = GRAPHS_DF.iloc[g_idx]
    graph = {
        "graph_id": row["graph_id"],
        "n": int(row["n"]),
        "edges": json.loads(row["edges_json"]),
    }
    tau = TAUS[int(row["n"])]

    # 1. Reconstruct exact Phase A uniform COBYLA discovery trajectory (2,560 shots)
    s_init, s_shot = np.random.SeedSequence([MASTER_SEED, g_idx, 0]).spawn(2)
    disc = run_cobyla_stage3_discovery(
        graph=graph,
        qaoa_depth=2,
        target_threshold=tau,
        init_seed_seq=s_init,
        shot_seed_seq=s_shot,
        cobyla_mode="standard_fixed_cobyla",
        total_discovery_budget=2560,
        max_candidates=16,
        pilot_shots=160,
        batch_shots=32,
        allocation_method="uniform",
        elimination_mode="none",
        epsilon=0.008,
        alpha_discovery=0.05,
        cs_method="howard_gamma_exponential_eb",
        m_intrinsic=5.0,
        rhobeg=0.10,
        tol=1e-4,
        initialization_method="wurtz_love_3regular_jitter",
        jitter_half_width=0.07,
        enable_early_stopping=False,
        enable_target_promising_stop=False,
    )

    # 2. Draw the exact independent Phase C confirmation stream at frozen_theta
    tag_hash = int(hashlib.sha256("uniform_disc_2560_seq_conf_att1".encode("utf-8")).hexdigest()[:8], 16)
    c_ss = np.random.SeedSequence([MASTER_SEED + 80000, g_idx, tag_hash, 1])
    csamp = CachedQuantumSampler(graph, p=2, shot_seed_seq=c_ss)
    losses, _ = csamp.draw_shots(np.asarray(disc["frozen_theta"], dtype=np.float64), 4096, return_indices=False)

    # 3. Run sequential confirmation with WSR aGRAPA hedged-betting CS
    c_res_wsr = run_sequential_confirmation_from_losses(
        losses,
        target_threshold=tau,
        alpha=0.05,
        batch_size=256,
        max_shots=4096,
        method="wsr_agrapa_betting",
        m_intrinsic=5.0,
        alpha_upper_fraction=0.90,
    )

    return {
        "graph_index": g_idx,
        "graph_id": row["graph_id"],
        "n": int(row["n"]),
        "disc_shots": 2560,
        "conf_shots_wsr": int(c_res_wsr["stopping_shots"]),
        "total_shots_wsr": 2560 + int(c_res_wsr["stopping_shots"]),
        "certified_wsr": bool(c_res_wsr["certified"]),
        "rejected_wsr": bool(c_res_wsr["target_rejected"]),
        "inconclusive_wsr": bool(c_res_wsr["inconclusive"]),
        "L_t": float(c_res_wsr["L_t"]),
        "U_t": float(c_res_wsr["U_t"]),
    }


def main() -> None:
    t0 = time.perf_counter()
    workers = max(1, min(8, os.cpu_count() or 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res_list = list(ex.map(eval_single_graph, range(len(GRAPHS_DF))))

    df_wsr = pd.DataFrame(res_list).sort_values("graph_index").reset_index(drop=True)
    u_eb = E2E_DF[E2E_DF["method_id"] == "uniform_disc_2560_seq_conf"].sort_values("graph_index").reset_index(drop=True)
    mode_u = E2E_DF[E2E_DF["method_id"] == "proposed_sota_uncoupled"].sort_values("graph_index").reset_index(drop=True)
    mode_h = E2E_DF[E2E_DF["method_id"] == "proposed_sota_hybrid_simplex"].sort_values("graph_index").reset_index(drop=True)

    df_wsr["true_mu"] = u_eb["true_expected_loss"]
    df_wsr["tau"] = u_eb["target_threshold"]
    df_wsr["false_cert"] = df_wsr["certified_wsr"] & (df_wsr["true_mu"] > df_wsr["tau"])

    out_csv = os.path.join(
        BASE_DIR, "results_stage3_corrected/processed/uniform_disc_2560_seq_conf_betting_500graphs.csv"
    )
    df_wsr.to_csv(out_csv, index=False)

    print(f"Completed 500-graph evaluation in {time.perf_counter() - t0:.2f} s -> {out_csv}")
    for name, d_s, c_s, t_s, cert, fc in [
        (
            "Uniform + seq (EB)",
            u_eb["discovery_shots"],
            u_eb["confirmation_shots"],
            u_eb["total_shots"],
            u_eb["certified"],
            u_eb["false_certification"],
        ),
        (
            "Uniform + seq (Betting)",
            df_wsr["disc_shots"],
            df_wsr["conf_shots_wsr"],
            df_wsr["total_shots_wsr"],
            df_wsr["certified_wsr"],
            df_wsr["false_cert"],
        ),
        (
            "Mode U (Deployable, Betting)",
            mode_u["discovery_shots"],
            mode_u["confirmation_shots"],
            mode_u["total_shots"],
            mode_u["certified"],
            mode_u["false_certification"],
        ),
        (
            "Mode H (Sim. CRN, Betting)",
            mode_h["discovery_shots"],
            mode_h["confirmation_shots"],
            mode_h["total_shots"],
            mode_h["certified"],
            mode_h["false_certification"],
        ),
    ]:
        n_g = len(t_s)
        print(
            f"{name:28s} | Disc: {d_s.mean():6.1f} +/- {d_s.std(ddof=1)/np.sqrt(n_g):4.1f} "
            f"| Conf: {c_s.mean():6.1f} +/- {c_s.std(ddof=1)/np.sqrt(n_g):4.1f} "
            f"| Total: {t_s.mean():6.1f} +/- {t_s.std(ddof=1)/np.sqrt(n_g):4.1f} "
            f"(Saved {(1.0 - t_s.mean()/6656.0)*100:4.1f}%) "
            f"| Cert: {cert.mean():.3f} | FalseCert: {int(fc.sum())}/{n_g}"
        )

    diff_u = mode_u["total_shots"] - df_wsr["total_shots_wsr"]
    _, p_t_u = ttest_rel(mode_u["total_shots"], df_wsr["total_shots_wsr"])
    _, p_w_u = wilcoxon(diff_u)
    print(
        f"\nPaired diff [Mode U - Uniform+seq(Betting)]: "
        f"{diff_u.mean():.1f} +/- {diff_u.std(ddof=1)/np.sqrt(500):.1f} shots "
        f"(paired t p = {p_t_u:.3e}, Wilcoxon p = {p_w_u:.3e})"
    )

    diff_h = mode_h["total_shots"] - df_wsr["total_shots_wsr"]
    _, p_t_h = ttest_rel(mode_h["total_shots"], df_wsr["total_shots_wsr"])
    _, p_w_h = wilcoxon(diff_h)
    print(
        f"Paired diff [Mode H - Uniform+seq(Betting)]: "
        f"{diff_h.mean():.1f} +/- {diff_h.std(ddof=1)/np.sqrt(500):.1f} shots "
        f"(paired t p = {p_t_h:.3e}, Wilcoxon p = {p_w_h:.3e})"
    )


if __name__ == "__main__":
    main()
