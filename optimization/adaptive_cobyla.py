"""
COBYLA Integration with Adaptive Shot Allocation, Candidate Elimination,
Two-Tier Simplex Anchoring, Loss-Ordered Quantum Coupling, and Discovery Stopping
for AMFS 2.0 (Stage 3 Corrected & State-of-the-Art Upgraded).

Repairs & Literature Upgrades Enforced:
1. Simultaneous family-wise alpha allocation (alpha_arm = alpha_discovery / max_candidates)
   via SimultaneousAlphaSchedule. Dynamic candidate additions beyond max_candidates raise ValueError.
2. Supports both strict [0, 1]-bounded Howard et al. (2021) Gamma-Exponential EB (`howard_gamma_exponential_eb`)
   and Waudby-Smith & Ramdas (2024, JRSSB, Theorem 1 & Prop 3) aGRAPA Hedged-Capital Betting CS (`wsr_agrapa_betting`).
3. Zero heuristic z-score elimination rules (all ad-hoc SE thresholds completely removed).
4. Sequential Holm Step-Down Alpha Recycling (Holm 1979) & Pairwise Difference Martingale Elimination
   (Waudby-Smith & Ramdas 2024) on Y_{i, l_t, s} = 0.5 + 0.5*(X_{i,s} - X_{l_t,s}) in [0, 1].
5. Two-Tier Simplex Anchoring (Powell 1994; Kaufmann & Kalyanakrishnan 2013; Hao et al. 2025):
   anchors initial 2p+1 simplex compass vertices at `simplex_anchor_shots` for low-noise gradient estimation,
   then applies adaptive racing across downhill trial candidates.
6. Monotone Loss-Ordered Inverse-CDF Coupling & Z2 Bit-Flip Symmetry (Farhi et al. 2014; Owen 2013):
   preserves exact marginal multinomial QAOA distribution of every candidate while reducing directional
   gradient variance across simplex vertices (`coupling_mode in {"none", "simplex_only", "full"}`).
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from simulator.qaoa import simulate_statevector
from simulator.statevector import extract_graph_info, get_cut_diagonal, get_probabilities
from optimization.initialization import generate_initial_parameters
from optimization.adaptive_allocation import (
    SimultaneousAlphaSchedule,
    CandidateArmState,
    compute_ugape_indices,
    select_next_candidate_to_sample,
    apply_candidate_elimination,
    select_final_candidate_from_pool,
)
from certification.confidence_sequences import wsr_log_wealth_minus


class CachedQuantumSampler:
    """
    Fast CDF-cached finite-shot quantum circuit sampler for a given graph and QAOA depth p.
    Computes statevector probabilities to draw genuine multinomial projective measurement shots,
    and also supports monotone loss-ordered inverse-CDF coupling (preserving exact marginal
    multinomial distributions while reducing pairwise comparison variance).
    NEVER computes, caches, or exposes exact expected losses or exact variances.
    """

    def __init__(
        self,
        graph: Dict[str, Any],
        p: int,
        shot_seed_seq: np.random.SeedSequence,
        shared_cdf_cache: Optional[Dict[bytes, np.ndarray]] = None,
    ):
        self.graph = graph
        self.p = int(p)
        self.n, self.edges = extract_graph_info(graph)
        self.m = len(self.edges)
        if self.m == 0:
            raise ValueError("Graph must have at least 1 edge.")
        self.cuts_diag = get_cut_diagonal(graph)
        self.losses_diag = 1.0 - (self.cuts_diag / float(self.m))
        self.shot_seed_seq = shot_seed_seq
        self._cdf_cache: Dict[bytes, np.ndarray] = shared_cdf_cache if shared_cdf_cache is not None else {}

        # Precompute loss-sorted basis permutation for monotone Fréchet-Hoeffding coupling
        self._sort_order = np.argsort(self.losses_diag, kind="mergesort")
        self._sorted_losses = self.losses_diag[self._sort_order]
        self._coupled_u_stream: Optional[np.ndarray] = None

    def _get_cdf(self, theta: np.ndarray) -> np.ndarray:
        arr = np.asarray(theta, dtype=np.float64)
        key = arr.tobytes()
        if key not in self._cdf_cache:
            state = simulate_statevector(self.graph, arr, p=self.p)
            probs = get_probabilities(state)
            # Enforce exact Z2 bit-flip symmetry P(z) = P(~z) of QAOA on MaxCut
            probs = 0.5 * (probs + probs[::-1])
            cdf = np.cumsum(probs)
            cdf[-1] = 1.0
            if len(self._cdf_cache) < 2048:
                self._cdf_cache[key] = cdf
            return cdf
        return self._cdf_cache[key]

    def _get_sorted_cdf(self, theta: np.ndarray) -> np.ndarray:
        arr = np.asarray(theta, dtype=np.float64)
        key = b"SORTED_" + arr.tobytes()
        if key not in self._cdf_cache:
            cdf_std = self._get_cdf(arr)
            probs = np.empty_like(cdf_std)
            probs[0] = cdf_std[0]
            probs[1:] = np.diff(cdf_std)
            s_probs = probs[self._sort_order]
            s_cdf = np.cumsum(s_probs)
            s_cdf[-1] = 1.0
            if len(self._cdf_cache) < 2048:
                self._cdf_cache[key] = s_cdf
            return s_cdf
        return self._cdf_cache[key]

    def draw_shots(
        self,
        theta: np.ndarray,
        n_shots: int,
        return_indices: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if n_shots < 1:
            return np.empty(0, dtype=np.float64), (np.empty(0, dtype=np.int64) if return_indices else None)
        child_ss = self.shot_seed_seq.spawn(1)[0]
        rng = np.random.default_rng(child_ss)
        cdf = self._get_cdf(theta)
        u = rng.random(int(n_shots))
        sampled_indices = np.searchsorted(cdf, u, side="right")
        sampled_indices = np.minimum(sampled_indices, cdf.size - 1)
        shot_losses = self.losses_diag[sampled_indices]
        return shot_losses, (sampled_indices if return_indices else None)

    def draw_coupled_shots(
        self,
        theta: np.ndarray,
        n_shots: int,
        offset: int = 0,
        return_indices: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Draws n_shots from the exact marginal multinomial QAOA distribution of `theta`
        coupled via monotone loss-ordered inverse transform on u_stream[offset : offset + n_shots].
        """
        if n_shots < 1:
            return np.empty(0, dtype=np.float64), (np.empty(0, dtype=np.int64) if return_indices else None)
        if self._coupled_u_stream is None or self._coupled_u_stream.size < offset + n_shots:
            child_ss = self.shot_seed_seq.spawn(1)[0]
            rng = np.random.default_rng(child_ss)
            req_len = max(16384, offset + n_shots + 1024)
            self._coupled_u_stream = rng.random(req_len)
        u = self._coupled_u_stream[int(offset) : int(offset + n_shots)]
        s_cdf = self._get_sorted_cdf(theta)
        s_pos = np.minimum(np.searchsorted(s_cdf, u, side="right"), s_cdf.size - 1)
        shot_losses = self._sorted_losses[s_pos]
        orig_indices = self._sort_order[s_pos] if return_indices else None
        return shot_losses, orig_indices


def run_cobyla_stage3_discovery(
    graph: Dict[str, Any],
    qaoa_depth: int,
    target_threshold: float,
    init_seed_seq: np.random.SeedSequence,
    shot_seed_seq: np.random.SeedSequence,
    cobyla_mode: str = "proposed_adaptive_cobyla",
    total_discovery_budget: int = 2560,
    max_candidates: int = 16,
    pilot_shots: int = 64,
    simplex_anchor_shots: Optional[int] = None,
    batch_shots: int = 32,
    duel_max_shots: int = 384,
    allocation_method: str = "ugape_eb",
    elimination_mode: str = "combined",
    epsilon: float = 0.008,
    alpha_discovery: float = 0.05,
    cs_method: str = "howard_gamma_exponential_eb",
    m_intrinsic: float = 5.0,
    rhobeg: float = 0.10,
    tol: float = 1e-4,
    initialization_method: str = "wurtz_love_3regular_jitter",
    jitter_half_width: float = 0.07,
    enable_early_stopping: bool = True,
    enable_target_promising_stop: bool = False,
    enable_holm_step_down: bool = False,
    enable_pairwise_betting: bool = False,
    coupling_mode: str = "none",
    shared_cdf_cache: Optional[Dict[bytes, np.ndarray]] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    init_rng = np.random.default_rng(init_seed_seq)
    theta_0 = generate_initial_parameters(
        p=int(qaoa_depth),
        rng=init_rng,
        jitter_half_width=float(jitter_half_width),
        method=initialization_method,
    )

    sampler = CachedQuantumSampler(
        graph=graph,
        p=int(qaoa_depth),
        shot_seed_seq=shot_seed_seq,
        shared_cdf_cache=shared_cdf_cache,
    )

    eff_rhobeg = float(rhobeg)
    eff_tol = float(tol)
    eff_max_candidates = int(max_candidates)
    eff_cs_method = str(cs_method)
    eff_holm = bool(enable_holm_step_down)
    eff_pairwise = bool(enable_pairwise_betting)
    eff_coupling = str(coupling_mode)
    eff_simplex_shots = int(simplex_anchor_shots) if simplex_anchor_shots is not None else int(pilot_shots)

    if cobyla_mode == "standard_fixed_cobyla":
        eff_allocation = "uniform"
        eff_elimination = "none"
        eff_early_stop = False
        b_per_eval = max(32, int(total_discovery_budget // max(1, eff_max_candidates)))
        eff_pilot = b_per_eval
        eff_simplex_shots = b_per_eval
    elif cobyla_mode == "hao_style_cobyla":
        eff_max_candidates = min(int(max_candidates), 2 * int(qaoa_depth) + 1 + 3)
        eff_allocation = "uniform"
        eff_elimination = "none"
        eff_early_stop = False
        b_per_eval = max(64, int(total_discovery_budget // max(1, eff_max_candidates)))
        eff_pilot = b_per_eval
        eff_simplex_shots = b_per_eval
    elif cobyla_mode == "variance_aware_cobyla":
        eff_allocation = "variance_only"
        eff_elimination = "none"
        eff_early_stop = bool(enable_early_stopping)
        eff_pilot = int(pilot_shots)
    elif cobyla_mode == "gap_aware_cobyla":
        eff_allocation = "gap_only"
        eff_elimination = str(elimination_mode)
        eff_early_stop = bool(enable_early_stopping)
        eff_pilot = int(pilot_shots)
    elif cobyla_mode == "proposed_adaptive_cobyla":
        eff_allocation = str(allocation_method)
        eff_elimination = str(elimination_mode)
        eff_early_stop = bool(enable_early_stopping)
        eff_pilot = int(pilot_shots)
    elif cobyla_mode == "proposed_certified_corrected":
        # Strictly guaranteed simultaneous FWER <= alpha_discovery:
        # - Common random numbers ONLY on predetermined initial simplex (simplex_only)
        # - Pre-allocated alpha_arm = alpha_discovery / K_max (eff_holm = False, eff_pairwise = False)
        # - Elimination & early stopping use ONLY individual simultaneous CS bounds [L_i^sim(t), U_i^sim(t)]
        eff_max_candidates = int(max_candidates) if max_candidates != 16 else 10
        eff_simplex_shots = int(simplex_anchor_shots) if simplex_anchor_shots is not None else 256
        eff_pilot = int(pilot_shots) if pilot_shots != 64 else 128
        eff_allocation = "top_two_challenger" if allocation_method == "ugape_eb" else str(allocation_method)
        eff_elimination = str(elimination_mode)
        eff_early_stop = bool(enable_early_stopping)
        eff_cs_method = "wsr_agrapa_betting"
        eff_holm = False
        eff_pairwise = False
        eff_coupling = str(coupling_mode)
    elif cobyla_mode in ("proposed_sota_uncoupled", "proposed_sota_hybrid_simplex", "proposed_sota_full_coupled"):
        # State-of-the-art 5-upgrade configurations:
        # 1. WSR aGRAPA Hedged-Capital Betting CS (`wsr_agrapa_betting`)
        # 2. Sequential Holm Step-Down Alpha Recycling (`eff_holm = True`)
        # 3. Pairwise Difference Martingale Elimination (`eff_pairwise = True`)
        # 4. Two-Tier Simplex Anchoring (`simplex_anchor_shots = 256`, `pilot_shots = 128`, `max_candidates = 10`) + LUCB (`top_two_challenger`)
        # 5. Monotone Loss-Ordered Quantum Coupling (`none` vs `simplex_only` vs `full`)
        eff_max_candidates = int(max_candidates) if max_candidates != 16 else 10
        eff_simplex_shots = int(simplex_anchor_shots) if simplex_anchor_shots is not None else 256
        eff_pilot = int(pilot_shots) if pilot_shots != 64 else 128
        eff_allocation = "top_two_challenger" if allocation_method == "ugape_eb" else str(allocation_method)
        eff_elimination = str(elimination_mode)
        eff_early_stop = bool(enable_early_stopping)
        eff_cs_method = "wsr_agrapa_betting"
        eff_holm = True
        eff_pairwise = True
        if cobyla_mode == "proposed_sota_uncoupled":
            eff_coupling = "none"
        elif cobyla_mode == "proposed_sota_hybrid_simplex":
            eff_coupling = "simplex_only"
        else:
            eff_coupling = "full"
    else:
        raise ValueError(f"Unsupported cobyla_mode '{cobyla_mode}'.")

    alpha_scheduler = SimultaneousAlphaSchedule(
        alpha_discovery=float(alpha_discovery),
        max_candidates=eff_max_candidates,
    )

    arms: List[CandidateArmState] = []
    incumbent_id: Optional[int] = None
    cumulative_shots = 0
    pilot_shots_total = 0
    elimination_log: List[Dict[str, Any]] = []
    allocation_history: List[Dict[str, Any]] = []
    stopping_reason = "budget_exhausted"
    early_stopped_flag = False

    n_simplex = 2 * int(qaoa_depth) + 1

    def _draw_for_arm(arm_idx: int, theta_vec: np.ndarray, n_shots: int, cur_arm_shots: int, tot_shots: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if eff_coupling == "none":
            return sampler.draw_shots(theta_vec, n_shots, return_indices=True)
        elif eff_coupling == "simplex_only":
            # Common random numbers ONLY for predetermined initial simplex vertices (arm_idx < n_simplex, cur_arm_shots == 0);
            # all adaptively generated COBYLA candidates and subsequent adaptive batches receive fresh independent streams.
            if arm_idx < n_simplex and cur_arm_shots == 0:
                return sampler.draw_coupled_shots(theta_vec, n_shots, offset=0, return_indices=True)
            else:
                return sampler.draw_shots(theta_vec, n_shots, return_indices=True)
        elif eff_coupling == "legacy_simplex_offset":
            if arm_idx < n_simplex and cur_arm_shots == 0:
                return sampler.draw_coupled_shots(theta_vec, n_shots, offset=0, return_indices=True)
            else:
                return sampler.draw_coupled_shots(theta_vec, n_shots, offset=tot_shots, return_indices=True)
        elif eff_coupling == "full":
            # Separate experimental mode: reuses common uniform sequence from cur_arm_shots across all candidates
            return sampler.draw_coupled_shots(theta_vec, n_shots, offset=cur_arm_shots, return_indices=True)
        else:
            raise ValueError(f"Unknown coupling_mode '{eff_coupling}'.")

    def _create_arm(theta_vec: np.ndarray, origin_label: str) -> CandidateArmState:
        alpha_i = alpha_scheduler.allocate_next_arm_alpha()
        cid = len(arms)
        arm = CandidateArmState(
            candidate_id=cid,
            theta=np.clip(theta_vec, 0.0, 2.0 * np.pi),
            origin=origin_label,
            alpha_arm=alpha_i,
            alpha_simultaneous=alpha_i,
            alpha_marginal=float(alpha_discovery),
            cs_method=eff_cs_method,
            m_intrinsic=float(m_intrinsic),
            initial_prediction=float(target_threshold),
            n_qubits=sampler.n,
        )
        arms.append(arm)
        return arm

    def _noisy_cobyla_objective(theta_trial: np.ndarray) -> float:
        nonlocal incumbent_id, cumulative_shots, pilot_shots_total, stopping_reason, early_stopped_flag

        if early_stopped_flag or len(arms) >= eff_max_candidates or cumulative_shots >= total_discovery_budget:
            if incumbent_id is not None:
                return float(arms[incumbent_id].empirical_mean + 0.05)
            return float(target_threshold + 0.05)

        idx = len(arms)
        target_b = int(eff_simplex_shots) if idx < n_simplex else int(eff_pilot)
        b_init = min(target_b, int(total_discovery_budget - cumulative_shots))
        if b_init < 2:
            if incumbent_id is not None:
                return float(arms[incumbent_id].empirical_mean + 0.05)
            return float(target_threshold + 0.05)

        if idx == 0:
            origin = "warm_start_initial"
        elif idx < n_simplex:
            origin = "cobyla_initial_simplex"
        else:
            origin = "cobyla_trial_step"

        arm = _create_arm(theta_trial, origin)
        losses, b_indices = _draw_for_arm(arm.candidate_id, arm.theta, b_init, 0, cumulative_shots)
        arm.add_observations(losses, is_pilot=True, basis_indices=b_indices)
        cumulative_shots += int(losses.size)
        pilot_shots_total += int(losses.size)

        if incumbent_id is None or arm.empirical_mean < arms[incumbent_id].empirical_mean:
            incumbent_id = arm.candidate_id

        active_now = [a.candidate_id for a in arms if not a.eliminated and a.sample_count > 0]
        active_now, elim_ev = apply_candidate_elimination(
            arms=arms,
            active_ids=active_now,
            elimination_mode=eff_elimination,
            target_threshold=target_threshold,
            step_index=len(allocation_history),
            epsilon_best=0.0,
            enable_holm_step_down=eff_holm,
            enable_pairwise_betting=eff_pairwise,
            alpha_discovery=alpha_discovery,
        )
        elimination_log.extend(elim_ev)

        if len(active_now) == 0 and eff_elimination != "none":
            stopping_reason = "all_candidates_rejected"
            early_stopped_flag = True
            return float(arm.empirical_mean)

        if eff_early_stop and enable_target_promising_stop and len(arms) >= (n_simplex + 2) and len(active_now) > 0:
            _, l_now, _ = compute_ugape_indices(arms, active_now, use_hoeffding=(eff_allocation == "gap_only"))
            lead_arm_now = arms[l_now]
            if lead_arm_now.sim_upper_bound <= float(target_threshold):
                stopping_reason = "target_feasible_promising"
                early_stopped_flag = True
            elif eff_holm and eff_cs_method == "wsr_agrapa_betting" and lead_arm_now.sample_count >= 256:
                # Exploratory heuristic early stopping when eff_holm=True
                alpha_lead = float(alpha_discovery) / float(len(active_now))
                if wsr_log_wealth_minus(lead_arm_now.observations, target_threshold, initial_prediction=target_threshold) >= np.log(1.0 / max(1e-12, 0.90 * alpha_lead)):
                    stopping_reason = "target_feasible_promising"
                    early_stopped_flag = True

        return float(arm.empirical_mean)

    minimize(
        _noisy_cobyla_objective,
        theta_0.copy(),
        method="COBYLA",
        options={
            "rhobeg": float(eff_rhobeg),
            "tol": float(eff_tol),
            "maxiter": int(eff_max_candidates) + 5,
            "disp": False,
        },
    )

    active_ids = [a.candidate_id for a in arms if not a.eliminated and a.sample_count > 0]
    active_ids, post_elim = apply_candidate_elimination(
        arms=arms,
        active_ids=active_ids,
        elimination_mode=eff_elimination,
        target_threshold=target_threshold,
        step_index=len(allocation_history),
        epsilon_best=0.0,
        enable_holm_step_down=eff_holm,
        enable_pairwise_betting=eff_pairwise,
        alpha_discovery=alpha_discovery,
    )
    elimination_log.extend(post_elim)

    adaptive_modes = (
        "proposed_certified_corrected",
        "proposed_adaptive_cobyla",
        "variance_aware_cobyla",
        "gap_aware_cobyla",
        "proposed_sota_uncoupled",
        "proposed_sota_hybrid_simplex",
        "proposed_sota_full_coupled",
    )
    if not early_stopped_flag and cobyla_mode in adaptive_modes:
        while cumulative_shots < total_discovery_budget:
            if len(active_ids) == 0:
                stopping_reason = "all_candidates_rejected"
                break

            B_indices, l_t, u_t = compute_ugape_indices(
                arms, active_ids, use_hoeffding=(eff_allocation == "gap_only")
            )
            leader_arm = arms[l_t]

            if eff_early_stop and (len(active_ids) == 1 or B_indices[l_t] <= float(epsilon)):
                if len(active_ids) == 1 or leader_arm.sim_radius <= float(epsilon) or leader_arm.sim_upper_bound <= float(target_threshold):
                    stopping_reason = "best_candidate_identified"
                    break

            if eff_early_stop and enable_target_promising_stop:
                if leader_arm.sim_upper_bound <= float(target_threshold):
                    stopping_reason = "target_feasible_promising"
                    break
                if eff_holm and eff_cs_method == "wsr_agrapa_betting" and leader_arm.sample_count >= 256:
                    alpha_lead = float(alpha_discovery) / float(len(active_ids))
                    if wsr_log_wealth_minus(leader_arm.observations, target_threshold, initial_prediction=target_threshold) >= np.log(1.0 / max(1e-12, 0.90 * alpha_lead)):
                        stopping_reason = "target_feasible_promising"
                        break

            pick_id = select_next_candidate_to_sample(
                arms=arms,
                active_ids=active_ids,
                allocation_method=eff_allocation,
                step_index=len(allocation_history) + 1,
                epsilon=epsilon,
            )
            b_step = min(int(batch_shots), int(total_discovery_budget - cumulative_shots))
            if b_step < 1:
                stopping_reason = "budget_exhausted"
                break

            chosen_arm = arms[pick_id]
            more_losses, more_indices = _draw_for_arm(
                chosen_arm.candidate_id, chosen_arm.theta, b_step, chosen_arm.sample_count, cumulative_shots
            )
            chosen_arm.add_observations(more_losses, is_pilot=False, basis_indices=more_indices)
            cumulative_shots += int(more_losses.size)

            allocation_history.append({
                "step": len(allocation_history) + 1,
                "phase": "pool_consolidation",
                "chosen_candidate_id": int(pick_id),
                "batch_shots": int(more_losses.size),
                "cumulative_shots": int(cumulative_shots),
                "active_count": int(len(active_ids)),
                "leader_id": int(l_t),
                "leader_B_sim_index": float(B_indices[l_t]),
            })

            active_ids, elim_ev = apply_candidate_elimination(
                arms=arms,
                active_ids=active_ids,
                elimination_mode=eff_elimination,
                target_threshold=target_threshold,
                step_index=len(allocation_history),
                epsilon_best=0.0,
                enable_holm_step_down=eff_holm,
                enable_pairwise_betting=eff_pairwise,
                alpha_discovery=alpha_discovery,
            )
            elimination_log.extend(elim_ev)

    selected_id = select_final_candidate_from_pool(arms, active_ids, eff_allocation)
    selected_arm = arms[selected_id]
    ranked_ids = sorted(
        [a.candidate_id for a in arms if a.sample_count > 0],
        key=lambda cid: (
            arms[cid].eliminated and arms[cid].elimination_reason == "target_rejected",
            arms[cid].empirical_mean,
            arms[cid].sim_upper_bound,
            cid,
        ),
    )

    frozen_theta = selected_arm.theta.copy()
    frozen_theta.setflags(write=False)

    eliminated_arms = [a for a in arms if a.eliminated]
    max_arm_shots = max((a.sample_count for a in arms), default=eff_pilot)
    shots_saved = sum(max(0, max_arm_shots - a.sample_count) for a in eliminated_arms)
    mp_bstr, mp_loss = selected_arm.most_probable_sampled_bitstring()

    dt = time.perf_counter() - t0
    return {
        "cobyla_mode": str(cobyla_mode),
        "allocation_method": str(eff_allocation),
        "elimination_mode": str(eff_elimination),
        "coupling_mode": str(eff_coupling),
        "holm_step_down": bool(eff_holm),
        "pairwise_betting": bool(eff_pairwise),
        "statistical_guarantee_status": (
            "exploratory_heuristic" if (eff_holm or eff_pairwise or eff_coupling in ("full", "legacy_simplex_offset"))
            else "guaranteed_simultaneous_fwer"
        ),
        "initial_theta": theta_0.copy(),
        "selected_candidate_id": int(selected_id),
        "ranked_candidate_ids": [int(cid) for cid in ranked_ids],
        "frozen_theta": frozen_theta,
        "selected_origin": str(selected_arm.origin),
        "selected_empirical_mean": float(selected_arm.empirical_mean),
        "selected_empirical_variance": float(selected_arm.empirical_variance),
        "selected_lower_bound": float(selected_arm.sim_lower_bound),
        "selected_upper_bound": float(selected_arm.sim_upper_bound),
        "selected_marginal_lower_bound": float(selected_arm.marginal_lower_bound),
        "selected_marginal_upper_bound": float(selected_arm.marginal_upper_bound),
        "selected_sample_count": int(selected_arm.sample_count),
        "alpha_discovery": float(alpha_discovery),
        "alpha_per_arm": float(arms[0].alpha_simultaneous) if arms else float(alpha_discovery / max(1, eff_max_candidates)),
        "total_alpha_allocated": float(alpha_scheduler.total_alpha_allocated),
        "cs_method": str(eff_cs_method),
        "rhobeg": float(eff_rhobeg),
        "tol": float(eff_tol),
        "jitter_half_width": float(jitter_half_width),
        "max_candidates": int(eff_max_candidates),
        "simplex_anchor_shots": int(eff_simplex_shots),
        "best_sampled_bitstring": str(selected_arm.best_bitstring_observed),
        "best_sampled_loss": float(selected_arm.min_loss_observed),
        "most_probable_sampled_bitstring": str(mp_bstr),
        "most_probable_sampled_loss": float(mp_loss),
        "pilot_shots_total": int(pilot_shots_total),
        "adaptive_discovery_shots": int(cumulative_shots - pilot_shots_total),
        "total_discovery_shots": int(cumulative_shots),
        "stopping_reason": str(stopping_reason),
        "num_candidates_evaluated": int(len(arms)),
        "num_candidates_eliminated": int(len(eliminated_arms)),
        "num_best_candidate_eliminated": int(sum(1 for a in eliminated_arms if a.elimination_reason == "best_candidate_eliminated")),
        "num_target_rejected_in_discovery": int(sum(1 for a in eliminated_arms if a.elimination_reason == "target_rejected")),
        "pct_candidates_eliminated": float(100.0 * len(eliminated_arms) / float(max(1, len(arms)))),
        "mean_shots_before_elimination": float(np.mean([a.sample_count for a in eliminated_arms])) if eliminated_arms else 0.0,
        "shots_saved_estimate": int(shots_saved),
        "classical_runtime_seconds": float(dt),
        "arms": arms,
        "elimination_log": elimination_log,
        "allocation_history": allocation_history,
    }
