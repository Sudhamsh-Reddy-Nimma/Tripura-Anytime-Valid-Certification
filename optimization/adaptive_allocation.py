"""
Phase A & Phase B Fixed-Pool Adaptive Shot Allocation, Candidate Elimination,
Discovery Stopping, and Candidate Freezing for AMFS 2.0 (Stage 3 Corrected & SOTA Upgraded).

Strictly enforces:
1. Simultaneous family-wise alpha allocation (alpha_arm = alpha_discovery / K or pre-allocated K_max schedule)
   for all multi-candidate decision rules (elimination, UGapE index, leader/challenger selection, early stopping).
2. Optional Sequential Holm Step-Down Alpha Recycling (Holm 1979; Kaufmann & Kalyanakrishnan 2013):
   when m_t = |A_t| <= K candidates remain active after permanent eliminations, active arms may be evaluated
   at alpha_arm(t) = alpha_discovery / |A_t| while strictly preserving FWER <= alpha_discovery.
3. Optional Pairwise Difference Martingale Elimination (Waudby-Smith & Ramdas 2024 Theorem 1):
   tests rescaled paired differences Y_{i, l_t, s} = 0.5 + 0.5*(X_{i,s} - X_{l_t,s}) in [0, 1] against 0.5 + eps/2.
4. Separate marginal confidence sequences (alpha_marginal = alpha_discovery) retained purely for single-arm diagnostics.
5. Pure confidence-sequence elimination with zero heuristic z-score fallbacks.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from simulator.observables import index_to_bitstring
from certification.confidence_sequences import (
    TimeUniformConfidenceSequence,
    confidence_sequence_radius,
    wsr_log_wealth_plus,
    wsr_log_wealth_minus,
)


ALL_ALLOCATION_METHODS = [
    "uniform",
    "variance_only",
    "gap_only",
    "variance_plus_gap",
    "ugape_eb",
    "top_two_challenger",
    "top_three",
]

ALL_ELIMINATION_MODES = [
    "none",
    "best_candidate_only",
    "target_rejection_only",
    "combined",
]


class SimultaneousAlphaSchedule:
    """
    Pre-allocated simultaneous error budget manager for K or K_max candidates.
    Guarantees sum_{i=1}^K alpha_i <= alpha_discovery at all times and blocks
    unscheduled dynamic candidate additions beyond K_max.
    """

    def __init__(
        self,
        alpha_discovery: float,
        max_candidates: int,
        custom_schedule: Optional[List[float]] = None,
    ):
        if not (0.0 < float(alpha_discovery) < 1.0):
            raise ValueError(f"alpha_discovery must be in (0, 1), got {alpha_discovery}.")
        if int(max_candidates) < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}.")

        self.alpha_discovery = float(alpha_discovery)
        self.max_candidates = int(max_candidates)

        if custom_schedule is not None:
            if len(custom_schedule) != self.max_candidates:
                raise ValueError(
                    f"custom_schedule length ({len(custom_schedule)}) must match max_candidates ({self.max_candidates})."
                )
            if any(float(a) <= 0.0 for a in custom_schedule):
                raise ValueError("All per-candidate alpha values in custom_schedule must be strictly positive.")
            total_sched = float(sum(float(a) for a in custom_schedule))
            if total_sched > self.alpha_discovery + 1e-12:
                raise ValueError(
                    f"custom_schedule sum ({total_sched:.8f}) exceeds alpha_discovery ({self.alpha_discovery:.8f})."
                )
            self._schedule = [float(a) for a in custom_schedule]
        else:
            alpha_each = self.alpha_discovery / float(self.max_candidates)
            self._schedule = [alpha_each for _ in range(self.max_candidates)]

        self._allocated_count = 0
        self._allocated_sum = 0.0

    def allocate_next_arm_alpha(self) -> float:
        if self._allocated_count >= self.max_candidates:
            raise ValueError(
                f"Cannot add candidate {self._allocated_count}: exceeds pre-allocated max_candidates={self.max_candidates} "
                f"without a valid pre-established alpha-spending schedule."
            )
        alpha_i = float(self._schedule[self._allocated_count])
        if self._allocated_sum + alpha_i > self.alpha_discovery + 1e-12:
            raise ValueError(
                f"Candidate alpha allocation would exceed alpha_discovery ({self._allocated_sum + alpha_i} > {self.alpha_discovery})."
            )
        self._allocated_count += 1
        self._allocated_sum += alpha_i
        return alpha_i

    @property
    def allocated_count(self) -> int:
        return int(self._allocated_count)

    @property
    def total_alpha_allocated(self) -> float:
        return float(self._allocated_sum)


class CandidateArmState:
    """
    Maintains strictly finite-shot observable statistics and both marginal and simultaneous
    time-uniform confidence sequences for a single QAOA parameter candidate.
    """

    def __init__(
        self,
        candidate_id: int,
        theta: np.ndarray,
        origin: str,
        alpha_arm: float,
        alpha_simultaneous: Optional[float] = None,
        alpha_marginal: Optional[float] = None,
        cs_method: str = "howard_gamma_exponential_eb",
        m_intrinsic: float = 5.0,
        initial_prediction: float = 0.26,
        n_qubits: int = 12,
    ):
        self.candidate_id = int(candidate_id)
        self.theta = np.asarray(theta, dtype=np.float64).copy()
        self.theta.setflags(write=False)
        self.origin = str(origin)
        self.n_qubits = int(n_qubits)
        self.cs_method = str(cs_method)
        self.m_intrinsic = float(m_intrinsic)
        self.initial_prediction = float(initial_prediction)

        if alpha_simultaneous is not None:
            self.alpha_simultaneous = float(alpha_simultaneous)
            self.alpha_marginal = float(alpha_marginal) if alpha_marginal is not None else float(max(alpha_arm, alpha_simultaneous))
        else:
            self.alpha_simultaneous = float(alpha_arm)
            self.alpha_marginal = float(alpha_marginal) if alpha_marginal is not None else float(alpha_arm)

        self.alpha_arm = float(self.alpha_simultaneous)

        if not (0.0 < self.alpha_simultaneous <= self.alpha_marginal < 1.0):
            raise ValueError(
                f"Invalid alpha levels: require 0 < alpha_simultaneous ({self.alpha_simultaneous}) "
                f"<= alpha_marginal ({self.alpha_marginal}) < 1."
            )

        self.cs_marginal = TimeUniformConfidenceSequence(
            alpha=float(self.alpha_marginal),
            method=self.cs_method,
            m_intrinsic=self.m_intrinsic,
            initial_prediction=initial_prediction,
            running_intersection=True,
        )
        self.cs_simultaneous = TimeUniformConfidenceSequence(
            alpha=float(self.alpha_simultaneous),
            method=self.cs_method,
            m_intrinsic=self.m_intrinsic,
            initial_prediction=initial_prediction,
            running_intersection=True,
        )

        self.pilot_shots: int = 0
        self.adaptive_shots: int = 0
        self.eliminated: bool = False
        self.elimination_reason: str = "none"
        self.elimination_step: Optional[int] = None
        self.shots_at_elimination: Optional[int] = None
        self.min_loss_observed: float = 1.0
        self.best_basis_idx: int = 0
        self.basis_counts: Dict[int, Tuple[int, float]] = {}

    @property
    def observations(self) -> np.ndarray:
        return self.cs_simultaneous.observations

    @property
    def sample_count(self) -> int:
        return int(self.cs_simultaneous.t)

    @property
    def empirical_mean(self) -> float:
        return float(self.cs_simultaneous.mean)

    @property
    def empirical_variance(self) -> float:
        return float(self.cs_simultaneous.sample_variance)

    @property
    def sim_lower_bound(self) -> float:
        return float(self.cs_simultaneous.L_t)

    @property
    def sim_upper_bound(self) -> float:
        return float(self.cs_simultaneous.U_t)

    @property
    def sim_radius(self) -> float:
        return float(0.5 * (self.cs_simultaneous.U_t - self.cs_simultaneous.L_t))

    @property
    def lower_bound(self) -> float:
        return self.sim_lower_bound

    @property
    def upper_bound(self) -> float:
        return self.sim_upper_bound

    @property
    def radius(self) -> float:
        return self.sim_radius

    @property
    def marginal_lower_bound(self) -> float:
        return float(self.cs_marginal.L_t)

    @property
    def marginal_upper_bound(self) -> float:
        return float(self.cs_marginal.U_t)

    @property
    def marginal_radius(self) -> float:
        return float(0.5 * (self.cs_marginal.U_t - self.cs_marginal.L_t))

    @property
    def hoeffding_radius(self) -> float:
        if self.sample_count < 1:
            return 1.0
        return confidence_sequence_radius(
            t=self.sample_count,
            v_hat_t=0.25 * self.sample_count,
            alpha=self.alpha_simultaneous,
            method="howard_normal_mixture_hoeffding",
            two_sided=True,
        )

    @property
    def hoeffding_lower_bound(self) -> float:
        return float(np.clip(self.empirical_mean - self.hoeffding_radius, 0.0, 1.0))

    @property
    def hoeffding_upper_bound(self) -> float:
        return float(np.clip(self.empirical_mean + self.hoeffding_radius, 0.0, 1.0))

    @property
    def best_bitstring_observed(self) -> str:
        return index_to_bitstring(int(self.best_basis_idx), self.n_qubits)

    def add_observations(
        self,
        shot_losses: np.ndarray,
        is_pilot: bool = False,
        basis_indices: Optional[np.ndarray] = None,
    ) -> None:
        arr = np.asarray(shot_losses, dtype=np.float64)
        n_shots = int(arr.size)
        if n_shots == 0:
            return
        self.cs_marginal.update_batch(arr)
        self.cs_simultaneous.update_batch(arr)

        if is_pilot:
            self.pilot_shots += n_shots
        else:
            self.adaptive_shots += n_shots

        min_pos = int(np.argmin(arr))
        if float(arr[min_pos]) < self.min_loss_observed:
            self.min_loss_observed = float(arr[min_pos])
            if basis_indices is not None and basis_indices.size == n_shots:
                self.best_basis_idx = int(basis_indices[min_pos])

        if basis_indices is not None and basis_indices.size == n_shots:
            u_idx, u_cnt = np.unique(basis_indices, return_counts=True)
            for b_idx, cnt in zip(u_idx, u_cnt):
                b_int = int(b_idx)
                prev_cnt, prev_loss = self.basis_counts.get(b_int, (0, float(arr[basis_indices == b_idx][0])))
                self.basis_counts[b_int] = (prev_cnt + int(cnt), prev_loss)

    def most_probable_sampled_bitstring(self) -> Tuple[str, float]:
        if not self.basis_counts:
            return self.best_bitstring_observed, self.min_loss_observed
        best_kv = max(self.basis_counts.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
        return index_to_bitstring(int(best_kv[0]), self.n_qubits), float(best_kv[1][1])


def compute_ugape_indices(
    arms: List[CandidateArmState],
    active_ids: List[int],
    use_hoeffding: bool = False,
) -> Tuple[Dict[int, float], int, Optional[int]]:
    if len(active_ids) == 0:
        raise ValueError("active_ids cannot be empty.")
    arm_map = {a.candidate_id: a for a in arms}
    if len(active_ids) == 1:
        only_id = active_ids[0]
        return {only_id: 0.0}, only_id, None

    L_vals = {}
    U_vals = {}
    rad_vals = {}
    for cid in active_ids:
        a = arm_map[cid]
        if use_hoeffding:
            L_vals[cid] = a.hoeffding_lower_bound
            U_vals[cid] = a.hoeffding_upper_bound
            rad_vals[cid] = a.hoeffding_radius
        else:
            L_vals[cid] = a.sim_lower_bound
            U_vals[cid] = a.sim_upper_bound
            rad_vals[cid] = a.sim_radius

    B_indices: Dict[int, float] = {}
    for cid in active_ids:
        min_other_L = min(L_vals[jid] for jid in active_ids if jid != cid)
        B_indices[cid] = float(max(0.0, U_vals[cid] - min_other_L))

    l_t = min(active_ids, key=lambda cid: (B_indices[cid], U_vals[cid], -rad_vals[cid], cid))
    other_ids = [cid for cid in active_ids if cid != l_t]
    u_t = min(other_ids, key=lambda cid: (L_vals[cid], -rad_vals[cid], cid))
    return B_indices, int(l_t), int(u_t)


def select_next_candidate_to_sample(
    arms: List[CandidateArmState],
    active_ids: List[int],
    allocation_method: str,
    step_index: int,
    epsilon: float = 0.008,
) -> int:
    if len(active_ids) == 0:
        raise ValueError("No active candidates to sample.")
    if len(active_ids) == 1:
        return int(active_ids[0])

    arm_map = {a.candidate_id: a for a in arms}

    if allocation_method == "uniform":
        return int(min(active_ids, key=lambda cid: (arm_map[cid].sample_count, cid)))

    elif allocation_method == "variance_only":
        v_floor = 1e-4
        return int(
            max(
                active_ids,
                key=lambda cid: (
                    np.sqrt((arm_map[cid].empirical_variance + v_floor) / float(max(1, arm_map[cid].sample_count))),
                    -cid,
                ),
            )
        )

    elif allocation_method == "gap_only":
        _, l_t, u_t = compute_ugape_indices(arms, active_ids, use_hoeffding=True)
        if u_t is None:
            return int(l_t)
        if arm_map[u_t].hoeffding_radius >= arm_map[l_t].hoeffding_radius:
            return int(u_t)
        return int(l_t)

    elif allocation_method == "variance_plus_gap":
        v_floor = 1e-4
        means = {cid: arm_map[cid].empirical_mean for cid in active_ids}
        leader_id = min(active_ids, key=lambda cid: (means[cid], cid))
        other_ids = [cid for cid in active_ids if cid != leader_id]
        runner_up_mean = min(means[cid] for cid in other_ids)

        best_cid = leader_id
        best_score = -1.0
        for cid in active_ids:
            a = arm_map[cid]
            if cid == leader_id:
                gap = max(float(epsilon), float(runner_up_mean - means[leader_id]))
            else:
                gap = max(float(epsilon), float(means[cid] - means[leader_id]))
            se_i = np.sqrt((a.empirical_variance + v_floor) / float(max(1, a.sample_count)))
            score = float(se_i / gap)
            if score > best_score:
                best_score = score
                best_cid = cid
        return int(best_cid)

    elif allocation_method == "ugape_eb":
        _, l_t, u_t = compute_ugape_indices(arms, active_ids, use_hoeffding=False)
        if u_t is None:
            return int(l_t)
        if arm_map[u_t].sim_radius >= arm_map[l_t].sim_radius:
            return int(u_t)
        return int(l_t)

    elif allocation_method == "top_two_challenger":
        v_floor = 1e-4
        leader_id = min(active_ids, key=lambda cid: (arm_map[cid].empirical_mean, arm_map[cid].sim_upper_bound, cid))
        lead_arm = arm_map[leader_id]
        se_lead2 = (lead_arm.empirical_variance + v_floor) / float(max(1, lead_arm.sample_count))

        best_challenger = None
        min_z = np.inf
        for cid in active_ids:
            if cid == leader_id:
                continue
            a = arm_map[cid]
            se_c2 = (a.empirical_variance + v_floor) / float(max(1, a.sample_count))
            z_stat = (a.empirical_mean - lead_arm.empirical_mean + float(epsilon)) / np.sqrt(se_lead2 + se_c2)
            if z_stat < min_z:
                min_z = z_stat
                best_challenger = cid

        if best_challenger is None:
            return int(leader_id)
        chal_arm = arm_map[best_challenger]
        sd_lead = np.sqrt(lead_arm.empirical_variance + v_floor)
        sd_chal = np.sqrt(chal_arm.empirical_variance + v_floor)
        target_frac_lead = sd_lead / (sd_lead + sd_chal)
        actual_frac_lead = lead_arm.sample_count / float(lead_arm.sample_count + chal_arm.sample_count)
        return int(leader_id if actual_frac_lead < target_frac_lead else best_challenger)

    elif allocation_method == "top_three":
        _, l_t, _ = compute_ugape_indices(arms, active_ids, use_hoeffding=False)
        other_ids = sorted(
            [cid for cid in active_ids if cid != l_t],
            key=lambda cid: (arm_map[cid].sim_lower_bound, -arm_map[cid].sim_radius, cid),
        )
        contenders = [l_t] + other_ids[:2]
        return int(max(contenders, key=lambda cid: (arm_map[cid].sim_radius, -arm_map[cid].sample_count, -cid)))

    else:
        raise ValueError(f"Unknown allocation method '{allocation_method}'.")


def apply_candidate_elimination(
    arms: List[CandidateArmState],
    active_ids: List[int],
    elimination_mode: str,
    target_threshold: float,
    step_index: int,
    epsilon_best: float = 0.0,
    enable_holm_step_down: bool = False,
    enable_pairwise_betting: bool = False,
    alpha_discovery: float = 0.05,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Strictly applies simultaneous confidence-sequence elimination rules:
      1. Target-rejection elimination: L_i^{sim}(t) > tau
      2. Best-candidate elimination:   L_i^{sim}(t) > min_{j in A_t} U_j^{sim}(t) + epsilon_best
         OR (when enable_pairwise_betting=True) exact Waudby-Smith & Ramdas (2024) Theorem 1
         betting martingale on the [0,1]-bounded paired difference Y_{i, l_t, s} = 0.5 + 0.5*(X_{i,s} - X_{l_t,s})
         exceeding 0.5 + epsilon_best / 2 at simultaneous Holm step-down level alpha_arm(t) = alpha_discovery / |A_t|.
    Zero heuristic z-score or paired-SE rules are permitted.
    """
    if elimination_mode == "none" or len(active_ids) == 0:
        return list(active_ids), []

    arm_map = {a.candidate_id: a for a in arms}
    tau = float(target_threshold)

    # Sequential Holm Step-Down Loop (Holm 1979): if an arm is eliminated, the active pool shrinks
    # and the simultaneous alpha per surviving arm can be recycled under closed testing.
    curr_active = list(active_ids)
    eliminated_events = []

    while len(curr_active) > 0:
        m_act = len(curr_active)
        alpha_holm = float(alpha_discovery) / float(m_act) if enable_holm_step_down else None
        thresh_bet = np.log(1.0 / max(1e-12, 0.5 * (alpha_holm if alpha_holm is not None else arm_map[curr_active[0]].alpha_simultaneous)))

        min_U_sim = min(arm_map[cid].sim_upper_bound for cid in curr_active)
        min_U_holder = min(curr_active, key=lambda cid: (arm_map[cid].sim_upper_bound, arm_map[cid].empirical_mean, cid))
        emp_leader_id = min(curr_active, key=lambda cid: (arm_map[cid].empirical_mean, arm_map[cid].sim_upper_bound, cid))
        emp_leader = arm_map[emp_leader_id]

        newly_eliminated = []
        next_active = []

        for cid in curr_active:
            a = arm_map[cid]
            reason = "none"

            # Check simultaneous bounds (with optional Holm step-down alpha recycling)
            if enable_holm_step_down and alpha_holm is not None and alpha_holm > a.alpha_simultaneous + 1e-12:
                sim_L_now, sim_U_now = a.cs_simultaneous.recompute_for_alpha(alpha_holm)
                sim_L_now = max(a.sim_lower_bound, sim_L_now)
            else:
                sim_L_now, sim_U_now = a.sim_lower_bound, a.sim_upper_bound

            if elimination_mode in ("target_rejection_only", "combined"):
                if sim_L_now > tau:
                    reason = "target_rejected"
                elif enable_pairwise_betting and a.cs_method == "wsr_agrapa_betting":
                    if wsr_log_wealth_plus(a.observations, tau, initial_prediction=tau, prior_variance=0.006) >= thresh_bet:
                        reason = "target_rejected"

            if reason == "none" and elimination_mode in ("best_candidate_only", "combined"):
                if cid != min_U_holder and sim_L_now > (min_U_sim + float(epsilon_best)):
                    reason = "best_candidate_eliminated"
                elif enable_pairwise_betting and cid != emp_leader_id:
                    t_min = min(a.sample_count, emp_leader.sample_count)
                    if t_min >= 64:
                        obs_i = a.observations[:t_min]
                        obs_l = emp_leader.observations[:t_min]
                        y_pair = 0.5 + 0.5 * (obs_i - obs_l)
                        null_m = 0.5 + 0.5 * float(epsilon_best)
                        if wsr_log_wealth_plus(y_pair, null_m, initial_prediction=0.5, prior_variance=0.002) >= thresh_bet:
                            reason = "best_candidate_eliminated"

            if reason != "none":
                a.eliminated = True
                a.elimination_reason = reason
                a.elimination_step = int(step_index)
                a.shots_at_elimination = int(a.sample_count)
                ev = {
                    "candidate_id": int(cid),
                    "reason": reason,
                    "step": int(step_index),
                    "shots_at_elimination": int(a.sample_count),
                    "sim_lower_bound": float(sim_L_now),
                    "sim_upper_bound": float(sim_U_now),
                    "marginal_lower_bound": float(a.marginal_lower_bound),
                    "marginal_upper_bound": float(a.marginal_upper_bound),
                    "min_sim_upper_bound": float(min_U_sim),
                    "target_threshold": float(tau),
                    "alpha_arm": float(alpha_holm if alpha_holm is not None else a.alpha_simultaneous),
                    "statistical_guarantee_status": (
                        "exploratory_heuristic" if (enable_holm_step_down or enable_pairwise_betting)
                        else "guaranteed_simultaneous_fwer"
                    ),
                }
                eliminated_events.append(ev)
                newly_eliminated.append(cid)
            else:
                next_active.append(cid)

        curr_active = next_active
        if not enable_holm_step_down or len(newly_eliminated) == 0:
            break

    return curr_active, eliminated_events


def select_final_candidate_from_pool(
    arms: List[CandidateArmState],
    active_ids: List[int],
    allocation_method: str,
) -> int:
    pool = active_ids if len(active_ids) > 0 else [a.candidate_id for a in arms if a.sample_count > 0]
    if len(pool) == 0:
        return 0
    if len(pool) == 1:
        return int(pool[0])

    arm_map = {a.candidate_id: a for a in arms}
    if allocation_method in ("ugape_eb", "gap_only"):
        _, l_t, _ = compute_ugape_indices(
            arms, pool, use_hoeffding=(allocation_method == "gap_only")
        )
        return int(l_t)

    max_shots = max(arm_map[cid].sample_count for cid in pool)
    min_viable_shots = max(1, max_shots // 3)
    viable_pool = [cid for cid in pool if arm_map[cid].sample_count >= min_viable_shots]
    if not viable_pool:
        viable_pool = pool
    return int(
        min(
            viable_pool,
            key=lambda cid: (arm_map[cid].empirical_mean, arm_map[cid].sim_upper_bound, cid),
        )
    )


def run_fixed_pool_adaptive_discovery(
    candidate_thetas: List[np.ndarray],
    candidate_origins: List[str],
    sample_fn: Callable[[int, np.ndarray, int], Tuple[np.ndarray, Optional[np.ndarray]]],
    total_discovery_budget: int,
    target_threshold: float,
    pilot_shots: int = 64,
    batch_shots: int = 32,
    allocation_method: str = "ugape_eb",
    elimination_mode: str = "combined",
    epsilon: float = 0.008,
    alpha_discovery: float = 0.05,
    cs_method: str = "howard_gamma_exponential_eb",
    m_intrinsic: float = 5.0,
    enable_early_stopping: bool = True,
    enable_target_promising_stop: bool = False,
    enable_holm_step_down: bool = False,
    enable_pairwise_betting: bool = False,
    n_qubits: int = 12,
    custom_alpha_schedule: Optional[List[float]] = None,
) -> Dict[str, Any]:
    K = len(candidate_thetas)
    if K == 0:
        raise ValueError("Candidate pool must contain at least 1 candidate.")
    if allocation_method not in ALL_ALLOCATION_METHODS:
        raise ValueError(f"Unsupported allocation method '{allocation_method}'.")
    if elimination_mode not in ALL_ELIMINATION_MODES:
        raise ValueError(f"Unsupported elimination mode '{elimination_mode}'.")

    alpha_scheduler = SimultaneousAlphaSchedule(
        alpha_discovery=float(alpha_discovery),
        max_candidates=K,
        custom_schedule=custom_alpha_schedule,
    )

    arms: List[CandidateArmState] = []
    for i in range(K):
        alpha_i = alpha_scheduler.allocate_next_arm_alpha()
        arms.append(
            CandidateArmState(
                candidate_id=i,
                theta=candidate_thetas[i],
                origin=candidate_origins[i] if i < len(candidate_origins) else "pool",
                alpha_arm=alpha_i,
                alpha_simultaneous=alpha_i,
                alpha_marginal=float(alpha_discovery),
                cs_method=cs_method,
                m_intrinsic=float(m_intrinsic),
                initial_prediction=float(target_threshold),
                n_qubits=n_qubits,
            )
        )

    cumulative_shots = 0
    pilot_expenditure = 0
    elimination_log: List[Dict[str, Any]] = []
    allocation_history: List[Dict[str, Any]] = []

    b_pilot = int(pilot_shots)
    for arm in arms:
        if cumulative_shots + b_pilot > total_discovery_budget:
            rem = max(0, total_discovery_budget - cumulative_shots)
            if rem < 2:
                break
            b_use = rem
        else:
            b_use = b_pilot
        losses, b_indices = sample_fn(arm.candidate_id, arm.theta, b_use)
        arm.add_observations(losses, is_pilot=True, basis_indices=b_indices)
        cumulative_shots += int(losses.size)
        pilot_expenditure += int(losses.size)

    active_ids = [arm.candidate_id for arm in arms if arm.sample_count > 0]
    step_idx = 0

    active_ids, elim_ev = apply_candidate_elimination(
        arms=arms,
        active_ids=active_ids,
        elimination_mode=elimination_mode,
        target_threshold=target_threshold,
        step_index=step_idx,
        epsilon_best=0.0,
        enable_holm_step_down=enable_holm_step_down,
        enable_pairwise_betting=enable_pairwise_betting,
        alpha_discovery=alpha_discovery,
    )
    elimination_log.extend(elim_ev)

    stopping_reason = "budget_exhausted"

    while cumulative_shots < total_discovery_budget:
        if len(active_ids) == 0:
            stopping_reason = "all_candidates_rejected"
            break

        B_indices, l_t, u_t = compute_ugape_indices(
            arms, active_ids, use_hoeffding=(allocation_method == "gap_only")
        )
        leader_arm = arms[l_t]

        if enable_early_stopping and B_indices[l_t] <= float(epsilon):
            if len(active_ids) > 1 or leader_arm.sim_radius <= float(epsilon) or leader_arm.sim_upper_bound <= float(target_threshold):
                stopping_reason = "best_candidate_identified"
                break

        if enable_early_stopping and enable_target_promising_stop:
            if leader_arm.sim_upper_bound <= float(target_threshold):
                stopping_reason = "target_feasible_promising"
                break
            if cs_method == "wsr_agrapa_betting" and leader_arm.sample_count >= 192:
                alpha_lead = float(alpha_discovery) / float(len(active_ids)) if enable_holm_step_down else leader_arm.alpha_simultaneous
                if wsr_log_wealth_minus(leader_arm.observations, target_threshold, initial_prediction=target_threshold) >= np.log(1.0 / max(1e-12, 0.5 * alpha_lead)):
                    stopping_reason = "target_feasible_promising"
                    break

        step_idx += 1
        chosen_id = select_next_candidate_to_sample(
            arms=arms,
            active_ids=active_ids,
            allocation_method=allocation_method,
            step_index=step_idx,
            epsilon=epsilon,
        )
        b_alloc = min(int(batch_shots), int(total_discovery_budget - cumulative_shots))
        if b_alloc < 1:
            stopping_reason = "budget_exhausted"
            break

        chosen_arm = arms[chosen_id]
        losses, b_indices = sample_fn(chosen_id, chosen_arm.theta, b_alloc)
        chosen_arm.add_observations(losses, is_pilot=False, basis_indices=b_indices)
        cumulative_shots += int(losses.size)

        allocation_history.append({
            "step": int(step_idx),
            "chosen_candidate_id": int(chosen_id),
            "batch_shots": int(losses.size),
            "cumulative_shots": int(cumulative_shots),
            "active_count": int(len(active_ids)),
            "leader_id": int(l_t),
            "leader_B_sim_index": float(B_indices[l_t]),
        })

        active_ids, elim_ev = apply_candidate_elimination(
            arms=arms,
            active_ids=active_ids,
            elimination_mode=elimination_mode,
            target_threshold=target_threshold,
            step_index=step_idx,
            epsilon_best=0.0,
            enable_holm_step_down=enable_holm_step_down,
            enable_pairwise_betting=enable_pairwise_betting,
            alpha_discovery=alpha_discovery,
        )
        elimination_log.extend(elim_ev)

    selected_id = select_final_candidate_from_pool(arms, active_ids, allocation_method)
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
    max_arm_shots = max((a.sample_count for a in arms), default=pilot_shots)
    shots_saved_vs_max = sum(max(0, max_arm_shots - a.sample_count) for a in eliminated_arms)

    return {
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
        "alpha_per_arm": float(arms[0].alpha_simultaneous),
        "total_alpha_allocated": float(alpha_scheduler.total_alpha_allocated),
        "cs_method": str(cs_method),
        "statistical_guarantee_status": (
            "exploratory_heuristic" if (enable_holm_step_down or enable_pairwise_betting)
            else "guaranteed_simultaneous_fwer"
        ),
        "pilot_shots_total": int(pilot_expenditure),
        "adaptive_discovery_shots": int(cumulative_shots - pilot_expenditure),
        "total_discovery_shots": int(cumulative_shots),
        "stopping_reason": str(stopping_reason),
        "num_candidates_total": int(K),
        "num_candidates_eliminated": int(len(eliminated_arms)),
        "num_best_candidate_eliminated": int(sum(1 for a in eliminated_arms if a.elimination_reason == "best_candidate_eliminated")),
        "num_target_rejected_in_discovery": int(sum(1 for a in eliminated_arms if a.elimination_reason == "target_rejected")),
        "pct_candidates_eliminated": float(100.0 * len(eliminated_arms) / float(max(1, K))),
        "mean_shots_before_elimination": float(np.mean([a.sample_count for a in eliminated_arms])) if eliminated_arms else 0.0,
        "shots_saved_estimate": int(shots_saved_vs_max),
        "arms": arms,
        "elimination_log": elimination_log,
        "allocation_history": allocation_history,
    }
