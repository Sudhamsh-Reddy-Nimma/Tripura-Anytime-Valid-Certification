"""
Time-Uniform Nonparametric Nonasymptotic Confidence Sequences and Sequential Confirmation
for AMFS 2.0 (Stage 3 Corrected & State-of-the-Art Upgraded).

Primary Theoretical Foundations:
1. Waudby-Smith and Ramdas (2024), "Estimating means of bounded random variables by betting",
   Journal of the Royal Statistical Society Series B, 86(1), 1-27 (arXiv:2010.09686).
   - Theorem 1 & Section 3.2 Proposition 3 (aGRAPA Hedged-Capital Betting Confidence Sequence).
   - Section 2.4 One-Sided & Asymmetric Error Spending (alpha_upper + alpha_lower = alpha).
2. Howard, Ramdas, McAuliffe, and Sekhon (2021), "Time-Uniform, Nonparametric, Nonasymptotic
   Confidence Sequences", Annals of Statistics 49(2), 1055-1080 (DOI: 10.1214/20-AOS1991).
   - Theorem 4 + Proposition 9 (Gamma-Exponential Empirical-Bernstein CS & Stitched EB CS).

STRICT GROUND-TRUTH ISOLATION:
This module NEVER imports or accesses exact statevector expectations, exact variances, or MaxCut optima.
"""

from typing import Dict, Any, List, Tuple, Optional, Sequence
import numpy as np
from scipy.special import gammaln, gammainc, zeta, lambertw
from scipy.optimize import brentq


_GE_CACHE: Dict[Tuple[int, float, float, float], float] = {}


def compute_predictable_variance_process(observations: np.ndarray, initial_prediction: float = 0.25) -> Tuple[float, float, float]:
    arr = np.asarray(observations, dtype=np.float64)
    t = arr.size
    if t < 1:
        raise ValueError("At least 1 observation is required.")
    mean_t = float(np.mean(arr))
    var_t = float(np.var(arr, ddof=1)) if t >= 2 else 0.0
    var_t = max(0.0, var_t)

    if t == 1:
        v_hat_t = float((arr[0] - float(np.clip(initial_prediction, 0.0, 1.0))) ** 2)
    else:
        csum = np.cumsum(arr[:-1])
        counts = np.arange(1, t, dtype=np.float64)
        preds = np.empty(t, dtype=np.float64)
        preds[0] = float(np.clip(initial_prediction, 0.0, 1.0))
        preds[1:] = csum / counts
        v_hat_t = float(np.sum((arr - preds) ** 2))

    return mean_t, var_t, max(0.0, v_hat_t)


def tune_rho_lambert_w(m: float, alpha_two_sided: float, l0: float = 1.0) -> float:
    m_clean = max(1e-4, float(m))
    alpha_clean = float(np.clip(alpha_two_sided, 1e-12, 0.999))
    arg = -(alpha_clean ** 2) / (np.e * (l0 ** 2))
    w_val = float(np.real(lambertw(arg, k=-1)))
    denom = max(1e-6, -w_val - 1.0)
    return float(m_clean / denom)


def howard_stitched_eb_boundary(
    v_hat: float,
    alpha_tail: float,
    m: float = 1.0,
    eta: float = 2.0,
    s: float = 1.4,
    c: float = 1.0,
    l0: float = 1.0,
) -> float:
    if alpha_tail <= 0.0 or alpha_tail >= 1.0:
        raise ValueError(f"alpha_tail must be in (0, 1), got {alpha_tail}.")
    v_eff = max(float(v_hat), float(m), 1e-9)
    m_eff = max(float(m), 1e-9)
    k1 = (eta ** 0.25 + eta ** (-0.25)) / np.sqrt(2.0)
    k2 = (np.sqrt(eta) + 1.0) / 2.0
    log_eta_ratio = max(0.0, np.log(v_eff / m_eff) / np.log(eta))
    ell = s * np.log(log_eta_ratio + 1.0) + np.log((l0 * float(zeta(s))) / float(alpha_tail))
    s_bound = np.sqrt((k1 ** 2) * v_eff * ell + (k2 ** 2) * (c ** 2) * (ell ** 2)) + k2 * c * ell
    return float(s_bound)


def _log_gamma_exp_mixture(s_val: float, v: float, rho: float, c: float = 1.0) -> float:
    c2 = c * c
    a0 = rho / c2
    av = (v + rho) / c2
    x0 = a0
    xv = (c * s_val + v + rho) / c2
    if xv <= 0.0:
        return -np.inf
    inc0 = max(float(gammainc(a0, x0)), 1e-300)
    incv = max(float(gammainc(av, xv)), 1e-300)
    term1 = a0 * np.log(a0) - float(gammaln(a0)) - np.log(inc0)
    term2 = float(gammaln(av)) + np.log(incv) - av * np.log(xv)
    term3 = (c * s_val + v) / c2
    return float(term1 + term2 + term3)


def howard_gamma_exponential_eb_boundary(
    v_hat: float,
    alpha_tail: float,
    m: float = 5.0,
    c: float = 1.0,
    l0: float = 1.0,
) -> float:
    if alpha_tail <= 0.0 or alpha_tail >= 1.0:
        raise ValueError(f"alpha_tail must be in (0, 1), got {alpha_tail}.")
    v = max(0.0, float(v_hat))
    rho = tune_rho_lambert_w(m=float(m), alpha_two_sided=min(0.999, 2.0 * float(alpha_tail)), l0=l0)

    v_key = int(np.ceil(v * 200.0))
    cache_key = (v_key, round(float(alpha_tail), 8), round(rho, 6), round(c, 4))
    if cache_key in _GE_CACHE:
        return _GE_CACHE[cache_key]

    v_eval = v_key / 200.0
    target = np.log(float(l0) / float(alpha_tail))
    if _log_gamma_exp_mixture(0.0, v_eval, rho, c) >= target:
        return 0.0

    hi = max(0.5, howard_stitched_eb_boundary(v_eval, alpha_tail, m=max(0.5, m), c=c, l0=l0) * 1.25)
    while _log_gamma_exp_mixture(hi, v_eval, rho, c) < target and hi < 1e7:
        hi *= 2.0

    s_star = float(brentq(lambda s_val: _log_gamma_exp_mixture(s_val, v_eval, rho, c) - target, 0.0, hi, xtol=1e-6))
    if len(_GE_CACHE) < 100000:
        _GE_CACHE[cache_key] = s_star
    return s_star


def howard_normal_mixture_hoeffding_boundary(
    t: int,
    alpha_two_sided: float,
    m_shots: float = 256.0,
    l0: float = 1.0,
) -> float:
    if t < 1:
        raise ValueError(f"Sample size t must be >= 1, got {t}.")
    if alpha_two_sided <= 0.0 or alpha_two_sided >= 1.0:
        raise ValueError(f"alpha_two_sided must be in (0, 1), got {alpha_two_sided}.")
    v_t = 0.25 * float(t)
    m_v = 0.25 * max(1.0, float(m_shots))
    rho = tune_rho_lambert_w(m=m_v, alpha_two_sided=alpha_two_sided, l0=l0)
    u_val = np.sqrt((v_t + rho) * np.log((l0 ** 2) * (v_t + rho) / ((alpha_two_sided ** 2) * rho)))
    return float(u_val)


def howard_empirical_normal_mixture_boundary(
    t: int,
    v_hat_t: float,
    alpha_two_sided: float,
    m_intrinsic: float = 0.5,
    v_floor_per_shot: float = 0.004,
    l0: float = 1.0,
) -> float:
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}.")
    v_eff = max(float(v_hat_t), float(v_floor_per_shot) * float(t))
    rho = tune_rho_lambert_w(m=max(0.05, float(m_intrinsic)), alpha_two_sided=alpha_two_sided, l0=l0)
    u_val = np.sqrt((v_eff + rho) * np.log((l0 ** 2) * (v_eff + rho) / ((alpha_two_sided ** 2) * rho)))
    return float(u_val)


# ==============================================================================
# Waudby-Smith & Ramdas (2024, JRSSB, Theorem 1 & Section 3.2 Proposition 3)
# aGRAPA Hedged-Capital Betting Martingales & Confidence Sequences
# ==============================================================================
def wsr_predictable_estimators(
    observations: np.ndarray,
    initial_prediction: float = 0.26,
    prior_variance: float = 0.006,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes strictly F_{s-1}-measurable predictable mean mu_hat_{s-1} and
    predictable variance sigma_hat_{s-1}^2 for s = 1..t (Waudby-Smith & Ramdas 2024, Eq. 14 & 22).
    """
    x = np.asarray(observations, dtype=np.float64)
    t = x.size
    if t == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    init_m = float(np.clip(initial_prediction, 1e-4, 1.0 - 1e-4))
    p_var = max(1e-5, float(prior_variance))
    idx = np.arange(1, t + 1, dtype=np.float64)
    csum = np.cumsum(x)
    mu_hat = np.empty(t, dtype=np.float64)
    mu_hat[0] = init_m
    if t > 1:
        mu_hat[1:] = (init_m + csum[:-1]) / idx[:-1]
    sq_dev = (x - mu_hat) ** 2
    csum_v = np.cumsum(sq_dev)
    sig2_hat = np.empty(t, dtype=np.float64)
    sig2_hat[0] = p_var
    if t > 1:
        sig2_hat[1:] = (p_var + csum_v[:-1]) / idx[:-1]
    return mu_hat, sig2_hat


def wsr_log_wealth_plus(
    observations: np.ndarray,
    m_null: float,
    c: float = 0.75,
    initial_prediction: float = 0.26,
    prior_variance: float = 0.006,
    mu_hat: Optional[np.ndarray] = None,
    sig2_hat: Optional[np.ndarray] = None,
) -> float:
    """
    Log-wealth log K_t^+(m_null) = sum_{s=1}^t log(1 + lam_s^+(m_null) * (X_s - m_null))
    testing H_0: mu <= m_null against H_1: mu > m_null (lower confidence bound / target rejection).
    By Ville's inequality (1939) and Waudby-Smith & Ramdas (2024) Theorem 1,
    P(exists t >= 1: log K_t^+(mu) >= log(1/alpha_lower)) <= alpha_lower.
    """
    x = np.asarray(observations, dtype=np.float64)
    t = x.size
    m_val = float(np.clip(m_null, 1e-6, 1.0 - 1e-6))
    if t < 2 or float(np.mean(x)) <= m_val:
        return 0.0
    if mu_hat is None or sig2_hat is None:
        mu_hat, sig2_hat = wsr_predictable_estimators(x, initial_prediction=initial_prediction, prior_variance=prior_variance)
    diff = mu_hat - m_val
    lam = np.clip(np.maximum(0.0, diff / (sig2_hat + diff * diff)), 0.0, float(c) / m_val)
    return float(np.sum(np.log(1.0 + lam * (x - m_val))))


def wsr_log_wealth_minus(
    observations: np.ndarray,
    m_null: float,
    c: float = 0.75,
    initial_prediction: float = 0.26,
    prior_variance: float = 0.006,
    mu_hat: Optional[np.ndarray] = None,
    sig2_hat: Optional[np.ndarray] = None,
) -> float:
    """
    Log-wealth log K_t^-(m_null) = sum_{s=1}^t log(1 - lam_s^-(m_null) * (X_s - m_null))
    testing H_0: mu >= m_null against H_1: mu < m_null (upper confidence bound / target certification).
    By Ville's inequality (1939) and Waudby-Smith & Ramdas (2024) Theorem 1,
    P(exists t >= 1: log K_t^-(mu) >= log(1/alpha_upper)) <= alpha_upper.
    """
    x = np.asarray(observations, dtype=np.float64)
    t = x.size
    m_val = float(np.clip(m_null, 1e-6, 1.0 - 1e-6))
    if t < 2 or float(np.mean(x)) >= m_val:
        return 0.0
    if mu_hat is None or sig2_hat is None:
        mu_hat, sig2_hat = wsr_predictable_estimators(x, initial_prediction=initial_prediction, prior_variance=prior_variance)
    diff = m_val - mu_hat
    lam = np.clip(np.maximum(0.0, diff / (sig2_hat + diff * diff)), 0.0, float(c) / (1.0 - m_val))
    return float(np.sum(np.log(1.0 - lam * (x - m_val))))


def wsr_agrapa_betting_bounds(
    observations: np.ndarray,
    alpha_upper: float = 0.025,
    alpha_lower: float = 0.025,
    c: float = 0.75,
    initial_prediction: float = 0.26,
    prior_variance: float = 0.006,
    mu_hat: Optional[np.ndarray] = None,
    sig2_hat: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Computes the outer confidence interval [L_t, U_t] of the Waudby-Smith & Ramdas (2024)
    aGRAPA Hedged-Capital Betting Confidence Sequence via two-stage coarse-to-fine vectorized inversion.
    """
    x = np.asarray(observations, dtype=np.float64)
    if x.size < 2:
        return 0.0, 1.0
    if mu_hat is None or sig2_hat is None:
        mu_hat, sig2_hat = wsr_predictable_estimators(x, initial_prediction=initial_prediction, prior_variance=prior_variance)

    emp_mean = float(np.clip(np.mean(x), 1e-4, 1.0 - 1e-4))
    thresh_u = np.log(1.0 / max(1e-12, float(alpha_upper)))
    thresh_l = np.log(1.0 / max(1e-12, float(alpha_lower)))

    # 1. Upper bound U_t on [emp_mean, 0.999]
    m_coarse_u = np.linspace(emp_mean, 0.999, 17)
    diff_u = m_coarse_u[:, None] - mu_hat[None, :]
    lam_m = np.clip(np.maximum(0.0, diff_u / (sig2_hat[None, :] + diff_u * diff_u)), 0.0, float(c) / (1.0 - m_coarse_u[:, None]))
    log_km = np.sum(np.log(1.0 - lam_m * (x[None, :] - m_coarse_u[:, None])), axis=1)
    valid_idx_u = np.where(log_km < thresh_u)[0]
    if valid_idx_u.size == 0:
        U_t = emp_mean
    elif valid_idx_u[-1] == 16:
        U_t = 1.0
    else:
        k_hi = int(valid_idx_u[-1])
        m_fine_u = np.linspace(m_coarse_u[k_hi], m_coarse_u[k_hi + 1], 17)
        diff_uf = m_fine_u[:, None] - mu_hat[None, :]
        lam_mf = np.clip(np.maximum(0.0, diff_uf / (sig2_hat[None, :] + diff_uf * diff_uf)), 0.0, float(c) / (1.0 - m_fine_u[:, None]))
        log_kmf = np.sum(np.log(1.0 - lam_mf * (x[None, :] - m_fine_u[:, None])), axis=1)
        vf_u = np.where(log_kmf < thresh_u)[0]
        U_t = float(m_fine_u[vf_u[-1] + 1] if (vf_u.size > 0 and vf_u[-1] < 16) else m_fine_u[-1])

    # 2. Lower bound L_t on [0.001, emp_mean]
    m_coarse_l = np.linspace(0.001, emp_mean, 17)
    diff_l = mu_hat[None, :] - m_coarse_l[:, None]
    lam_p = np.clip(np.maximum(0.0, diff_l / (sig2_hat[None, :] + diff_l * diff_l)), 0.0, float(c) / m_coarse_l[:, None])
    log_kp = np.sum(np.log(1.0 + lam_p * (x[None, :] - m_coarse_l[:, None])), axis=1)
    valid_idx_l = np.where(log_kp < thresh_l)[0]
    if valid_idx_l.size == 0:
        L_t = emp_mean
    elif valid_idx_l[0] == 0:
        L_t = 0.0
    else:
        k_lo = int(valid_idx_l[0])
        m_fine_l = np.linspace(m_coarse_l[k_lo - 1], m_coarse_l[k_lo], 17)
        diff_lf = mu_hat[None, :] - m_fine_l[:, None]
        lam_pf = np.clip(np.maximum(0.0, diff_lf / (sig2_hat[None, :] + diff_lf * diff_lf)), 0.0, float(c) / m_fine_l[:, None])
        log_kpf = np.sum(np.log(1.0 + lam_pf * (x[None, :] - m_fine_l[:, None])), axis=1)
        vf_l = np.where(log_kpf < thresh_l)[0]
        L_t = float(m_fine_l[vf_l[0] - 1] if (vf_l.size > 0 and vf_l[0] > 0) else m_fine_l[0])

    return float(np.clip(L_t, 0.0, 1.0)), float(np.clip(U_t, 0.0, 1.0))


def confidence_sequence_radius(
    t: int,
    v_hat_t: float,
    alpha: float = 0.05,
    method: str = "howard_gamma_exponential_eb",
    two_sided: bool = True,
    m_intrinsic: float = 5.0,
) -> float:
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}.")
    alpha_clean = float(np.clip(alpha, 1e-12, 0.999))
    alpha_tail = (alpha_clean / 2.0) if two_sided else alpha_clean

    if method == "howard_gamma_exponential_eb":
        u_val = howard_gamma_exponential_eb_boundary(v_hat_t, alpha_tail=alpha_tail, m=m_intrinsic, c=1.0)
    elif method == "howard_stitched_eb":
        u_val = howard_stitched_eb_boundary(v_hat_t, alpha_tail=alpha_tail, m=max(0.5, m_intrinsic), c=1.0)
    elif method == "howard_normal_mixture_hoeffding":
        u_val = howard_normal_mixture_hoeffding_boundary(t, alpha_two_sided=alpha_clean if two_sided else min(0.999, 2.0 * alpha_clean))
    elif method == "howard_empirical_normal_mixture":
        u_val = howard_empirical_normal_mixture_boundary(
            t=t,
            v_hat_t=v_hat_t,
            alpha_two_sided=alpha_clean if two_sided else min(0.999, 2.0 * alpha_clean),
            m_intrinsic=m_intrinsic,
        )
    elif method == "wsr_agrapa_betting":
        # Approximate scalar radius fallback when only (t, v_hat_t) is passed without full array
        u_val = howard_gamma_exponential_eb_boundary(v_hat_t, alpha_tail=alpha_tail, m=max(0.5, m_intrinsic), c=0.65)
    else:
        raise ValueError(f"Unknown confidence sequence method '{method}'.")

    return float(u_val / float(t))


class TimeUniformConfidenceSequence:
    """
    Stateful streaming tracker for a time-uniform confidence sequence [L_t, U_t]
    on bounded observations X_1, X_2, ... in [0, 1] supporting both:
    - Waudby-Smith & Ramdas (2024) Theorem 1 & Prop 3 (`wsr_agrapa_betting`)
    - Howard et al. (2021) Theorem 4 + Prop 9 (`howard_gamma_exponential_eb`, `howard_stitched_eb`, etc.)
    and supporting asymmetric upper/lower error spending (`alpha_upper_fraction`).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        method: str = "howard_gamma_exponential_eb",
        m_intrinsic: float = 5.0,
        initial_prediction: float = 0.25,
        running_intersection: bool = True,
        alpha_upper_fraction: float = 0.50,
    ):
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not (0.0 < float(alpha_upper_fraction) < 1.0):
            raise ValueError(f"alpha_upper_fraction must be in (0, 1), got {alpha_upper_fraction}.")
        self.alpha = float(alpha)
        self.alpha_upper_fraction = float(alpha_upper_fraction)
        self.method = str(method)
        self.m_intrinsic = float(m_intrinsic)
        self.initial_prediction = float(np.clip(initial_prediction, 0.0, 1.0))
        self.running_intersection = bool(running_intersection)

        self.t: int = 0
        self.sum_x: float = 0.0
        self.sum_x2: float = 0.0
        self.v_hat_t: float = 0.0
        self.L_t: float = 0.0
        self.U_t: float = 1.0
        self.raw_L_t: float = 0.0
        self.raw_U_t: float = 1.0
        self.radius_t: float = 1.0
        self._obs_list: List[np.ndarray] = []
        self._cached_obs: np.ndarray = np.empty(0, dtype=np.float64)
        self.history: List[Dict[str, float]] = []

    @property
    def observations(self) -> np.ndarray:
        if len(self._obs_list) > 0:
            if len(self._obs_list) > 1:
                self._cached_obs = np.concatenate(self._obs_list)
                self._obs_list = [self._cached_obs]
            else:
                self._cached_obs = self._obs_list[0]
        return self._cached_obs

    @property
    def mean(self) -> float:
        return float(self.sum_x / self.t) if self.t > 0 else self.initial_prediction

    @property
    def sample_variance(self) -> float:
        if self.t < 2:
            return 0.0
        mean_val = self.sum_x / self.t
        var_val = (self.sum_x2 - self.t * (mean_val ** 2)) / float(self.t - 1)
        return float(max(0.0, var_val))

    def recompute_for_alpha(self, new_alpha: float) -> Tuple[float, float]:
        """
        Re-evaluates [L_t, U_t] at a dynamically updated simultaneous alpha level
        (e.g. under Sequential Holm Step-Down alpha recycling alpha_arm(t) = alpha_discovery / |A_t|).
        """
        if self.t < 1:
            return 0.0, 1.0
        a_clean = float(np.clip(new_alpha, 1e-12, 0.999))
        a_u = a_clean * self.alpha_upper_fraction
        a_l = a_clean * (1.0 - self.alpha_upper_fraction)
        if self.method == "wsr_agrapa_betting":
            l_val, u_val = wsr_agrapa_betting_bounds(
                self.observations,
                alpha_upper=a_u,
                alpha_lower=a_l,
                initial_prediction=self.initial_prediction,
            )
            return l_val, u_val
        else:
            r_u = confidence_sequence_radius(self.t, self.v_hat_t, alpha=a_u, method=self.method, two_sided=False, m_intrinsic=self.m_intrinsic)
            r_l = confidence_sequence_radius(self.t, self.v_hat_t, alpha=a_l, method=self.method, two_sided=False, m_intrinsic=self.m_intrinsic)
            mean_now = self.sum_x / float(self.t)
            return float(np.clip(mean_now - r_l, 0.0, 1.0)), float(np.clip(mean_now + r_u, 0.0, 1.0))

    def update_batch(self, batch_observations: Sequence[float]) -> Tuple[float, float]:
        arr = np.asarray(batch_observations, dtype=np.float64)
        if arr.size == 0:
            return self.L_t, self.U_t
        if np.any(arr < -1e-9) or np.any(arr > 1.0 + 1e-9):
            raise ValueError("All observations must lie in [0, 1].")
        arr = np.clip(arr, 0.0, 1.0)
        self._obs_list.append(arr)

        n_new = arr.size
        idx_prev = np.arange(self.t, self.t + n_new, dtype=np.float64)
        csum_new = self.sum_x + np.cumsum(arr)
        sum_prev = np.empty(n_new, dtype=np.float64)
        sum_prev[0] = self.sum_x
        if n_new > 1:
            sum_prev[1:] = csum_new[:-1]

        preds = np.where(idx_prev == 0, self.initial_prediction, sum_prev / np.maximum(1.0, idx_prev))
        v_increment = float(np.sum((arr - preds) ** 2))

        self.t += int(n_new)
        self.sum_x = float(csum_new[-1])
        self.sum_x2 += float(np.sum(arr ** 2))
        self.v_hat_t += max(0.0, v_increment)

        mean_now = self.sum_x / float(self.t)
        raw_L, raw_U = self.recompute_for_alpha(self.alpha)
        self.raw_L_t = float(raw_L)
        self.raw_U_t = float(raw_U)
        self.radius_t = float(0.5 * (raw_U - raw_L))

        if self.running_intersection:
            self.L_t = max(self.L_t, self.raw_L_t)
            self.U_t = min(self.U_t, self.raw_U_t)
            if self.L_t > self.U_t:
                self.L_t = self.raw_L_t
                self.U_t = self.raw_U_t
        else:
            self.L_t = self.raw_L_t
            self.U_t = self.raw_U_t

        self.history.append({
            "t": int(self.t),
            "mean": float(mean_now),
            "sample_variance": float(self.sample_variance),
            "v_hat_t": float(self.v_hat_t),
            "radius_t": float(self.radius_t),
            "raw_L_t": float(self.raw_L_t),
            "raw_U_t": float(self.raw_U_t),
            "L_t": float(self.L_t),
            "U_t": float(self.U_t),
        })
        return self.L_t, self.U_t


def compute_alpha_spending_schedule(
    alpha_total: float = 0.05,
    max_attempts: int = 2,
    method: str = "geometric_80_20",
) -> List[float]:
    if alpha_total <= 0.0 or alpha_total >= 1.0:
        raise ValueError(f"alpha_total must be in (0, 1), got {alpha_total}.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    if max_attempts == 1:
        return [float(alpha_total)]

    if method == "bonferroni":
        return [float(alpha_total) / float(max_attempts) for _ in range(max_attempts)]
    elif method == "geometric_80_20":
        if max_attempts == 2:
            return [0.80 * float(alpha_total), 0.20 * float(alpha_total)]
        weights = np.array([0.80 * (0.20 ** j) for j in range(max_attempts)], dtype=np.float64)
        weights /= np.sum(weights)
        return [float(w * alpha_total) for w in weights]
    else:
        raise ValueError(f"Unknown alpha-spending schedule method '{method}'.")


def run_sequential_confirmation_from_losses(
    shot_losses: np.ndarray,
    target_threshold: float,
    alpha: float = 0.05,
    batch_size: int = 256,
    max_shots: int = 4096,
    method: str = "howard_gamma_exponential_eb",
    m_intrinsic: float = 5.0,
    alpha_upper_fraction: float = 0.50,
) -> Dict[str, Any]:
    """
    Executes Phase C Anytime-Valid Sequential Confirmation on fresh projective measurement losses.
    Supports both:
    - `wsr_agrapa_betting` (Waudby-Smith & Ramdas 2024 Theorem 1 + Prop 3, with fast direct log-wealth check at tau)
    - `howard_gamma_exponential_eb` (Howard et al. 2021 Theorem 4 + Prop 9)
    and supports asymmetric error spending (`alpha_upper_fraction`, e.g. 0.90 on U_t and 0.10 on L_t).
    """
    arr = np.asarray(shot_losses, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("shot_losses cannot be empty.")
    b_size = int(batch_size)
    b_max = min(int(max_shots), int(arr.size))
    if b_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    tau = float(target_threshold)

    if method == "wsr_agrapa_betting":
        x = np.clip(arr[:b_max], 0.0, 1.0)
        mu_hat, sig2_hat = wsr_predictable_estimators(x, initial_prediction=tau, prior_variance=0.012)
        alpha_u = float(alpha) * float(alpha_upper_fraction)
        alpha_l = float(alpha) * (1.0 - float(alpha_upper_fraction))
        thresh_u = np.log(1.0 / max(1e-12, alpha_u))
        thresh_l = np.log(1.0 / max(1e-12, alpha_l))

        diff_tau = tau - mu_hat
        lam_m = np.clip(np.maximum(0.0, diff_tau / (sig2_hat + diff_tau * diff_tau)), 0.0, 0.75 / max(1e-6, 1.0 - tau))
        lam_p = np.clip(np.maximum(0.0, -diff_tau / (sig2_hat + diff_tau * diff_tau)), 0.0, 0.75 / max(1e-6, tau))

        log_km_cum = np.cumsum(np.log(1.0 - lam_m * (x - tau)))
        log_kp_cum = np.cumsum(np.log(1.0 + lam_p * (x - tau)))
        csum_x = np.cumsum(x)

        decision = "INCONCLUSIVE"
        certified = False
        target_rejected = False
        inconclusive = True
        stop_t = b_max
        n_batches = 0
        L_t, U_t = 0.0, 1.0

        cursor = 0
        while cursor < b_max:
            next_c = min(cursor + b_size, b_max)
            n_batches += 1
            emp_m = float(csum_x[next_c - 1] / float(next_c))
            if emp_m <= tau and log_km_cum[next_c - 1] >= thresh_u:
                l_cand, u_cand = wsr_agrapa_betting_bounds(
                    x[:next_c], alpha_upper=alpha_u, alpha_lower=alpha_l,
                    initial_prediction=tau, prior_variance=0.012,
                    mu_hat=mu_hat[:next_c], sig2_hat=sig2_hat[:next_c],
                )
                if u_cand <= tau:
                    decision = "CERTIFIED"
                    certified = True
                    inconclusive = False
                    stop_t = next_c
                    L_t, U_t = l_cand, u_cand
                    break
            elif emp_m > tau and log_kp_cum[next_c - 1] >= thresh_l:
                l_cand, u_cand = wsr_agrapa_betting_bounds(
                    x[:next_c], alpha_upper=alpha_u, alpha_lower=alpha_l,
                    initial_prediction=tau, prior_variance=0.012,
                    mu_hat=mu_hat[:next_c], sig2_hat=sig2_hat[:next_c],
                )
                if l_cand > tau:
                    decision = "TARGET_REJECTED"
                    target_rejected = True
                    inconclusive = False
                    stop_t = next_c
                    L_t, U_t = l_cand, u_cand
                    break
            cursor = next_c

        sub_x = x[:stop_t]
        if inconclusive:
            L_t, U_t = wsr_agrapa_betting_bounds(
                sub_x, alpha_upper=alpha_u, alpha_lower=alpha_l,
                initial_prediction=tau, prior_variance=0.012,
                mu_hat=mu_hat[:stop_t], sig2_hat=sig2_hat[:stop_t],
            )
        c_mean = float(np.mean(sub_x))
        c_var = float(np.var(sub_x, ddof=1)) if stop_t >= 2 else 0.0
        v_hat_stop = float(np.sum((sub_x - mu_hat[:stop_t]) ** 2))
        return {
            "decision": decision,
            "certified": certified,
            "target_rejected": target_rejected,
            "inconclusive": inconclusive,
            "stopping_shots": int(stop_t),
            "batches_evaluated": int(n_batches),
            "max_shots": int(b_max),
            "batch_size": int(b_size),
            "alpha": float(alpha),
            "alpha_upper_fraction": float(alpha_upper_fraction),
            "method": str(method),
            "target_threshold": float(tau),
            "confirmation_mean": c_mean,
            "confirmation_variance": c_var,
            "v_hat_t": v_hat_stop,
            "L_t": float(L_t),
            "U_t": float(U_t),
            "radius_t": float(0.5 * (U_t - L_t)),
            "certification_slack": float(tau - U_t),
            "rejection_slack": float(L_t - tau),
            "history": [],
        }

    cs = TimeUniformConfidenceSequence(
        alpha=alpha,
        method=method,
        m_intrinsic=m_intrinsic,
        initial_prediction=tau,
        running_intersection=True,
        alpha_upper_fraction=alpha_upper_fraction,
    )

    decision = "INCONCLUSIVE"
    certified = False
    target_rejected = False
    inconclusive = True
    cursor = 0
    n_batches = 0

    while cursor < b_max:
        next_cursor = min(cursor + b_size, b_max)
        batch = arr[cursor:next_cursor]
        L_t, U_t = cs.update_batch(batch)
        cursor = next_cursor
        n_batches += 1

        if U_t <= tau:
            decision = "CERTIFIED"
            certified = True
            target_rejected = False
            inconclusive = False
            break
        elif L_t > tau:
            decision = "TARGET_REJECTED"
            certified = False
            target_rejected = True
            inconclusive = False
            break

    return {
        "decision": decision,
        "certified": certified,
        "target_rejected": target_rejected,
        "inconclusive": inconclusive,
        "stopping_shots": int(cs.t),
        "batches_evaluated": int(n_batches),
        "max_shots": int(b_max),
        "batch_size": int(b_size),
        "alpha": float(alpha),
        "alpha_upper_fraction": float(alpha_upper_fraction),
        "method": str(method),
        "target_threshold": float(tau),
        "confirmation_mean": float(cs.mean),
        "confirmation_variance": float(cs.sample_variance),
        "v_hat_t": float(cs.v_hat_t),
        "L_t": float(cs.L_t),
        "U_t": float(cs.U_t),
        "radius_t": float(cs.radius_t),
        "certification_slack": float(tau - cs.U_t),
        "rejection_slack": float(cs.L_t - tau),
        "history": cs.history,
    }
