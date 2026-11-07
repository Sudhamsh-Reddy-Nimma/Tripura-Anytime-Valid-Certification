"""
Reproducible data generator for the new synthetic validation panels requested in the PRA figure specification:
1. Fig. 2(c): Equal-mean vs. QAOA-calibrated vs. Disjoint-support selection bias (Theorem 3.1a,b & Remark 3.2).
2. Fig. 3(b): Empirical Pr(U_{k_hat} <= tau) vs. 1 - prod_{k=1}^K (1 - p_k) under homogeneous vs. heterogeneous radii (Theorem 3.1c).
3. Fig. 4(b): Phase C adversarial boundary stress test at mu(theta_hat) = tau + 1e-4 across alpha in {0.01, 0.02, 0.03, 0.04, 0.05} (Theorem 3.8a).
4. Fig. 5(a,e): Conditional mean shift E[X_{k,s} | F_{n_k-1}] - mu(theta_k) and adaptive-arm Phase A coverage under Mode U, Mode H, and Mode C (Remark 2.4).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm
import certification.confidence_sequences as cs

OUT_DIR = "results_stage3_corrected/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi

rng = np.random.default_rng(20261106)

# ---------------------------------------------------------------------------
# 1. Fig. 2(c): Synthetic Contrast (Equal-Mean vs QAOA-Calibrated vs Disjoint)
# ---------------------------------------------------------------------------
rows_2c = []
n_rep = 2000
for K in [2, 10, 25, 50, 100]:
    B = 256
    sigma = 0.07  # matches QAOA variance ~ 0.0049
    # (i) Equal-mean Gaussian/Bounded arm pool: mu_1 = ... = mu_K = 0.30, var = 0.0049
    draws_eq = rng.normal(loc=0.30, scale=sigma / np.sqrt(B), size=(n_rep, K))
    k_hat_eq = np.argmin(draws_eq, axis=1)
    bias_eq = draws_eq[np.arange(n_rep), k_hat_eq] - 0.30
    opt_eq = -bias_eq
    th_bound = np.sqrt(2.0 * np.log(max(K, 1)) * (sigma ** 2) / B) if K > 1 else 0.0
    rows_2c.append({
        "regime": "equal_mean_homogeneous",
        "K": K, "B": B,
        "mean_bias": float(np.mean(bias_eq)),
        "mean_optimism": float(np.mean(opt_eq)),
        "ci_lower": float(np.mean(opt_eq) - 1.96 * np.std(opt_eq, ddof=1) / np.sqrt(n_rep)),
        "ci_upper": float(np.mean(opt_eq) + 1.96 * np.std(opt_eq, ddof=1) / np.sqrt(n_rep)),
        "theory_bound": float(th_bound),
    })

    # (ii) Mildly heterogeneous QAOA-like gaps: mu_k = 0.30 + 0.012 * (k / K)
    mus_het = 0.30 + 0.012 * np.linspace(0.0, 1.0, K)
    draws_het = rng.normal(loc=mus_het[None, :], scale=sigma / np.sqrt(B), size=(n_rep, K))
    k_hat_het = np.argmin(draws_het, axis=1)
    opt_het = mus_het[k_hat_het] - draws_het[np.arange(n_rep), k_hat_het]
    rows_2c.append({
        "regime": "qaoa_heterogeneous_gaps",
        "K": K, "B": B,
        "mean_bias": float(-np.mean(opt_het)),
        "mean_optimism": float(np.mean(opt_het)),
        "ci_lower": float(np.mean(opt_het) - 1.96 * np.std(opt_het, ddof=1) / np.sqrt(n_rep)),
        "ci_upper": float(np.mean(opt_het) + 1.96 * np.std(opt_het, ddof=1) / np.sqrt(n_rep)),
        "theory_bound": float(th_bound),
    })

    # (iii) Disjoint-support counterexample (Remark 3.2): X_1 ~ U(0, 0.1), X_{2..K} ~ U(0.2, 0.3)
    # Single-shot or B-shot average has strictly disjoint support [0, 0.1] vs [0.2, 0.3]
    x1 = np.mean(rng.uniform(0.0, 0.1, size=(n_rep, B)), axis=1, keepdims=True)
    x_rest = np.mean(rng.uniform(0.2, 0.3, size=(n_rep, K - 1, B)), axis=2) if K > 1 else np.empty((n_rep, 0))
    draws_disj = np.hstack([x1, x_rest])
    mus_disj = np.array([0.05] + [0.25] * (K - 1))
    k_hat_disj = np.argmin(draws_disj, axis=1)  # always index 0!
    opt_disj = mus_disj[k_hat_disj] - draws_disj[np.arange(n_rep), k_hat_disj]
    rows_2c.append({
        "regime": "disjoint_support_counterexample",
        "K": K, "B": B,
        "mean_bias": float(-np.mean(opt_disj)),
        "mean_optimism": float(np.mean(opt_disj)),
        "ci_lower": float(np.mean(opt_disj) - 1.96 * np.std(opt_disj, ddof=1) / np.sqrt(n_rep)),
        "ci_upper": float(np.mean(opt_disj) + 1.96 * np.std(opt_disj, ddof=1) / np.sqrt(n_rep)),
        "theory_bound": float(th_bound),
    })

pd.DataFrame(rows_2c).to_csv(os.path.join(OUT_DIR, "fig2c_synthetic_contrast.csv"), index=False)

# ---------------------------------------------------------------------------
# 2. Fig. 3(b): Pr(U_{k_hat} <= tau) vs 1 - prod(1 - p_k) (Theorem 3.1c)
# ---------------------------------------------------------------------------
rows_3b = []
n_mc = 4000
z_alpha = float(norm.ppf(0.95))
for K in [5, 10, 15, 25, 40, 60]:
    sigma = 0.08
    mu_true = 0.30
    tau = 0.296  # slightly below mu_true so all arms are infeasible (mu_k > tau)
    # Case 1: Homogeneous radii (B_1 = ... = B_K = 256)
    B_hom = np.full(K, 256.0)
    r_hom = z_alpha * sigma / np.sqrt(B_hom)
    se_hom = sigma / np.sqrt(B_hom)
    p_k_hom = norm.cdf((tau - r_hom - mu_true) / se_hom)
    prod_formula_hom = float(1.0 - np.prod(1.0 - p_k_hom))
    X_hom = rng.normal(loc=mu_true, scale=se_hom[None, :], size=(n_mc, K))
    U_hom = X_hom + r_hom[None, :]
    k_hat_hom = np.argmin(X_hom, axis=1)
    emp_sel_pass_hom = int(np.sum(U_hom[np.arange(n_mc), k_hat_hom] <= tau))
    lo_h, hi_h = clopper_pearson(emp_sel_pass_hom, n_mc)
    rows_3b.append({
        "radii_regime": "homogeneous_radii",
        "K": K,
        "product_formula_prob": prod_formula_hom,
        "empirical_selected_prob": emp_sel_pass_hom / n_mc,
        "cp_lower": lo_h,
        "cp_upper": hi_h,
    })

    # Case 2: Heterogeneous radii (unequal shot allocations B_k in [32, 1024])
    # Noisy low-shot arms have wide radii r_k but frequently win argmin_k X_k!
    B_het = np.geomspace(32.0, 1024.0, K)
    r_het = z_alpha * sigma / np.sqrt(B_het)
    se_het = sigma / np.sqrt(B_het)
    p_k_het = norm.cdf((tau - r_het - mu_true) / se_het)
    prod_formula_het = float(1.0 - np.prod(1.0 - p_k_het))
    X_het = rng.normal(loc=mu_true, scale=se_het[None, :], size=(n_mc, K))
    U_het = X_het + r_het[None, :]
    k_hat_het = np.argmin(X_het, axis=1)
    emp_sel_pass_het = int(np.sum(U_het[np.arange(n_mc), k_hat_het] <= tau))
    lo_ht, hi_ht = clopper_pearson(emp_sel_pass_het, n_mc)
    rows_3b.append({
        "radii_regime": "heterogeneous_radii",
        "K": K,
        "product_formula_prob": prod_formula_het,
        "empirical_selected_prob": emp_sel_pass_het / n_mc,
        "cp_lower": lo_ht,
        "cp_upper": hi_ht,
    })

pd.DataFrame(rows_3b).to_csv(os.path.join(OUT_DIR, "fig3b_heterogeneous_radii_validation.csv"), index=False)

# ---------------------------------------------------------------------------
# 3. Fig. 4(b): Phase C Adversarial Boundary Stress Test at mu = tau + 1e-4
# ---------------------------------------------------------------------------
rows_4b = []
n_boundary = 500
tau_b = 0.2600
mu_b = tau_b + 1e-4  # strictly infeasible by 10^-4
# Beta distribution with mean mu_b and variance ~ 0.0048 (matching QAOA MaxCut loss)
var_b = 0.0048
common_fac = mu_b * (1.0 - mu_b) / var_b - 1.0
a_beta = mu_b * common_fac
b_beta = (1.0 - mu_b) * common_fac

for alpha_val in [0.01, 0.02, 0.03, 0.04, 0.05]:
    false_certs = 0
    covered = 0
    for g in range(n_boundary):
        stream = rng.beta(a_beta, b_beta, size=1024)
        # Sequential checkpoints at t = 256, 512, 768, 1024 using actual WSR aGRAPA betting CS
        certified = False
        cov_ok = True
        for t in [256, 512, 768, 1024]:
            L_t, U_t = cs.wsr_agrapa_betting_bounds(
                stream[:t], alpha_upper=0.99 * alpha_val, alpha_lower=0.01 * alpha_val
            )
            if not (L_t <= mu_b <= U_t):
                cov_ok = False
            if U_t <= tau_b:
                certified = True
                break
        if certified:
            false_certs += 1
        if cov_ok:
            covered += 1
    fc_lo, fc_hi = clopper_pearson(false_certs, n_boundary)
    cov_lo, cov_hi = clopper_pearson(covered, n_boundary)
    rows_4b.append({
        "confirmation_type": "boundary_stress_mu_eq_tau_plus_1e-4",
        "alpha": alpha_val,
        "nominal_confidence": 1.0 - alpha_val,
        "independent_attempts_G": n_boundary,
        "false_certifications": false_certs,
        "observed_false_certification_rate": false_certs / n_boundary,
        "clopper_pearson_95_lower_bound": fc_lo,
        "clopper_pearson_95_upper_bound": fc_hi,
        "cs_coverage": covered / n_boundary,
        "cs_coverage_ci_lower": cov_lo,
        "cs_coverage_ci_upper": cov_hi,
    })

pd.DataFrame(rows_4b).to_csv(os.path.join(OUT_DIR, "fig4b_boundary_stress_validation.csv"), index=False)

# ---------------------------------------------------------------------------
# 4. Fig. 5(a,e): Conditional Mean Shift E[X|F] - mu & Adaptive-Arm Coverage
# ---------------------------------------------------------------------------
# Simulate adaptive candidate selection where adaptive arm theta_k is chosen at parameter
# minimizing empirical loss on the first B_init = 64 uniform draws U_1..U_64.
# If those same U_1..U_64 are reused (Mode C), E[X | F] - mu is strictly negative!
# If fresh draws are used on adaptive arms (Mode U and Mode H), E[X | F] - mu is centered at 0!
n_sim = 1000
B_adapt = 128
m_grid = 12
cond_shifts_fresh_U = []
cond_shifts_hybrid_H = []
cond_shifts_reused_C = []

cov_U, cov_H, cov_C = 0, 0, 0
alpha_arm = 0.05

for i in range(n_sim):
    # Shared uniform stream used during simplex exploration
    u_shared = rng.uniform(0.0, 1.0, size=B_adapt)
    # Candidate proposals parameterized by sensitivity beta_m in [0.22, 0.30]
    mus_cand = np.linspace(0.24, 0.28, m_grid)
    # Each proposal's inverse-CDF transform with local jitter that depends on u_shared[:64]
    noise_matrix = rng.normal(0.0, 0.018, size=(m_grid, B_adapt))
    # Optimizer picks adaptive proposal m* that looked best on the shared stream prefix
    obs_shared = mus_cand[:, None] + 0.08 * (u_shared[None, :] - 0.5) + noise_matrix
    m_star = int(np.argmin(np.mean(obs_shared[:, :64], axis=1)))
    mu_true_star = float(mus_cand[m_star])

    # Mode C: Reuses u_shared[0:B_adapt] on the adaptively chosen arm m_star
    x_mode_C = np.clip(obs_shared[m_star, :B_adapt], 0.01, 0.99)
    cond_shifts_reused_C.append(float(np.mean(x_mode_C) - mu_true_star))
    L_c, U_c = cs.wsr_agrapa_betting_bounds(x_mode_C, alpha_upper=alpha_arm/2, alpha_lower=alpha_arm/2)
    if L_c <= mu_true_star <= U_c:
        cov_C += 1

    # Mode H (Flagship): Simplex used CRN, but adaptive arm m_star gets a FRESH stream
    u_fresh_H = rng.uniform(0.0, 1.0, size=B_adapt)
    noise_H = rng.normal(0.0, 0.018, size=B_adapt)
    x_mode_H = np.clip(mu_true_star + 0.08 * (u_fresh_H - 0.5) + noise_H, 0.01, 0.99)
    cond_shifts_hybrid_H.append(float(np.mean(x_mode_H) - mu_true_star))
    L_h, U_h = cs.wsr_agrapa_betting_bounds(x_mode_H, alpha_upper=alpha_arm/2, alpha_lower=alpha_arm/2)
    if L_h <= mu_true_star <= U_h:
        cov_H += 1

    # Mode U (Uncoupled): Fresh stream everywhere
    u_fresh_U = rng.uniform(0.0, 1.0, size=B_adapt)
    noise_U = rng.normal(0.0, 0.018, size=B_adapt)
    x_mode_U = np.clip(mu_true_star + 0.08 * (u_fresh_U - 0.5) + noise_U, 0.01, 0.99)
    cond_shifts_fresh_U.append(float(np.mean(x_mode_U) - mu_true_star))
    L_u, U_u = cs.wsr_agrapa_betting_bounds(x_mode_U, alpha_upper=alpha_arm/2, alpha_lower=alpha_arm/2)
    if L_u <= mu_true_star <= U_u:
        cov_U += 1

df_shifts = pd.DataFrame({
    "trial": np.arange(n_sim),
    "shift_mode_U": cond_shifts_fresh_U,
    "shift_mode_H": cond_shifts_hybrid_H,
    "shift_mode_C": cond_shifts_reused_C,
})
df_shifts.to_csv(os.path.join(OUT_DIR, "fig5_crn_breakage_shifts.csv"), index=False)

summary_5a = []
for m_id, cov_cnt, shifts in [
    ("proposed_sota_uncoupled", cov_U, cond_shifts_fresh_U),
    ("proposed_sota_hybrid_simplex", cov_H, cond_shifts_hybrid_H),
    ("proposed_sota_full_coupled", cov_C, cond_shifts_reused_C),
]:
    lo, hi = clopper_pearson(cov_cnt, n_sim)
    summary_5a.append({
        "method_id": m_id,
        "adaptive_arm_cs_coverage": cov_cnt / n_sim,
        "coverage_cp_lower": lo,
        "coverage_cp_upper": hi,
        "mean_conditional_shift": float(np.mean(shifts)),
        "sd_conditional_shift": float(np.std(shifts, ddof=1)),
    })
pd.DataFrame(summary_5a).to_csv(os.path.join(OUT_DIR, "fig5_adaptive_arm_coverage_summary.csv"), index=False)
print("Successfully generated all supplementary validation CSVs.")
