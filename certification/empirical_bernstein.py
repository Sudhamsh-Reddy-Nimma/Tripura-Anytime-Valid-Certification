"""
Empirical-Bernstein Confidence Bounds (Maurer & Pontil 2009, Theorem 4) for AMFS 2.0 (Stage 1).

Mathematical Formulation (Section 7):
For independent random variables X_1, ..., X_B in [0, 1] with sample mean X_bar and
unbiased sample variance V_hat = (1 / (B - 1)) * sum_{i=1}^B (X_i - X_bar)^2:

    U_EB = X_bar + sqrt(2 * V_hat * ln(2 / delta) / B) + (7 * ln(2 / delta)) / (3 * (B - 1))

Clipped to the valid loss domain [0, 1].

For Bonferroni multiplicity correction over K predeclared candidate evaluations (Section 8, Method 2):
    delta_K = delta / K
    U_EB_Bonf = empirical_bernstein_upper_bound(X_bar, V_hat, B, delta / K)

Two-sided interval (1 - delta nominal coverage):
Uses delta_two_sided = delta / 2 in each tail, i.e., log term ln(4 / delta), clipped to [0, 1].
"""

from typing import Tuple
import numpy as np


def validate_bound_inputs(mean: float, variance: float, B: int, delta: float) -> Tuple[float, float, int, float]:
    """
    Validate inputs for bounded confidence interval calculations on [0, 1].
    """
    if not np.isfinite(mean) or mean < -1e-9 or mean > 1.0 + 1e-9:
        raise ValueError(f"Sample mean must be finite in [0, 1], got {mean}.")
    if not np.isfinite(variance) or variance < -1e-12:
        raise ValueError(f"Sample variance must be finite and non-negative, got {variance}.")
    if not isinstance(B, (int, np.integer)) or int(B) < 2:
        raise ValueError(f"Shot budget B must be an integer >= 2, got {B}.")
    if not np.isfinite(delta) or delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"Confidence failure probability delta must be in (0, 1), got {delta}.")

    clean_mean = float(np.clip(mean, 0.0, 1.0))
    clean_var = float(max(0.0, variance))
    clean_B = int(B)
    clean_delta = float(delta)
    return clean_mean, clean_var, clean_B, clean_delta


def empirical_bernstein_radius(variance: float, B: int, delta: float) -> float:
    """
    Compute the unclipped empirical-Bernstein radius:
        r_EB = sqrt(2 * V_hat * ln(2 / delta) / B) + 7 * ln(2 / delta) / (3 * (B - 1))
    """
    _, clean_var, clean_B, clean_delta = validate_bound_inputs(0.5, variance, B, delta)
    log_term = np.log(2.0 / clean_delta)
    std_term = np.sqrt((2.0 * clean_var * log_term) / float(clean_B))
    rem_term = (7.0 * log_term) / (3.0 * float(clean_B - 1))
    return float(std_term + rem_term)


def empirical_bernstein_upper_bound(
    mean: float,
    variance: float,
    B: int,
    delta: float,
    clip: bool = True,
) -> float:
    """
    Calculate the one-sided empirical-Bernstein upper confidence bound U_EB (Method 1).
    """
    clean_mean, clean_var, clean_B, clean_delta = validate_bound_inputs(mean, variance, B, delta)
    radius = empirical_bernstein_radius(clean_var, clean_B, clean_delta)
    ub = clean_mean + radius
    if clip:
        return float(np.clip(ub, 0.0, 1.0))
    return float(ub)


def empirical_bernstein_two_sided_interval(
    mean: float,
    variance: float,
    B: int,
    delta: float,
    clip: bool = True,
) -> Tuple[float, float]:
    """
    Calculate the two-sided empirical-Bernstein confidence interval [L_EB, U_EB]
    at overall failure probability delta (allocating delta/2 to each tail -> ln(4/delta)).
    """
    clean_mean, clean_var, clean_B, clean_delta = validate_bound_inputs(mean, variance, B, delta)
    radius = empirical_bernstein_radius(clean_var, clean_B, clean_delta / 2.0)
    lb = clean_mean - radius
    ub = clean_mean + radius
    if clip:
        lb = float(np.clip(lb, 0.0, 1.0))
        ub = float(np.clip(ub, 0.0, 1.0))
    return float(lb), float(ub)


def bonferroni_empirical_bernstein_upper_bound(
    mean: float,
    variance: float,
    B: int,
    delta: float,
    K: int,
    clip: bool = True,
) -> float:
    """
    Calculate the Bonferroni multiplicity-adjusted empirical-Bernstein upper bound (Method 2)
    with delta_K = delta / K.
    """
    if not isinstance(K, (int, np.integer)) or int(K) < 1:
        raise ValueError(f"Candidate prefix K must be a positive integer, got {K}.")
    _, _, _, clean_delta = validate_bound_inputs(mean, variance, B, delta)
    delta_k = clean_delta / float(int(K))
    return empirical_bernstein_upper_bound(mean, variance, B, delta_k, clip=clip)


def bonferroni_empirical_bernstein_two_sided_interval(
    mean: float,
    variance: float,
    B: int,
    delta: float,
    K: int,
    clip: bool = True,
) -> Tuple[float, float]:
    """
    Calculate the two-sided Bonferroni empirical-Bernstein interval at delta_K = delta / K.
    """
    if not isinstance(K, (int, np.integer)) or int(K) < 1:
        raise ValueError(f"Candidate prefix K must be a positive integer, got {K}.")
    _, _, _, clean_delta = validate_bound_inputs(mean, variance, B, delta)
    delta_k = clean_delta / float(int(K))
    return empirical_bernstein_two_sided_interval(mean, variance, B, delta_k, clip=clip)
