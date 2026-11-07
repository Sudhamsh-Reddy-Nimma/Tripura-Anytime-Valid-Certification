"""
Non-Guaranteed Asymptotic / Parametric Reference Confidence Bounds (Section 8, Method 4).

Mathematical Formulation:
- Student-t reference bound:
    U_t = X_bar + t_{1 - delta, df=B-1} * sqrt(V_hat / B)
- Gaussian reference bound:
    U_z = X_bar + z_{1 - delta} * sqrt(V_hat / B)

IMPORTANT SCIENTIFIC NOTE:
These are non-guaranteed reference methods. They are NOT distribution-free and do NOT
possess a finite-sample coverage guarantee, particularly under adaptive selection.
Their purpose is strictly to demonstrate how less conservative asymptotic/normal bounds
fail under finite-shot adaptive candidate selection.
"""

from typing import Tuple
import numpy as np
from scipy.stats import norm, t as student_t
from certification.empirical_bernstein import validate_bound_inputs


def reference_upper_bound(
    mean: float,
    variance: float,
    B: int,
    delta: float,
    distribution: str = "student_t",
    clip: bool = True,
) -> float:
    """
    Calculate the one-sided non-guaranteed reference upper bound (Student-t or Gaussian).
    """
    clean_mean, clean_var, clean_B, clean_delta = validate_bound_inputs(mean, variance, B, delta)
    se = np.sqrt(clean_var / float(clean_B))
    if distribution == "student_t":
        crit = float(student_t.ppf(1.0 - clean_delta, df=clean_B - 1))
    elif distribution == "gaussian":
        crit = float(norm.ppf(1.0 - clean_delta))
    else:
        raise ValueError(f"Unsupported reference distribution '{distribution}'.")
    ub = clean_mean + crit * se
    if clip:
        return float(np.clip(ub, 0.0, 1.0))
    return float(ub)


def reference_two_sided_interval(
    mean: float,
    variance: float,
    B: int,
    delta: float,
    distribution: str = "student_t",
    clip: bool = True,
) -> Tuple[float, float]:
    """
    Calculate the two-sided non-guaranteed reference interval [L_ref, U_ref]
    at nominal confidence 1 - delta.
    """
    clean_mean, clean_var, clean_B, clean_delta = validate_bound_inputs(mean, variance, B, delta)
    se = np.sqrt(clean_var / float(clean_B))
    if distribution == "student_t":
        crit = float(student_t.ppf(1.0 - clean_delta / 2.0, df=clean_B - 1))
    elif distribution == "gaussian":
        crit = float(norm.ppf(1.0 - clean_delta / 2.0))
    else:
        raise ValueError(f"Unsupported reference distribution '{distribution}'.")
    lb = clean_mean - crit * se
    ub = clean_mean + crit * se
    if clip:
        lb = float(np.clip(lb, 0.0, 1.0))
        ub = float(np.clip(ub, 0.0, 1.0))
    return float(lb), float(ub)
