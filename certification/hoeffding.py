"""
Hoeffding Upper Confidence Bound for Bounded Observations in [0, 1] (Section 8, Method 3).

Mathematical Formulation:
    U_H = X_bar + sqrt(ln(1 / delta) / (2 * B))
Clipped to [0, 1].
"""

from typing import Tuple
import numpy as np
from certification.empirical_bernstein import validate_bound_inputs


def hoeffding_radius(B: int, delta: float) -> float:
    """
    Compute the unclipped one-sided Hoeffding radius:
        r_H = sqrt(ln(1 / delta) / (2 * B))
    """
    _, _, clean_B, clean_delta = validate_bound_inputs(0.5, 0.0, B, delta)
    return float(np.sqrt(np.log(1.0 / clean_delta) / (2.0 * float(clean_B))))


def hoeffding_upper_bound(
    mean: float,
    B: int,
    delta: float,
    clip: bool = True,
) -> float:
    """
    Calculate the one-sided Hoeffding upper bound U_H (Method 3).
    """
    clean_mean, _, clean_B, clean_delta = validate_bound_inputs(mean, 0.0, B, delta)
    ub = clean_mean + hoeffding_radius(clean_B, clean_delta)
    if clip:
        return float(np.clip(ub, 0.0, 1.0))
    return float(ub)


def hoeffding_two_sided_interval(
    mean: float,
    B: int,
    delta: float,
    clip: bool = True,
) -> Tuple[float, float]:
    """
    Calculate the two-sided Hoeffding interval [L_H, U_H] at overall failure probability delta
    (using delta / 2 in each tail -> sqrt(ln(2 / delta) / (2 * B))).
    """
    clean_mean, _, clean_B, clean_delta = validate_bound_inputs(mean, 0.0, B, delta)
    radius = hoeffding_radius(clean_B, clean_delta / 2.0)
    lb = clean_mean - radius
    ub = clean_mean + radius
    if clip:
        lb = float(np.clip(lb, 0.0, 1.0))
        ub = float(np.clip(ub, 0.0, 1.0))
    return float(lb), float(ub)
