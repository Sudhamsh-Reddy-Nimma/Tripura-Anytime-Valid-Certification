"""
Certification Decisions and Bound Slack Calculations for AMFS 2.0 (Stage 1).

Note: Consistent with Section 5 and Section 21, this module does NOT import `analysis.ground_truth`.
It computes certification bounds, certification decisions (U <= tau), and bound slack (U - mu_hat)
strictly from finite-shot observations.
"""

from typing import Dict, Any, List
import numpy as np
from certification.empirical_bernstein import (
    empirical_bernstein_upper_bound,
    empirical_bernstein_two_sided_interval,
    bonferroni_empirical_bernstein_upper_bound,
    bonferroni_empirical_bernstein_two_sided_interval,
)
from certification.hoeffding import (
    hoeffding_upper_bound,
    hoeffding_two_sided_interval,
)
from certification.reference_bounds import (
    reference_upper_bound,
    reference_two_sided_interval,
)

ALL_CERTIFICATION_METHODS = [
    "naive_eb",
    "bonferroni_eb",
    "hoeffding",
    "reference_t",
]


def evaluate_certification_method(
    method: str,
    observed_mean: float,
    observed_variance: float,
    B: int,
    delta: float,
    K: int,
    target_threshold: float,
) -> Dict[str, Any]:
    """
    Compute upper bound, two-sided lower/upper bounds, certification decision, and bound slack
    for a specified certification method:
    - 'naive_eb': Method 1 (Naive in-sample empirical Bernstein)
    - 'bonferroni_eb': Method 2 (Bonferroni empirical Bernstein with delta_K = delta / K)
    - 'hoeffding': Method 3 (Hoeffding upper bound)
    - 'reference_t': Method 4 (Student-t non-guaranteed reference bound)
    - 'reference_z': Method 4 variant (Gaussian non-guaranteed reference bound)
    """
    if method == "naive_eb":
        ub = empirical_bernstein_upper_bound(observed_mean, observed_variance, B, delta)
        lb_2s, ub_2s = empirical_bernstein_two_sided_interval(observed_mean, observed_variance, B, delta)
    elif method == "bonferroni_eb":
        ub = bonferroni_empirical_bernstein_upper_bound(observed_mean, observed_variance, B, delta, K)
        lb_2s, ub_2s = bonferroni_empirical_bernstein_two_sided_interval(
            observed_mean, observed_variance, B, delta, K
        )
    elif method == "hoeffding":
        ub = hoeffding_upper_bound(observed_mean, B, delta)
        lb_2s, ub_2s = hoeffding_two_sided_interval(observed_mean, B, delta)
    elif method in ("reference_t", "student_t_ref"):
        ub = reference_upper_bound(observed_mean, observed_variance, B, delta, distribution="student_t")
        lb_2s, ub_2s = reference_two_sided_interval(
            observed_mean, observed_variance, B, delta, distribution="student_t"
        )
    elif method in ("reference_z", "gaussian_ref"):
        ub = reference_upper_bound(observed_mean, observed_variance, B, delta, distribution="gaussian")
        lb_2s, ub_2s = reference_two_sided_interval(
            observed_mean, observed_variance, B, delta, distribution="gaussian"
        )
    else:
        raise ValueError(f"Unknown certification method: {method}")

    tau = float(target_threshold)
    certified = bool(ub <= tau)
    bound_slack = float(ub - observed_mean)
    se = float(np.sqrt(max(0.0, observed_variance) / float(B)))
    if se > 1e-15:
        bound_slack_se = float(bound_slack / se)
    else:
        bound_slack_se = np.nan

    return {
        "certification_method": method,
        "delta": float(delta),
        "observed_mean": float(observed_mean),
        "observed_variance": float(observed_variance),
        "upper_confidence_bound": float(ub),
        "lower_confidence_bound": float(lb_2s),
        "two_sided_upper_bound": float(ub_2s),
        "target_threshold": tau,
        "certified": certified,
        "bound_slack": bound_slack,
        "bound_slack_standard_errors": bound_slack_se,
    }


def evaluate_all_certification_methods(
    observed_mean: float,
    observed_variance: float,
    B: int,
    delta: float,
    K: int,
    target_threshold: float,
    methods: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate all four required certification methods on a candidate's summary statistics.
    """
    if methods is None:
        methods = ALL_CERTIFICATION_METHODS
    return [
        evaluate_certification_method(
            m, observed_mean, observed_variance, B, delta, K, target_threshold
        )
        for m in methods
    ]
