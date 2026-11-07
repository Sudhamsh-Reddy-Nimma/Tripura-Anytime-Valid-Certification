"""
Offline Ground-Truth Exact Analysis Layer for AMFS 2.0 (Stage 1).

CRITICAL ARCHITECTURAL RULE (Section 5 & Section 21):
This module calculates exact statevector expectations and loss distributions
 strictly for post-experiment offline ground-truth analysis.
It MUST NEVER be imported by `optimization/` or `certification/` modules.
"""

from typing import Dict, Any, Sequence
import numpy as np
from simulator.statevector import extract_graph_info, get_cut_diagonal, get_probabilities
from simulator.qaoa import simulate_statevector


def exact_expectation(
    graph: Any,
    theta: Sequence[float],
    p: int = None,
    target_threshold: float = None,
) -> Dict[str, Any]:
    """
    Calculate exact ground-truth QAOA quantities from the statevector distribution:
    - exact_expected_cut: E[C(x)] = <psi(theta)| H_C |psi(theta)>
    - exact_expected_loss: mu(theta) = E[L(x)] = 1 - E[C(x)] / m
    - exact_loss_variance: Var(L(x)) = sum_x P(x) (L(x) - mu(theta))^2
    - exact_cost_hamiltonian_expectation: <H_C>
    - exact_target_success_probability: P_{x ~ P_theta}(L(x) <= tau) on an individual shot
    """
    n, edges = extract_graph_info(graph)
    m = len(edges)
    if m == 0:
        raise ValueError("Graph must have at least one edge.")

    state = simulate_statevector(graph, theta, p=p)
    probs = get_probabilities(state)
    cuts = get_cut_diagonal(graph)
    losses = 1.0 - (cuts / float(m))

    expected_cut = float(np.dot(probs, cuts))
    expected_loss = float(np.dot(probs, losses))
    loss_variance = float(np.dot(probs, (losses - expected_loss) ** 2))
    loss_variance = max(0.0, loss_variance)

    if target_threshold is not None:
        tau = float(target_threshold)
        shot_success_prob = float(np.sum(probs[losses <= tau + 1e-15]))
    else:
        shot_success_prob = np.nan

    return {
        "exact_expected_cut": expected_cut,
        "exact_cost_hamiltonian_expectation": expected_cut,
        "exact_expected_loss": expected_loss,
        "exact_loss_variance": loss_variance,
        "exact_target_success_probability": shot_success_prob,
    }
