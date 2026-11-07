"""
QAOA Circuit Simulation & Parameter Convention Handling for AMFS 2.0 (Stage 1).

Parameter vector convention:
    theta = (gamma_1, ..., gamma_p, beta_1, ..., beta_p) of length 2p.
Gate application order for each layer ell = 1, ..., p:
    1. Apply cost unitary U_C(gamma_ell) = exp(-i * gamma_ell * H_C)
    2. Apply mixer unitary U_M(beta_ell) = exp(-i * beta_ell * H_M)
Full QAOA state:
    |psi(theta)> = prod_{ell=1}^p (U_M(beta_ell) U_C(gamma_ell)) |+>^{otimes n}
"""

from typing import Tuple, Any, Sequence
import numpy as np
from simulator.statevector import (
    extract_graph_info,
    initialize_state,
    apply_cost_layer,
    apply_mixer_layer,
)


def unpack_theta(theta: Sequence[float], p: int = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Validate and unpack parameter vector theta = (gamma_1, ..., gamma_p, beta_1, ..., beta_p).
    """
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or arr.size % 2 != 0:
        raise ValueError(f"Parameter vector theta must be 1D of even positive length 2p, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Parameter vector theta contains NaN or infinite values.")
    inferred_p = arr.size // 2
    if p is not None and int(p) != inferred_p:
        raise ValueError(f"Specified depth p={p} does not match len(theta)=2*{inferred_p}.")
    gammas = arr[:inferred_p]
    betas = arr[inferred_p:]
    return gammas, betas, inferred_p


def simulate_statevector(graph: Any, theta: Sequence[float], p: int = None) -> np.ndarray:
    """
    Simulate the full p-layer QAOA statevector for unweighted MaxCut on `graph`.
    """
    n, edges = extract_graph_info(graph)
    if len(edges) == 0:
        raise ValueError("Graph must have at least one edge for MaxCut QAOA.")
    gammas, betas, depth_p = unpack_theta(theta, p)
    state = initialize_state(n)
    for ell in range(depth_p):
        state = apply_cost_layer(state, graph, gammas[ell])
        state = apply_mixer_layer(state, betas[ell])
    return state
