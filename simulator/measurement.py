"""
Finite-Shot Computational-Basis Measurement Pipeline for AMFS 2.0 (Stage 1 & Stage 2).

STRICT GROUND-TRUTH SEPARATION (Stage 2 Section 14):
This module computes ONLY finite-shot sampled outcomes and sample statistics.
It NEVER calculates, caches, or returns exact ground-truth expectations.
"""

from typing import Dict, Any, List, Sequence
import numpy as np
from simulator.statevector import extract_graph_info, get_cut_diagonal, get_probabilities
from simulator.qaoa import simulate_statevector
from simulator.observables import index_to_bitstring, compute_sample_statistics


def sample_bitstrings(
    probabilities: np.ndarray,
    B: int,
    rng: np.random.Generator,
    return_indices: bool = False,
) -> List[str]:
    """
    Draw B independent computational-basis bitstring measurements from `probabilities`
    using the provided NumPy Generator `rng`.
    """
    if B <= 0:
        raise ValueError(f"Shot budget B must be a positive integer, got {B}.")
    probs = np.asarray(probabilities, dtype=np.float64)
    dim = probs.size
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Probability array length {dim} must be a positive power of 2.")
    if np.any(probs < -1e-12) or not np.isclose(np.sum(probs), 1.0, atol=1e-8):
        raise ValueError("Probabilities must be non-negative and sum to 1.")
    probs = np.maximum(probs, 0.0)
    probs /= np.sum(probs)

    n = int(np.log2(dim))
    sampled_indices = rng.choice(dim, size=int(B), replace=True, p=probs)
    if return_indices:
        return sampled_indices
    return [index_to_bitstring(idx, n) for idx in sampled_indices]


def sample_loss_counts(
    graph: Any,
    theta: Sequence[float],
    B: int,
    rng: np.random.Generator,
    p: int = None,
    build_histogram: bool = True,
) -> Dict[str, Any]:
    """
    Simulate QAOA state for (graph, theta), draw B independent measurement shots,
    and return strictly finite-shot statistics. Never computes or attaches exact expectations.
    """
    if B < 2:
        raise ValueError(f"Shot budget B must be >= 2 for unbiased variance estimation, got {B}.")
    n, edges = extract_graph_info(graph)
    m = len(edges)
    if m == 0:
        raise ValueError("Graph must have at least one edge.")

    state = simulate_statevector(graph, theta, p=p)
    probs = get_probabilities(state)
    cuts_diag = get_cut_diagonal(graph)
    losses_diag = 1.0 - (cuts_diag / float(m))

    sampled_indices = sample_bitstrings(probs, B, rng, return_indices=True)
    shot_cuts = cuts_diag[sampled_indices]
    shot_losses = losses_diag[sampled_indices]

    mean_cut = float(np.mean(shot_cuts))
    mean_loss, var_loss, se_loss = compute_sample_statistics(shot_losses)
    min_loss = float(np.min(shot_losses))
    max_loss = float(np.max(shot_losses))

    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    unique_bitstring_count = int(unique_indices.size)

    histogram = []
    if build_histogram:
        for idx, cnt in zip(unique_indices, counts):
            bstr = index_to_bitstring(int(idx), n)
            c_val = float(cuts_diag[int(idx)])
            l_val = float(losses_diag[int(idx)])
            histogram.append({
                "basis_index": int(idx),
                "bitstring": bstr,
                "count": int(cnt),
                "shot_budget": int(B),
                "observed_frequency": float(cnt / float(B)),
                "cut_value": c_val,
                "normalized_loss": l_val,
            })

    return {
        "shot_budget": int(B),
        "observed_mean_cut": mean_cut,
        "observed_mean_loss": mean_loss,
        "observed_variance_loss": var_loss,
        "observed_standard_error": se_loss,
        "minimum_observed_loss": min_loss,
        "maximum_observed_loss": max_loss,
        "unique_bitstring_count": unique_bitstring_count,
        "sampled_indices": sampled_indices,
        "shot_cuts": shot_cuts,
        "shot_losses": shot_losses,
        "unique_indices": unique_indices,
        "unique_counts": counts,
        "histogram": histogram,
    }
