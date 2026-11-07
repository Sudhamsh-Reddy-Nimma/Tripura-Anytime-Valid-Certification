"""
Custom NumPy Statevector Engine for AMFS 2.0 (Stage 1 & Stage 2).

Mathematical Convention:
- State space: 2^n complex128 vector in computational basis |x_0 x_1 ... x_{n-1}>,
  where x_q in {0, 1} and basis index k = sum_{q=0}^{n-1} x_q * 2^{n-1-q} (big-endian).
- Initial state: |+>^{otimes n} = (1 / sqrt(2^n)) * sum_{x in {0,1}^n} |x>.
- Cost Hamiltonian: H_C = sum_{(i,j) in E} (I - Z_i Z_j) / 2.
  For basis state |x>, H_C |x> = C(x) |x>, where C(x) is the unweighted MaxCut value.
- Cost Unitary: U_C(gamma) = exp(-i * gamma * H_C), diagonal with entries exp(-i * gamma * C(x)).
- Mixer Hamiltonian: H_M = sum_{i=0}^{n-1} X_i.
- Mixer Unitary: U_M(beta) = exp(-i * beta * H_M) = prod_{i=0}^{n-1} exp(-i * beta * X_i),
  where exp(-i * beta * X) = [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]].
"""

from typing import Tuple, List, Dict, Any
import numpy as np


def _build_hamming_matrix(k: int) -> np.ndarray:
    idx = np.arange(1 << k, dtype=np.uint32)
    xm = idx[:, None] ^ idx[None, :]
    pop = np.zeros_like(xm, dtype=np.int8)
    for b in range(k):
        pop += ((xm >> b) & 1).astype(np.int8)
    return pop


_POP3 = _build_hamming_matrix(3)
_POP4 = _build_hamming_matrix(4)


def _fast_block_rx(k: int, pop_mat: np.ndarray, beta: float) -> np.ndarray:
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    powers = np.array([(c ** (k - p)) * (s ** p) for p in range(k + 1)], dtype=np.complex128)
    return powers[pop_mat]


def extract_graph_info(graph: Any) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Extract number of vertices n and list of undirected edges (u, v) from a graph representation.
    """
    if isinstance(graph, dict):
        if "n" not in graph or "edges" not in graph:
            raise ValueError("Graph dict must contain 'n' and 'edges' keys.")
        n = int(graph["n"])
        edges = [(int(u), int(v)) for u, v in graph["edges"]]
    elif hasattr(graph, "n") and hasattr(graph, "edges"):
        n = int(graph.n)
        edges = [(int(u), int(v)) for u, v in graph.edges]
    elif isinstance(graph, np.ndarray):
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("Adjacency matrix must be square 2D array.")
        n = int(graph.shape[0])
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if graph[i, j] != 0:
                    edges.append((i, j))
    else:
        raise TypeError(f"Unsupported graph representation of type {type(graph)}.")

    if n <= 0:
        raise ValueError(f"Number of vertices n must be positive, got {n}.")
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError(f"Edge ({u}, {v}) out of bounds for n={n}.")
        if u == v:
            raise ValueError(f"Self-loop ({u}, {v}) is not permitted in unweighted MaxCut.")
    return n, edges


_CUT_DIAGONAL_CACHE: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], np.ndarray] = {}


def get_cut_diagonal(graph: Any) -> np.ndarray:
    """
    Precompute or retrieve the cached diagonal vector C(x) of length 2^n
    for all computational basis states x in {0, 1}^n.
    """
    if isinstance(graph, dict) and "_cut_diagonal" in graph and graph["_cut_diagonal"] is not None:
        return graph["_cut_diagonal"]
    if hasattr(graph, "_cut_diagonal") and getattr(graph, "_cut_diagonal") is not None:
        return getattr(graph, "_cut_diagonal")

    n, edges = extract_graph_info(graph)
    canonical_edges = tuple(sorted((min(u, v), max(u, v)) for u, v in edges))
    cache_key = (n, canonical_edges)

    if cache_key in _CUT_DIAGONAL_CACHE:
        diag = _CUT_DIAGONAL_CACHE[cache_key]
    else:
        dim = 1 << n
        indices = np.arange(dim, dtype=np.uint32)
        cuts = np.zeros(dim, dtype=np.float64)
        for u, v in canonical_edges:
            bit_u = (indices >> (n - 1 - u)) & 1
            bit_v = (indices >> (n - 1 - v)) & 1
            cuts += (bit_u ^ bit_v).astype(np.float64)
        diag = cuts
        if len(_CUT_DIAGONAL_CACHE) < 512:
            _CUT_DIAGONAL_CACHE[cache_key] = diag

    if isinstance(graph, dict):
        graph["_cut_diagonal"] = diag
    elif hasattr(graph, "_cut_diagonal"):
        setattr(graph, "_cut_diagonal", diag)
    return diag


def initialize_state(n: int) -> np.ndarray:
    """
    Initialize the uniform superposition state |+>^{otimes n} of dimension 2^n.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"Number of qubits n must be a positive integer, got {n}.")
    dim = 1 << int(n)
    amp = 1.0 / np.sqrt(dim)
    return np.full(dim, amp, dtype=np.complex128)


def apply_cost_layer(state: np.ndarray, graph: Any, gamma: float) -> np.ndarray:
    """
    Apply the MaxCut cost unitary U_C(gamma) = exp(-i * gamma * H_C) to the statevector.
    Exploits the diagonal structure of H_C in the computational basis.
    """
    if not np.isscalar(gamma) and (isinstance(gamma, np.ndarray) and gamma.ndim > 0):
        raise ValueError("gamma must be a scalar float.")
    gamma_val = float(gamma)
    cuts = get_cut_diagonal(graph)
    if state.shape != cuts.shape:
        raise ValueError(f"State dimension {state.shape} does not match graph state space {cuts.shape}.")
    phases = np.exp(-1j * gamma_val * cuts)
    return state * phases


def apply_mixer_layer(state: np.ndarray, beta: float) -> np.ndarray:
    """
    Apply the transverse-field mixer unitary U_M(beta) = prod_{j=0}^{n-1} exp(-i * beta * X_j).
    Uses exact block-Kronecker L1-cache-resident BLAS contractions for n=12 (4+4+4) and n=14 (4+4+3+3),
    and 3D contiguous tensor reshape for general n.
    """
    if not np.isscalar(beta) and (isinstance(beta, np.ndarray) and beta.ndim > 0):
        raise ValueError("beta must be a scalar float.")
    beta_val = float(beta)
    dim = state.shape[0]
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"State length {dim} must be a positive power of 2.")
    n = int(np.log2(dim))

    if n == 12:
        u4 = _fast_block_rx(4, _POP4, beta_val)
        p = u4 @ state.reshape(16, 256)
        p = p.reshape(256, 16) @ u4
        return (u4 @ p.reshape(16, 16, 16)).reshape(dim)
    elif n == 14:
        u4 = _fast_block_rx(4, _POP4, beta_val)
        u3 = _fast_block_rx(3, _POP3, beta_val)
        p = u4 @ state.reshape(16, 1024)
        p = p.reshape(2048, 8) @ u3
        p = (u4 @ p.reshape(16, 16, 64)).reshape(256, 8, 8)
        return (u3 @ p).reshape(dim)

    c = np.cos(beta_val)
    s = -1j * np.sin(beta_val)

    psi = state.copy()
    for axis in range(n):
        left = 1 << axis
        right = 1 << (n - 1 - axis)
        v = psi.reshape(left, 2, right)
        v0 = v[:, 0, :].copy()
        v1 = v[:, 1, :]
        v[:, 0, :] = c * v0 + s * v1
        v[:, 1, :] = s * v0 + c * v1

    return psi


def get_probabilities(state: np.ndarray) -> np.ndarray:
    """
    Calculate computational-basis measurement probabilities P(x) = |<x|psi>|^2.
    """
    probs = (state.real ** 2) + (state.imag ** 2)
    total = float(np.sum(probs))
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"Statevector is not normalized (sum of probabilities = {total}).")
    probs = np.maximum(probs, 0.0)
    probs /= np.sum(probs)
    return probs
