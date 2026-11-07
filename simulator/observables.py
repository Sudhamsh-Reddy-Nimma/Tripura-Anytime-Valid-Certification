"""
Observable Evaluation (MaxCut Value and Normalized Loss) for AMFS 2.0 (Stage 1).

Mathematical definitions (Section 3 of Specification):
- Bitstring x in {0,1}^n represents a vertex partition.
- Spin z_i = (-1)^{x_i} in {+1, -1}.
- Cut value: C(x) = sum_{(i,j) in E} (1 - z_i z_j) / 2.
- Normalized loss: L(x) = 1 - C(x) / m, where m = |E| is the number of edges.
  Therefore 0 <= L(x) <= 1, and lower loss is better.
"""

from typing import Union, Sequence, Any, Tuple
import numpy as np
from simulator.statevector import extract_graph_info, get_cut_diagonal


def bitstring_to_index(bitstring: Union[str, Sequence[int], int], n: int) -> int:
    """
    Convert a bitstring (e.g., '0101', [0, 1, 0, 1], or integer index) to its basis index in [0, 2^n - 1].
    """
    if isinstance(bitstring, (int, np.integer)):
        idx = int(bitstring)
        if idx < 0 or idx >= (1 << n):
            raise ValueError(f"Basis index {idx} out of range for n={n}.")
        return idx
    elif isinstance(bitstring, str):
        if len(bitstring) != n or any(c not in "01" for c in bitstring):
            raise ValueError(f"Bitstring '{bitstring}' must be a binary string of length n={n}.")
        return int(bitstring, 2)
    elif isinstance(bitstring, (list, tuple, np.ndarray)):
        if len(bitstring) != n:
            raise ValueError(f"Bitstring sequence length {len(bitstring)} does not match n={n}.")
        idx = 0
        for b in bitstring:
            if b not in (0, 1):
                raise ValueError(f"Bitstring elements must be 0 or 1, got {b}.")
            idx = (idx << 1) | int(b)
        return idx
    else:
        raise TypeError(f"Unsupported bitstring type: {type(bitstring)}")


def index_to_bitstring(index: int, n: int) -> str:
    """
    Format an integer basis state index in [0, 2^n - 1] as a binary string of length n.
    """
    return format(int(index), f"0{n}b")


def evaluate_cut(bitstring: Union[str, Sequence[int], int], graph: Any) -> float:
    """
    Calculate the unweighted MaxCut value C(x) for a bitstring x on `graph`.
    """
    n, edges = extract_graph_info(graph)
    idx = bitstring_to_index(bitstring, n)
    cuts = get_cut_diagonal(graph)
    return float(cuts[idx])


def evaluate_normalized_loss(bitstring: Union[str, Sequence[int], int], graph: Any) -> float:
    """
    Calculate the normalized MaxCut loss L(x) = 1 - C(x) / m for a bitstring x on `graph`.
    """
    n, edges = extract_graph_info(graph)
    m = len(edges)
    if m == 0:
        raise ValueError("Cannot compute normalized loss on a graph with 0 edges.")
    cut_val = evaluate_cut(bitstring, graph)
    return float(1.0 - (cut_val / m))


def compute_sample_statistics(losses: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate sample mean, unbiased sample variance (ddof=1), and standard error
    from an array of B shot losses.
    """
    arr = np.asarray(losses, dtype=np.float64)
    B = arr.size
    if B < 1:
        raise ValueError("At least 1 shot is required to compute sample mean.")
    mean = float(np.mean(arr))
    if B < 2:
        return mean, 0.0, 0.0
    var = float(np.var(arr, ddof=1))
    # Numerical guard against tiny negative zero
    var = max(0.0, var)
    se = float(np.sqrt(var / B))
    return mean, var, se
