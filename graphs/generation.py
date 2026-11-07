"""
Connected Random 3-Regular Graph Generation for AMFS 2.0 (Stage 1).
"""

from typing import List, Tuple, Dict, Any, Set
import numpy as np
from graphs.validation import (
    validate_3_regular_connected_graph,
    compute_graph_hash,
    compute_isomorphism_class_id,
)


def sample_random_3_regular_graph(
    n: int,
    rng: np.random.Generator,
    max_attempts: int = 10000,
) -> List[Tuple[int, int]]:
    if n < 4 or (3 * n) % 2 != 0:
        raise ValueError(f"3-regular graph requires even n >= 4, got n={n}.")

    stubs_template = np.repeat(np.arange(n, dtype=np.int32), 3)
    for _ in range(max_attempts):
        stubs = rng.permutation(stubs_template)
        pairs = stubs.reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue
        u = np.minimum(pairs[:, 0], pairs[:, 1])
        v = np.maximum(pairs[:, 0], pairs[:, 1])
        edges = list(zip(u.tolist(), v.tolist()))
        if len(set(edges)) != len(edges):
            continue
        edges.sort()
        candidate = {"n": n, "edges": edges}
        if validate_3_regular_connected_graph(candidate):
            return edges

    raise RuntimeError(f"Failed to sample a connected 3-regular graph on n={n} in {max_attempts} attempts.")


def generate_graph_ensemble(
    n: int,
    count: int,
    split_name: str,
    seed_seq: np.random.SeedSequence,
    existing_isomorphism_ids: Set[str] = None,
    id_prefix: str = None,
) -> List[Dict[str, Any]]:
    if existing_isomorphism_ids is None:
        existing_isomorphism_ids = set()
    else:
        existing_isomorphism_ids = set(existing_isomorphism_ids)

    if id_prefix is None:
        prefix = "PILOT" if split_name.lower() == "pilot" else "CONF"
    else:
        prefix = id_prefix

    graphs: List[Dict[str, Any]] = []
    seen_iso: Set[str] = set(existing_isomorphism_ids)
    attempt = 0

    while len(graphs) < count:
        attempt += 1
        if attempt > 5000:
            raise RuntimeError(f"Could not find {count} distinct non-isomorphic 3-regular graphs for n={n}.")
        child_ss = seed_seq.spawn(1)[0]
        rng = np.random.default_rng(child_ss)
        edges = sample_random_3_regular_graph(n, rng)
        iso_id = compute_isomorphism_class_id(n, edges)
        if iso_id in seen_iso:
            continue
        seen_iso.add(iso_id)
        idx = len(graphs) + 1
        graph_id = f"G_{prefix}_N{n}_{idx:02d}"
        g_hash = compute_graph_hash(n, edges)
        seed_int = int(child_ss.generate_state(1, dtype=np.uint64)[0])
        graphs.append({
            "graph_id": graph_id,
            "graph_hash": g_hash,
            "graph_size_n": int(n),
            "edge_count_m": len(edges),
            "graph_degree": 3,
            "graph_is_connected": True,
            "graph_isomorphism_class_id": iso_id,
            "graph_split": split_name,
            "graph_generation_seed": str(seed_int),
            "graph_sampling_method": "pairing_model_uniform_rejection_non_isomorphic",
            "n": int(n),
            "edges": edges,
        })

    return graphs
