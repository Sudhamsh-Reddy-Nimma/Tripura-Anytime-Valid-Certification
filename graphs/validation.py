"""
Graph Validation & Exact Canonical Isomorphism Checks for AMFS 2.0 (Stage 1).
"""

from typing import List, Tuple, Dict, Any
import hashlib
import numpy as np


def build_adjacency_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"Vertex count n must be positive, got {n}.")
    adj = np.zeros((n, n), dtype=np.int8)
    seen = set()
    for u, v in edges:
        u_i, v_i = int(u), int(v)
        if u_i < 0 or u_i >= n or v_i < 0 or v_i >= n:
            raise ValueError(f"Edge ({u_i}, {v_i}) out of range for n={n}.")
        if u_i == v_i:
            raise ValueError(f"Self-loop ({u_i}, {v_i}) detected.")
        e = (min(u_i, v_i), max(u_i, v_i))
        if e in seen:
            raise ValueError(f"Duplicate edge {e} detected.")
        seen.add(e)
        adj[u_i, v_i] = 1
        adj[v_i, u_i] = 1
    return adj


def is_connected(n: int, edges: List[Tuple[int, int]]) -> bool:
    if n <= 1:
        return True
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[int(u)].append(int(v))
        adj_list[int(v)].append(int(u))

    visited = [False] * n
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        curr = queue.pop(0)
        for nxt in adj_list[curr]:
            if not visited[nxt]:
                visited[nxt] = True
                count += 1
                queue.append(nxt)
    return count == n


def is_d_regular(n: int, edges: List[Tuple[int, int]], degree: int = 3) -> bool:
    try:
        adj = build_adjacency_matrix(n, edges)
    except ValueError:
        return False
    degrees = np.sum(adj, axis=1)
    return bool(np.all(degrees == degree))


def validate_3_regular_connected_graph(graph: Dict[str, Any]) -> bool:
    n = int(graph["n"])
    edges = graph["edges"]
    if len(edges) != (3 * n) // 2:
        return False
    return is_d_regular(n, edges, degree=3) and is_connected(n, edges)


def compute_graph_hash(n: int, edges: List[Tuple[int, int]]) -> str:
    canonical_edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in edges)
    payload = f"n={n}|edges={canonical_edges}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def compute_isomorphism_class_id(n: int, edges: List[Tuple[int, int]]) -> str:
    adj = build_adjacency_matrix(n, edges).astype(np.int64)

    def refine_colors(adj_mat: np.ndarray, init_colors: Tuple[int, ...]) -> Tuple[int, ...]:
        colors = list(init_colors)
        for _ in range(n):
            sigs = []
            for v in range(n):
                nbr_c = tuple(sorted(colors[u] for u in range(n) if adj_mat[v, u]))
                sigs.append((colors[v], nbr_c))
            unique_sigs = sorted(set(sigs))
            sig_map = {s: idx for idx, s in enumerate(unique_sigs)}
            new_colors = [sig_map[s] for s in sigs]
            if new_colors == colors:
                break
            colors = new_colors
        return tuple(colors)

    A2 = adj @ adj
    A3 = A2 @ adj
    A4 = A2 @ A2
    A5 = A4 @ adj
    v_invs = []
    for i in range(n):
        inv = (
            int(A2[i, i]),
            int(A3[i, i]),
            int(A4[i, i]),
            int(A5[i, i]),
            tuple(sorted(int(x) for x in A2[i])),
            tuple(sorted(int(x) for x in A3[i])),
            tuple(sorted(int(x) for x in A4[i])),
        )
        v_invs.append(inv)
    unique_invs = sorted(set(v_invs))
    inv_map = {val: idx for idx, val in enumerate(unique_invs)}
    base_colors = refine_colors(adj, tuple(inv_map[val] for val in v_invs))

    def search_canonical_orderings(colors: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        if len(set(colors)) == n:
            order = tuple(np.argsort(colors))
            return [order]
        counts: Dict[int, List[int]] = {}
        for v, c in enumerate(colors):
            counts.setdefault(c, []).append(v)
        non_singleton = [(len(vs), c, vs) for c, vs in counts.items() if len(vs) > 1]
        non_singleton.sort(key=lambda x: (x[0], x[1]))
        _, target_c, target_vertices = non_singleton[0]

        leaf_orders = []
        for v_star in target_vertices:
            ind_colors = tuple(
                (colors[v] * 2 + (1 if v == v_star else 0)) for v in range(n)
            )
            refined = refine_colors(adj, ind_colors)
            leaf_orders.extend(search_canonical_orderings(refined))
        return leaf_orders

    candidate_orders = search_canonical_orderings(base_colors)
    best_upper_bits = None
    for order in candidate_orders:
        perm_adj = adj[np.ix_(order, order)]
        upper = tuple(int(perm_adj[i, j]) for i in range(n) for j in range(i + 1, n))
        if best_upper_bits is None or upper > best_upper_bits:
            best_upper_bits = upper

    bit_str = "".join(str(b) for b in best_upper_bits)
    digest = hashlib.sha256(f"n={n}|canon={bit_str}".encode("utf-8")).hexdigest()[:12]
    return f"ISO_N{n}_{digest}"
