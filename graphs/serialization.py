"""
Graph Serialization and Schema Handling for `graphs.csv` (Section 12.A).
"""

from typing import List, Dict, Any
import json
import os
import pandas as pd

GRAPHS_CSV_COLUMNS = [
    "graph_id",
    "graph_hash",
    "graph_size_n",
    "edge_count_m",
    "graph_degree",
    "graph_is_connected",
    "graph_isomorphism_class_id",
    "graph_split",
    "graph_generation_seed",
    "graph_sampling_method",
    "graph_adjacency_encoding",
]


def encode_graph_adjacency(edges: List[tuple]) -> str:
    canonical = [[int(min(u, v)), int(max(u, v))] for u, v in sorted(edges)]
    return json.dumps(canonical, separators=(",", ":"))


def decode_graph_adjacency(encoding: str) -> List[tuple]:
    raw = json.loads(encoding)
    return [(int(u), int(v)) for u, v in raw]


def serialize_graphs_to_dataframe(graphs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for g in graphs:
        rows.append({
            "graph_id": g["graph_id"],
            "graph_hash": g["graph_hash"],
            "graph_size_n": int(g["graph_size_n"]),
            "edge_count_m": int(g["edge_count_m"]),
            "graph_degree": int(g["graph_degree"]),
            "graph_is_connected": bool(g["graph_is_connected"]),
            "graph_isomorphism_class_id": g["graph_isomorphism_class_id"],
            "graph_split": g["graph_split"],
            "graph_generation_seed": str(g["graph_generation_seed"]),
            "graph_sampling_method": g["graph_sampling_method"],
            "graph_adjacency_encoding": encode_graph_adjacency(g["edges"]),
        })
    return pd.DataFrame(rows, columns=GRAPHS_CSV_COLUMNS)


def save_graphs_csv(graphs: List[Dict[str, Any]], csv_path: str, append: bool = False) -> pd.DataFrame:
    df = serialize_graphs_to_dataframe(graphs)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if append and os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    return df
