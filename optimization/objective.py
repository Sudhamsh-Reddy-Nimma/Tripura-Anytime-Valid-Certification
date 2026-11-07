"""
Stateful Finite-Shot QAOA Objective Wrapper for AMFS 2.0 (Stage 1).
"""

import time
from typing import Dict, Any
import numpy as np
from simulator.measurement import sample_loss_counts
from optimization.trajectory import OptimizationTrajectory


class FiniteShotQAOAObjective:
    def __init__(
        self,
        graph: Dict[str, Any],
        qaoa_depth: int,
        shot_budget: int,
        shot_seed_seq: np.random.SeedSequence,
        trajectory: OptimizationTrajectory,
        max_evaluations: int = 100,
        store_full_shots: bool = False,
        store_histogram: bool = False,
    ):
        self.graph = graph
        self.qaoa_depth = int(qaoa_depth)
        self.shot_budget = int(shot_budget)
        self.shot_seed_seq = shot_seed_seq
        self.trajectory = trajectory
        self.max_evaluations = int(max_evaluations)
        self.store_full_shots = bool(store_full_shots)
        self.store_histogram = bool(store_histogram)

    def evaluate_single(self, theta: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        child_ss = self.shot_seed_seq.spawn(1)[0]
        eval_seed = int(child_ss.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(child_ss)

        meas = sample_loss_counts(
            self.graph,
            theta,
            self.shot_budget,
            rng,
            p=self.qaoa_depth,
            build_histogram=self.store_histogram,
        )
        dt = time.perf_counter() - t0
        rec = self.trajectory.record_evaluation(
            theta=np.asarray(theta, dtype=np.float64),
            measurement_result=meas,
            runtime_seconds=dt,
            eval_seed=eval_seed,
            store_full_shots=self.store_full_shots,
            store_histogram=self.store_histogram,
            graph_edges_m=len(self.graph["edges"]),
            graph_n=int(self.graph["n"]),
        )
        return rec

    def __call__(self, theta: np.ndarray) -> float:
        rec = self.evaluate_single(theta)
        return float(rec["observed_mean_loss"])
