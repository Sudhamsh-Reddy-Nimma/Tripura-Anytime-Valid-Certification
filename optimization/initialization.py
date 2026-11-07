"""
QAOA Parameter Initialization Policies for 3-Regular MaxCut (Section 6).
"""

from typing import Tuple, Dict
import numpy as np

WURTZ_LOVE_3REGULAR_ANGLES: Dict[int, Dict[str, Tuple[float, ...]]] = {
    1: {
        "gamma": (0.616,),
        "beta": (0.393,),
    },
    2: {
        "gamma": (0.488, 0.898),
        "beta": (0.555, 0.293),
    },
}


def generate_initial_parameters(
    p: int,
    rng: np.random.Generator,
    jitter_half_width: float = 0.05,
    method: str = "wurtz_love_3regular_jitter",
) -> np.ndarray:
    if p not in WURTZ_LOVE_3REGULAR_ANGLES and method.startswith("wurtz_love"):
        raise ValueError(f"Wurtz-Love fixed angles only configured for p in {{1, 2}}, got p={p}.")

    if method == "wurtz_love_3regular_jitter":
        base_gamma = np.array(WURTZ_LOVE_3REGULAR_ANGLES[p]["gamma"], dtype=np.float64)
        base_beta = np.array(WURTZ_LOVE_3REGULAR_ANGLES[p]["beta"], dtype=np.float64)
        if jitter_half_width > 0.0:
            j_gamma = rng.uniform(-jitter_half_width, jitter_half_width, size=p)
            j_beta = rng.uniform(-jitter_half_width, jitter_half_width, size=p)
        else:
            j_gamma = np.zeros(p, dtype=np.float64)
            j_beta = np.zeros(p, dtype=np.float64)
        gammas = np.clip(base_gamma + j_gamma, 0.0, 2.0 * np.pi)
        betas = np.clip(base_beta + j_beta, 0.0, np.pi)
        return np.concatenate([gammas, betas])
    elif method == "uniform_random":
        gammas = rng.uniform(0.0, 2.0 * np.pi, size=p)
        betas = rng.uniform(0.0, np.pi, size=p)
        return np.concatenate([gammas, betas])
    else:
        raise ValueError(f"Unknown initialization method '{method}'.")
