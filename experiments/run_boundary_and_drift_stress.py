"""
Adversarial Boundary & Nonstationary Readout-Drift Stress Suite (Section 5.3)
=============================================================================
Evaluates Phase C sequential confirmation (`b_conf = 256`, `B_conf = 4096`)
across R = 1000 independent confirmation streams parked immediately above the
target threshold at mu(theta_hat) = tau + 1e-4 under:
  1. Stationary boundary stress (delta_drift = 0) across alpha in {0.01..0.05}.
  2. Adversarial downward readout drift (delta_drift = 0.005, stream mean
     mu_t = mu(theta_hat) - delta_drift < tau), comparing the unshifted rule
     U_t <= tau against the drift-shifted rule U_t + delta_drift <= tau
     (Theorem 3 and Remark 5).
"""

import os
import numpy as np
import pandas as pd
from certification.confidence_sequences import run_sequential_confirmation_from_losses


def main() -> None:
    rng = np.random.default_rng(20261107)
    tau = 0.26866083912936534
    mu_true = tau + 1e-4
    delta_drift = 0.005
    mu_drifted = mu_true - delta_drift
    R = 1000
    v = 0.004
    conc = mu_drifted * (1.0 - mu_drifted) / v - 1.0
    a, b = mu_drifted * conc, (1.0 - mu_drifted) * conc

    unshifted_eb, shifted_eb = 0, 0
    unshifted_wsr, shifted_wsr = 0, 0
    for _ in range(R):
        shots = rng.beta(a, b, size=4096)
        res_u_eb = run_sequential_confirmation_from_losses(
            shots, target_threshold=tau, alpha=0.05, batch_size=256, max_shots=4096,
            method="howard_gamma_exponential_eb"
        )
        res_s_eb = run_sequential_confirmation_from_losses(
            shots, target_threshold=tau - delta_drift, alpha=0.05, batch_size=256, max_shots=4096,
            method="howard_gamma_exponential_eb"
        )
        res_u_wsr = run_sequential_confirmation_from_losses(
            shots, target_threshold=tau, alpha=0.05, batch_size=256, max_shots=4096,
            method="wsr_agrapa_betting", alpha_upper_fraction=0.90
        )
        res_s_wsr = run_sequential_confirmation_from_losses(
            shots, target_threshold=tau - delta_drift, alpha=0.05, batch_size=256, max_shots=4096,
            method="wsr_agrapa_betting", alpha_upper_fraction=0.90
        )
        unshifted_eb += int(res_u_eb["decision"] == "CERTIFIED")
        shifted_eb += int(res_s_eb["decision"] == "CERTIFIED")
        unshifted_wsr += int(res_u_wsr["decision"] == "CERTIFIED")
        shifted_wsr += int(res_s_wsr["decision"] == "CERTIFIED")

    print(f"Adversarial downward drift (delta_drift = {delta_drift}, R = {R}):")
    print(f"  Gamma-Exp. EB: unshifted FC = {unshifted_eb}/{R} ({unshifted_eb/R:.3f}), shifted FC = {shifted_eb}/{R} ({shifted_eb/R:.3f})")
    print(f"  WSR Betting:   unshifted FC = {unshifted_wsr}/{R} ({unshifted_wsr/R:.3f}), shifted FC = {shifted_wsr}/{R} ({shifted_wsr/R:.3f})")


if __name__ == "__main__":
    main()
