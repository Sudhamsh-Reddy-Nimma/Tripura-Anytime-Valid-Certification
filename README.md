# Tripura: Discover-Freeze-Certify for Anytime-Valid Certification of Finite-Shot Variational Quantum Optimization

Official simulation, sequential-certification, figure-generation, and manuscript repository for **Tripura**.

## Repository Structure

```text
Tripura/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── stage2.yaml                        # 100-graph selection-bias study configuration
│   └── stage3_corrected.yaml              # 500-graph end-to-end study configuration (Modes U, H, C + baselines)
├── simulator/                             # Exact statevector QAOA MaxCut simulator & Z2-symmetric CDF sampler
├── graphs/                                # Non-isomorphic 3-regular graph ensemble generator
├── optimization/
│   ├── initialization.py                  # Wurtz-Love warm start + F_0 jitter initialization
│   ├── cobyla_wrapper.py                  # Selection-bias study (Phase 2) COBYLA wrapper
│   ├── adaptive_allocation.py             # SimultaneousAlphaSchedule, LUCB racing, and safe CS elimination
│   └── adaptive_cobyla.py                 # Phase A discovery engine supporting Modes U, H (simplex CRN), and C
├── certification/
│   ├── confidence_sequences.py            # Howard et al. (2021) EB and Waudby-Smith & Ramdas (2024) aGRAPA betting CS
│   ├── empirical_bernstein.py             # Fixed and time-uniform empirical-Bernstein boundaries
│   ├── hoeffding.py                       # Hoeffding reference bounds
│   ├── decisions.py                       # Phase C sequential stopping rules (CERTIFIED / REJECTED / INCONCLUSIVE)
│   └── stage3_cs.py                       # Multi-attempt and calibration utilities
├── analysis/
│   ├── ground_truth.py                    # Exact QAOA expectation and variance evaluation
│   ├── stage2_analysis.py                 # Selection-bias and in-sample vs. frozen certification summaries
│   └── stage3_analysis.py                 # End-to-end 500-graph summaries, Clopper-Pearson bounds, and Holm tests
├── experiments/
│   ├── run_stage2.py                      # Reproduces the 100-graph selection-bias campaign (300 trajectories/cell)
│   ├── run_stage3.py                      # Reproduces the 500-graph end-to-end campaign (Modes U, H, C + baselines)
│   ├── run_stage3_minimal_validation.py   # 50-graph Phase A safety audit (pre-allocated C1/C2 vs. Holm/pairwise)
│   ├── run_uniform_seq_betting_gap.py     # Closes the confirmation-engine gap: Uniform + sequential with betting CS
│   └── run_boundary_and_drift_stress.py   # 1,000-stream boundary (mu = tau + 1e-4) and downward drift (delta = 0.005) tests
├── scripts/
│   ├── generate_figures.py                # Generates all publication PDF and 600-DPI PNG figures
│   ├── paper.mplstyle                     # Matplotlib publication stylesheet
│   └── make_tables.py                     # Programmatically generates all 16 LaTeX tables from CSV/JSON outputs
├── results_stage2_raw/                    # Raw 100-graph selection-bias logs
├── results_stage2_processed/              # Processed selection-bias and in-sample certification summaries
├── results_stage3_corrected/              # Raw, processed, validation, and metadata archives for the 500-graph study
│   └── processed/uniform_disc_2560_seq_conf_betting_500graphs.csv
└── paper/
    ├── tripura.tex                        # Self-contained Quantum manuscript (4 inline tables + 4 figures)
    ├── tripura.pdf                        # Compiled manuscript PDF
    ├── figures/                           # All vector PDF figures (fig1..fig6, figS1)
    └── tables_full/                       # Full 16-table LaTeX suite (main + supplementary)
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the 500-graph Confirmation-Engine Attribution Gap experiment (`Uniform + sequential` with Betting CS):**
   ```bash
   python3 experiments/run_uniform_seq_betting_gap.py
   ```
3. **Run the Boundary & Nonstationary Drift stress tests:**
   ```bash
   python3 experiments/run_boundary_and_drift_stress.py
   ```
4. **Regenerate all publication figures and LaTeX tables:**
   ```bash
   python3 scripts/generate_figures.py
   python3 scripts/make_tables.py
   ```
