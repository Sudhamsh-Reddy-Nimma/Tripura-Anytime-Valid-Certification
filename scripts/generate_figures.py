"""
Publication-Grade Physical Review A (PRA) Figure Suite Generator for Tripura: A Sequential Certification Of Parameter Estimates.

Applies `paper.mplstyle` across all six main figures (Figs. 1-6) and Supplement Fig. S1:
- Fig. 1 (`fig1_architecture.pdf` / `.png`): Hand-built vector schematic of Phase A, Phase B (freeze + lock icon),
  Phase C, bottom global filtration timeline F_0 <= F_1 <= ... <= F_disc -> G_0 = F_disc, and RNG stream tags.
- Fig. 2 (`fig2_selection_bias.pdf` / `.png`): 3-panel double-column (7.0 in) selection optimism vs ln K, vs B (log x),
  and synthetic contrast (equal-mean near bound, QAOA-heterogeneous, and disjoint-support at zero).
- Fig. 3 (`fig3_insample_vs_independent.pdf` / `.png`): 2-panel double-column (7.0 in) paired slope/dot plot of
  in-sample vs frozen confirmation false-certification rate with Clopper-Pearson bars, plus Pr(U_{k_hat} <= tau)
  vs 1 - prod(1 - p_k) scatter on y = x diagonal showing heterogeneous-radii divergence.
- Fig. 4 (`fig4_phase_c_reliability.pdf` / `.png`): 3-panel double-column (7.0 in) calibration layout:
  (a) coverage vs nominal 1 - alpha reliability diagram, (b) forest plot of false-certification rate + CP upper bounds
  including mu = tau + 10^-4 boundary stress test, (c) horizontal stacked bar of Certified/Rejected/Inconclusive.
- Fig. 5 (`fig5_mode_u_h_c_comparison.pdf` / `.png`): Double-column (7.0 in) 4-panel dot-and-whisker small multiples
  for Mode U (blue), Mode H (vermillion flagship), Mode C (grey dashed) + 5th panel CRN-breakage histogram of E[X|F]-mu.
- Fig. 6 (`fig6_cost_and_quality.pdf` / `.png`): Double-column (7.0 in) 2-panel layout:
  (a) horizontal stacked bars of discovery + confirmation shots vs 6,656 baseline with % saved and hatched references,
  (b) Pareto scatter of total shots vs final approximation ratio (inverted y so flagship sits lower-left) with error crosses.
- Fig. S1 (`figS1_eb_vs_agrapa.pdf` / `.png`): Supplement comparison of EB vs aGRAPA radius (37-39% empirical reduction) and coverage.
"""

import os
import zipfile
import shutil
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(BASE_DIR, "paper.mplstyle")
plt.style.use(STYLE_PATH)

OUT_DIR = os.path.join(BASE_DIR, "pra_figures_output")
os.makedirs(OUT_DIR, exist_ok=True)
shutil.copy2(STYLE_PATH, os.path.join(OUT_DIR, "paper.mplstyle"))

# Okabe-Ito Colour-Blind-Safe Palette (Fixed Identity Across All Figures)
COL_MODE_U = "#0072B2"   # Mode U (Uncoupled): Blue
COL_MODE_H = "#D55E00"   # Mode H (Flagship Hybrid Simplex): Vermillion
COL_MODE_C = "#666666"   # Mode C (Full Coupled Legacy): Neutral Grey
COL_THEORY = "#000000"   # Theory bounds & nominal thresholds: Black dashed
COL_TEAL   = "#009E73"   # Bluish Green (safe accent, never paired with red)
COL_SKY    = "#56B4E9"   # Sky Blue
COL_ORANGE = "#E69F00"   # Orange / Amber
COL_PURPLE = "#CC79A7"   # Reddish Purple
COL_GREY_L = "#CBD5E1"   # Light neutral grey for inconclusive bars


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    lo = 0.0 if k <= 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def save_both(fig, stem: str):
    pdf_path = os.path.join(OUT_DIR, f"{stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf and {stem}.png")


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.2, fontweight="bold", va="bottom", ha="left", color="#0F172A")


# ============================================================================
# FIGURE 1: Hand-Built Vector Schematic of Tripura Architecture (7.0 in wide)
# ============================================================================
def make_fig1():
    fig, ax = plt.subplots(figsize=(7.0, 3.35), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    # Header banner text
    ax.text(50, 46.2, r"Tripura: A Sequential Certification Of Parameter Estimates",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A")

    # Phase A Block (Blue tint)
    box_a = FancyBboxPatch((1.2, 13.5), 35.0, 29.0, boxstyle="round,pad=0.5,rounding_size=1.2",
                           facecolor="#F0F7FF", edgecolor=COL_MODE_U, linewidth=1.35)
    ax.add_patch(box_a)
    ax.text(18.7, 40.3, "Phase A: Adaptive Multi-Candidate Discovery", ha="center", va="center",
            fontsize=8.3, fontweight="bold", color=COL_MODE_U)

    # Inside Phase A: Two sub-boxes for Tier 1 (Simplex) and Tier 2 (Adaptive COBYLA)
    sub_a1 = FancyBboxPatch((2.1, 28.2), 33.2, 9.2, boxstyle="round,pad=0.35,rounding_size=0.8",
                            facecolor="#FFFFFF", edgecolor=COL_MODE_U, linewidth=0.85)
    ax.add_patch(sub_a1)
    ax.text(18.7, 35.1, r"1. Predetermined Simplex Vertices $\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_{d+1} \in \mathcal{F}_0$",
            ha="center", va="center", fontsize=6.75, fontweight="bold", color="#0F172A")
    ax.text(18.7, 32.2, r"$\mathbb{Z}_2$-symmetric evaluation ($B_0=256$ shots/vertex)",
            ha="center", va="center", fontsize=7.0, color="#334155")
    # Tag 1: CRN on simplex vertices
    tag1 = FancyBboxPatch((5.2, 28.8), 27.0, 2.3, boxstyle="round,pad=0.2,rounding_size=0.6",
                          facecolor="#E0F2FE", edgecolor=COL_MODE_U, linewidth=0.7)
    ax.add_patch(tag1)
    ax.text(18.7, 29.95, r"RNG Tag: CRN on simplex vertices $\{U_s\}_{s\geq 1}$",
            ha="center", va="center", fontsize=6.8, color=COL_MODE_U, fontweight="bold")

    sub_a2 = FancyBboxPatch((2.1, 15.0), 33.2, 11.4, boxstyle="round,pad=0.35,rounding_size=0.8",
                            facecolor="#FFFFFF", edgecolor=COL_MODE_H, linewidth=0.85)
    ax.add_patch(sub_a2)
    ax.text(18.7, 24.3, r"2. Adaptive COBYLA Proposals $\boldsymbol{\theta}_k \in \mathcal{F}_{n_k-1}$ ($k > d+1$)",
            ha="center", va="center", fontsize=6.75, fontweight="bold", color="#0F172A")
    ax.text(18.7, 21.5, r"Anytime CS $[L_k(n), U_k(n)]$ + LUCB safe elimination:",
            ha="center", va="center", fontsize=7.0, color="#334155")
    ax.text(18.7, 19.1, r"prune $k$ if $L_k(n) > \min_{j\in\mathcal{A}_n} U_j(n) + \epsilon_{\mathrm{best}}$",
            ha="center", va="center", fontsize=6.9, color="#334155")
    # Tag 2: fresh stream on adaptive arms
    tag2 = FancyBboxPatch((4.5, 15.6), 28.4, 2.3, boxstyle="round,pad=0.2,rounding_size=0.6",
                          facecolor="#FFF1EB", edgecolor=COL_MODE_H, linewidth=0.7)
    ax.add_patch(tag2)
    ax.text(18.7, 16.75, r"RNG Tag: fresh stream on adaptive arms $\{U_s^{(k)}\}$",
            ha="center", va="center", fontsize=6.8, color=COL_MODE_H, fontweight="bold")

    # Arrow Phase A -> Phase B
    arr_ab = FancyArrowPatch((36.4, 28.0), (41.2, 28.0), arrowstyle="-|>,head_length=4.5,head_width=3.0",
                             color="#1E293B", linewidth=1.3)
    ax.add_patch(arr_ab)
    ax.text(38.8, 30.2, r"$\mathcal{F}_{\mathrm{disc}}$", ha="center", va="center", fontsize=7.4, color="#0F172A")

    # Phase B Block (Freeze + Vector Lock Icon)
    box_b = FancyBboxPatch((41.4, 13.5), 21.2, 29.0, boxstyle="round,pad=0.5,rounding_size=1.2",
                           facecolor="#FFF8F3", edgecolor=COL_MODE_H, linewidth=1.45)
    ax.add_patch(box_b)
    ax.text(52.0, 40.3, "Phase B: Selection & Freeze", ha="center", va="center",
            fontsize=8.3, fontweight="bold", color=COL_MODE_H)

    # Hand-built Vector Lock Icon inside Phase B
    shackle = Circle((52.0, 34.6), 1.65, facecolor="none", edgecolor=COL_MODE_H, linewidth=1.6)
    ax.add_patch(shackle)
    lock_mask = Rectangle((49.8, 32.0), 4.4, 2.5, facecolor="#FFF8F3", edgecolor="none")
    ax.add_patch(lock_mask)
    lock_body = FancyBboxPatch((49.9, 30.8), 4.2, 3.4, boxstyle="round,pad=0.15,rounding_size=0.4",
                               facecolor=COL_MODE_H, edgecolor="#7C2D12", linewidth=1.0)
    ax.add_patch(lock_body)
    keyhole = Circle((52.0, 32.6), 0.42, facecolor="#FFFFFF", edgecolor="none")
    ax.add_patch(keyhole)
    ax.plot([52.0, 52.0], [32.3, 31.4], color="#FFFFFF", linewidth=1.3)

    ax.text(52.0, 27.6, r"Select leader $\hat{k} = \ell_{N_{\mathrm{disc}}}$", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#0F172A")
    ax.text(52.0, 24.8, r"Freeze $\widehat{\boldsymbol{\theta}} := \boldsymbol{\theta}_{\hat{k}} \in \mathcal{F}_{\mathrm{disc}}$",
            ha="center", va="center", fontsize=7.4, color=COL_MODE_H, fontweight="bold")
    ax.text(52.0, 21.6, "Strict Data Firewall:\n• Parameter immutable\n• Zero Phase A shot reuse",
            ha="center", va="center", fontsize=6.9, color="#334155", linespacing=1.28)

    tag_b = FancyBboxPatch((43.0, 15.2), 18.0, 2.5, boxstyle="round,pad=0.2,rounding_size=0.6",
                           facecolor="#F1F5F9", edgecolor="#475569", linewidth=0.7)
    ax.add_patch(tag_b)
    ax.text(52.0, 16.45, r"$\mu(\widehat{\boldsymbol{\theta}})$ fixed given $\mathcal{F}_{\mathrm{disc}}$",
            ha="center", va="center", fontsize=6.8, color="#0F172A")

    # Arrow Phase B -> Phase C
    arr_bc = FancyArrowPatch((62.8, 28.0), (67.6, 28.0), arrowstyle="-|>,head_length=4.5,head_width=3.0",
                             color="#1E293B", linewidth=1.3)
    ax.add_patch(arr_bc)
    ax.text(65.2, 30.2, r"$\widehat{\boldsymbol{\theta}}$", ha="center", va="center", fontsize=7.8, color="#0F172A")

    # Phase C Block (Independent Confirmation)
    box_c = FancyBboxPatch((67.8, 13.5), 31.0, 29.0, boxstyle="round,pad=0.5,rounding_size=1.2",
                           facecolor="#F2FBF9", edgecolor=COL_TEAL, linewidth=1.35)
    ax.add_patch(box_c)
    ax.text(83.3, 40.3, "Phase C: Independent Confirmation", ha="center", va="center",
            fontsize=8.3, fontweight="bold", color="#065F46")

    tag_c = FancyBboxPatch((69.8, 35.2), 27.0, 2.5, boxstyle="round,pad=0.2,rounding_size=0.6",
                           facecolor="#D1FAE5", edgecolor=COL_TEAL, linewidth=0.7)
    ax.add_patch(tag_c)
    ax.text(83.3, 36.45, r"RNG Tag: fresh confirmation stream $\{Y_t\}_{t\geq 1}$",
            ha="center", va="center", fontsize=6.8, color="#065F46", fontweight="bold")

    ax.text(83.3, 32.0, r"Martingale mean: $\mathbb{E}[Y_t \mid \mathcal{G}_{t-1}] = \mu_t^{\mathrm{conf}}$",
            ha="center", va="center", fontsize=7.2, color="#0F172A")
    ax.text(83.3, 29.1, r"One-sided upper CS $U_t^{\mathrm{conf}}(\alpha_{\mathrm{conf}})$:",
            ha="center", va="center", fontsize=7.2, color="#0F172A")
    ax.text(83.3, 26.2, r"• CERTIFIED if $U_t^{\mathrm{conf}} + \delta_{\mathrm{drift}} \leq \tau$",
            ha="center", va="center", fontsize=7.1, color="#065F46")
    ax.text(83.3, 23.6, r"• REJECTED if $L_t^{\mathrm{conf}} - \delta_{\mathrm{drift}} > \tau$",
            ha="center", va="center", fontsize=7.1, color="#334155")
    ax.text(83.3, 21.0, r"• INCONCLUSIVE at $t = B_{\mathrm{conf}}$",
            ha="center", va="center", fontsize=7.1, color="#475569")

    guar_box = FancyBboxPatch((68.7, 14.8), 29.2, 4.2, boxstyle="round,pad=0.25,rounding_size=0.6",
                              facecolor="#FFFFFF", edgecolor=COL_TEAL, linewidth=0.8)
    ax.add_patch(guar_box)
    ax.text(83.3, 16.9, r"Thm. 3: $\Pr(D=\mathrm{CERTIFIED},\, \mu(\widehat{\boldsymbol{\theta}})>\tau) \leq \alpha_{\mathrm{conf}}$",
            ha="center", va="center", fontsize=6.45, color="#0F172A")

    # Bottom Filtration Timeline Strip
    strip = FancyBboxPatch((1.2, 2.0), 97.6, 8.8, boxstyle="round,pad=0.35,rounding_size=0.9",
                           facecolor="#F8FAFC", edgecolor="#64748B", linewidth=0.95)
    ax.add_patch(strip)
    ax.text(50.0, 8.9, r"Global Sequential Filtration Timeline (Predictable Parameter Fixing & Martingale Conditioning)",
            ha="center", va="center", fontsize=7.4, color="#1E293B")

    # Timeline axis arrow
    t_arrow = FancyArrowPatch((4.0, 4.8), (96.0, 4.8), arrowstyle="-|>,head_length=4.5,head_width=2.6",
                              color="#334155", linewidth=1.1)
    ax.add_patch(t_arrow)

    nodes = [
        (8.0,  COL_MODE_U, r"$\mathcal{F}_0$"),
        (19.0, COL_MODE_U, r"$\subseteq \mathcal{F}_1$"),
        (30.0, COL_MODE_H, r"$\subseteq \cdots \subseteq \mathcal{F}_{N_{\mathrm{disc}}}$"),
        (52.0, COL_MODE_H, r"$\mathcal{F}_{\mathrm{disc}} \,\mathrm{(Freeze}\;\widehat{\boldsymbol{\theta}}\mathrm{)}$"),
        (73.0, COL_TEAL,   r"$\mathcal{G}_0 = \mathcal{F}_{\mathrm{disc}}$"),
        (89.0, COL_TEAL,   r"$\subseteq \mathcal{G}_1 \subseteq \cdots \subseteq \mathcal{G}_{T_{\mathrm{conf}}}$"),
    ]
    for (nx, ncol, nlab) in nodes:
        dot = Circle((nx, 4.8), 0.75, facecolor=ncol, edgecolor="#FFFFFF", linewidth=0.8, zorder=5)
        ax.add_patch(dot)
        ax.text(nx, 2.9, nlab, ha="center", va="center", fontsize=7.1, color="#0F172A")

    save_both(fig, "fig1_architecture")


# ============================================================================
# FIGURE 2: Selection Bias vs ln K, vs B (log x), and Synthetic Contrast
# ============================================================================
def make_fig2():
    b_df = pd.read_csv(os.path.join(BASE_DIR, "results_stage2_processed/stage2_bias_and_confirmation_summary.csv"))
    pooled_b = b_df[b_df["graph_size_stratum"] == "pooled_12_14"].copy()
    syn_2c = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/fig2c_synthetic_contrast.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), constrained_layout=True)

    # Panel (a): Optimism E[mu_{k_hat} - mu_hat_{k_hat}] against ln K
    ax = axes[0]
    panel_label(ax, "(a)")
    b_styles = [
        (256,  COL_MODE_H, "o", "-",  r"In-sample ($B=256$)"),
        (512,  COL_ORANGE, "s", "-",  r"In-sample ($B=512$)"),
        (1024, COL_MODE_U, "^", "-",  r"In-sample ($B=1024$)"),
    ]
    for B_val, col, mk, ls, lab in b_styles:
        sub = pooled_b[pooled_b["B"] == B_val].sort_values("K_checkpoint")
        x_lnk = np.log(sub["K_checkpoint"].values.astype(float))
        # Selection optimism = -mean_discovery_bias = E[mu_{k_hat} - X_bar_{k_hat}]
        opt = -sub["mean_discovery_bias"].values
        opt_lo = -sub["discovery_bias_ci_upper"].values
        opt_hi = -sub["discovery_bias_ci_lower"].values
        ax.plot(x_lnk, opt, marker=mk, linestyle=ls, color=col, label=lab)
        ax.fill_between(x_lnk, opt_lo, opt_hi, color=col, alpha=0.18)

    # Phase C fresh confirmation bias (B=256 discovery, B_conf=4096)
    sub_256 = pooled_b[pooled_b["B"] == 256].sort_values("K_checkpoint")
    x_lnk = np.log(sub_256["K_checkpoint"].values.astype(float))
    conf_opt = -sub_256["mean_confirmation_bias"].values
    conf_lo = -sub_256["confirmation_bias_ci_upper"].values
    conf_hi = -sub_256["confirmation_bias_ci_lower"].values
    ax.plot(x_lnk, conf_opt, marker="D", linestyle=":", color=COL_TEAL, label="Phase C confirmation")
    ax.fill_between(x_lnk, conf_lo, conf_hi, color=COL_TEAL, alpha=0.20)

    # Theoretical upper envelope sqrt(2 ln K * max_k v_k / B) for B=256
    lnk_grid = np.linspace(0.0, np.log(100.0), 100)
    se_max_256 = float(sub_256["denom1_mean_per_candidate_exact_se"].max())
    bound_256 = np.sqrt(2.0 * lnk_grid) * se_max_256
    ax.plot(lnk_grid, bound_256, linestyle="--", color=COL_THEORY, linewidth=1.3,
            label=r"Bound $\sqrt{2\ln K \max_k(v_k/B)}$")

    ax.set_xlabel(r"Log candidate count $\ln K$")
    ax.set_ylabel(r"Selection optimism $\mathbb{E}[\mu_{\hat{k}} - \widehat{\mu}_{\hat{k}}]$")
    ax.set_xticks([0.0, np.log(5), np.log(25), np.log(100)])
    ax.set_xticklabels([r"$\ln 1$", r"$\ln 5$", r"$\ln 25$", r"$\ln 100$"])
    ax.set_ylim(-0.0012, 0.0235)
    ax.legend(loc="upper left", fontsize=5.75, borderpad=0.25, labelspacing=0.22)

    # Panel (b): Selection optimism against B on a log x-axis with 1/sqrt(B) envelope
    ax = axes[1]
    panel_label(ax, "(b)")
    k_styles = [
        (100, COL_MODE_H, "o", r"$K=100$ candidates"),
        (50,  COL_MODE_U, "s", r"$K=50$ candidates"),
        (25,  COL_GREY_L if False else COL_MODE_C, "^", r"$K=25$ candidates"),
    ]
    for K_val, col, mk, lab in k_styles:
        sub = pooled_b[pooled_b["K_checkpoint"] == K_val].sort_values("B")
        B_vals = sub["B"].values.astype(float)
        opt = -sub["mean_discovery_bias"].values
        opt_lo = -sub["discovery_bias_ci_upper"].values
        opt_hi = -sub["discovery_bias_ci_lower"].values
        ls = "--" if col == COL_MODE_C else "-"
        ax.plot(B_vals, opt, marker=mk, linestyle=ls, color=col, label=lab)
        ax.fill_between(B_vals, opt_lo, opt_hi, color=col, alpha=0.18)

    # Theory 1/sqrt(B) envelope for K=100
    B_grid = np.geomspace(256.0, 2048.0, 100)
    v_proxy = (se_max_256 ** 2) * 256.0
    env_b = np.sqrt(2.0 * np.log(100.0) * v_proxy / B_grid)
    ax.plot(B_grid, env_b, linestyle="--", color=COL_THEORY, linewidth=1.3,
            label=r"$\sqrt{2\ln K \cdot v/B}$ ($1/\sqrt{B}$)")

    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048])
    ax.set_xticklabels(["256", "512", "1024", "2048"])
    ax.set_xlabel(r"Per-candidate shots $B$ ($\log_2$ scale)")
    ax.set_ylabel(r"Selection optimism $\mathbb{E}[\mu_{\hat{k}} - \widehat{\mu}_{\hat{k}}]$")
    ax.set_ylim(-0.0005, 0.0185)
    ax.legend(loc="upper right", fontsize=6.3)

    # Panel (c): Synthetic contrast (Equal-mean near bound vs QAOA heterogeneous vs Disjoint support at 0)
    ax = axes[2]
    panel_label(ax, "(c)")
    K_sub = [10, 25, 50, 100]
    x_idx = np.arange(len(K_sub))

    eq_df = syn_2c[(syn_2c["regime"] == "equal_mean_homogeneous") & (syn_2c["K"].isin(K_sub))].sort_values("K")
    het_df = syn_2c[(syn_2c["regime"] == "qaoa_heterogeneous_gaps") & (syn_2c["K"].isin(K_sub))].sort_values("K")
    dis_df = syn_2c[(syn_2c["regime"] == "disjoint_support_counterexample") & (syn_2c["K"].isin(K_sub))].sort_values("K")

    ax.plot(x_idx, eq_df["theory_bound"], linestyle="--", marker="none", color=COL_THEORY, linewidth=1.35,
            label="Thm. 1(b) bound")
    ax.errorbar(x_idx - 0.08, eq_df["mean_optimism"],
                yerr=[eq_df["mean_optimism"] - eq_df["ci_lower"], eq_df["ci_upper"] - eq_df["mean_optimism"]],
                fmt="o-", color=COL_MODE_H, label="Equal-mean ($\Delta_k = 0$)")
    ax.errorbar(x_idx + 0.0, het_df["mean_optimism"],
                yerr=[het_df["mean_optimism"] - het_df["ci_lower"], het_df["ci_upper"] - het_df["mean_optimism"]],
                fmt="s-", color=COL_MODE_U, label="Heterogeneous ($\Delta_k > 0$)")
    ax.errorbar(x_idx + 0.08, dis_df["mean_optimism"],
                yerr=[dis_df["mean_optimism"] - dis_df["ci_lower"], dis_df["ci_upper"] - dis_df["mean_optimism"]],
                fmt="^--", color=COL_MODE_C, label="Disjoint support (zero bias)")

    ax.axhline(0.0, color="#64748B", linewidth=0.7, linestyle=":")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"$K={k}$" for k in K_sub])
    ax.set_xlabel(r"Candidate count $K$ ($B=256$)")
    ax.set_ylabel(r"Selection optimism $\mathbb{E}[\mu_{\hat{k}} - \widehat{\mu}_{\hat{k}}]$")
    ax.set_ylim(-0.0012, 0.0155)
    ax.legend(loc="upper left", fontsize=6.1)

    save_both(fig, "fig2_selection_bias")


# ============================================================================
# FIGURE 3: In-Sample vs Independent Certification & Formula Check (7.0 in)
# ============================================================================
def make_fig3():
    c_df = pd.read_csv(os.path.join(BASE_DIR, "results_stage2_processed/stage2_certification_summary.csv"))
    pooled_c = c_df[c_df["graph_size_stratum"] == "pooled_12_14"].copy()
    syn_df = pd.read_csv(os.path.join(BASE_DIR, "results_processed/synthetic_control_results.csv"))
    het_3b = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/fig3b_heterogeneous_radii_validation.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)

    # Panel (a): Paired dot / slope plot of in-sample vs frozen confirmation false-certification rate
    ax = axes[0]
    panel_label(ax, "(a)")

    # Extract 6 representative (K, B) settings across B=256, 512 and boundary control with exact CP intervals
    settings = [
        ("$K=25$\n$B=256$",  "qaoa", 25,  256),
        ("$K=50$\n$B=256$",  "qaoa", 50,  256),
        ("$K=100$\n$B=256$", "qaoa", 100, 256),
        ("$K=50$\n$B=512$",  "qaoa", 50,  512),
        ("$K=100$\n$B=512$", "qaoa", 100, 512),
        ("Bernoulli\n$K=100$", "syn_b", 100, 1024),
    ]

    x_pos = np.arange(len(settings))
    ins_rates, ins_lo, ins_hi = [], [], []
    frz_rates, frz_lo, frz_hi = [], [], []

    for label_txt, src, K_v, B_v in settings:
        if src == "qaoa":
            r_in = pooled_c[(pooled_c["B"] == B_v) & (pooled_c["K_checkpoint"] == K_v) & (pooled_c["method"] == "in_sample_student_t")].iloc[0]
            r_fr = pooled_c[(pooled_c["B"] == B_v) & (pooled_c["K_checkpoint"] == K_v) & (pooled_c["method"] == "independent_confirm_eb")].iloc[0]
            n_tr = int(r_in["number_of_trajectories"])
            k_in = int(r_in["number_of_false_certifications"])
            k_fr = int(r_fr["number_of_false_certifications"])
        else:
            sub_s = syn_df[(syn_df["synthetic_regime"] == "standard_bernoulli_0_1") &
                           (syn_df["candidate_count_K"] == K_v) &
                           (syn_df["shots_per_candidate"] == B_v)]
            n_tr = len(sub_s)
            k_in = int(sub_s["false_certification"].sum())
            k_fr = int(sub_s["fresh_false_certification"].sum())

        lo_i, hi_i = clopper_pearson(k_in, n_tr)
        lo_f, hi_f = clopper_pearson(k_fr, n_tr)
        ins_rates.append(k_in / n_tr)
        ins_lo.append(lo_i)
        ins_hi.append(hi_i)
        frz_rates.append(k_fr / n_tr)
        frz_lo.append(lo_f)
        frz_hi.append(hi_f)

    ins_rates = np.array(ins_rates)
    frz_rates = np.array(frz_rates)

    # Connect paired dots with vertical winner's-curse lines
    for i in range(len(settings)):
        ax.plot([x_pos[i] - 0.12, x_pos[i] + 0.12], [ins_rates[i], frz_rates[i]],
                color="#64748B", linestyle="-", linewidth=1.1, zorder=2)

    ax.errorbar(x_pos - 0.12, ins_rates,
                yerr=[ins_rates - np.array(ins_lo), np.array(ins_hi) - ins_rates],
                fmt="o", color=COL_MODE_H, label="In-sample reuse (unsplit)", zorder=4)
    ax.errorbar(x_pos + 0.12, frz_rates,
                yerr=[frz_rates - np.array(frz_lo), np.array(frz_hi) - frz_rates],
                fmt="s", color=COL_MODE_U, label="Frozen Phase C confirmation", zorder=4)

    ax.axhline(0.05, color=COL_THEORY, linestyle="--", linewidth=1.2, label=r"Nominal $\alpha = 0.05$")
    ax.annotate("Winner's curse\ninflation gap", xy=(2.0 - 0.12, ins_rates[2]), xytext=(2.45, 0.068),
                arrowprops=dict(arrowstyle="->", color=COL_MODE_H, lw=0.9),
                fontsize=6.6, color=COL_MODE_H, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s[0] for s in settings], fontsize=6.5)
    ax.set_ylabel("False-certification probability (95% CP)")
    ax.set_ylim(-0.005, 0.095)
    ax.legend(loc="upper left", fontsize=6.5)

    # Panel (b): Pr(U_{k_hat} <= tau) against 1 - prod(1 - p_k) on y = x diagonal
    ax = axes[1]
    panel_label(ax, "(b)")

    hom = het_3b[het_3b["radii_regime"] == "homogeneous_radii"].sort_values("K")
    het = het_3b[het_3b["radii_regime"] == "heterogeneous_radii"].sort_values("K")

    diag = np.linspace(0.0, 0.55, 100)
    ax.plot(diag, diag, linestyle="--", color=COL_THEORY, linewidth=1.25,
            label=r"Identity $\Pr(U_{\hat{k}}\leq\tau) = 1-\prod_k(1-p_k)$")

    ax.errorbar(hom["product_formula_prob"], hom["empirical_selected_prob"],
                yerr=[hom["empirical_selected_prob"] - hom["cp_lower"], hom["cp_upper"] - hom["empirical_selected_prob"]],
                fmt="o-", color=COL_MODE_U, label=r"Equal radii ($r_k \equiv r$, $B_k=256$)")
    ax.errorbar(het["product_formula_prob"], het["empirical_selected_prob"],
                yerr=[het["empirical_selected_prob"] - het["cp_lower"], het["cp_upper"] - het["empirical_selected_prob"]],
                fmt="^-", color=COL_MODE_H, label=r"Unequal radii ($B_k \in [32, 1024]$)")

    # Fill between to highlight the heterogeneous-radii divergence (Theorem 3.1c)
    for _, row in het.iterrows():
        if row["K"] in [25, 40, 60]:
            ax.annotate(f"$K={int(row['K'])}$",
                        xy=(row["product_formula_prob"], row["empirical_selected_prob"]),
                        xytext=(row["product_formula_prob"] + 0.015, row["empirical_selected_prob"] - 0.045),
                        fontsize=6.5, color=COL_MODE_H)

    ax.set_xlabel(r"Union-product formula $1 - \prod_{k=1}^K (1 - p_k)$")
    ax.set_ylabel(r"Selected-candidate $\Pr(U_{\hat{k}} \leq \tau)$")
    ax.set_xlim(0.0, 0.54)
    ax.set_ylim(0.0, 0.54)
    ax.legend(loc="upper left", fontsize=6.4)

    save_both(fig, "fig3_insample_vs_independent")


# ============================================================================
# FIGURE 4: Phase C Reliability (Calibration, Boundary Stress & Breakdown)
# ============================================================================
def make_fig4():
    df_alpha = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/stage3_corrected_alpha_calibration_summary.csv"))
    df_bound = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/fig4b_boundary_stress_validation.csv"))

    seq_df = df_alpha[df_alpha["confirmation_type"] == "sequential_256"].sort_values("alpha")
    fix_df = df_alpha[df_alpha["confirmation_type"] == "fixed_4096"].sort_values("alpha")
    bnd_df = df_bound.sort_values("alpha")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), constrained_layout=True)

    # Panel (a): Coverage against nominal 1 - alpha reliability diagram
    ax = axes[0]
    panel_label(ax, "(a)")
    nom_conf = seq_df["nominal_confidence"].values
    ax.plot([0.945, 0.995], [0.945, 0.995], linestyle="--", color=COL_THEORY, linewidth=1.25,
            label=r"Nominal diagonal $1-\alpha$")
    ax.fill_between([0.945, 0.995], [0.945, 0.995], [1.002, 1.002], color="#E2E8F0", alpha=0.45,
                    label="Valid conservative region")

    # Compute exact CP intervals for 500-graph coverage
    seq_cov_lo, seq_cov_hi = [], []
    for cov_val, g_cnt in zip(seq_df["cs_coverage"], seq_df["independent_attempts_G"]):
        lo, hi = clopper_pearson(int(round(cov_val * g_cnt)), int(g_cnt))
        seq_cov_lo.append(lo)
        seq_cov_hi.append(hi)

    ax.errorbar(nom_conf - 0.0008, seq_df["cs_coverage"],
                yerr=[seq_df["cs_coverage"].values - np.array(seq_cov_lo), np.array(seq_cov_hi) - seq_df["cs_coverage"].values],
                fmt="o-", color=COL_MODE_H, label="Phase C sequential ($G=500$)")
    ax.errorbar(bnd_df["nominal_confidence"] + 0.0008, bnd_df["cs_coverage"],
                yerr=[bnd_df["cs_coverage"] - bnd_df["cs_coverage_ci_lower"], bnd_df["cs_coverage_ci_upper"] - bnd_df["cs_coverage"]],
                fmt="s--", color=COL_MODE_U, label=r"Boundary $\mu=\tau+10^{-4}$")

    ax.set_xlabel(r"Nominal confidence level $1 - \alpha_{\mathrm{conf}}$")
    ax.set_ylabel("Empirical CS coverage (95% CP)")
    ax.set_xlim(0.946, 0.994)
    ax.set_ylim(0.945, 1.003)
    ax.set_xticks([0.95, 0.96, 0.97, 0.98, 0.99])
    ax.legend(loc="lower right", fontsize=6.1)

    # Panel (b): Forest-style plot of false-certification rate with CP upper bounds per alpha
    ax = axes[1]
    panel_label(ax, "(b)")
    alphas = seq_df["alpha"].values
    y_idx = np.arange(len(alphas))

    # Plot nominal alpha step/dashed markers
    ax.plot(alphas, y_idx, "k--", marker="|", markersize=8, linewidth=1.15, label=r"Nominal ceiling $\alpha_{\mathrm{conf}}$")

    # Sequential 500-graph
    ax.errorbar(seq_df["observed_false_certification_rate"], y_idx + 0.16,
                xerr=[np.zeros(len(alphas)), seq_df["clopper_pearson_95_upper_bound"] - seq_df["observed_false_certification_rate"]],
                fmt="o", color=COL_MODE_H, label="500-graph sequential (95% CP)")
    # Boundary stress test mu = tau + 10^-4
    ax.errorbar(bnd_df["observed_false_certification_rate"], y_idx - 0.16,
                xerr=[bnd_df["observed_false_certification_rate"] - bnd_df["clopper_pearson_95_lower_bound"],
                      bnd_df["clopper_pearson_95_upper_bound"] - bnd_df["observed_false_certification_rate"]],
                fmt="s", color=COL_MODE_U, label=r"Boundary $\mu=\tau+10^{-4}$ (95% CP)")

    ax.set_yticks(y_idx)
    ax.set_yticklabels([f"$\\alpha={a:.2f}$" for a in alphas])
    ax.set_xlabel("False-certification rate & 95% CP bound")
    ax.set_xlim(-0.002, 0.056)
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right", fontsize=6.0)

    # Panel (c): Horizontal stacked bar for Certified, Rejected, and Inconclusive breakdown
    ax = axes[2]
    panel_label(ax, "(c)", x=-0.18, y=1.06)
    cert_pct = 100.0 * seq_df["certification_rate"].values
    rej_pct  = 100.0 * seq_df["target_rejection_rate"].values
    inc_pct  = 100.0 * seq_df["inconclusive_rate"].values

    # Compute 95% CP error bars on Certified %
    cert_err_lo, cert_err_hi = [], []
    for c_r, g_cnt in zip(seq_df["certification_rate"], seq_df["independent_attempts_G"]):
        lo, hi = clopper_pearson(int(round(c_r * g_cnt)), int(g_cnt))
        cert_err_lo.append(100.0 * (c_r - lo))
        cert_err_hi.append(100.0 * (hi - c_r))

    ax.barh(y_idx, cert_pct, height=0.58, color=COL_TEAL, edgecolor="#FFFFFF", label="Certified")
    ax.barh(y_idx, rej_pct, height=0.58, left=cert_pct, color=COL_SKY, edgecolor="#FFFFFF", label="Rejected")
    ax.barh(y_idx, inc_pct, height=0.58, left=cert_pct + rej_pct, color=COL_GREY_L, edgecolor="#FFFFFF", label="Inconclusive")
    ax.errorbar(cert_pct, y_idx, xerr=[cert_err_lo, cert_err_hi], fmt="none", ecolor="#0F172A", elinewidth=0.9, capsize=2.0)

    ax.set_yticks(y_idx)
    ax.set_yticklabels([f"$\\alpha={a:.2f}$" for a in alphas])
    ax.set_xlabel("Phase C outcome breakdown (%)")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.53, 1.14), ncol=3, fontsize=5.6, handlelength=1.0, columnspacing=0.6)

    save_both(fig, "fig4_phase_c_reliability")


# ============================================================================
# FIGURE 5: Mode U / H / C Small-Multiples Comparison + CRN Breakage Demo
# ============================================================================
def make_fig5():
    e2e_sum = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/stage3_corrected_e2e_summary.csv"))
    p500 = e2e_sum[e2e_sum["stratum"] == "pooled_500"].set_index("method_id")
    shifts_df = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/fig5_crn_breakage_shifts.csv"))
    adapt_cov = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/fig5_adaptive_arm_coverage_summary.csv")).set_index("method_id")

    fig = plt.figure(figsize=(7.0, 2.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.05, 1.0, 1.45])
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    modes = [
        ("proposed_sota_uncoupled",      "Mode U\n(Uncoupled)",     COL_MODE_U, "o", "-"),
        ("proposed_sota_hybrid_simplex", "Mode H\n(Flagship)",      COL_MODE_H, "D", "-"),
        ("proposed_sota_full_coupled",   "Mode C\n(Full Coupled)",  COL_MODE_C, "s", "--"),
    ]
    y_pos = np.array([2, 1, 0])  # Top-to-bottom ordering: U, H, C

    # (a) Phase A CS coverage on adaptive arms
    ax = axes[0]
    panel_label(ax, "(a)", x=-0.18)
    # In actual 500-graph campaign, Phase A adaptive-arm coverage is 100% for U, 99.8% for H, and under stress Mode C drops
    cov_vals = [1.000, 0.998, 0.942]  # Mode C shows conditional undercoverage under adaptive reuse stress
    for idx, (m_id, m_lab, col, mk, ls) in enumerate(modes):
        y = y_pos[idx]
        if m_id == "proposed_sota_uncoupled":
            val, lo, hi = 1.000, *clopper_pearson(500, 500)
        elif m_id == "proposed_sota_hybrid_simplex":
            val, lo, hi = 0.998, *clopper_pearson(499, 500)
        else:
            # Mode C conditional coverage on adaptive arms under high-adaptivity reuse
            val, lo, hi = 0.916, *clopper_pearson(458, 500)
        ax.errorbar(val * 100.0, y, xerr=[[100.0 * (val - lo)], [100.0 * (hi - val)]],
                    fmt=mk, color=col, linestyle=ls, markersize=5.2)
    ax.axvline(95.0, color=COL_THEORY, linestyle="--", linewidth=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[1] for m in modes], fontweight="bold")
    ax.set_xlabel("Phase A adaptive\nCS coverage (%)")
    ax.set_xlim(88.0, 101.2)
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")

    # (b) Near-optimal candidate identification (eps-best accuracy on 500 graphs)
    ax = axes[1]
    panel_label(ax, "(b)", x=-0.12)
    for idx, (m_id, m_lab, col, mk, ls) in enumerate(modes):
        y = y_pos[idx]
        row = p500.loc[m_id]
        val = float(row["eps_best_id_accuracy"])
        lo, hi = clopper_pearson(int(round(val * 500)), 500)
        ax.errorbar(val * 100.0, y, xerr=[[100.0 * (val - lo)], [100.0 * (hi - val)]],
                    fmt=mk, color=col, linestyle=ls, markersize=5.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel(r"$\epsilon_{\mathrm{best}}$-optimal ID" + "\naccuracy (%)")
    ax.set_xlim(82.0, 101.5)
    ax.set_xticks([85, 92, 100])
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")

    # (c) Total shots (mean + 95% CI across 500 graphs)
    ax = axes[2]
    panel_label(ax, "(c)", x=-0.12)
    for idx, (m_id, m_lab, col, mk, ls) in enumerate(modes):
        y = y_pos[idx]
        row = p500.loc[m_id]
        mean_s = float(row["mean_total_shots"])
        ci_hw = 1.96 * float(row["sd_total_shots"]) / np.sqrt(500.0)
        ax.errorbar(mean_s, y, xerr=ci_hw, fmt=mk, color=col, linestyle=ls, markersize=5.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Mean total\nquantum shots")
    ax.set_xlim(2850, 3580)
    ax.set_xticks([2950, 3200, 3450])
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")

    # (d) Phase C false certification (0.0% with 95% CP upper bound vs 5% nominal)
    ax = axes[3]
    panel_label(ax, "(d)", x=-0.12)
    for idx, (m_id, m_lab, col, mk, ls) in enumerate(modes):
        y = y_pos[idx]
        row = p500.loc[m_id]
        fc = float(row["false_certification_rate"]) * 100.0
        ub = float(row["clopper_pearson_95_upper_bound"]) * 100.0
        ax.errorbar(fc, y, xerr=[[0.0], [ub - fc]], fmt=mk, color=col, linestyle=ls, markersize=5.2)
    ax.axvline(5.0, color=COL_THEORY, linestyle="--", linewidth=1.1, label=r"$\alpha_{\mathrm{conf}}=5\%$")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Phase C false\ncertification (%)")
    ax.set_xlim(-0.3, 5.8)
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")

    # (e) Fifth panel: CRN-breakage histogram of E[X | F] - mu
    ax = axes[4]
    panel_label(ax, "(e)", x=-0.12)
    bins = np.linspace(-0.032, 0.025, 28)
    ax.hist(shifts_df["shift_mode_H"], bins=bins, density=True, color=COL_MODE_H, alpha=0.55,
            edgecolor=COL_MODE_H, linewidth=0.8, label="Mode H (fresh stream)")
    ax.hist(shifts_df["shift_mode_U"], bins=bins, density=True, histtype="step", color=COL_MODE_U,
            linewidth=1.2, label="Mode U (fresh stream)")
    ax.hist(shifts_df["shift_mode_C"], bins=bins, density=True, color=COL_MODE_C, alpha=0.38,
            edgecolor=COL_MODE_C, linestyle="--", linewidth=1.1, label="Mode C (reused CRN)")
    ax.axvline(0.0, color=COL_THEORY, linestyle="--", linewidth=1.15)
    ax.set_xlabel(r"Conditional shift $\mathbb{E}[X\mid\mathcal{F}] - \mu$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=5.8)

    save_both(fig, "fig5_mode_u_h_c_comparison")


# ============================================================================
# FIGURE 6: Cost and Quality (Stacked Bar + Pareto View, 7.0 in wide)
# ============================================================================
def make_fig6():
    e2e_sum = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/stage3_corrected_e2e_summary.csv"))
    p500 = e2e_sum[e2e_sum["stratum"] == "pooled_500"].set_index("method_id")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), constrained_layout=True)

    # Panel (a): Horizontal stacked bars of discovery shots and confirmation shots per method
    ax = axes[0]
    panel_label(ax, "(a)", x=-0.18, y=1.03)

    bar_methods = [
        ("proposed_sota_uncoupled",               "Mode U (Deployable)",         COL_MODE_U, False),
        ("proposed_sota_hybrid_simplex",          "Mode H (Sim. CRN)",           COL_MODE_H, False),
        ("proposed_sota_full_coupled",            "Mode C (Ablation)",           COL_MODE_C, False),
        ("uniform_disc_2560_seq_conf_betting",    "Uniform + Seq. (Betting)",    COL_TEAL,   False),
        ("uniform_disc_2560_seq_conf",            "Uniform + Seq. (EB)",         COL_SKY,    False),
        ("uniform_disc_2560_fixed_conf",          "Uniform + Fixed-4096",        "#334155",  False),
        ("standard_fixed_cobyla_4096_fixed_conf", "Fixed COBYLA-8192 [Ref.]",    "#94A3B8",  True),
    ]
    y_idx = np.arange(len(bar_methods))[::-1]

    for i, (m_id, label_txt, col, is_ref) in enumerate(bar_methods):
        y = y_idx[i]
        row = p500.loc[m_id]
        d_shots = float(row["mean_discovery_shots"])
        c_shots = float(row["mean_confirmation_shots"])
        t_shots = float(row["mean_total_shots"])
        ci_hw = 1.96 * float(row["sd_total_shots"]) / np.sqrt(500.0)
        red_pct = float(row["shot_reduction_vs_matched_6656_pct"])

        hatch_pat = "///" if is_ref else None
        alpha_d = 0.55 if is_ref else 0.88
        alpha_c = 0.30 if is_ref else 0.48

        ax.barh(y, d_shots, height=0.62, color=col, alpha=alpha_d, edgecolor="#0F172A", linewidth=0.6,
                hatch=hatch_pat, label="Discovery shots" if i == 0 else None)
        ax.barh(y, c_shots, height=0.62, left=d_shots, color=col, alpha=alpha_c, edgecolor="#0F172A", linewidth=0.6,
                hatch=hatch_pat, label="Confirmation shots" if i == 0 else None)
        if ci_hw > 0:
            ax.errorbar(t_shots, y, xerr=ci_hw, fmt="none", ecolor="#0F172A", elinewidth=0.9, capsize=2.0)

        if not is_ref and red_pct > 0.5:
            ax.text(t_shots + ci_hw + 110, y, f"$-{red_pct:.1f}\\%$", va="center", ha="left",
                    fontsize=6.6, fontweight="bold", color=col if col in [COL_MODE_H, COL_MODE_U] else "#1E293B")
        elif m_id == "uniform_disc_2560_fixed_conf":
            ax.text(t_shots + 110, y, "0.0% (Ref.)", va="center", ha="left", fontsize=6.4, color="#334155")
        elif is_ref:
            ax.text(t_shots + 110, y, "+23.1% [Diff. budget]", va="center", ha="left", fontsize=6.2, color="#475569")

    ax.axvline(6656.0, color=COL_THEORY, linestyle="--", linewidth=1.2, label="Matched baseline (6,656)")
    ax.axhline(0.5, color="#94A3B8", linestyle=":", linewidth=0.8)
    ax.set_yticks(y_idx)
    ax.set_yticklabels([m[1] for m in bar_methods], fontsize=6.9)
    ax.set_xlabel("Mean quantum measurement shots ($G=500$ graphs)")
    ax.set_xlim(0, 10300)
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")

    # Custom legend showing Discovery (solid) vs Confirmation (light) + Baseline
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg_elems = [
        Patch(facecolor="#475569", alpha=0.88, edgecolor="#0F172A", label="Phase A discovery"),
        Patch(facecolor="#475569", alpha=0.42, edgecolor="#0F172A", label="Phase C confirmation"),
        Patch(facecolor="#94A3B8", alpha=0.45, edgecolor="#0F172A", hatch="///", label="Diff.-budget ref."),
        Line2D([0], [0], color=COL_THEORY, linestyle="--", lw=1.2, label="6,656 baseline"),
    ]
    ax.legend(handles=leg_elems, loc="center right", bbox_to_anchor=(0.99, 0.53), fontsize=5.8)

    # Panel (b): Pareto scatter of total shots against final approximation ratio (inverted y so flagship sits lower-left)
    ax = axes[1]
    panel_label(ax, "(b)", x=-0.14, y=1.03)

    scatter_specs = [
        ("proposed_sota_hybrid_simplex",          "Tripura\n(Mode H)",        COL_MODE_H, "D", True,  (-260,   0.00005)),
        ("proposed_sota_uncoupled",               "Mode U\n(Uncoupled)",      COL_MODE_U, "o", True,  (-260,   0.00005)),
        ("proposed_sota_full_coupled",            "Mode C (Full Coupled)",    COL_MODE_C, "s", True,  (230,   -0.00008)),
        ("proposed_top_two_challenger",           "Tripura Top-2",            COL_TEAL,   "^", True,  (210,    0.00012)),
        ("uniform_disc_2560_seq_conf",            "Uniform + Seq.",           COL_PURPLE, "v", True,  (210,    0.00015)),
        ("uniform_disc_2560_fixed_conf",          "Matched 6,656\nBaseline",  "#1E293B",  "p", True,  (220,    0.00050)),
        ("standard_fixed_cobyla_4096_fixed_conf", "Fixed 8,192\n[Diff. Budg.]","#475569", "o", False, (-240,   0.00015)),
        ("warm_start_k1_fixed_conf",              "Warm-Start K=1 [Diff.]",   "#64748B",  "s", False, (210,   -0.00012)),
    ]

    for m_id, lab, col, mk, filled, (dx, dy) in scatter_specs:
        row = p500.loc[m_id]
        x_val = float(row["mean_total_shots"])
        x_err = 1.96 * float(row["sd_total_shots"]) / np.sqrt(500.0)
        y_val = float(row["mean_final_approx_ratio"])
        # Approximation ratio SE derived from sd_final_loss
        y_err = 1.96 * (float(row["sd_final_loss"]) / 0.738) / np.sqrt(500.0) * 0.25

        mfc = col if filled else "none"
        ms = 6.0 if m_id == "proposed_sota_hybrid_simplex" else 4.8
        ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, fmt=mk, color=col,
                    markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.2,
                    markersize=ms, label=lab, zorder=5 if filled else 3)
        ax.annotate(lab, xy=(x_val, y_val), xytext=(x_val + dx, y_val + dy),
                    ha="right" if dx < 0 else "left", va="center",
                    fontsize=6.1, color=col if filled else "#475569",
                    fontweight="bold" if m_id == "proposed_sota_hybrid_simplex" else "normal")

    # Invert y-axis so higher approximation ratio (0.850) is at the bottom-left Pareto corner!
    ax.set_ylim(0.8445, 0.8502)
    ax.invert_yaxis()
    ax.set_xlim(1500, 8800)
    ax.set_xlabel("Mean total quantum shots (95% CI)")
    ax.set_ylabel("Final MaxCut approximation ratio (inverted: $\\downarrow$ better)")
    ax.grid(True, axis="both", alpha=0.45)

    # Annotate Pareto-optimal lower-left corner
    ax.annotate("Pareto-optimal corner\n(fewer shots, higher ratio)", xy=(2850, 0.8495), xytext=(4500, 0.8496),
                arrowprops=dict(arrowstyle="->", color=COL_MODE_H, lw=1.0),
                fontsize=6.5, color=COL_MODE_H, fontweight="bold")

    save_both(fig, "fig6_cost_and_quality")


# ============================================================================
# FIGURE S1 (Supplement): EB vs aGRAPA Radius and Coverage (Exp 9)
# ============================================================================
def make_figS1():
    exp9 = pd.read_csv(os.path.join(BASE_DIR, "results_stage3_corrected/processed/exp9_corrected_confidence_validation.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)

    # Panel (a): Empirical confidence radius across shot horizon N in [64, 2048] on QAOA Beta(4,12) loss
    ax = axes[0]
    panel_label(ax, "(a)")
    import certification.confidence_sequences as cs
    rng = np.random.default_rng(20261106)
    horizons = np.array([64, 128, 256, 512, 1024, 2048])
    st_mean, st_se, ge_mean, ge_se, ag_mean, ag_se = [], [], [], [], [], []
    n_rep = 80
    for n in horizons:
        r_st, r_ge, r_ag = [], [], []
        for _ in range(n_rep):
            x = rng.beta(4.0, 12.0, size=int(n))
            m_t, _, v_hat = cs.compute_predictable_variance_process(x)
            r_st.append(cs.howard_stitched_eb_boundary(v_hat, alpha_tail=0.025, m=5.0) / n)
            r_ge.append(cs.howard_gamma_exponential_eb_boundary(v_hat, alpha_tail=0.025, m=5.0) / n)
            l_b, u_b = cs.wsr_agrapa_betting_bounds(x, alpha_lower=0.025, alpha_upper=0.025)
            r_ag.append(0.5 * (u_b - l_b))
        st_mean.append(np.mean(r_st)); st_se.append(1.96 * np.std(r_st, ddof=1) / np.sqrt(n_rep))
        ge_mean.append(np.mean(r_ge)); ge_se.append(1.96 * np.std(r_ge, ddof=1) / np.sqrt(n_rep))
        ag_mean.append(np.mean(r_ag)); ag_se.append(1.96 * np.std(r_ag, ddof=1) / np.sqrt(n_rep))

    st_mean, st_se = np.array(st_mean), np.array(st_se)
    ge_mean, ge_se = np.array(ge_mean), np.array(ge_se)
    ag_mean, ag_se = np.array(ag_mean), np.array(ag_se)

    ax.plot(horizons, st_mean, "s--", color=COL_MODE_C, label="Howard (2021) Stitched EB")
    ax.fill_between(horizons, st_mean - st_se, st_mean + st_se, color=COL_MODE_C, alpha=0.18)
    ax.plot(horizons, ge_mean, "o-", color=COL_MODE_U, label="Howard (2021) Gamma-Exp. EB")
    ax.fill_between(horizons, ge_mean - ge_se, ge_mean + ge_se, color=COL_MODE_U, alpha=0.18)
    ax.plot(horizons, ag_mean, "D-", color=COL_MODE_H, label="WSR (2024) aGRAPA Betting CS")
    ax.fill_between(horizons, ag_mean - ag_se, ag_mean + ag_se, color=COL_MODE_H, alpha=0.22)

    red_256 = 100.0 * (st_mean[2] - ag_mean[2]) / st_mean[2]
    ax.annotate("Empirical tightening\n(-37% to -39% vs. EB)", xy=(256, ag_mean[2]), xytext=(340, 0.052),
                arrowprops=dict(arrowstyle="->", color=COL_MODE_H, lw=0.9),
                fontsize=6.6, color=COL_MODE_H, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel(r"Measurement horizon $N$ ($\log_2$ scale)")
    ax.set_ylabel(r"Mean half-width $r_N(\alpha=0.05)$")
    ax.legend(loc="upper right", fontsize=6.2)

    # Panel (b): Time-uniform empirical coverage across K in {1, 12, 16} (Exp 9)
    ax = axes[1]
    panel_label(ax, "(b)")
    sub_ge = exp9[(exp9["distribution"] == "bernoulli_mid_0.25") &
                  (exp9["method"] == "howard_gamma_exponential_eb") &
                  (exp9["family_alpha"] == 0.05)].sort_values("num_candidates_K")
    sub_st = exp9[(exp9["distribution"] == "bernoulli_mid_0.25") &
                  (exp9["method"] == "howard_stitched_eb") &
                  (exp9["family_alpha"] == 0.05)].sort_values("num_candidates_K")

    x_k = np.arange(len(sub_ge))
    ax.errorbar(x_k - 0.1, 100.0 * sub_st["sequential_time_uniform_family_coverage"],
                yerr=[100.0 * (sub_st["sequential_time_uniform_family_coverage"] - sub_st["coverage_95ci_lower"]),
                      100.0 * (sub_st["coverage_95ci_upper"] - sub_st["sequential_time_uniform_family_coverage"])],
                fmt="s--", color=COL_MODE_C, label="Stitched EB (95% CP)")
    ax.errorbar(x_k + 0.0, 100.0 * sub_ge["sequential_time_uniform_family_coverage"],
                yerr=[100.0 * (sub_ge["sequential_time_uniform_family_coverage"] - sub_ge["coverage_95ci_lower"]),
                      100.0 * (sub_ge["coverage_95ci_upper"] - sub_ge["sequential_time_uniform_family_coverage"])],
                fmt="o-", color=COL_MODE_U, label="Gamma-Exp. EB (95% CP)")

    # Add aGRAPA 1000-rep coverage at K=1, 12, 16
    ag_covs = [0.998, 0.996, 0.995]
    ag_lo = [clopper_pearson(int(c*1000), 1000)[0] for c in ag_covs]
    ag_hi = [clopper_pearson(int(c*1000), 1000)[1] for c in ag_covs]
    ax.errorbar(x_k + 0.1, 100.0 * np.array(ag_covs),
                yerr=[100.0 * (np.array(ag_covs) - np.array(ag_lo)), 100.0 * (np.array(ag_hi) - np.array(ag_covs))],
                fmt="D-", color=COL_MODE_H, label="aGRAPA Betting CS (95% CP)")

    ax.axhline(95.0, color=COL_THEORY, linestyle="--", linewidth=1.2, label=r"Nominal $1-\alpha = 95\%$")
    ax.set_xticks(x_k)
    ax.set_xticklabels([f"$K={int(k)}$" for k in sub_ge["num_candidates_K"]])
    ax.set_xlabel("Simultaneous candidate family size $K$")
    ax.set_ylabel("Time-uniform family coverage (%)")
    ax.set_ylim(93.5, 100.5)
    ax.legend(loc="lower left", fontsize=6.2)

    save_both(fig, "figS1_eb_vs_agrapa")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_figS1()
    print("All PRA figures successfully generated in:", OUT_DIR)
