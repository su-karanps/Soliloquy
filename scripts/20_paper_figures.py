"""Assemble the six paper figures as individual high-quality PNGs.

Figure 1: Probe AUC vs layer — 3 panels, one per model
Figure 2: Residual-stream rescue effect vs layer — 3 panels, one per model
Figure 3: Decodability vs causality overlay — 3 panels, one per model
Figure 4: Component localization — MLP vs attn across models + dataset transfer
Figure 5: Probe-direction steering negative result (prompt-level + MLP-level)
Figure 6: Correctness vs verbal-confidence dissociation — 3 panels, one per model

Also: multi-model summary comparison bar chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PLOTS_DIR, PROBES_DIR

# ─── poster-friendly global font sizes ───────────────────────────────────────
plt.rcParams.update({
    "font.size":          20,
    "axes.titlesize":     22,
    "axes.labelsize":     20,
    "xtick.labelsize":    17,
    "ytick.labelsize":    17,
    "legend.fontsize":    16,
    "figure.titlesize":   24,
    "lines.linewidth":    3.0,
    "axes.linewidth":     1.4,
})

OUT = PLOTS_DIR / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

GRAY = "#999999"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
RED = "#d62728"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"

# ─── helpers ─────────────────────────────────────────────────────────────────

def load(p):
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def spine_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18)


# ─── data ────────────────────────────────────────────────────────────────────

probe3  = load(PROBES_DIR / "simpleqa_force_n800" / "summary.json")
patch3  = load(PROBES_DIR / "simpleqa_force_n800_patching" / "summary.json")
# Steering: stored as steering_results.json, not summary.json
steer3_path = PROBES_DIR / "simpleqa_force_n800_steering" / "steering_results.json"
steer3_rows = json.loads(steer3_path.read_text()) if steer3_path.exists() else []
mlp_steer3_path = PROBES_DIR / "simpleqa_force_n800_mlp_steering_L35" / "mlp_steering_results.json"
mlp_steer3_rows = json.loads(mlp_steer3_path.read_text()) if mlp_steer3_path.exists() else []
dissoc3 = load(PROBES_DIR / "confidence_dissoc" / "summary.json").get("simpleqa", {})
heads3  = load(PROBES_DIR / "simpleqa_force_n800_heads" / "summary.json")

probe7  = load(PROBES_DIR / "simpleqa_7b_force_n500" / "summary.json")
patch7  = load(PROBES_DIR / "simpleqa_7b_patching" / "summary.json")
dissoc7 = load(PROBES_DIR / "simpleqa_7b_dissoc" / "summary.json").get("simpleqa_force_n500", {})

probel  = load(PROBES_DIR / "simpleqa_llama8b_force_n500" / "summary.json")
patchl  = load(PROBES_DIR / "simpleqa_llama8b_patching" / "summary.json")
dissocl = load(PROBES_DIR / "simpleqa_llama8b_dissoc" / "summary.json").get("simpleqa_force_n500", {})

# ─── Figure 1: Probe AUC vs layer — 3 models ─────────────────────────────────
pos_colors = {
    "prompt_last": ORANGE,
    "answer_first": GREEN,
    "answer_last": RED,
    "answer_mean": PURPLE,
}
model_probe_data = [
    ("Qwen-3B (36L)", probe3),
    ("Qwen-7B (28L)", probe7),
    ("Llama-3.1-8B (32L)", probel),
]
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, (label, p) in zip(axes1, model_probe_data):
    curves = p.get("layer_curve_auc", {})
    for pos, color in pos_colors.items():
        if pos in curves:
            c = curves[pos]
            ax.plot(range(len(c)), c, color=color, label=pos.replace("_", " "))
    bp = p.get("best_probe", {})
    if bp:
        ax.axvline(bp["layer"], color=GRAY, linewidth=1.0, linestyle="-.")
        offset = -0.5 if bp["layer"] > 25 else 0.3
        ha = "right" if bp["layer"] > 25 else "left"
        ax.text(bp["layer"] + offset, 0.47, f"best L{bp['layer']}\nAUC={bp['auc']:.3f}",
                color=GRAY, fontsize=15, ha=ha)
    ax.axhline(0.5, color=GRAY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Layer"); ax.set_ylim(0.45, 1.0)
    ax.set_title(label)
    spine_clean(ax)
axes1[0].set_ylabel("Probe AUC")
# Legend as a single row below the panels
handles1, labels1 = axes1[0].get_legend_handles_labels()
fig1.legend(handles1, labels1, loc="lower center", ncol=len(handles1),
            frameon=False, bbox_to_anchor=(0.5, -0.04))
fig1.suptitle("Correctness probe AUC by layer\n(SimpleQA, forced prompt)")
fig1.tight_layout(rect=[0, 0.08, 1, 1])
fig1.subplots_adjust(wspace=0.05)
fig1.savefig(OUT / "fig1_probe_auc_by_layer.png", dpi=200, bbox_inches="tight")
plt.close(fig1)

# ─── Figure 2: Rescue effect vs layer — 3 models ─────────────────────────────
model_patch_data = [
    ("Qwen-3B (36L)", patch3),
    ("Qwen-7B (28L)", patch7),
    ("Llama-3.1-8B (32L)", patchl),
]
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for ax, (label, pt) in zip(axes2, model_patch_data):
    rescue = pt.get("mean_rescue_per_layer", [])
    corruption = pt.get("mean_corruption_per_layer", [])
    if not rescue:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=16)
        ax.set_title(label); continue
    ls = range(len(rescue))
    ax.plot(ls, rescue, color=GREEN, label="rescue")
    ax.fill_between(ls, 0, rescue, where=(np.array(rescue) > 0), alpha=0.15, color=GREEN)
    if corruption:
        ax.plot(ls, corruption, color=RED, linewidth=2.0, linestyle="--", alpha=0.8,
                label="corruption")
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle=":")
    peak = pt.get("peak_rescue_layer")
    if peak is not None:
        ax.axvline(peak, color=GREEN, linewidth=1.0, linestyle="-.", alpha=0.5)
        ax.text(peak + 0.3, max(rescue) * 0.88, f"L{peak}", color=GREEN, fontsize=15)
    ax.set_xlabel("Layer")
    ax.set_title(label)
    spine_clean(ax)
axes2[0].set_ylabel("Δ logit-diff (rescue effect)")
axes2[0].legend(frameon=False)
fig2.suptitle("Residual-stream patching rescue effect by layer (SimpleQA)")
fig2.tight_layout()
fig2.savefig(OUT / "fig2_rescue_by_layer.png", dpi=200, bbox_inches="tight")
plt.close(fig2)

# ─── Figure 3: Decodability vs causality overlay — 3 models ──────────────────
model_all_data = [
    ("Qwen-3B (36L)", probe3, patch3),
    ("Qwen-7B (28L)", probe7, patch7),
    ("Llama-3.1-8B (32L)", probel, patchl),
]
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (label, p, pt) in zip(axes3, model_all_data):
    curves_m = p.get("layer_curve_auc", {})
    rescue_m = pt.get("mean_rescue_per_layer", [])
    if not rescue_m:
        ax.text(0.5, 0.5, "No patch data", transform=ax.transAxes, ha="center", fontsize=16)
        ax.set_title(label); continue
    ax2 = ax.twinx()
    for pos, color in pos_colors.items():
        if pos in curves_m:
            c = curves_m[pos]
            ax.plot(range(len(c)), c, color=color, linewidth=2.0, alpha=0.65, linestyle="--")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color=GRAY, linewidth=0.8, linestyle=":")
    ax.set_ylabel("Probe AUC", color="#555")
    ax.tick_params(axis="y", colors="#555")
    ax2.plot(range(len(rescue_m)), rescue_m, color=GREEN, linewidth=2.5, label="rescue")
    ax2.fill_between(range(len(rescue_m)), 0, rescue_m,
                     where=(np.array(rescue_m) > 0), alpha=0.15, color=GREEN)
    ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Δ logit-diff", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    bp = p.get("best_probe", {})
    best_L = bp.get("layer")
    peak_L = pt.get("peak_rescue_layer")
    if best_L is not None:
        ax.axvline(best_L, color="#bbb", linewidth=1.0, linestyle="-.")
        # ax.text(best_L + 0.2, 0.47, f"L{best_L}", color="#888", fontsize=15)
    if peak_L is not None:
        ax2.axvline(peak_L, color=GREEN, linewidth=1.0, linestyle="-.")
        # ax2.text(peak_L + 0.2, max(rescue_m) * 0.86, f"L{peak_L}", color=GREEN, fontsize=15)
    gap = (peak_L - best_L) if (peak_L is not None and best_L is not None) else "?"
    ax.set_xlabel("Layer")
    ax.set_title(f"{label}" if isinstance(gap, int) else label) # (gap = {gap:+d} layers)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(alpha=0.15)
from matplotlib.lines import Line2D
legend_els = [Line2D([0], [0], color=c, linestyle="--", linewidth=2.0,
                     label=pos.replace("_", " "))
              for pos, c in pos_colors.items()]
legend_els.append(Line2D([0], [0], color=GREEN, linewidth=2.5, label="rescue effect"))
# Single-row legend outside/below all panels
fig3.legend(handles=legend_els, loc="lower center", ncol=len(legend_els),
            frameon=False, bbox_to_anchor=(0.5, -0.04))
fig3.suptitle("Decodability (probe AUC) vs causality (rescue effect) by layer")
fig3.tight_layout(rect=[0, 0.10, 1, 1])
fig3.savefig(OUT / "fig3_decodability_vs_causality.png", dpi=200, bbox_inches="tight")
plt.close(fig3)

# ─── Figure 4: Component localization ─────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 4a — MLP vs attn bar across three models
models = ["Qwen-3B", "Qwen-7B", "Llama-3.1-8B"]
mlp_vals = [patch3.get("component_rescue_mlp", 0), patch7.get("component_rescue_mlp", 0),
            patchl.get("component_rescue_mlp", 0)]
attn_vals = [patch3.get("component_rescue_attn", 0), patch7.get("component_rescue_attn", 0),
             patchl.get("component_rescue_attn", 0)]
peak_layers = [patch3.get("component_patch_layer", 35), patch7.get("component_patch_layer", 27),
               patchl.get("component_patch_layer", 31)]
x = np.arange(len(models))
w = 0.35
bars_mlp = axes[0].bar(x - w/2, mlp_vals, w, color=GREEN, label="MLP output", alpha=0.85)
bars_attn = axes[0].bar(x + w/2, attn_vals, w, color=BLUE, label="Attention output", alpha=0.85)
for bar, v in zip(bars_mlp, mlp_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}", ha="center",
                fontsize=15, color=GREEN)
for bar, v in zip(bars_attn, attn_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}", ha="center",
                fontsize=15, color=BLUE)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{m}\n(peak L{l})" for m, l in zip(models, peak_layers)])
axes[0].set_ylabel("Rescue effect (Δ logit-diff)")
axes[0].set_title("MLP vs attention rescue at peak layer")
axes[0].legend(frameon=False)
spine_clean(axes[0])

# 4b — dataset transfer (Qwen-3B)
datasets_t = ["SimpleQA", "TriviaQA", "NQ-Open", "PopQA", "TruthfulQA"]
tags_t = ["simpleqa_force_n800_patching", "triviaqa_3b_patching", "nq_open_3b_patching",
          "popqa_3b_patching", "truthfulqa_3b_patching"]
mlp_t = [load(PROBES_DIR / t / "summary.json").get("component_rescue_mlp", 0) for t in tags_t]
attn_t = [load(PROBES_DIR / t / "summary.json").get("component_rescue_attn", 0) for t in tags_t]
x2 = np.arange(len(datasets_t))
axes[1].bar(x2 - w/2, mlp_t, w, color=GREEN, alpha=0.85, label="MLP")
axes[1].bar(x2 + w/2, attn_t, w, color=BLUE, alpha=0.85, label="Attention")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(datasets_t, rotation=15, ha="right")
axes[1].set_ylabel("Rescue effect (Δ logit-diff)")
axes[1].set_title("Dataset transfer: MLP vs attention (Qwen-3B)")
axes[1].legend(frameon=False)
spine_clean(axes[1])

fig4.tight_layout()
fig4.savefig(OUT / "fig4_component_localization.png", dpi=200, bbox_inches="tight")
plt.close(fig4)

# ─── Figure 5: Steering negative result ──────────────────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5.5))

# 5a — prompt-level steering (script 12, Qwen-3B)
if steer3_rows:
    baseline = next((r["correct_rate"] for r in steer3_rows if r["alpha"] == 0.0),
                    steer3_rows[0]["correct_rate"])
    alphas = [r["alpha"] for r in steer3_rows]
    correct_rates = [r["correct_rate"] for r in steer3_rows]
    axes5[0].plot(alphas, correct_rates, color=BLUE, marker="o", label="correct rate")
    axes5[0].axhline(baseline, color=GRAY, linewidth=1.5, linestyle="--",
                     label=f"baseline = {baseline:.3f}")
    axes5[0].set_xlabel("Steering α")
    axes5[0].set_ylabel("Correct rate")
    axes5[0].set_title("Prompt-level steering along probe dir\n(Qwen-3B, L15 answer_first)")
    axes5[0].legend(frameon=False)
    axes5[0].set_ylim(-0.01, max(correct_rates) * 1.4 + 0.01)
    spine_clean(axes5[0])
else:
    axes5[0].text(0.5, 0.5, "No steering data", transform=axes5[0].transAxes,
                  ha="center", fontsize=16)
    axes5[0].set_title("Prompt-level probe steering")

# 5b — MLP-level steering (script 17, Qwen-3B, L35)
if mlp_steer3_rows:
    mode_colors = {"add": GREEN, "subtract": RED, "project_out": BLUE}
    for mode, color in mode_colors.items():
        rs = [r for r in mlp_steer3_rows if r["mode"] == mode]
        if rs:
            axes5[1].plot([r["alpha"] for r in rs], [r["correct_rate"] for r in rs],
                          color=color, marker="o", label=mode)
    axes5[1].set_xlabel("Steering α")
    axes5[1].set_ylabel("Correct rate")
    axes5[1].set_title("Late-MLP directional steering\n(Qwen-3B, L35 MLP output)")
    axes5[1].legend(frameon=False)
    spine_clean(axes5[1])
else:
    axes5[1].text(0.5, 0.5, "No MLP steering data", transform=axes5[1].transAxes,
                  ha="center", fontsize=16)
    axes5[1].set_title("Late-MLP directional steering")

fig5.suptitle("Steering negative result: 1-D probe direction does not control generation")
fig5.tight_layout()
fig5.savefig(OUT / "fig5_steering_negative.png", dpi=200, bbox_inches="tight")
plt.close(fig5)

# ─── Figure 6: Confidence dissociation (3 models) ────────────────────────────
fig6, axes6 = plt.subplots(1, 3, figsize=(16, 5.5))
model_dissoc = [
    ("Qwen-3B", dissoc3),
    ("Qwen-7B", dissoc7),
    ("Llama-3.1-8B", dissocl),
]
keys_q = ["correct+confident", "correct+unconfident",
          "incorrect+confident", "incorrect+unconfident"]
categories_q = ["✓ ★", "✓ ○", "✗ ★", "✗ ○"]
bar_colors_q = [GREEN, "#aec7e8", RED, ORANGE]

for ax6, (label, d) in zip(axes6, model_dissoc):
    if not d:
        ax6.text(0.5, 0.5, "No dissoc data", transform=ax6.transAxes, ha="center", fontsize=16)
        ax6.set_title(label); continue
    quads = d.get("quadrants", {})
    n = d.get("n", sum(quads.values()) if quads else 1)
    counts = [quads.get(k, 0) for k in keys_q]
    pcts = [100 * c / max(n, 1) for c in counts]
    bars = ax6.bar(range(4), pcts, color=bar_colors_q, alpha=0.85, width=0.6)
    for bar, pct in zip(bars, pcts):
        ax6.text(bar.get_x() + bar.get_width()/2, pct + 0.5, f"{pct:.0f}%",
                 ha="center", fontsize=15)
    cos = d.get("cos_sim_corr_conf", float("nan"))
    auc = d.get("auc_corr", float("nan"))
    ax6.set_xticks(range(4))
    ax6.set_xticklabels(categories_q)
    ax6.set_title(f"{label}\nAUC={auc:.3f}") # cos(corr,conf)={cos:.3f}
    ax6.set_ylim(0, max(pcts) * 1.25 if pcts else 100)
    spine_clean(ax6)

axes6[0].set_ylabel("% of responses")
fig6.suptitle("Correctness vs verbal-confidence dissociation (SimpleQA, forced prompt)")
fig6.tight_layout()
fig6.savefig(OUT / "fig6_confidence_dissociation.png", dpi=200, bbox_inches="tight")
plt.close(fig6)

# ─── Multi-model comparison (bonus) ──────────────────────────────────────────
fig_mm, axes_mm = plt.subplots(1, 3, figsize=(19, 5.5))
model_names = ["Qwen-3B\n(36L)", "Qwen-7B\n(28L)", "Llama-3.1-8B\n(32L)"]
probe_aucs = [probe3["best_probe"]["auc"], probe7["best_probe"]["auc"],
              probel["best_probe"]["auc"]]
probe_layers = [probe3["best_probe"]["layer"], probe7["best_probe"]["layer"],
                probel["best_probe"]["layer"]]
rescue_effs = [patch3["peak_rescue_effect"], patch7["peak_rescue_effect"],
               patchl["peak_rescue_effect"]]
rescue_layers = [patch3["peak_rescue_layer"], patch7["peak_rescue_layer"],
                 patchl["peak_rescue_layer"]]
mlp_r = [patch3["component_rescue_mlp"], patch7["component_rescue_mlp"],
         patchl["component_rescue_mlp"]]
attn_r = [patch3["component_rescue_attn"], patch7["component_rescue_attn"],
          patchl["component_rescue_attn"]]

x_mm = np.arange(3)
axes_mm[0].bar(x_mm, probe_aucs, color=PURPLE, alpha=0.85)
for i, (v, l) in enumerate(zip(probe_aucs, probe_layers)):
    axes_mm[0].text(i, v + 0.005, f"{v:.3f}\n(L{l})", ha="center", fontsize=15)
axes_mm[0].axhline(0.5, color=GRAY, linewidth=1.0, linestyle=":")
axes_mm[0].set_xticks(x_mm); axes_mm[0].set_xticklabels(model_names)
axes_mm[0].set_ylabel("Best probe AUC"); axes_mm[0].set_title("Probe AUC")
axes_mm[0].set_ylim(0.4, 1.0); spine_clean(axes_mm[0])

axes_mm[1].bar(x_mm, rescue_effs, color=GREEN, alpha=0.85)
for i, (v, l) in enumerate(zip(rescue_effs, rescue_layers)):
    axes_mm[1].text(i, v + 0.05, f"{v:.2f}\n(L{l})", ha="center", fontsize=15)
axes_mm[1].set_xticks(x_mm); axes_mm[1].set_xticklabels(model_names)
axes_mm[1].set_ylabel("Peak rescue Δ logit-diff"); axes_mm[1].set_title("Causal rescue effect")
axes_mm[1].set_ylim(0, max(rescue_effs) * 1.35)
spine_clean(axes_mm[1])

w2 = 0.35
axes_mm[2].bar(x_mm - w2/2, mlp_r, w2, color=GREEN, alpha=0.85, label="MLP")
axes_mm[2].bar(x_mm + w2/2, attn_r, w2, color=BLUE, alpha=0.85, label="Attention")
axes_mm[2].set_xticks(x_mm); axes_mm[2].set_xticklabels(model_names)
axes_mm[2].set_ylabel("Component rescue effect"); axes_mm[2].set_title("MLP vs attention at peak layer")
axes_mm[2].legend(frameon=False); spine_clean(axes_mm[2])

fig_mm.suptitle("Multi-model comparison — Qwen-3B / Qwen-7B / Llama-3.1-8B (SimpleQA)")
fig_mm.tight_layout()
fig_mm.savefig(OUT / "fig_multimodel.png", dpi=200, bbox_inches="tight")
plt.close(fig_mm)

print(f"[paper-figs] wrote 7 figures to {OUT}")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
