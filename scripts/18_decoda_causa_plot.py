"""Figure: decodability (probe AUC) vs causality (rescue effect) per layer.

The headline conceptual figure for the project. Overlays:
  - per-layer correctness-probe AUC (decodability)
  - per-layer residual-stream patching rescue effect (causality)

Reveals the mismatch: signal becomes decodable early/mid, but only causally
controllable in late layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PLOTS_DIR, PROBES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-tag", required=True)
    ap.add_argument("--patch-tag", required=True,
                    help="out-tag passed to 14_patch_residual.py")
    ap.add_argument("--out-tag", default=None,
                    help="defaults to <probe-tag>_decoda_causa")
    ap.add_argument("--positions", nargs="+",
                    default=["prompt_last", "answer_first", "answer_last", "answer_mean"])
    args = ap.parse_args()

    out_tag = args.out_tag or f"{args.probe_tag}_decoda_causa"

    probe_summary = json.loads((PROBES_DIR / args.probe_tag / "summary.json").read_text())
    patch_summary = json.loads((PROBES_DIR / args.patch_tag / "summary.json").read_text())

    layer_curve = probe_summary.get("layer_curve_auc") or {}
    rescue = patch_summary["mean_rescue_per_layer"]
    corruption = patch_summary.get("mean_corruption_per_layer")
    n_layers = len(rescue)

    plots_dir = PLOTS_DIR / out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    # Left axis: probe AUC for each position (lighter, dashed)
    position_colors = {
        "prompt_last":  "#bcbd22",
        "answer_first": "#17becf",
        "answer_last":  "#9467bd",
        "answer_mean":  "#e377c2",
    }
    for pos in args.positions:
        if pos not in layer_curve:
            continue
        curve = layer_curve[pos]
        ax1.plot(range(len(curve)), curve, color=position_colors.get(pos, "grey"),
                 linewidth=1.2, alpha=0.7, label=f"probe AUC ({pos})", linestyle="--")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Probe AUC (decodability)", color="#444")
    ax1.set_ylim(0.45, 1.0)
    ax1.axhline(0.5, color="#bbb", linewidth=0.6, linestyle=":")
    ax1.tick_params(axis="y", colors="#444")

    # Right axis: rescue effect (causality), solid green
    ax2.plot(range(n_layers), rescue, color="#2ca02c", linewidth=2.4,
             label="rescue effect (causal)")
    ax2.fill_between(range(n_layers), 0, rescue,
                     where=(np.array(rescue) > 0), alpha=0.15, color="#2ca02c")
    if corruption is not None:
        ax2.plot(range(n_layers), corruption, color="#d62728", linewidth=1.6,
                 linestyle=":", alpha=0.7, label="corruption effect (causal)")
    ax2.set_ylabel("Δ logit-diff (causality)", color="#2ca02c")
    ax2.tick_params(axis="y", colors="#2ca02c")
    ax2.axhline(0, color="#bbb", linewidth=0.6, linestyle=":")

    # Mark the peak-decodable layer (best probe across positions) and peak-causal layer
    best_probe = probe_summary["best_probe"]
    peak_rescue = patch_summary["peak_rescue_layer"]
    ax1.axvline(best_probe["layer"], color="#444", linewidth=0.8, linestyle="-.", alpha=0.4)
    ax1.text(best_probe["layer"] + 0.4, 0.47,
             f"peak probe: L{best_probe['layer']}",
             color="#444", fontsize=8)
    ax2.axvline(peak_rescue, color="#2ca02c", linewidth=0.8, linestyle="-.", alpha=0.4)
    ax2.text(peak_rescue + 0.4, ax2.get_ylim()[1] * 0.92,
             f"peak rescue: L{peak_rescue}",
             color="#2ca02c", fontsize=8)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False,
               loc="upper left", ncol=2)

    ax1.set_title(f"Decodability vs causality across layers — {args.probe_tag}")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    out_path = plots_dir / "decodability_vs_causality.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[decoda-causa] wrote {out_path}")

    # Numerical summary
    summary = {
        "probe_tag": args.probe_tag,
        "patch_tag": args.patch_tag,
        "peak_probe_layer": best_probe["layer"],
        "peak_probe_auc": best_probe["auc"],
        "peak_probe_position": best_probe["position"],
        "peak_rescue_layer": peak_rescue,
        "peak_rescue_effect": patch_summary["peak_rescue_effect"],
        "layer_gap": peak_rescue - best_probe["layer"],
    }
    (PROBES_DIR / out_tag).mkdir(parents=True, exist_ok=True)
    (PROBES_DIR / out_tag / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[decoda-causa] peak probe L{summary['peak_probe_layer']} "
          f"vs peak rescue L{summary['peak_rescue_layer']} (gap = {summary['layer_gap']:+d})")


if __name__ == "__main__":
    main()
