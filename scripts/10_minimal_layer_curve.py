"""Minimal layer-AUC curve: a single position, no title, no legend, no baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PLOTS_DIR, PROBES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="simpleqa_force_n800",
                    help="Probe out-tag (a subdirectory name under PROBES_DIR with summary.json)")
    ap.add_argument("--position", default="prompt_last")
    ap.add_argument("--out-name", default=None,
                    help="Output PNG name; defaults to <tag>_<position>_minimal.png")
    ap.add_argument("--ylim", nargs=2, type=float, default=[0.4, 1.0])
    ap.add_argument("--color", default="#1f77b4")
    args = ap.parse_args()

    summary_path = PROBES_DIR / args.tag / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    info = json.loads(summary_path.read_text())
    curve = info["layer_curve_auc"][args.position]
    n_layers = len(curve)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(n_layers), curve, color=args.color, marker="o", linewidth=3, markersize=6)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("AUC", fontsize=14)
    ax.set_ylim(*args.ylim)
    ax.grid(alpha=0.25, linewidth=1)
    fig.tight_layout()

    out_name = args.out_name or f"{args.tag}_{args.position}_minimal.png"
    out_path = PLOTS_DIR / args.tag / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[minimal-curve] wrote {out_path}")


if __name__ == "__main__":
    main()
