"""Make a single cross-dataset overview figure: probe vs baselines on each dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PLOTS_DIR, PROBES_DIR

DATASETS = [
    ("simpleqa", "simpleqa_force_n800"),
    ("triviaqa", "triviaqa_greedy_n500"),
    ("nq_open", "nq_open_greedy_n500"),
    ("popqa", "popqa_force_n500"),
    ("truthfulqa", "truthfulqa_greedy_n300"),
]

# Order baselines explicitly so colours are stable across datasets.
BASELINE_ORDER = [
    "probe",
    "mean_logprob",
    "min_logprob",
    "neg_first_entropy",
    "neg_mean_entropy",
    "mean_margin",
    "first_margin",
    "verbal_conf",
    "self_consistency",
]


def main():
    rows = []
    for name, tag in DATASETS:
        s = PROBES_DIR / tag / "summary.json"
        if not s.exists():
            continue
        info = json.loads(s.read_text())
        baselines = dict(info["baselines"])
        baselines["probe"] = info["best_probe"]["auc"]

        # Also try to pull verbal-conf and self-consistency from verbal_compare
        vc_summary = PROBES_DIR / "verbal_compare" / "summary.json"
        if vc_summary.exists():
            vc = json.loads(vc_summary.read_text())
            if name in vc:
                vc_aucs = vc[name]["aucs"]
                if "verbal_conf" in vc_aucs:
                    baselines["verbal_conf"] = vc_aucs["verbal_conf"]
        rows.append((name, baselines))

    # Build matrix [datasets x baselines]
    names = [r[0] for r in rows]
    bs = [b for b in BASELINE_ORDER if any(b in r[1] for r in rows)]
    M = np.full((len(names), len(bs)), np.nan)
    for i, (_, bd) in enumerate(rows):
        for j, k in enumerate(bs):
            v = bd.get(k)
            M[i, j] = float(v) if v is not None else np.nan

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(bs)), 0.55 * len(names) + 1.5))
    cmap = plt.get_cmap("tab10")
    width = 0.85 / len(bs)
    x = np.arange(len(names))
    for j, k in enumerate(bs):
        ax.bar(x + (j - len(bs) / 2) * width + width / 2, M[:, j], width,
               label=k, color=cmap(j % 10), edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC (held-out)")
    ax.set_title("Best layer probe vs confidence baselines, per dataset")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "overview_probe_vs_baselines.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[overview] wrote {out}")


if __name__ == "__main__":
    main()
