"""Plotting utilities (matplotlib only, no seaborn dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


POSITION_STYLES = {
    "prompt_last": dict(color="#1f77b4", marker="o"),
    "answer_first": dict(color="#ff7f0e", marker="s"),
    "answer_last": dict(color="#2ca02c", marker="^"),
    "answer_mean": dict(color="#d62728", marker="D"),
}


def plot_layer_curve(
    layer_curve: Mapping[str, list[float]],
    baselines: Mapping[str, float] | None,
    title: str,
    out_path: Path,
    ylabel: str = "AUC",
    ylim: tuple[float, float] | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    n_layers = None
    for pos, vals in layer_curve.items():
        n_layers = len(vals)
        style = POSITION_STYLES.get(pos, dict(marker="x"))
        ax.plot(range(n_layers), vals, label=pos, linewidth=1.6, markersize=4, **style)
    if baselines:
        # Plot baselines as horizontal lines.
        cmap = plt.get_cmap("tab10")
        for i, (name, val) in enumerate(baselines.items()):
            ax.axhline(val, linestyle="--", linewidth=1.0, color=cmap(i % 10),
                       alpha=0.75, label=f"{name}={val:.3f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_baseline_bars(values: Mapping[str, float], title: str, out_path: Path):
    items = sorted(values.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(items))))
    ax.barh(names, vals, color="#4c72b0")
    for y, v in enumerate(vals):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.4, 1.0)
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("AUC")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)


def plot_transfer_matrix(matrix: np.ndarray, row_labels, col_labels, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(col_labels), 1.0 + 0.6 * len(row_labels)))
    im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap="viridis")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "-"
            ax.text(j, i, txt, ha="center", va="center", color="white" if v < 0.75 else "black", fontsize=9)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Test")
    ax.set_ylabel("Train")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="AUC")
    fig.savefig(out_path)
    plt.close(fig)


def plot_calibration(probs: np.ndarray, labels: np.ndarray, title: str, out_path: Path, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    bin_acc, bin_conf, bin_cnt = [], [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_cnt.append(0)
        else:
            bin_acc.append(float(labels[m].mean()))
            bin_conf.append(float(probs[m].mean()))
            bin_cnt.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.scatter(bin_conf, bin_acc, s=[max(8, c) for c in bin_cnt], color="#4c72b0", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical correctness")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
