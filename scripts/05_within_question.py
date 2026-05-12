"""Experiment 2: within-question paired controls.

For sampled generations, restrict to questions that produced BOTH at least one
correct and at least one incorrect (non-abstain) answer. Train a probe in two ways:

A) Random qid split: train/test split by qid (no leakage).
B) Within-question pairing: include both correct and incorrect generations of
   *the same* qid in BOTH train and test, so the probe must discriminate
   correctness within a fixed question.

Compares (A) and (B) AUC at every layer/position to test whether the
"correctness" signal survives when topic/difficulty is controlled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PLOTS_DIR, POSITIONS, PROBES_DIR, TABLES_DIR
from src.plotting import plot_layer_curve
from src.probes import train_probe
from src.utils import read_jsonl, stratified_qid_split


def load_features(rows, position: str):
    X, y, qids, gen_idx = [], [], [], []
    for r in rows:
        path = r.get("hidden_states_path")
        if not path:
            continue
        h = torch.load(path, map_location="cpu", weights_only=False)
        X.append(h[position].numpy())  # (L+1, H)
        y.append(int(r["is_correct"]))
        qids.append(r["qid"])
        gen_idx.append(r["gen_idx"])
    return np.stack(X, 0), np.array(y), qids, gen_idx


def filter_within_question(rows):
    """Keep only generations from qids that have both correct and incorrect (non-abstain) answers."""
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("did_abstain"):
            continue
        if not r.get("hidden_states_path"):
            continue
        by_q[r["qid"]].append(r)
    out = []
    for q, group in by_q.items():
        labels = {r["is_correct"] for r in group}
        if labels == {True, False} or labels == {False, True} or labels == {0, 1}:
            out.extend(group)
    return out


def within_question_split(qids: list[str], gen_idx: list[int], labels: list[int], seed: int = 0):
    """For each qid, randomly send half of (correct, incorrect) into train, half into test.

    This guarantees each qid contributes to both sides so the probe must discriminate
    *within* the same question.
    """
    rng = np.random.default_rng(seed)
    by_q: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qids):
        by_q[q].append(i)
    train, test = [], []
    for q, idxs in by_q.items():
        pos = [i for i in idxs if labels[i] == 1]
        neg = [i for i in idxs if labels[i] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos_tr = max(1, len(pos) // 2)
        n_neg_tr = max(1, len(neg) // 2)
        train.extend(pos[:n_pos_tr])
        test.extend(pos[n_pos_tr:])
        train.extend(neg[:n_neg_tr])
        test.extend(neg[n_neg_tr:])
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--positions", nargs="+", default=list(POSITIONS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    rows_all = read_jsonl(gen_dir / "graded.jsonl")
    rows = filter_within_question(rows_all)
    n_q = len({r["qid"] for r in rows})
    n_c = sum(1 for r in rows if r["is_correct"])
    n_i = sum(1 for r in rows if not r["is_correct"])
    print(f"[within] qids with mixed outcomes: {n_q}, gens={len(rows)} (correct={n_c}, incorrect={n_i})")
    if n_q < 5 or n_c < 5 or n_i < 5:
        print("[within] not enough mixed-outcome qids; aborting")
        return

    qids = [r["qid"] for r in rows]
    gidx = [r["gen_idx"] for r in rows]
    labels = [int(r["is_correct"]) for r in rows]
    y = np.array(labels)

    feats = {}
    for pos in args.positions:
        feats[pos] = load_features(rows, pos)[0]
    n_layers = next(iter(feats.values())).shape[1]
    print(f"[within] L+1={n_layers}")

    # (A) random split by qid (so a given qid is either fully in train or test)
    tr_a, te_a = stratified_qid_split(qids, labels, test_frac=0.3, seed=args.seed)
    # (B) within-question split: each qid contributes to both train and test
    tr_b, te_b = within_question_split(qids, gidx, labels, seed=args.seed)

    layer_curve_a = {p: [] for p in args.positions}
    layer_curve_b = {p: [] for p in args.positions}
    per_layer = []
    for pos in args.positions:
        for L in range(n_layers):
            X = feats[pos][:, L, :]
            ra = train_probe(X[tr_a], y[tr_a], X[te_a], y[te_a])
            rb = train_probe(X[tr_b], y[tr_b], X[te_b], y[te_b])
            layer_curve_a[pos].append(ra["auc"])
            layer_curve_b[pos].append(rb["auc"])
            per_layer.append({
                "position": pos, "layer": L,
                "auc_qid_split": ra["auc"], "auc_within_q_split": rb["auc"],
                "delta": rb["auc"] - ra["auc"],
            })

    best_a = max(per_layer, key=lambda r: r["auc_qid_split"] if not np.isnan(r["auc_qid_split"]) else 0)
    best_b = max(per_layer, key=lambda r: r["auc_within_q_split"] if not np.isnan(r["auc_within_q_split"]) else 0)
    print(f"[within] best qid-split AUC: {best_a['auc_qid_split']:.3f} @ {best_a['position']} L{best_a['layer']}")
    print(f"[within] best within-q AUC : {best_b['auc_within_q_split']:.3f} @ {best_b['position']} L{best_b['layer']}")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_questions": n_q, "n_correct": n_c, "n_incorrect": n_i,
        "best_qid_split": best_a, "best_within_q_split": best_b,
        "layer_curve_qid_split": layer_curve_a,
        "layer_curve_within_q_split": layer_curve_b,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import csv
    with (tables_dir / "within_question_probes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer[0].keys()))
        w.writeheader()
        w.writerows(per_layer)

    plot_layer_curve(layer_curve_a, baselines=None,
                     title=f"qid-split AUC (questions disjoint) — {args.out_tag}",
                     out_path=plots_dir / "layer_curve_qid_split.png", ylim=(0.4, 1.0))
    plot_layer_curve(layer_curve_b, baselines=None,
                     title=f"within-question AUC (same qids in train+test) — {args.out_tag}",
                     out_path=plots_dir / "layer_curve_within_q_split.png", ylim=(0.4, 1.0))

    # Comparison plot: best-position curve under both split strategies
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    # plot answer_last (a common best) under both
    cmap = plt.get_cmap("tab10")
    for k, pos in enumerate(args.positions):
        ax.plot(range(n_layers), layer_curve_a[pos], color=cmap(k), linestyle="--", linewidth=1.0,
                label=f"{pos} (qid-split)", alpha=0.7)
        ax.plot(range(n_layers), layer_curve_b[pos], color=cmap(k), linestyle="-", linewidth=1.6,
                label=f"{pos} (within-q)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_title(f"Within-question vs qid-split AUC — {args.out_tag}")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_within_vs_qid.png", dpi=160)
    plt.close(fig)

    print(f"[within] wrote {out_dir/'summary.json'} and plots in {plots_dir}")


if __name__ == "__main__":
    main()
