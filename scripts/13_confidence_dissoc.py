"""Experiment 10: Distinguish correctness from verbalized confidence.

Trains two parallel probes at the best correctness-probe (layer, position):
  probe_A: predict is_correct
  probe_B: predict high_verbal_conf (verbal_conf >= 50)

Compares their weight directions via cosine similarity and generates a
dissociation plot showing how far apart the directions are.

Also computes:
  - cases in each quadrant (correct x confident, correct x unconfident, etc.)
  - logit-lens predictions for each category across layers
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PLOTS_DIR, PROBES_DIR, TABLES_DIR
from src.utils import read_jsonl, stratified_qid_split


def load_features_for_rows(rows, position, layer):
    X, y_corr, y_conf, qids = [], [], [], []
    for r in rows:
        path = r.get("hidden_states_path")
        if not path:
            continue
        vc = r.get("verbal_conf")
        if vc is None:
            continue
        h = torch.load(path, map_location="cpu", weights_only=False)
        X.append(h[position][layer].numpy())
        y_corr.append(int(r["is_correct"]))
        y_conf.append(int(vc >= 50))
        qids.append(r["qid"])
    return np.stack(X, 0), np.array(y_corr), np.array(y_conf), qids


def fit_direction(X_train, y_train):
    sc = StandardScaler()
    Xsc = sc.fit_transform(X_train)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(Xsc, y_train)
    coef = clf.coef_.squeeze()
    coef_norm = coef / (np.linalg.norm(coef) + 1e-12)
    return coef_norm, sc, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dirs", nargs="+", required=True,
                    help="list of <name>:<gen_dir>")
    ap.add_argument("--probe-tags", nargs="+", required=True,
                    help="matching probe out-tags (same order as gen-dirs)")
    ap.add_argument("--out-tag", default="confidence_dissoc")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for spec, probe_tag in zip(args.gen_dirs, args.probe_tags):
        if ":" in spec:
            name, gd_path = spec.split(":", 1)
        else:
            name, gd_path = Path(spec).name, spec
        gd = Path(gd_path)

        # Load best probe layer/position
        summary = json.loads((PROBES_DIR / probe_tag / "summary.json").read_text())
        best = summary["best_probe"]
        layer, position = best["layer"], best["position"]
        print(f"[dissoc] {name}: best probe layer={layer}, pos={position}")

        # Load graded + verbal_conf
        graded = {r["qid"]: r for r in read_jsonl(gd / "graded.jsonl")}
        if (gd / "verbal_conf.jsonl").exists():
            for r in read_jsonl(gd / "verbal_conf.jsonl"):
                qid = r["qid"]
                if qid in graded:
                    graded[qid]["verbal_conf"] = r.get("verbal_conf")
        rows = [r for r in graded.values()
                if r.get("hidden_states_path") and not r.get("did_abstain")
                and r.get("verbal_conf") is not None]
        if len(rows) < 20:
            print(f"[dissoc] {name}: not enough rows with verbal_conf, skipping")
            continue

        X, y_corr, y_conf, qids = load_features_for_rows(rows, position, layer)
        if len(set(y_corr)) < 2 or len(set(y_conf)) < 2:
            print(f"[dissoc] {name}: insufficient class balance, skipping")
            continue

        # 70/30 qid-level split
        tr, te = stratified_qid_split(qids, list(y_corr), test_frac=0.3, seed=0)

        dir_corr, sc_corr, clf_corr = fit_direction(X[tr], y_corr[tr])
        dir_conf, sc_conf, clf_conf = fit_direction(X[tr], y_conf[tr])

        cos_sim = float(np.dot(dir_corr, dir_conf))

        # AUCs
        from sklearn.metrics import roc_auc_score
        Xte_sc_corr = sc_corr.transform(X[te])
        Xte_sc_conf = sc_conf.transform(X[te])
        auc_corr_on_corr = roc_auc_score(y_corr[te], clf_corr.predict_proba(Xte_sc_corr)[:, 1])
        auc_conf_on_conf = roc_auc_score(y_conf[te], clf_conf.predict_proba(Xte_sc_conf)[:, 1])
        # Cross: correctness probe predicting verbal-conf, and vice versa
        auc_corr_probe_on_conf = roc_auc_score(y_conf[te], clf_corr.predict_proba(Xte_sc_corr)[:, 1])
        auc_conf_probe_on_corr = roc_auc_score(y_corr[te], clf_conf.predict_proba(Xte_sc_conf)[:, 1])

        print(f"  cos_sim(corr, conf) = {cos_sim:.3f}")
        print(f"  AUC correctness probe → correctness = {auc_corr_on_corr:.3f}")
        print(f"  AUC verbal-conf probe → verbal-conf = {auc_conf_on_conf:.3f}")
        print(f"  AUC correctness probe → verbal-conf  = {auc_corr_probe_on_conf:.3f}")
        print(f"  AUC verbal-conf probe → correctness  = {auc_conf_probe_on_corr:.3f}")

        # Quadrant counts
        quadrants = {
            "correct+confident": int(((y_corr == 1) & (y_conf == 1)).sum()),
            "correct+unconfident": int(((y_corr == 1) & (y_conf == 0)).sum()),
            "incorrect+confident": int(((y_corr == 0) & (y_conf == 1)).sum()),
            "incorrect+unconfident": int(((y_corr == 0) & (y_conf == 0)).sum()),
        }
        print(f"  quadrants: {quadrants}")

        all_results[name] = {
            "layer": layer, "position": position,
            "cos_sim_corr_conf": cos_sim,
            "auc_corr": auc_corr_on_corr,
            "auc_conf": auc_conf_on_conf,
            "auc_corr_probe_on_conf": auc_corr_probe_on_conf,
            "auc_conf_probe_on_corr": auc_conf_probe_on_corr,
            "quadrants": quadrants,
            "n": len(rows),
        }

        # Direction similarity bar chart
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        metrics = ["corr→corr", "conf→conf", "corr→conf", "conf→corr"]
        aucs = [auc_corr_on_corr, auc_conf_on_conf, auc_corr_probe_on_conf, auc_conf_probe_on_corr]
        colors = ["#2ca02c", "#1f77b4", "#9467bd", "#9467bd"]
        axes[0].barh(metrics, aucs, color=colors)
        axes[0].axvline(0.5, color="grey", linewidth=0.8, linestyle=":")
        axes[0].set_xlim(0.4, 1.0)
        axes[0].set_title("Probe cross-prediction AUCs")
        axes[0].set_xlabel("AUC")

        # Quadrant distribution
        qnames = list(quadrants.keys())
        qcounts = list(quadrants.values())
        qcolors = ["#2ca02c", "#bcbd22", "#d62728", "#7f7f7f"]
        axes[1].barh(qnames, qcounts, color=qcolors)
        axes[1].set_title(f"Quadrant counts (cosine similarity = {cos_sim:.3f})")
        axes[1].set_xlabel("Count")

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(f"Correctness vs verbal-confidence dissociation — {name}", fontsize=10)
        fig.tight_layout()
        ds_plots = plots_dir / name
        ds_plots.mkdir(parents=True, exist_ok=True)
        fig.savefig(ds_plots / "dissociation.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
    with (tables_dir / "dissociation.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "layer", "position", "cos_sim", "auc_corr", "auc_conf",
                    "auc_corr_on_conf", "auc_conf_on_corr"])
        for name, res in all_results.items():
            w.writerow([name, res["layer"], res["position"], f"{res['cos_sim_corr_conf']:.4f}",
                        f"{res['auc_corr']:.4f}", f"{res['auc_conf']:.4f}",
                        f"{res['auc_corr_probe_on_conf']:.4f}", f"{res['auc_conf_probe_on_corr']:.4f}"])
    print(f"[dissoc] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
