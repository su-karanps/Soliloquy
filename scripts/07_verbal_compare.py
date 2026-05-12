"""Experiment 4 extension: verbalized confidence vs probe vs logprob.

For each dataset that has verbal_conf.jsonl, compares the probe (trained on
held-out within-dataset 70/30 split, best layer at the chosen position) against
mean-logprob and verbalized confidence, all evaluated on the same test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PLOTS_DIR, PROBES_DIR, TABLES_DIR
from src.plotting import plot_baseline_bars
from src.probes import baseline_auc, train_probe
from src.utils import read_jsonl, stratified_qid_split
from src import baselines as B


def load_rows(gen_dir: Path, with_verbal: bool = True):
    rows = read_jsonl(gen_dir / "graded.jsonl")
    rows = [r for r in rows if r.get("hidden_states_path") and not r.get("did_abstain")]
    if with_verbal:
        verb_path = gen_dir / "verbal_conf.jsonl"
        if verb_path.exists():
            verb_map = {(o["qid"], o["gen_idx"]): o["verbal_conf"] for o in read_jsonl(verb_path)}
            for r in rows:
                r["verbal_conf"] = verb_map.get((r["qid"], r["gen_idx"]))
    return rows


def features(rows, position: str, layer: int):
    X = []
    for r in rows:
        h = torch.load(r["hidden_states_path"], map_location="cpu", weights_only=False)
        X.append(h[position][layer].numpy())
    return np.stack(X, 0)


def best_layer_probe(rows, position: str, y, train_idx, test_idx):
    h = torch.load(rows[0]["hidden_states_path"], map_location="cpu", weights_only=False)
    n_layers = h[position].shape[0]
    best_auc, best_L, best_res = -1.0, 0, None
    for L in range(n_layers):
        X = features(rows, position, L)
        res = train_probe(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        if not np.isnan(res["auc"]) and res["auc"] > best_auc:
            best_auc, best_L, best_res = res["auc"], L, res
    return best_L, best_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dirs", nargs="+", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--position", default="answer_last")
    args = ap.parse_args()

    results = {}
    qual_examples = []

    for spec in args.gen_dirs:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name, path = Path(spec).name, spec
        gd = Path(path)
        rows = load_rows(gd, with_verbal=True)
        if len(rows) < 20:
            print(f"[verbal-cmp] skipping {name}: only {len(rows)} non-abstain rows")
            continue
        qids = [r["qid"] for r in rows]
        labels = np.array([int(r["is_correct"]) for r in rows])
        train_idx, test_idx = stratified_qid_split(qids, list(labels), test_frac=0.3, seed=0)

        best_L, res = best_layer_probe(rows, args.position, labels, train_idx, test_idx)
        probe_p = np.array(res["p_test"])
        y_test = np.array(res["y_test"])

        # Baselines on test split
        lp = np.array([B.mean_logprob(r) or 0.0 for r in rows])[test_idx]
        ent = np.array([B.mean_entropy_neg(r) or 0.0 for r in rows])[test_idx]
        marg = np.array([B.mean_margin(r) or 0.0 for r in rows])[test_idx]
        vc = np.array([
            (r.get("verbal_conf") if r.get("verbal_conf") is not None else 50)
            for r in rows
        ])[test_idx].astype(float)

        aucs = {
            f"probe@{args.position}_L{best_L}": baseline_auc(probe_p, y_test),
            "mean_logprob": baseline_auc(lp, y_test),
            "neg_mean_entropy": baseline_auc(ent, y_test),
            "mean_margin": baseline_auc(marg, y_test),
            "verbal_conf": baseline_auc(vc, y_test),
        }
        print(f"[verbal-cmp] {name}: {aucs}")

        out_dir = PROBES_DIR / args.out_tag / name
        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = PLOTS_DIR / args.out_tag / name
        plots_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = TABLES_DIR / args.out_tag / name
        tables_dir.mkdir(parents=True, exist_ok=True)

        results[name] = {
            "best_probe_layer": int(best_L),
            "aucs": aucs,
            "n_test": int(len(test_idx)),
        }

        plot_baseline_bars(aucs, title=f"{name}: probe vs baselines", out_path=plots_dir / "auc_bars.png")

        # Qualitative dissociation examples on this dataset's test split
        # Where: probe says wrong (probe_p < 0.3) but model is confident
        test_rows = [rows[i] for i in test_idx]
        for i, r in enumerate(test_rows):
            p_prob = float(probe_p[i])
            p_lp = float(np.exp(lp[i])) if lp[i] is not None else None
            p_vc = float(vc[i]) / 100.0
            if (p_vc >= 0.85 or (p_lp is not None and p_lp >= 0.8)) and p_prob < 0.3 and not r["is_correct"]:
                qual_examples.append({
                    "dataset": name,
                    "question": r["question"],
                    "model_answer": r["answer_text"],
                    "gold": r["gold_answers"],
                    "is_correct": bool(r["is_correct"]),
                    "probe_p_correct": p_prob,
                    "mean_logprob": float(lp[i]),
                    "verbal_conf": float(vc[i]) if vc[i] is not None else None,
                })

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    (out_dir / "qualitative_dissociation.json").write_text(
        json.dumps(qual_examples[:50], indent=2, ensure_ascii=False)
    )
    print(f"[verbal-cmp] saved summary to {out_dir/'summary.json'} "
          f"(qualitative examples: {len(qual_examples)})")


if __name__ == "__main__":
    main()
