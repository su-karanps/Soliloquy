"""Experiment 3: cross-dataset transfer of the correctness probe.

Trains a probe (at one position, one layer) on each dataset and evaluates on every
other dataset's held-out non-abstain generations.
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
from src.plotting import plot_transfer_matrix
from src.probes import baseline_auc, train_probe
from src.utils import read_jsonl
from src import baselines as B


def load_dataset_rows(gen_dir: Path):
    rows = read_jsonl(gen_dir / "graded.jsonl")
    rows = [r for r in rows if r.get("hidden_states_path") and not r.get("did_abstain")]
    return rows


def features_for(rows, position: str, layer: int):
    X, y = [], []
    for r in rows:
        h = torch.load(r["hidden_states_path"], map_location="cpu", weights_only=False)
        X.append(h[position][layer].numpy())
        y.append(int(r["is_correct"]))
    return np.stack(X, 0), np.array(y)


def logprob_score(rows):
    return np.array([B.mean_logprob(r) or 0.0 for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dirs", nargs="+", required=True,
                    help="list of <gen_dir> tagged as <name>:<gen_dir>")
    ap.add_argument("--position", default="answer_last")
    ap.add_argument("--layer", type=int, default=None,
                    help="layer index; if None, picks the best layer on the first dataset")
    ap.add_argument("--out-tag", required=True)
    args = ap.parse_args()

    name_to_rows: dict[str, list[dict]] = {}
    for spec in args.gen_dirs:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name, path = Path(spec).name, spec
        gd = Path(path)
        rows = load_dataset_rows(gd)
        if len(rows) < 10:
            print(f"[transfer] skipping {name}: only {len(rows)} non-abstain rows")
            continue
        name_to_rows[name] = rows

    if not name_to_rows:
        print("[transfer] no usable datasets; aborting")
        return

    # Determine layer
    chosen_layer = args.layer
    if chosen_layer is None:
        first_name = next(iter(name_to_rows))
        first_rows = name_to_rows[first_name]
        # Load just one example to get n_layers
        h = torch.load(first_rows[0]["hidden_states_path"], map_location="cpu", weights_only=False)
        n_layers = h[args.position].shape[0]
        # Crude best-layer search on first dataset using random 70/30 split
        X, y = features_for(first_rows, args.position, layer=0)
        # actually we need to re-load X for each layer
        from src.utils import stratified_qid_split
        qids = [r["qid"] for r in first_rows]
        tr, te = stratified_qid_split(qids, list(y), test_frac=0.3, seed=0)
        best_auc, best_L = -1.0, 0
        for L in range(n_layers):
            XL, _ = features_for(first_rows, args.position, L)
            res = train_probe(XL[tr], y[tr], XL[te], y[te])
            if not np.isnan(res["auc"]) and res["auc"] > best_auc:
                best_auc, best_L = res["auc"], L
        chosen_layer = best_L
        print(f"[transfer] auto-selected layer={chosen_layer} (AUC={best_auc:.3f} on {first_name})")
    print(f"[transfer] using position={args.position}, layer={chosen_layer}")

    # Train probes on each dataset (with held-out within-dataset eval also)
    names = list(name_to_rows.keys())
    n = len(names)
    auc_matrix = np.full((n, n), np.nan)
    logprob_matrix = np.full((n, n), np.nan)
    n_matrix = np.zeros((n, n), dtype=int)

    # Train probes: we'll just train on full feature set for each train dataset (no held-out
    # within-dataset) and additionally compute within-dataset 70/30 for the diagonal.
    from src.utils import stratified_qid_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        feats[name] = features_for(name_to_rows[name], args.position, chosen_layer)

    for i, train_name in enumerate(names):
        Xtr, ytr = feats[train_name]
        # held-out diagonal
        qids = [r["qid"] for r in name_to_rows[train_name]]
        tr_idx, te_idx = stratified_qid_split(qids, list(ytr), test_frac=0.3, seed=0)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr[tr_idx])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr_s, ytr[tr_idx])

        for j, test_name in enumerate(names):
            Xte, yte = feats[test_name]
            if i == j:
                Xev = sc.transform(Xtr[te_idx])
                yev = ytr[te_idx]
                # logprob baseline on diagonal: use same split
                lp = logprob_score(name_to_rows[train_name])[te_idx]
            else:
                Xev = sc.transform(Xte)
                yev = yte
                lp = logprob_score(name_to_rows[test_name])
            if len(set(yev)) < 2:
                continue
            p = clf.predict_proba(Xev)[:, 1]
            auc_matrix[i, j] = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yev, p))
            logprob_matrix[i, j] = baseline_auc(lp, yev)
            n_matrix[i, j] = len(yev)

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "position": args.position,
        "layer": chosen_layer,
        "datasets": names,
        "auc_matrix": auc_matrix.tolist(),
        "logprob_matrix": logprob_matrix.tolist(),
        "n_matrix": n_matrix.tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import csv
    with (tables_dir / "transfer.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["train", "test", "probe_auc", "logprob_auc", "delta", "n_test"])
        for i, tr in enumerate(names):
            for j, te in enumerate(names):
                pa = auc_matrix[i, j]
                la = logprob_matrix[i, j]
                d = (pa - la) if (not np.isnan(pa) and not np.isnan(la)) else float("nan")
                w.writerow([tr, te, f"{pa:.4f}", f"{la:.4f}", f"{d:.4f}", n_matrix[i, j]])

    plot_transfer_matrix(auc_matrix, names, names,
                         title=f"Cross-dataset probe AUC — pos={args.position}, L={chosen_layer}",
                         out_path=plots_dir / "transfer_probe_auc.png")
    plot_transfer_matrix(logprob_matrix, names, names,
                         title="Cross-dataset mean-logprob AUC",
                         out_path=plots_dir / "transfer_logprob_auc.png")
    plot_transfer_matrix(auc_matrix - logprob_matrix, names, names,
                         title="Probe AUC − logprob AUC",
                         out_path=plots_dir / "transfer_delta.png")
    print(f"[transfer] wrote {out_dir / 'summary.json'} and plots in {plots_dir}")


if __name__ == "__main__":
    main()
