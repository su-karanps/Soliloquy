"""Experiment 1: layer x position correctness probes + baselines (Experiment 4).

Trains an l2-logistic probe per (position, layer), reports AUC/Acc/F1/ECE,
and compares against confidence baselines computed from generation metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler  # noqa: F401 (also used in probe direction save)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import baselines as B
from src.config import PLOTS_DIR, POSITIONS, PROBES_DIR, TABLES_DIR
from src.grading import normalize
from src.plotting import plot_baseline_bars, plot_calibration, plot_layer_curve
from src.probes import baseline_auc, train_probe
from src.utils import read_jsonl, stratified_qid_split


def collect_features(rows, position: str, layer: int, n_layers: int):
    X, y, qids = [], [], []
    for r in rows:
        path = r.get("hidden_states_path")
        if not path:
            continue
        h = torch.load(path, map_location="cpu", weights_only=False)
        vec = h[position][layer].numpy()
        X.append(vec)
        y.append(int(r["is_correct"]))
        qids.append(r["qid"])
    return np.stack(X, 0), np.array(y), qids


def collect_all_features(rows, position: str):
    """Load all layers at once, returns (N, L, H) array."""
    X, y, qids = [], [], []
    for r in rows:
        path = r.get("hidden_states_path")
        if not path:
            continue
        h = torch.load(path, map_location="cpu", weights_only=False)
        X.append(h[position].numpy())  # (L+1, H)
        y.append(int(r["is_correct"]))
        qids.append(r["qid"])
    return np.stack(X, 0), np.array(y), qids


def filter_rows(rows, *, drop_abstain: bool, require_hidden: bool = True):
    out = []
    for r in rows:
        if require_hidden and not r.get("hidden_states_path"):
            continue
        if drop_abstain and r.get("did_abstain"):
            continue
        out.append(r)
    return out


def compute_baselines(rows) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {
        "mean_logprob": [],
        "min_logprob": [],
        "neg_first_entropy": [],
        "neg_mean_entropy": [],
        "first_margin": [],
        "mean_margin": [],
    }
    for r in rows:
        out["mean_logprob"].append(B.mean_logprob(r) or 0.0)
        out["min_logprob"].append(B.min_logprob(r) or 0.0)
        out["neg_first_entropy"].append(B.first_entropy_neg(r) or 0.0)
        out["neg_mean_entropy"].append(B.mean_entropy_neg(r) or 0.0)
        out["first_margin"].append(B.first_margin(r) or 0.0)
        out["mean_margin"].append(B.mean_margin(r) or 0.0)
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, help="directory containing graded.jsonl")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positions", nargs="+", default=list(POSITIONS))
    ap.add_argument("--drop-abstain", action="store_true", default=True)
    ap.add_argument("--keep-abstain", dest="drop_abstain", action="store_false")
    ap.add_argument("--standardize", action="store_true", default=True)
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    rows_all = read_jsonl(gen_dir / "graded.jsonl")
    rows = filter_rows(rows_all, drop_abstain=args.drop_abstain)
    n_corr = sum(1 for r in rows if r["is_correct"])
    n_inc = len(rows) - n_corr
    print(f"[probe] dataset={gen_dir.name} usable={len(rows)} "
          f"(correct={n_corr}, incorrect={n_inc}, abstain_dropped={sum(1 for r in rows_all if r.get('did_abstain'))})")
    if n_corr < 5 or n_inc < 5:
        print("[probe] insufficient class balance; aborting")
        return

    qids = [r["qid"] for r in rows]
    labels = [int(r["is_correct"]) for r in rows]
    train_idx, test_idx = stratified_qid_split(qids, labels, args.test_frac, seed=args.seed)
    if not train_idx or not test_idx:
        print("[probe] empty split; aborting")
        return

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load all features once
    print("[probe] loading hidden states ...")
    feats: dict[str, np.ndarray] = {}
    for pos in args.positions:
        X, y, _ = collect_all_features(rows, pos)
        feats[pos] = X
    n_layers = next(iter(feats.values())).shape[1]
    y = np.array(labels)
    print(f"[probe] L+1={n_layers}, hidden_dim={next(iter(feats.values())).shape[-1]}")

    # ----- Probes per layer per position -----
    layer_curve: dict[str, list[float]] = {p: [] for p in args.positions}
    per_layer_records = []
    for pos in args.positions:
        X = feats[pos]
        for layer in range(n_layers):
            Xtr, Xte = X[train_idx, layer, :], X[test_idx, layer, :]
            ytr, yte = y[train_idx], y[test_idx]
            res = train_probe(Xtr, ytr, Xte, yte, standardize=args.standardize)
            layer_curve[pos].append(res["auc"])
            per_layer_records.append({
                "position": pos, "layer": layer,
                "auc": res["auc"], "acc": res["acc"], "f1": res["f1"],
                "logloss": res["logloss"], "ece": res["ece"],
            })
            if layer in (0, n_layers // 2, n_layers - 1):
                print(f"  pos={pos} L={layer:02d} AUC={res['auc']:.3f} Acc={res['acc']:.3f} ECE={res['ece']:.3f}")

    # ----- Confidence baselines -----
    base_scores = compute_baselines(rows)
    base_aucs = {k: baseline_auc(v[test_idx], y[test_idx]) for k, v in base_scores.items()}

    # Self-consistency baseline only if we have >1 generation per qid
    sc = B.self_consistency_score(rows, normalize_fn=normalize)
    sc_arr = np.array([sc.get((r["qid"], r["gen_idx"]), 0.0) for r in rows])
    if np.unique([r["gen_idx"] for r in rows]).size > 1:
        base_aucs["self_consistency"] = baseline_auc(sc_arr[test_idx], y[test_idx])

    # Best probe summary
    best = max(per_layer_records, key=lambda r: (r["auc"] if not np.isnan(r["auc"]) else 0))
    print(f"[probe] best probe: pos={best['position']} L={best['layer']} AUC={best['auc']:.3f}")
    print(f"[probe] baselines: {base_aucs}")

    # ----- Save outputs -----
    summary = {
        "gen_dir": str(gen_dir),
        "out_tag": args.out_tag,
        "n": len(rows),
        "n_correct": int(n_corr),
        "n_incorrect": int(n_inc),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "drop_abstain": args.drop_abstain,
        "best_probe": best,
        "baselines": base_aucs,
        "layer_curve_auc": layer_curve,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # CSV of per-layer probes
    import csv
    with (tables_dir / "layer_probes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer_records[0].keys()))
        w.writeheader()
        w.writerows(per_layer_records)

    # Baseline AUCs CSV
    with (tables_dir / "baselines.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "auc"])
        for k, v in sorted(base_aucs.items(), key=lambda kv: -kv[1]):
            w.writerow([k, v])
        w.writerow(["BEST_PROBE", best["auc"]])

    # Plots
    plot_layer_curve(
        layer_curve,
        baselines=base_aucs,
        title=f"Correctness-probe AUC by layer — {args.out_tag}",
        out_path=plots_dir / "layer_curve.png",
        ylabel="AUC",
        ylim=(0.4, 1.0),
    )
    plot_baseline_bars(
        {**{f"probe@{best['position']}_L{best['layer']}": best["auc"]}, **base_aucs},
        title=f"Best probe vs confidence baselines — {args.out_tag}",
        out_path=plots_dir / "baselines_bar.png",
    )

    # Calibration of best probe (need its predictions again) + save probe direction
    Xtr = feats[best["position"]][train_idx, best["layer"], :]
    Xte = feats[best["position"]][test_idx, best["layer"], :]
    # Fit scaler on train so we can reproduce the exact direction used
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    res_best = train_probe(Xtr, y[train_idx], Xte, y[test_idx], standardize=args.standardize)
    plot_calibration(
        np.array(res_best["p_test"]),
        np.array(res_best["y_test"]),
        title=f"Probe calibration — {args.out_tag} (pos={best['position']}, L={best['layer']})",
        out_path=plots_dir / "calibration_best_probe.png",
    )

    # Save the probe direction for downstream causal experiments.
    # The direction is the unit-normalised logistic regression weight (in standardised space).
    # To apply: z = (x - scaler_mean) / scaler_std; score = z @ coef + intercept
    coef = np.array(res_best["coef"], dtype=np.float32)   # (hidden_dim,)
    coef_norm = coef / (np.linalg.norm(coef) + 1e-12)
    np.savez(
        out_dir / "best_probe_direction.npz",
        coef=coef,
        coef_norm=coef_norm,
        intercept=np.float32(res_best["intercept"]),
        scaler_mean=sc.mean_.astype(np.float32),
        scaler_std=sc.scale_.astype(np.float32),
        position=np.array([best["position"]], dtype=object),
        layer=np.array([best["layer"]], dtype=np.int32),
    )
    print(f"[probe] wrote {out_dir / 'summary.json'} and plots in {plots_dir}")
    print(f"[probe] wrote probe direction to {out_dir / 'best_probe_direction.npz'}")


if __name__ == "__main__":
    main()
