"""Extract and display qualitative probe failure cases.

Four quadrants based on (probe prediction, actual correctness):
  True positive  — probe says correct, model IS correct           (good)
  False positive — probe says correct, model is WRONG             (probe failure: overconfident)
  True negative  — probe says wrong,   model IS wrong             (good)
  False negative — probe says wrong,   model is CORRECT           (probe failure: underconfident)

Within each failure quadrant, also shows verbal confidence (when available) and
the best-matched gold answer, so you can see what the probe was confused by.

Writes a markdown file with ranked examples and a confusion-matrix figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PLOTS_DIR, PROBES_DIR, TABLES_DIR
from src.utils import read_jsonl


def load_probe_predictions(rows, probe_data):
    """Apply saved probe to rows, return array of p(correct) for each row."""
    coef = probe_data["coef"].astype(np.float32)
    intercept = float(probe_data["intercept"])
    mean = probe_data["scaler_mean"].astype(np.float32)
    std = probe_data["scaler_std"].astype(np.float32)
    position = str(probe_data["position"][0])
    layer = int(probe_data["layer"][0])

    probs = []
    for r in rows:
        hs_path = r.get("hidden_states_path")
        if not hs_path or not Path(hs_path).exists():
            probs.append(float("nan"))
            continue
        hs = torch.load(hs_path, map_location="cpu", weights_only=False)
        h = hs[position][layer].numpy().astype(np.float32)
        z = (h - mean) / (std + 1e-8)
        logit = float(np.dot(z, coef) + intercept)
        prob = float(1 / (1 + np.exp(-logit)))
        probs.append(prob)
    return np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--probe-tag", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--n-examples", type=int, default=20,
                    help="Number of examples per failure quadrant to include")
    ap.add_argument("--probe-threshold", type=float, default=0.5,
                    help="p(correct) threshold to classify probe as 'predicts correct'")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    rows = [r for r in read_jsonl(gen_dir / "graded.jsonl") if not r.get("did_abstain")]
    print(f"[failures] {len(rows)} non-abstain rows")

    # Load verbal confidence if available
    vc_map: dict[str, int | None] = {}
    for vc_file in [gen_dir / "verbal_conf.jsonl", gen_dir / "verbal_confidence.jsonl"]:
        if vc_file.exists():
            for line in vc_file.read_text().splitlines():
                try:
                    obj = json.loads(line)
                    vc_map[f"{obj['qid']}_{obj['gen_idx']}"] = obj.get("verbal_conf")
                except Exception:
                    pass
            break
    print(f"[failures] loaded verbal confidence for {len(vc_map)} examples")

    # Load probe direction
    probe_data = np.load(
        PROBES_DIR / args.probe_tag / "best_probe_direction.npz", allow_pickle=True
    )
    print(f"[failures] probe at L{int(probe_data['layer'][0])}, pos={probe_data['position'][0]}")

    # Get probe predictions
    print("[failures] computing probe predictions...")
    probs = load_probe_predictions(rows, probe_data)
    valid_mask = ~np.isnan(probs)
    rows_v = [r for r, v in zip(rows, valid_mask) if v]
    probs_v = probs[valid_mask]

    y_true = np.array([int(r["is_correct"]) for r in rows_v])
    y_pred = (probs_v >= args.probe_threshold).astype(int)

    tp = np.where((y_pred == 1) & (y_true == 1))[0]
    fp = np.where((y_pred == 1) & (y_true == 0))[0]  # probe overconfident
    tn = np.where((y_pred == 0) & (y_true == 0))[0]
    fn = np.where((y_pred == 0) & (y_true == 1))[0]  # probe underconfident

    print(f"[failures] TP={len(tp)} FP={len(fp)} TN={len(tn)} FN={len(fn)}")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Confusion matrix figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n = len(rows_v)

    # 4-quadrant bar
    labels = ["TP\n(probe✓, ans✓)", "FP\n(probe✓, ans✗)",
              "TN\n(probe✗, ans✗)", "FN\n(probe✗, ans✓)"]
    counts = [len(tp), len(fp), len(tn), len(fn)]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
    bars = axes[0].bar(labels, [100 * c / n for c in counts], color=colors, alpha=0.85, width=0.6)
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.4, f"{c}", ha="center", fontsize=9)
    axes[0].set_ylabel("% of non-abstain examples")
    axes[0].set_title(f"Probe prediction quadrants\n(threshold={args.probe_threshold:.1f}, n={n})")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(alpha=0.2)

    # p(correct) histogram by actual correctness
    axes[1].hist(probs_v[y_true == 1], bins=30, color="#2ca02c", alpha=0.6, label="actually correct")
    axes[1].hist(probs_v[y_true == 0], bins=30, color="#d62728", alpha=0.6, label="actually wrong")
    axes[1].axvline(args.probe_threshold, color="grey", linewidth=1, linestyle="--",
                    label=f"threshold {args.probe_threshold:.1f}")
    axes[1].set_xlabel("Probe p(correct)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Probe score distribution by ground truth")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(plots_dir / "probe_failure_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── Qualitative markdown report ─────────────────────────────────────────
    def fmt_examples(indices, title, rows_v, probs_v, vc_map, n_max):
        # Sort: FPs by highest probe score (most overconfident); FNs by lowest score
        is_fp = "overconfident" in title.lower() or "false pos" in title.lower()
        sorted_idx = sorted(indices,
                            key=lambda i: -probs_v[i] if is_fp else probs_v[i])[:n_max]
        lines = [f"### {title} (n={len(indices)})\n"]
        for i in sorted_idx:
            r = rows_v[i]
            vc = vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}")
            vc_str = f"{vc}" if vc is not None else "n/a"
            gold = r["gold_answers"][0] if r["gold_answers"] else "?"
            lp = np.mean(r.get("token_logprobs", [0]) or [0])
            lines.append(
                f"- **Q**: {r['question']}\n"
                f"  - **Model answer**: `{r['answer_text']}`  "
                f"**Gold**: `{gold}`\n"
                f"  - Probe p(correct)={probs_v[i]:.3f}  "
                f"mean-logprob={lp:.2f}  verbal_conf={vc_str}\n"
            )
        return "\n".join(lines)

    md_lines = [
        f"# Probe failure analysis — {args.probe_tag}",
        "",
        f"Probe at L{int(probe_data['layer'][0])}, pos={probe_data['position'][0]}, "
        f"threshold={args.probe_threshold}",
        f"n={n}  TP={len(tp)} ({100*len(tp)//n}%)  FP={len(fp)} ({100*len(fp)//n}%)  "
        f"TN={len(tn)} ({100*len(tn)//n}%)  FN={len(fn)} ({100*len(fn)//n}%)",
        "",
        "**False Positives (probe overconfident)**: probe says the model is correct, but it is wrong.",
        "These are the 'hidden failures' — the probe is fooled in the same way the model is.",
        "",
        fmt_examples(fp, "False Positives — probe overconfident (probe✓, ans✗)",
                     rows_v, probs_v, vc_map, args.n_examples),
        "",
        "**False Negatives (probe underconfident)**: probe says the model is wrong, but it is right.",
        "These show cases where the model generates a correct answer despite the probe predicting incorrectness.",
        "",
        fmt_examples(fn, "False Negatives — probe underconfident (probe✗, ans✓)",
                     rows_v, probs_v, vc_map, args.n_examples),
    ]

    md_path = out_dir / "probe_failures.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[failures] wrote {md_path}")
    print(f"[failures] plot: {plots_dir / 'probe_failure_distribution.png'}")

    # Summary JSON
    summary = {
        "n": n,
        "tp": len(tp), "fp": len(fp), "tn": len(tn), "fn": len(fn),
        "precision": len(tp) / max(len(tp) + len(fp), 1),
        "recall": len(tp) / max(len(tp) + len(fn), 1),
        "fp_rate": len(fp) / max(len(fp) + len(tn), 1),
        "fn_rate": len(fn) / max(len(fn) + len(tp), 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
