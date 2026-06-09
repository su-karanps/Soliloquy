"""Generate answer + confidence jointly in one pass, then compare with
the existing separate-pass verbal confidence numbers.

Writes:
  <gen_dir>/verbal_conf_joint.jsonl   — answer + confidence from single generation
  probes/<out_tag>/joint_vs_separate.json  — comparison statistics
  plots/<out_tag>/joint_vs_separate.png    — scatter + distribution plot

Key comparisons:
  1. Do the two formats agree on confidence level? (correlation)
  2. Is the dissociation with actual correctness stronger/weaker with joint format?
  3. Does the quadrant breakdown (correct+confident / incorrect+confident etc.) shift?
  4. How often do the answer texts agree between formats? (answer consistency check)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DEFAULT_MODEL, PLOTS_DIR, PROBES_DIR
from src.generation import GenConfig, run_joint_generation
from src.grading import grade
from src.utils import read_jsonl

HIGH_CONF = 70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")

    gen_dir = Path(args.gen_dir)

    # ── Step 1: run joint generation ────────────────────────────────────────
    joint_path = gen_dir / "verbal_conf_joint.jsonl"
    cfg = GenConfig(model_name=args.model, max_new_tokens=40)
    questions_jsonl = gen_dir / "generations.jsonl"

    print("[joint] generating joint answer+confidence...")
    run_joint_generation(questions_jsonl, joint_path, cfg, overwrite=args.overwrite)
    print(f"[joint] wrote {joint_path}")

    # ── Step 2: load both conditions and the graded results ─────────────────
    graded = {
        f"{r['qid']}_{r.get('gen_idx', 0)}": r
        for r in read_jsonl(gen_dir / "graded.jsonl")
        if not r.get("did_abstain")
    }

    # Separate-pass confidence
    sep_vc: dict[str, int | None] = {}
    for vc_file in [gen_dir / "verbal_conf.jsonl", gen_dir / "verbal_confidence.jsonl"]:
        if vc_file.exists():
            for line in vc_file.read_text().splitlines():
                try:
                    obj = json.loads(line)
                    sep_vc[f"{obj['qid']}_{obj['gen_idx']}"] = obj.get("verbal_conf")
                except Exception:
                    pass
            break

    # Joint-pass confidence
    joint_rows = {
        f"{r['qid']}_{r['gen_idx']}": r
        for r in read_jsonl(joint_path)
    }

    # ── Step 3: compute comparison stats ────────────────────────────────────
    pairs = []  # (key, is_correct, sep_vc, joint_vc, answer_matches)
    for key, gr in graded.items():
        svc = sep_vc.get(key)
        jrow = joint_rows.get(key)
        if svc is None or jrow is None or jrow.get("verbal_conf") is None:
            continue
        jvc = jrow["verbal_conf"]
        # Check if joint answer agrees with graded answer
        j_ans = jrow.get("answer_text", "")
        orig_ans = gr.get("answer_text", "")
        gold = gr.get("gold_answers", [])
        j_correct = grade(j_ans, gold)["is_correct"] if j_ans else False
        pairs.append({
            "key": key,
            "is_correct": gr["is_correct"],
            "joint_correct": j_correct,
            "sep_vc": svc,
            "joint_vc": jvc,
            "answer_matches": j_ans.strip().lower() == orig_ans.strip().lower(),
        })

    print(f"[joint] {len(pairs)} matched pairs")

    sep_arr = np.array([p["sep_vc"] for p in pairs])
    joint_arr = np.array([p["joint_vc"] for p in pairs])
    corr = float(np.corrcoef(sep_arr, joint_arr)[0, 1])
    answer_match_rate = float(np.mean([p["answer_matches"] for p in pairs]))
    joint_correct_rate = float(np.mean([p["joint_correct"] for p in pairs]))
    orig_correct_rate = float(np.mean([p["is_correct"] for p in pairs]))

    print(f"[joint] correlation(sep_vc, joint_vc) = {corr:.3f}")
    print(f"[joint] answer match rate = {answer_match_rate:.3f}")
    print(f"[joint] orig correct rate = {orig_correct_rate:.3f}  "
          f"joint correct rate = {joint_correct_rate:.3f}")

    def quadrants(vc_arr, is_correct_arr):
        n = len(vc_arr)
        cc = np.sum((vc_arr >= HIGH_CONF) & is_correct_arr)
        cu = np.sum((vc_arr < HIGH_CONF) & is_correct_arr)
        ic = np.sum((vc_arr >= HIGH_CONF) & ~is_correct_arr)
        iu = np.sum((vc_arr < HIGH_CONF) & ~is_correct_arr)
        return {k: int(v) for k, v in
                zip(["correct+confident", "correct+unconfident",
                     "incorrect+confident", "incorrect+unconfident"],
                    [cc, cu, ic, iu])}

    correct_arr = np.array([p["is_correct"] for p in pairs])
    joint_correct_arr = np.array([p["joint_correct"] for p in pairs])

    quads_sep = quadrants(sep_arr, correct_arr)
    quads_joint = quadrants(joint_arr, joint_correct_arr)

    print("[joint] quadrants (separate-pass vc, original correctness):", quads_sep)
    print("[joint] quadrants (joint vc, joint correctness):", quads_joint)

    summary = {
        "n_pairs": len(pairs),
        "correlation_sep_joint": corr,
        "answer_match_rate": answer_match_rate,
        "orig_correct_rate": orig_correct_rate,
        "joint_correct_rate": joint_correct_rate,
        "mean_sep_vc": float(sep_arr.mean()),
        "mean_joint_vc": float(joint_arr.mean()),
        "quadrants_sep_vc": quads_sep,
        "quadrants_joint_vc": quads_joint,
    }

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "joint_vs_separate.json").write_text(json.dumps(summary, indent=2))

    # ── Step 4: plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: scatter sep vs joint confidence
    c = ["#2ca02c" if p["is_correct"] else "#d62728" for p in pairs]
    axes[0].scatter(sep_arr, joint_arr, c=c, alpha=0.35, s=14)
    axes[0].plot([0, 100], [0, 100], color="#aaa", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Separate-pass verbal conf")
    axes[0].set_ylabel("Joint-pass verbal conf")
    axes[0].set_title(f"Confidence agreement\n(r={corr:.2f})")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panel 2: quadrant bars — sep vs joint
    keys_q = ["correct+confident", "correct+unconfident",
               "incorrect+confident", "incorrect+unconfident"]
    labels_q = ["corr+conf", "corr+unconf", "incorr+conf", "incorr+unconf"]
    colors_q = ["#2ca02c", "#aec7e8", "#d62728", "#ff7f0e"]
    n = len(pairs)
    x = np.arange(4)
    w = 0.38
    sep_pcts = [100 * quads_sep[k] / max(n, 1) for k in keys_q]
    joint_pcts = [100 * quads_joint[k] / max(n, 1) for k in keys_q]
    bars1 = axes[1].bar(x - w/2, sep_pcts, w, color=colors_q, alpha=0.55, label="separate-pass")
    bars2 = axes[1].bar(x + w/2, joint_pcts, w, color=colors_q, alpha=0.9, label="joint-pass",
                        edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_q, fontsize=11, rotation=15, ha="right")
    axes[1].set_ylabel("% of responses")
    axes[1].set_title("Confidence quadrants:\nseparate vs joint prompt")
    axes[1].legend(frameon=False, fontsize=11)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # Panel 3: distribution of verbal conf by correctness for each condition
    correct_mask = correct_arr.astype(bool)
    for arr, label, ls in [(sep_arr, "separate", "dashed"), (joint_arr, "joint", "solid")]:
        axes[2].hist(arr[correct_mask], bins=20, alpha=0.25 if ls == "dashed" else 0.45,
                     color="#2ca02c", linestyle=ls,
                     label=f"correct ({label})", density=True)
        axes[2].hist(arr[~correct_mask], bins=20, alpha=0.25 if ls == "dashed" else 0.45,
                     color="#d62728", linestyle=ls,
                     label=f"incorrect ({label})", density=True)
    axes[2].axvline(HIGH_CONF, color="#aaa", linewidth=1, linestyle=":")
    axes[2].set_xlabel("Verbal confidence")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Confidence distribution\nby correctness and format")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.suptitle("Joint vs separate-pass verbal confidence — " + args.out_tag, fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "joint_vs_separate.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[joint] summary → {out_dir / 'joint_vs_separate.json'}")
    print(f"[joint] plot    → {plots_dir / 'joint_vs_separate.png'}")


if __name__ == "__main__":
    main()
