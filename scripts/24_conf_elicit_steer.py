"""Steer the verbal-confidence direction DURING the confidence elicitation pass.

Previous experiment (script 22) steered during answer generation, then re-asked
confidence in a fresh unsteered pass — so the steered hidden state never touched
the confidence output. This script fixes that.

Protocol:
  For each example, the confidence-elicitation prompt is:
    "Question: {Q}\nYour answer: {A}\nHow confident are you (0–100)? Confidence:"
  We run that prompt through the model with the verbal-confidence direction steered
  at the MLP of a target layer, and record the generated number.

  Directions tested:
    -α × verbal_conf_dir  (push away from 'confident' representation → expect ↓)
    +α × verbal_conf_dir  (push toward 'confident' representation → expect ↑)
    -α × corr_dir         (push toward 'incorrect' representation → control)

  Groups:
    confidently-wrong  (is_correct=False, verbal_conf ≥ threshold)
    confidently-correct (is_correct=True,  verbal_conf ≥ threshold) — control

Outputs:
  - conf_elicit_results.json  per-group, per-direction, per-alpha summary
  - conf_elicit_examples.json qualitative examples with steered vs original conf
  - plots/conf_elicit_steer_curves.png
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DEFAULT_MODEL, PLOTS_DIR, PROBES_DIR
from src.utils import read_jsonl


def fit_verbal_conf_probe(rows_all, vc_map, probe_layer, probe_pos, threshold):
    """Fit a logistic regression on hidden states to predict verbal conf >= threshold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = [], []
    for r in rows_all:
        if r.get("did_abstain"):
            continue
        vc = vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}")
        if vc is None:
            continue
        hs_path = r.get("hidden_states_path")
        if not hs_path or not Path(hs_path).exists():
            continue
        hs = torch.load(hs_path, map_location="cpu", weights_only=False)
        h = hs[probe_pos][probe_layer].numpy().astype(np.float32)
        X.append(h)
        y.append(1 if vc >= threshold else 0)

    if len(X) < 20 or sum(y) < 5:
        return None, None

    sc = StandardScaler()
    Xs = sc.fit_transform(np.stack(X))
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(Xs, np.array(y))
    coef = clf.coef_.squeeze()
    direction = torch.tensor((coef / (np.linalg.norm(coef) + 1e-12)).astype(np.float32))
    return direction, sc


@contextlib.contextmanager
def mlp_steer_hook(module: nn.Module, direction: torch.Tensor, alpha: float):
    """Add alpha * direction to every token position of the MLP output."""
    d = direction.clone()

    def hook(mod, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        h = h.clone()
        h = h + alpha * d.to(h.device, h.dtype)
        return (h,) + out[1:] if is_tuple else h

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def build_conf_prompt(tok, question: str, answer: str) -> str:
    content = (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"How confident are you that your answer is correct? "
        f"Reply with a single integer from 0 to 100.\n"
        f"Confidence:"
    )
    messages = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return content


def parse_conf(text: str) -> int | None:
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        v = int(m.group(1))
        return v if 0 <= v <= 100 else None
    return None


@torch.no_grad()
def generate_conf(model, tok, prompt: str, mlp, direction, alpha, max_new=6) -> int | None:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with mlp_steer_hook(mlp, direction, alpha):
        out = model.generate(enc.input_ids, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    raw = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return parse_conf(raw), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--correctness-probe-tag", required=True)
    ap.add_argument("--steer-layer", type=int, default=35,
                    help="MLP layer to steer at (default: peak rescue layer)")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 4.0, 8.0, 16.0, 32.0])
    ap.add_argument("--max-examples", type=int, default=60)
    ap.add_argument("--high-conf-threshold", type=int, default=70)
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")

    gen_dir = Path(args.gen_dir)
    rows_all = list(read_jsonl(gen_dir / "graded.jsonl"))

    # Load verbal confidence
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
    print(f"[conf-elicit] verbal conf for {len(vc_map)} rows")

    # Load correctness probe direction
    corr_data = np.load(
        PROBES_DIR / args.correctness_probe_tag / "best_probe_direction.npz",
        allow_pickle=True
    )
    corr_dir = torch.tensor(corr_data["coef_norm"].astype(np.float32))
    probe_layer = int(corr_data["layer"][0])
    probe_pos = str(corr_data["position"][0])
    print(f"[conf-elicit] correctness probe at L{probe_layer}, pos={probe_pos}")

    # Fit verbal-confidence probe direction
    vc_dir, vc_scaler = fit_verbal_conf_probe(
        rows_all, vc_map, probe_layer, probe_pos, args.high_conf_threshold
    )
    if vc_dir is None:
        print("[conf-elicit] too few examples with verbal confidence; aborting")
        return
    cos_sim = float(torch.dot(corr_dir, vc_dir).item())
    print(f"[conf-elicit] cos(correctness_dir, verbal_conf_dir) = {cos_sim:.3f}")

    # Select examples
    rows_non_abs = [r for r in rows_all if not r.get("did_abstain")]
    def has_vc(r):
        return vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}") is not None
    def get_vc(r):
        return vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}", 0) or 0

    conf_wrong = [r for r in rows_non_abs if not r["is_correct"]
                  and has_vc(r) and get_vc(r) >= args.high_conf_threshold]
    conf_correct = [r for r in rows_non_abs if r["is_correct"]
                    and has_vc(r) and get_vc(r) >= args.high_conf_threshold]
    print(f"[conf-elicit] confidently-wrong: {len(conf_wrong)}, "
          f"confidently-correct: {len(conf_correct)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer = min(args.steer_layer, n_layers - 1)
    mlp = model.model.layers[layer].mlp
    print(f"[conf-elicit] steering at L{layer} MLP")

    directions = [
        ("neg_vc",   -vc_dir,   "−α × verbal_conf_dir"),
        ("pos_vc",   +vc_dir,   "+α × verbal_conf_dir"),
        ("neg_corr", -corr_dir, "−α × correctness_dir (control)"),
    ]

    all_results = []
    all_examples = []

    for group_name, group in [("conf_wrong", conf_wrong[:args.max_examples]),
                               ("conf_correct", conf_correct[:args.max_examples])]:
        for dir_key, direction, dir_label in directions:
            group_row = {"group": group_name, "direction": dir_key,
                         "dir_label": dir_label, "by_alpha": []}
            for alpha in args.alphas:
                deltas, new_confs, orig_confs = [], [], []
                examples_this = []
                for r in tqdm(group, desc=f"{group_name}/{dir_key} α={alpha:+.0f}"):
                    orig_vc = get_vc(r)
                    prompt = build_conf_prompt(tok, r["question"], r["answer_text"])
                    new_vc, raw = generate_conf(model, tok, prompt, mlp, direction, alpha)
                    if new_vc is not None:
                        deltas.append(new_vc - orig_vc)
                        new_confs.append(new_vc)
                        orig_confs.append(orig_vc)
                    if alpha == max(args.alphas) and len(examples_this) < 10:
                        examples_this.append({
                            "group": group_name, "direction": dir_key,
                            "alpha": alpha,
                            "question": r["question"],
                            "answer": r["answer_text"],
                            "gold": r["gold_answers"][0] if r["gold_answers"] else "?",
                            "is_correct": r["is_correct"],
                            "orig_vc": orig_vc,
                            "new_vc": new_vc,
                            "delta": new_vc - orig_vc if new_vc is not None else None,
                            "raw_output": raw,
                        })
                mean_delta = float(np.mean(deltas)) if deltas else float("nan")
                mean_new = float(np.mean(new_confs)) if new_confs else float("nan")
                n_down = sum(1 for d in deltas if d < -5)
                n_up = sum(1 for d in deltas if d > 5)
                print(f"  {group_name}/{dir_key} α={alpha:+.0f}: "
                      f"mean_delta={mean_delta:+.1f}  "
                      f"↓>{5}: {n_down}/{len(deltas)}  ↑>{5}: {n_up}/{len(deltas)}")
                group_row["by_alpha"].append({
                    "alpha": alpha, "mean_delta": mean_delta,
                    "mean_new_conf": mean_new,
                    "n_measured": len(deltas),
                    "n_decreased_gt5": n_down,
                    "n_increased_gt5": n_up,
                })
                all_examples.extend(examples_this)
            all_results.append(group_row)

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "conf_elicit_results.json").write_text(json.dumps(all_results, indent=2))
    (out_dir / "conf_elicit_examples.json").write_text(json.dumps(all_examples, indent=2))

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dir_styles = {
        "neg_vc":   ("#d62728", "solid",  "−α × verbal_conf_dir"),
        "pos_vc":   ("#2ca02c", "solid",  "+α × verbal_conf_dir"),
        "neg_corr": ("#1f77b4", "dashed", "−α × correctness_dir"),
    }

    for ax, group_name, title in [
        (axes[0], "conf_wrong",   "Confidently-wrong examples"),
        (axes[1], "conf_correct", "Confidently-correct (control)"),
    ]:
        for row in all_results:
            if row["group"] != group_name:
                continue
            color, ls, label = dir_styles[row["direction"]]
            alphas_p = [x["alpha"] for x in row["by_alpha"]]
            deltas_p = [x["mean_delta"] for x in row["by_alpha"]]
            ax.plot(alphas_p, deltas_p, color=color, linestyle=ls,
                    marker="o", linewidth=2.2, label=label)
        ax.axhline(0, color="#aaa", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Steering α", fontsize=14)
        ax.set_ylabel("Mean Δ verbal confidence", fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.legend(frameon=False, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.18)

    fig.suptitle(
        f"Effect of L{layer}-MLP steering on stated confidence\n"
        f"(steered DURING confidence elicitation pass)",
        fontsize=13
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "conf_elicit_steer_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── Markdown qualitative report ─────────────────────────────────────────
    lines = [f"# Confidence elicitation steering — qualitative examples\n",
             f"Steered at L{layer} MLP during the confidence question forward pass.\n"]
    for ex in all_examples:
        if ex["direction"] != "neg_vc":
            continue
        delta_str = f"{ex['delta']:+d}" if ex["delta"] is not None else "n/a"
        lines += [
            f"- **Q**: {ex['question']}",
            f"  - **Answer**: `{ex['answer']}`  **Gold**: `{ex['gold']}`  "
            f"correct={ex['is_correct']}",
            f"  - Orig conf={ex['orig_vc']}  Steered conf={ex['new_vc']}  Δ={delta_str}",
            f"  - Raw output: `{ex['raw_output']}`",
            "",
        ]
    (out_dir / "qualitative_report.md").write_text("\n".join(lines))
    print(f"\n[conf-elicit] results → {out_dir}")
    print(f"[conf-elicit] plot   → {plots_dir / 'conf_elicit_steer_curves.png'}")


if __name__ == "__main__":
    main()
