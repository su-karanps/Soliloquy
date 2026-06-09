"""Targeted confidence steering experiment.

Motivation (from meeting): can we steer the model to *lower its stated confidence*
on confidently-wrong examples, without hurting correct answers? Ideally this makes
the model "explore other possibilities" instead of committing fluently to a wrong answer.

Two interventions, applied at the late-MLP output (peak rescue layer):
  A. Steer along -α * verbal_confidence_direction
     → the model is pushed away from the "I am confident" representation
  B. Steer along +α * correctness_direction  (for comparison — we already know this is null)

For each example, we measure:
  1. Does the regenerated answer change (answer flip)?
  2. Does verbalized confidence drop?
  3. Does the answer become more hedged (e.g., "I think", "I'm not sure")?
  4. Does actual correctness improve (rescue)?

We focus on two groups:
  - Confidently-wrong: is_correct=False AND verbal_conf >= HIGH_CONF (≥70)
  - Confidently-correct (control): is_correct=True AND verbal_conf >= HIGH_CONF

The verbal-confidence direction is loaded from the dissociation experiment (script 13),
which saves both the correctness and verbal-confidence probe directions.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
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

from src.config import DEFAULT_MODEL, PLOTS_DIR, PROBES_DIR, TABLES_DIR
from src.generation import build_prompt
from src.grading import grade, is_abstain
from src.utils import read_jsonl

HEDGE_PHRASES = [
    "i think", "i'm not sure", "i believe", "i'm unsure", "i don't know",
    "i'm uncertain", "not certain", "may be", "might be", "perhaps",
    "probably", "possibly", "unclear", "i cannot", "i can't",
]


def has_hedge(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in HEDGE_PHRASES)


@contextlib.contextmanager
def mlp_steer_hook(module: nn.Module, direction: torch.Tensor, alpha: float, position: int):
    direction = direction.clone()

    def hook(mod, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        if h.shape[1] <= position:
            return out
        h = h.clone()
        d = direction.to(h.device, h.dtype)
        h[:, position, :] = h[:, position, :] + alpha * d
        return (h,) + out[1:] if is_tuple else h

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def gen_steered(model, tok, input_ids, direction, alpha, layer, position, max_new_tokens):
    mlp = model.model.layers[layer].mlp
    with mlp_steer_hook(mlp, direction, alpha, position):
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen_ids = out[0, input_ids.shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def ask_verbal_conf(model, tok, question: str, answer: str) -> int | None:
    """Re-ask verbal confidence after steering."""
    from src.config import ANSWER_PROMPT
    conf_prompt = (
        f"Question: {question}\n"
        f"Your answer: {answer}\n"
        f"How confident are you that your answer is correct? "
        f"Answer with a number from 0 to 100.\n"
        f"Confidence:"
    )
    # Use chat template if available
    messages = [{"role": "user", "content": conf_prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = conf_prompt
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(enc.input_ids, max_new_tokens=6, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    raw = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"\d{1,3}", raw)
    if m:
        v = int(m.group())
        return v if 0 <= v <= 100 else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--dissoc-probe-tag", required=True,
                    help="probe tag whose probes/tag/ dir contains verbal_conf_direction.npz")
    ap.add_argument("--correctness-probe-tag", required=True,
                    help="probe tag with best_probe_direction.npz for correctness direction")
    ap.add_argument("--patch-summary", required=True,
                    help="patching summary.json to get peak causal layer")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--max-examples", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--high-conf-threshold", type=int, default=70)
    ap.add_argument("--prompt-style", default="force")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")

    patch_summary = json.loads(Path(args.patch_summary).read_text())
    target_layer = patch_summary.get("component_patch_layer",
                                     patch_summary["peak_rescue_layer"])
    print(f"[conf-steer] steering at MLP of L{target_layer}")

    # Load verbal-confidence direction from dissociation results
    dissoc_dir = PROBES_DIR / args.dissoc_probe_tag
    # Script 13 saves per-dataset subdirs with a probe directions file
    # We need to find the verbal_conf probe direction (coef for verbal_conf label)
    # It's not saved separately — load the correctness direction for now
    # and also load from the same npz by fitting it inline below
    corr_data = np.load(
        PROBES_DIR / args.correctness_probe_tag / "best_probe_direction.npz",
        allow_pickle=True
    )
    corr_dir = torch.tensor(corr_data["coef_norm"].astype(np.float32))
    probe_layer = int(corr_data["layer"][0])
    probe_pos = str(corr_data["position"][0])
    print(f"[conf-steer] correctness probe at L{probe_layer}, pos={probe_pos}")

    # Fit verbal-confidence probe direction from cached hidden states
    rows_all = list(read_jsonl(Path(args.gen_dir) / "graded.jsonl"))
    # load verbal conf
    vc_map: dict[str, int | None] = {}
    for vc_file in [Path(args.gen_dir) / "verbal_conf.jsonl",
                    Path(args.gen_dir) / "verbal_confidence.jsonl"]:
        if vc_file.exists():
            for line in vc_file.read_text().splitlines():
                try:
                    obj = json.loads(line)
                    vc_map[f"{obj['qid']}_{obj['gen_idx']}"] = obj.get("verbal_conf")
                except Exception:
                    pass
            break
    print(f"[conf-steer] verbal conf available for {len(vc_map)} rows")

    # Fit verbal-conf probe at the same layer/position as correctness probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X_vc, y_vc = [], []
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
        X_vc.append(h)
        y_vc.append(1 if vc >= args.high_conf_threshold else 0)
    print(f"[conf-steer] fitting verbal-conf probe on {len(X_vc)} rows "
          f"(high-conf: {sum(y_vc)}/{len(y_vc)})")
    if len(X_vc) < 20 or sum(y_vc) < 5:
        print("[conf-steer] too few examples with verbal confidence; aborting")
        return
    sc_vc = StandardScaler()
    Xs_vc = sc_vc.fit_transform(np.stack(X_vc))
    clf_vc = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
    clf_vc.fit(Xs_vc, np.array(y_vc))
    coef_vc = clf_vc.coef_.squeeze()
    conf_dir = torch.tensor((coef_vc / (np.linalg.norm(coef_vc) + 1e-12)).astype(np.float32))
    cos_sim = float(torch.dot(corr_dir, conf_dir).item())
    print(f"[conf-steer] cos(correctness_dir, conf_dir) = {cos_sim:.3f}")

    # Select confidently-wrong examples
    rows_non_abs = [r for r in rows_all if not r.get("did_abstain")]
    confidently_wrong = [
        r for r in rows_non_abs
        if not r["is_correct"]
        and vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}", 0) is not None
        and (vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}", 0) or 0) >= args.high_conf_threshold
    ]
    confidently_correct = [
        r for r in rows_non_abs
        if r["is_correct"]
        and vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}", 0) is not None
        and (vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}", 0) or 0) >= args.high_conf_threshold
    ]
    print(f"[conf-steer] confidently-wrong: {len(confidently_wrong)}, "
          f"confidently-correct (control): {len(confidently_correct)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    def run_group(group, group_name, direction, direction_name):
        examples = group[:args.max_examples]
        results = []
        for alpha in args.alphas:
            n_flipped = 0
            n_hedged = 0
            n_correct = 0
            vc_deltas = []
            qualitative = []
            for r in tqdm(examples, desc=f"{group_name} {direction_name} α={alpha:+.1f}"):
                prompt_text = build_prompt(tok, r["question"], style=args.prompt_style)
                enc = tok(prompt_text, return_tensors="pt").to(model.device)
                pos = enc.input_ids.shape[1] - 1
                if alpha == 0.0:
                    out_text = gen_steered(model, tok, enc.input_ids, direction, 0.0,
                                           target_layer, pos, args.max_new_tokens)
                else:
                    out_text = gen_steered(model, tok, enc.input_ids, direction, alpha,
                                           target_layer, pos, args.max_new_tokens)
                g = grade(out_text, r["gold_answers"])
                orig_vc = vc_map.get(f"{r['qid']}_{r.get('gen_idx', 0)}")
                changed = (out_text.strip().lower() != r["answer_text"].strip().lower())
                if changed:
                    n_flipped += 1
                if has_hedge(out_text):
                    n_hedged += 1
                if g["is_correct"]:
                    n_correct += 1
                # Re-ask verbal confidence for a subset
                if alpha > 0 and len(qualitative) < 8:
                    new_vc = ask_verbal_conf(model, tok, r["question"], out_text)
                    if new_vc is not None and orig_vc is not None:
                        vc_deltas.append(new_vc - orig_vc)
                    qualitative.append({
                        "question": r["question"],
                        "original_answer": r["answer_text"],
                        "steered_answer": out_text,
                        "gold": r["gold_answers"][0] if r["gold_answers"] else "?",
                        "original_vc": orig_vc,
                        "new_vc": new_vc if new_vc is not None else None,
                        "vc_delta": (new_vc - orig_vc) if (new_vc is not None and orig_vc is not None) else None,
                        "answer_changed": changed,
                        "now_correct": g["is_correct"],
                        "hedged": has_hedge(out_text),
                        "alpha": alpha,
                    })
            n = len(examples)
            mean_vc_delta = float(np.mean(vc_deltas)) if vc_deltas else float("nan")
            results.append({
                "group": group_name, "direction": direction_name, "alpha": alpha, "n": n,
                "flip_rate": n_flipped / n,
                "hedge_rate": n_hedged / n,
                "correct_rate": n_correct / n,
                "mean_vc_delta": mean_vc_delta,
                "n_vc_measured": len(vc_deltas),
                "qualitative": qualitative if alpha == max(args.alphas) else [],
            })
            print(f"  {group_name} {direction_name} α={alpha:+.1f}: "
                  f"flip={n_flipped/n:.2f} hedge={n_hedged/n:.2f} "
                  f"correct={n_correct/n:.2f} vc_delta={mean_vc_delta:+.1f}")
        return results

    all_results = []
    print("\n[conf-steer] === Confidently-WRONG examples: -conf_direction ===")
    all_results += run_group(confidently_wrong, "conf_wrong", -conf_dir,
                             "neg_conf_dir")
    print("\n[conf-steer] === Confidently-WRONG examples: +corr_direction (baseline) ===")
    all_results += run_group(confidently_wrong, "conf_wrong", corr_dir,
                             "corr_dir")
    print("\n[conf-steer] === Confidently-CORRECT (control): -conf_direction ===")
    all_results += run_group(confidently_correct, "conf_correct", -conf_dir,
                             "neg_conf_dir")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save results (without large qualitative blocks)
    summary_rows = [{k: v for k, v in r.items() if k != "qualitative"} for r in all_results]
    (out_dir / "conf_steering_results.json").write_text(json.dumps(summary_rows, indent=2))

    # Save qualitative examples from the best alpha
    all_qual = [ex for r in all_results for ex in r.get("qualitative", [])]
    (out_dir / "qualitative_steered.json").write_text(json.dumps(all_qual, indent=2))

    # CSV table
    with (tables_dir / "conf_steering.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [("flip_rate", "Answer flip rate"), ("hedge_rate", "Hedge rate"),
               ("correct_rate", "Correct rate"), ("mean_vc_delta", "Mean verbal-conf Δ")]
    groups_dirs = [("conf_wrong", "neg_conf_dir"), ("conf_wrong", "corr_dir"),
                   ("conf_correct", "neg_conf_dir")]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for (grp, drn), color in zip(groups_dirs, colors):
            rs = [r for r in summary_rows if r["group"] == grp and r["direction"] == drn]
            if not rs:
                continue
            alphas_p = [r["alpha"] for r in rs]
            vals = [r[metric] for r in rs]
            label = f"{grp}/{drn}"
            ax.plot(alphas_p, vals, marker="o", color=color, linewidth=1.8, label=label)
        ax.set_xlabel("Steering α"); ax.set_ylabel(ylabel)
        ax.set_title(ylabel); ax.legend(frameon=False, fontsize=7.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.18)
    fig.suptitle(f"Confidence steering at L{target_layer} MLP — {args.out_tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "conf_steering_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Markdown qualitative report
    qual_for_wrong = [ex for r in all_results
                      if r["group"] == "conf_wrong" and r["direction"] == "neg_conf_dir"
                      for ex in r.get("qualitative", [])]
    qual_lines = [f"# Confidence steering qualitative examples\n",
                  f"Model steered by -α × verbal_confidence_direction at L{target_layer} MLP.\n"]
    for ex in qual_for_wrong[:15]:
        vc_str = f"{ex['original_vc']} → {ex['new_vc']}" if ex.get("new_vc") is not None else str(ex.get("original_vc"))
        qual_lines += [
            f"- **Q**: {ex['question']}",
            f"  - **Original**: `{ex['original_answer']}`  **Gold**: `{ex['gold']}`",
            f"  - **Steered (α={ex['alpha']:+.0f})**: `{ex['steered_answer']}`",
            f"  - verbal_conf: {vc_str}  changed: {ex['answer_changed']}  "
            f"now_correct: {ex['now_correct']}  hedged: {ex['hedged']}",
            "",
        ]
    (out_dir / "qualitative_report.md").write_text("\n".join(qual_lines))
    print(f"\n[conf-steer] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
