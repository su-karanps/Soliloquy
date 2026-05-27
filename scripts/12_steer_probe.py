"""Experiment 7 (causal): Directional probe steering.

Takes the correctness-probe direction from 03_probe_layers.py and adds
alpha * direction to the residual stream at (best_layer, prompt_last) during
generation.  Measures:
  - whether the generated answer flips from incorrect → correct
  - P(abstain): does the model start saying "I don't know"?
  - logit(correct_tok) - logit(model_original_tok) at the first answer step
  - mean token log-prob (surrogate for confidence)

Sweep alpha from negative (steering away from correctness) to positive (toward).
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DEFAULT_MODEL, PLOTS_DIR, PROBES_DIR, TABLES_DIR
from src.generation import build_prompt
from src.grading import grade, is_abstain
from src.patching import steer_and_generate
from src.utils import read_jsonl


def load_direction(probe_dir: Path) -> dict:
    d = np.load(probe_dir / "best_probe_direction.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True,
                    help="dir with graded.jsonl (greedy run, so one gen per qid)")
    ap.add_argument("--probe-tag", required=True,
                    help="out-tag that was passed to 03_probe_layers.py; used to load direction")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0])
    ap.add_argument("--max-examples", type=int, default=100,
                    help="Number of INCORRECT questions to steer")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")

    probe_dir = PROBES_DIR / args.probe_tag
    direction_data = load_direction(probe_dir)
    coef_norm = torch.tensor(direction_data["coef_norm"], dtype=torch.float32)
    best_layer = int(direction_data["layer"][0])
    best_position_name = str(direction_data["position"][0])
    print(f"[steer] loaded probe direction: layer={best_layer}, pos={best_position_name}")

    gen_dir = Path(args.gen_dir)
    rows = read_jsonl(gen_dir / "graded.jsonl")
    # Only steer on incorrect non-abstain questions (these are the interesting ones)
    incorrect_rows = [r for r in rows if not r["is_correct"] and not r.get("did_abstain")]
    incorrect_rows = incorrect_rows[:args.max_examples]
    print(f"[steer] steering on {len(incorrect_rows)} incorrect questions")
    if not incorrect_rows:
        print("[steer] nothing to steer; aborting")
        return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    # Determine the prompt_last position (= prompt_len - 1)
    # We always steer at the LAST PROMPT TOKEN regardless of best_position_name
    # because that is the position that influences the first generated token.
    steer_layer = best_layer

    alpha_results: dict[float, dict] = {}

    for alpha in args.alphas:
        n_correct_after = 0
        n_abstain_after = 0
        logit_diffs = []
        mean_logprobs = []

        for r in tqdm(incorrect_rows, desc=f"alpha={alpha:+.1f}"):
            prompt_text = build_prompt(tok, r["question"], style="force")
            enc = tok(prompt_text, return_tensors="pt").to(model.device)
            prompt_len = enc.input_ids.shape[1]
            position = prompt_len - 1  # last prompt token

            if alpha == 0.0:
                # Baseline: just run generation without steering
                import torch.nn.functional as F
                with torch.no_grad():
                    out = model.generate(
                        enc.input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                gen_ids = out.sequences[0, prompt_len:]
                answer_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                token_logprobs = []
                for step, score in enumerate(out.scores):
                    tid = out.sequences[0, prompt_len + step].item()
                    lp = float(F.log_softmax(score[0].float(), dim=-1)[tid].item())
                    token_logprobs.append(lp)
                first_logits = out.scores[0][0].cpu().float() if out.scores else None
            else:
                result = steer_and_generate(
                    model, tok, enc.input_ids,
                    direction=coef_norm,
                    alpha=alpha,
                    layer=steer_layer,
                    position=position,
                    max_new_tokens=args.max_new_tokens,
                )
                answer_text = result["answer_text"]
                token_logprobs = result["token_logprobs"]
                first_logits = result["first_token_logits"]

            g = grade(answer_text, r["gold_answers"])
            if g["is_correct"]:
                n_correct_after += 1
            if g["did_abstain"] or is_abstain(answer_text):
                n_abstain_after += 1

            # Logit diff at first generated token: gold vs original wrong answer
            if first_logits is not None and r["gold_answers"] and r["answer_text"]:
                gold_ids = tok.encode(r["gold_answers"][0].strip(), add_special_tokens=False)
                wrong_ids = tok.encode(r["answer_text"].strip(), add_special_tokens=False)
                if gold_ids and wrong_ids:
                    ld = float(first_logits[gold_ids[0]].item() - first_logits[wrong_ids[0]].item())
                    logit_diffs.append(ld)

            if token_logprobs:
                mean_logprobs.append(float(np.mean(token_logprobs)))

        n = len(incorrect_rows)
        alpha_results[alpha] = {
            "alpha": alpha,
            "n": n,
            "correct_rate": n_correct_after / n,
            "abstain_rate": n_abstain_after / n,
            "mean_logit_diff": float(np.mean(logit_diffs)) if logit_diffs else float("nan"),
            "mean_logprob": float(np.mean(mean_logprobs)) if mean_logprobs else float("nan"),
        }
        print(f"  alpha={alpha:+.1f}  correct={n_correct_after/n:.3f}  "
              f"abstain={n_abstain_after/n:.3f}  ld={alpha_results[alpha]['mean_logit_diff']:.2f}")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows_out = list(alpha_results.values())
    (out_dir / "steering_results.json").write_text(json.dumps(rows_out, indent=2))

    with (tables_dir / "steering.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "n", "correct_rate", "abstain_rate",
                                          "mean_logit_diff", "mean_logprob"])
        w.writeheader()
        w.writerows(rows_out)

    alphas = [r["alpha"] for r in rows_out]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].plot(alphas, [r["correct_rate"] for r in rows_out], marker="o", color="#2ca02c")
    axes[0].axvline(0, color="grey", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("α"); axes[0].set_ylabel("P(correct after steering)")
    axes[0].set_title("Correct rate vs α")

    axes[1].plot(alphas, [r["abstain_rate"] for r in rows_out], marker="o", color="#1f77b4")
    axes[1].axvline(0, color="grey", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("α"); axes[1].set_ylabel("P(abstain)")
    axes[1].set_title("Abstain rate vs α")

    axes[2].plot(alphas, [r["mean_logit_diff"] for r in rows_out], marker="o", color="#ff7f0e")
    axes[2].axhline(0, color="grey", linewidth=0.8, linestyle=":")
    axes[2].axvline(0, color="grey", linewidth=0.8, linestyle=":")
    axes[2].set_xlabel("α"); axes[2].set_ylabel("logit(gold) − logit(wrong)")
    axes[2].set_title("Logit diff at first answer token vs α")

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Probe-direction steering — {args.out_tag} (layer={steer_layer})", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "steering_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[steer] wrote {out_dir / 'steering_results.json'} and plots in {plots_dir}")


if __name__ == "__main__":
    main()
