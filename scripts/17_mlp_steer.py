"""Experiment 7 v2: Directional steering at the peak-rescue MLP, not the prompt.

12_steer_probe.py steered the residual stream at the probe's preferred (early)
layer and got a null result. This script tests the alternative: steer at the
LATE-layer MLP output (where 14_patch_residual.py found the causal bottleneck).

Three modes per α:
  - add:        h ← h + α · correctness_direction
  - subtract:   h ← h − α · correctness_direction
  - project_out: h ← h − (h · d) d  (remove the correctness component entirely)

Applied at the LAST PROMPT TOKEN position of the MLP output at the specified
layer. Measures change in correct-answer rate, abstain rate, mean logprob,
and verbalized confidence.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
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


@contextlib.contextmanager
def mlp_steer_hook(module: nn.Module, direction: torch.Tensor, mode: str, alpha: float,
                   position: int):
    """Apply directional steering to the MLP output at `position`."""
    direction = direction.clone()

    def hook(mod, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        if h.shape[1] <= position:  # KV-cached generation step — skip
            return out
        h = h.clone()
        d = direction.to(h.device, h.dtype)
        if mode == "add":
            h[:, position, :] = h[:, position, :] + alpha * d
        elif mode == "subtract":
            h[:, position, :] = h[:, position, :] - alpha * d
        elif mode == "project_out":
            v = h[:, position, :]                       # (B, D)
            proj = (v * d).sum(dim=-1, keepdim=True) * d  # (B, D)
            h[:, position, :] = v - alpha * proj
        else:
            raise ValueError(mode)
        if is_tuple:
            return (h,) + out[1:]
        return h

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def gen_with_steer(model, tok, input_ids, direction, mode, alpha, layer, position,
                   max_new_tokens):
    mlp = model.model.layers[layer].mlp
    with mlp_steer_hook(mlp, direction, mode, alpha, position):
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    prompt_len = input_ids.shape[1]
    gen_ids = out.sequences[0, prompt_len:]
    answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
    import torch.nn.functional as F
    lps = []
    for step, score in enumerate(out.scores):
        tid = out.sequences[0, prompt_len + step].item()
        lps.append(float(F.log_softmax(score[0].float(), dim=-1)[tid].item()))
    return {"answer_text": answer, "token_logprobs": lps,
            "first_logits": out.scores[0][0].cpu().float() if out.scores else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--probe-tag", required=True,
                    help="probe out-tag (loads best_probe_direction.npz)")
    ap.add_argument("--patch-summary", required=True,
                    help="14_patch_residual.py summary.json for peak layer")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--max-examples", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--target", choices=["incorrect", "correct"], default="incorrect",
                    help="Steer to flip *incorrect* generations (rescue) or *correct* (corrupt)")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")

    # Load the probe direction (unit vector in standardised hidden-state space).
    pdata = np.load(PROBES_DIR / args.probe_tag / "best_probe_direction.npz", allow_pickle=True)
    coef_norm = torch.tensor(pdata["coef_norm"], dtype=torch.float32)
    print(f"[mlp-steer] loaded probe direction "
          f"(L{int(pdata['layer'][0])}, pos={pdata['position'][0]})")

    # Load peak MLP layer from patching results.
    patch_summary = json.loads(Path(args.patch_summary).read_text())
    target_layer = patch_summary.get("component_patch_layer", patch_summary["peak_rescue_layer"])
    print(f"[mlp-steer] steering at MLP of L{target_layer} (from patching summary)")

    gen_dir = Path(args.gen_dir)
    rows = read_jsonl(gen_dir / "graded.jsonl")
    if args.target == "incorrect":
        target_rows = [r for r in rows if not r["is_correct"] and not r.get("did_abstain")]
    else:
        target_rows = [r for r in rows if r["is_correct"] and not r.get("did_abstain")]
    target_rows = target_rows[:args.max_examples]
    print(f"[mlp-steer] target={args.target}: steering on {len(target_rows)} questions")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    results = []
    for mode in ["add", "subtract", "project_out"]:
        alphas = args.alphas if mode != "project_out" else [0.0, 0.5, 1.0]
        for alpha in alphas:
            n_correct = 0
            n_abstain = 0
            mean_lps = []
            n_flipped = 0
            for r in tqdm(target_rows, desc=f"{mode} α={alpha:+.1f}"):
                prompt_text = build_prompt(tok, r["question"], style="force")
                enc = tok(prompt_text, return_tensors="pt").to(model.device)
                pos = enc.input_ids.shape[1] - 1
                if alpha == 0.0 and mode == "add":
                    # Use cached baseline result
                    out = gen_with_steer(model, tok, enc.input_ids, coef_norm, "add", 0.0,
                                         target_layer, pos, args.max_new_tokens)
                else:
                    out = gen_with_steer(model, tok, enc.input_ids, coef_norm, mode, alpha,
                                         target_layer, pos, args.max_new_tokens)
                g = grade(out["answer_text"], r["gold_answers"])
                if g["is_correct"]:
                    n_correct += 1
                    if not r["is_correct"]:
                        n_flipped += 1
                if g["did_abstain"] or is_abstain(out["answer_text"]):
                    n_abstain += 1
                if out["token_logprobs"]:
                    mean_lps.append(float(np.mean(out["token_logprobs"])))

            n = len(target_rows)
            results.append({
                "mode": mode,
                "alpha": alpha,
                "target": args.target,
                "n": n,
                "correct_rate": n_correct / n,
                "abstain_rate": n_abstain / n,
                "flipped_to_correct_rate": n_flipped / n if args.target == "incorrect" else None,
                "mean_logprob": float(np.mean(mean_lps)) if mean_lps else float("nan"),
            })
            print(f"  {mode} α={alpha:+5.1f}  correct={n_correct/n:.3f}  "
                  f"abstain={n_abstain/n:.3f}  flipped={n_flipped/n:.3f}  "
                  f"meanLP={results[-1]['mean_logprob']:.2f}")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "mlp_steering_results.json").write_text(json.dumps(results, indent=2))
    with (tables_dir / "mlp_steering.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    colors = {"add": "#2ca02c", "subtract": "#d62728", "project_out": "#1f77b4"}
    for mode in ["add", "subtract", "project_out"]:
        rs = [r for r in results if r["mode"] == mode]
        if not rs:
            continue
        alphas = [r["alpha"] for r in rs]
        axes[0].plot(alphas, [r["correct_rate"] for r in rs], marker="o",
                     color=colors[mode], label=mode)
        axes[1].plot(alphas, [r["abstain_rate"] for r in rs], marker="o",
                     color=colors[mode], label=mode)
        if args.target == "incorrect":
            axes[2].plot(alphas, [r["flipped_to_correct_rate"] for r in rs], marker="o",
                         color=colors[mode], label=mode)

    axes[0].set_xlabel("α"); axes[0].set_ylabel("correct rate"); axes[0].set_title("Correct rate")
    axes[1].set_xlabel("α"); axes[1].set_ylabel("abstain rate"); axes[1].set_title("Abstain rate")
    axes[2].set_xlabel("α"); axes[2].set_ylabel("flipped-to-correct rate"); axes[2].set_title("Flip rate (incorrect→correct)")
    for ax in axes:
        ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"MLP-output steering at L{target_layer} — {args.out_tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "mlp_steering_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[mlp-steer] wrote {out_dir / 'mlp_steering_results.json'} and plots in {plots_dir}")


if __name__ == "__main__":
    main()
