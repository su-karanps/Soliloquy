"""Experiments 5 & 6: Residual-stream activation patching — rescue and corruption.

Setup: cross-question patching.
  - "correct" pool: questions where the model answered correctly (greedy run).
  - "incorrect" pool: questions where the model answered incorrectly (greedy run).

For each incorrect question Q_wrong:
  1. Pick a random correct question Q_right (different question).
  2. Run both through the model; cache the residual stream at each layer at position
     `prompt_last`.
  3. RESCUE: patch hidden state at (layer L, position prompt_last) of the Q_wrong
     run with that from Q_right.  Measure the change in logit_diff =
     logit(first_gold_token_for_Q_wrong) - logit(first_model_token_for_Q_wrong)
     at the position where the model will generate its first answer token.
  4. CORRUPTION: patch hidden state of Q_right with that from Q_wrong.  Measure
     how much logit_diff for Q_right degrades.

The key figure:
  x-axis: layer
  y-axis: rescue_effect = logit_diff(patched) − logit_diff(baseline_wrong)
  secondary: corruption_effect = logit_diff(patched) − logit_diff(baseline_right)

Additionally patch attn_out and mlp_out separately at the peak rescue layer to
localize which component carries the causal effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
from src.patching import (
    cache_component_outputs,
    cache_residual,
    patch_component_and_eval,
    patch_residual_and_eval,
)
from src.utils import read_jsonl


def get_first_token_id(tokenizer, text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def logit_diff_baseline(model, input_ids: torch.Tensor, gold_tok: int, wrong_tok: int) -> float:
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    logits = out.logits[0, -1, :].cpu().float()
    return float(logits[gold_tok].item() - logits[wrong_tok].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True,
                    help="dir with graded.jsonl (greedy, 1 gen per qid)")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pairs", type=int, default=60,
                    help="Number of (wrong, right) pairs to patch per direction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", default="force")
    ap.add_argument("--component-patch-at", type=int, default=None,
                    help="Layer at which to do attn/MLP component patching; "
                         "defaults to the peak rescue layer found by residual patching")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")
    random.seed(args.seed)
    np.random.seed(args.seed)

    gen_dir = Path(args.gen_dir)
    rows = [r for r in read_jsonl(gen_dir / "graded.jsonl") if not r.get("did_abstain")]
    correct_rows = [r for r in rows if r["is_correct"]]
    incorrect_rows = [r for r in rows if not r["is_correct"]]
    print(f"[patch] correct={len(correct_rows)} incorrect={len(incorrect_rows)}")

    n_pairs = min(args.max_pairs, len(incorrect_rows), len(correct_rows))
    if n_pairs < 5:
        print("[patch] too few pairs; aborting")
        return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    # Determine n_layers
    n_layers = model.config.num_hidden_layers
    print(f"[patch] model has {n_layers} layers")

    rescue_effects = np.zeros((n_pairs, n_layers))
    corruption_effects = np.zeros((n_pairs, n_layers))

    wrong_sample = random.sample(incorrect_rows, n_pairs)
    right_pool = correct_rows.copy()

    for i, r_wrong in enumerate(tqdm(wrong_sample, desc="patch-pairs")):
        r_right = random.choice(right_pool)

        gold_tok_wrong = get_first_token_id(tok, r_wrong["gold_answers"][0] if r_wrong["gold_answers"] else "")
        model_tok_wrong = get_first_token_id(tok, r_wrong["answer_text"])
        gold_tok_right = get_first_token_id(tok, r_right["gold_answers"][0] if r_right["gold_answers"] else "")
        model_tok_right = get_first_token_id(tok, r_right["answer_text"])

        if any(t is None for t in [gold_tok_wrong, model_tok_wrong, gold_tok_right, model_tok_right]):
            continue
        if gold_tok_wrong == model_tok_wrong:
            # already correct for this pair — skip
            continue

        prompt_wrong = build_prompt(tok, r_wrong["question"], style=args.prompt_style)
        prompt_right = build_prompt(tok, r_right["question"], style=args.prompt_style)

        enc_wrong = tok(prompt_wrong, return_tensors="pt").to(model.device)
        enc_right = tok(prompt_right, return_tensors="pt").to(model.device)

        pos_wrong = enc_wrong.input_ids.shape[1] - 1  # prompt_last position
        pos_right = enc_right.input_ids.shape[1] - 1

        # Cache full residual streams for both
        hs_wrong = cache_residual(model, enc_wrong.input_ids)  # (n_layers+1, seq_wrong, hidden)
        hs_right = cache_residual(model, enc_right.input_ids)  # (n_layers+1, seq_right, hidden)

        # Baselines (no patch)
        ld_wrong_base = logit_diff_baseline(model, enc_wrong.input_ids, gold_tok_wrong, model_tok_wrong)
        ld_right_base = logit_diff_baseline(model, enc_right.input_ids, gold_tok_right, model_tok_right)

        for L in range(n_layers):
            # RESCUE: inject hs_right[L, pos_right] into wrong run at pos_wrong
            patch_vec_rescue = hs_right[L + 1, pos_right, :].to(model.device)  # +1: layer 0 = embedding
            logits_rescued = patch_residual_and_eval(
                model, enc_wrong.input_ids, patch_vec_rescue, L, pos_wrong
            )
            ld_rescued = float(logits_rescued[gold_tok_wrong].item() - logits_rescued[model_tok_wrong].item())
            rescue_effects[i, L] = ld_rescued - ld_wrong_base

            # CORRUPTION: inject hs_wrong[L, pos_wrong] into right run at pos_right
            patch_vec_corrupt = hs_wrong[L + 1, pos_wrong, :].to(model.device)
            logits_corrupted = patch_residual_and_eval(
                model, enc_right.input_ids, patch_vec_corrupt, L, pos_right
            )
            ld_corrupted = float(logits_corrupted[gold_tok_right].item() - logits_corrupted[model_tok_right].item())
            corruption_effects[i, L] = ld_corrupted - ld_right_base

    mean_rescue = rescue_effects.mean(0)
    mean_corruption = corruption_effects.mean(0)
    sem_rescue = rescue_effects.std(0) / np.sqrt(n_pairs)
    sem_corruption = corruption_effects.std(0) / np.sqrt(n_pairs)
    peak_rescue_layer = int(np.argmax(mean_rescue))
    print(f"[patch] peak rescue layer: {peak_rescue_layer} (effect={mean_rescue[peak_rescue_layer]:.3f})")
    print(f"[patch] max corruption layer: {int(np.argmin(mean_corruption))} "
          f"(effect={mean_corruption.min():.3f})")

    # Component patching at peak rescue layer
    comp_layer = args.component_patch_at if args.component_patch_at is not None else peak_rescue_layer
    print(f"[patch] running component patching at layer {comp_layer}")
    comp_rescue = {"attn": [], "mlp": []}

    for i, r_wrong in enumerate(tqdm(wrong_sample[:n_pairs], desc="comp-patch")):
        r_right = random.choice(right_pool)
        gold_tok_wrong = get_first_token_id(tok, r_wrong["gold_answers"][0] if r_wrong["gold_answers"] else "")
        model_tok_wrong = get_first_token_id(tok, r_wrong["answer_text"])
        if any(t is None for t in [gold_tok_wrong, model_tok_wrong]):
            continue
        if gold_tok_wrong == model_tok_wrong:
            continue

        prompt_wrong = build_prompt(tok, r_wrong["question"], style=args.prompt_style)
        prompt_right = build_prompt(tok, r_right["question"], style=args.prompt_style)
        enc_wrong = tok(prompt_wrong, return_tensors="pt").to(model.device)
        enc_right = tok(prompt_right, return_tensors="pt").to(model.device)
        pos_wrong = enc_wrong.input_ids.shape[1] - 1

        ld_wrong_base = logit_diff_baseline(model, enc_wrong.input_ids, gold_tok_wrong, model_tok_wrong)

        # Cache attn/mlp outputs at comp_layer for the right (clean) run
        right_comps = cache_component_outputs(model, enc_right.input_ids, [comp_layer])
        pos_right = enc_right.input_ids.shape[1] - 1

        for comp in ["attn", "mlp"]:
            patch_vec = right_comps[comp][comp_layer][pos_right, :].to(model.device)
            logits = patch_component_and_eval(model, enc_wrong.input_ids, patch_vec, comp_layer, pos_wrong, comp)
            ld = float(logits[gold_tok_wrong].item() - logits[model_tok_wrong].item())
            comp_rescue[comp].append(ld - ld_wrong_base)

    mean_attn = float(np.mean(comp_rescue["attn"])) if comp_rescue["attn"] else float("nan")
    mean_mlp = float(np.mean(comp_rescue["mlp"])) if comp_rescue["mlp"] else float("nan")
    print(f"[patch] component rescue at L{comp_layer}: attn={mean_attn:.3f} mlp={mean_mlp:.3f}")

    # Save
    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_pairs": n_pairs,
        "peak_rescue_layer": peak_rescue_layer,
        "peak_rescue_effect": float(mean_rescue[peak_rescue_layer]),
        "component_patch_layer": comp_layer,
        "component_rescue_attn": mean_attn,
        "component_rescue_mlp": mean_mlp,
        "mean_rescue_per_layer": mean_rescue.tolist(),
        "mean_corruption_per_layer": mean_corruption.tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with (tables_dir / "patching.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["layer", "rescue_mean", "rescue_sem", "corruption_mean", "corruption_sem"])
        for L in range(n_layers):
            w.writerow([L, mean_rescue[L], sem_rescue[L], mean_corruption[L], sem_corruption[L]])

    # --- Main figure: rescue + corruption curves ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ls = range(n_layers)

    axes[0].plot(ls, mean_rescue, color="#2ca02c", linewidth=1.8)
    axes[0].fill_between(ls, mean_rescue - sem_rescue, mean_rescue + sem_rescue, alpha=0.15, color="#2ca02c")
    axes[0].axhline(0, color="grey", linewidth=0.8, linestyle=":")
    axes[0].axvline(peak_rescue_layer, color="#2ca02c", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Δ logit(gold) − logit(wrong) after patch")
    axes[0].set_title(f"Rescue effect (correct→incorrect, n={n_pairs})")
    axes[0].grid(alpha=0.2)

    axes[1].plot(ls, mean_corruption, color="#d62728", linewidth=1.8)
    axes[1].fill_between(ls, mean_corruption - sem_corruption, mean_corruption + sem_corruption,
                         alpha=0.15, color="#d62728")
    axes[1].axhline(0, color="grey", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Δ logit(gold) − logit(wrong) after patch")
    axes[1].set_title(f"Corruption effect (incorrect→correct, n={n_pairs})")
    axes[1].grid(alpha=0.2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Residual-stream patching — {args.out_tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "patching_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Component bar ---
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    bars = ax2.bar(
        ["full residual\n(peak L)", "attn_out\n(L{})".format(comp_layer), "mlp_out\n(L{})".format(comp_layer)],
        [float(mean_rescue[peak_rescue_layer]), mean_attn, mean_mlp],
        color=["#7f7f7f", "#1f77b4", "#ff7f0e"],
    )
    ax2.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Rescue effect (Δ logit diff)")
    ax2.set_title(f"Component patching — {args.out_tag}")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(plots_dir / "component_patch_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig2)

    print(f"[patch] wrote {out_dir / 'summary.json'} and plots in {plots_dir}")


if __name__ == "__main__":
    main()
