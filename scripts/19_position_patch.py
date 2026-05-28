"""Experiment 5+6 v3: position-specific residual patching.

The related paper (Orgad et al.) emphasizes that the truthfulness signal is
strongest at the **exact answer tokens**. This script tests whether the
*causal* signal also peaks at answer tokens by sweeping the patch position:

  - prompt_last       (position prompt_len - 1)
  - answer_first      (position prompt_len)
  - answer_last       (position prompt_len + answer_len - 1)
  - answer_mean       (mean over answer-token residuals; patched at answer_first)

For each (position, layer), measure the rescue effect from cross-question
patching (wrong run + correct run patch at that position/layer).
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
from src.patching import cache_residual, patch_residual_and_eval
from src.utils import read_jsonl


def first_token_id(tok, text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", default="force")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Layers to test; defaults to a coarse sweep")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = [r for r in read_jsonl(Path(args.gen_dir) / "graded.jsonl") if not r.get("did_abstain")]
    correct_rows = [r for r in rows if r["is_correct"]]
    incorrect_rows = [r for r in rows if not r["is_correct"]]
    print(f"[pos-patch] correct={len(correct_rows)} incorrect={len(incorrect_rows)}")

    n_pairs = min(args.max_pairs, len(correct_rows), len(incorrect_rows))
    if n_pairs < 5:
        print("[pos-patch] too few pairs; aborting")
        return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    test_layers = args.layers or list(range(0, n_layers, max(1, n_layers // 9)))
    if (n_layers - 1) not in test_layers:
        test_layers.append(n_layers - 1)
    test_layers = sorted(set(L for L in test_layers if 0 <= L < n_layers))
    print(f"[pos-patch] model has {n_layers} layers, testing {len(test_layers)}: {test_layers}")

    wrong_sample = random.sample(incorrect_rows, n_pairs)
    right_pool = correct_rows.copy()

    positions = ["prompt_last", "answer_first", "answer_last", "answer_mean"]
    # Results: per-position list of per-layer per-pair effects
    eff = {pos: np.zeros((n_pairs, len(test_layers))) for pos in positions}

    for i, r_W in enumerate(tqdm(wrong_sample, desc="pos-patch pairs")):
        r_C = random.choice(right_pool)

        c_tok = first_token_id(tok, r_C["answer_text"])
        w_tok = first_token_id(tok, r_W["answer_text"])
        gold_tok = first_token_id(tok, r_W["gold_answers"][0] if r_W["gold_answers"] else "")
        if any(t is None for t in [c_tok, w_tok, gold_tok]):
            continue

        # Build the wrong run as prompt_W only (so prompt_last is the last position).
        prompt_text_W = build_prompt(tok, r_W["question"], style=args.prompt_style)
        prompt_ids_W = tok(prompt_text_W, return_tensors="pt").input_ids.to(model.device)
        prompt_len_W = prompt_ids_W.shape[1]

        # For "answer_first", "answer_last", "answer_mean" — we also extend with the wrong
        # answer's tokens so the target position exists in the wrong run.
        w_ans_ids = tok.encode(r_W["answer_text"], add_special_tokens=False)
        if not w_ans_ids:
            continue
        w_ans_tensor = torch.tensor([w_ans_ids], device=model.device)
        seq_W_full = torch.cat([prompt_ids_W, w_ans_tensor], dim=1)
        a_first_W = prompt_len_W
        a_last_W = prompt_len_W + len(w_ans_ids) - 1

        # Cache wrong-run hidden states for the FULL wrong sequence.
        hs_W_full = cache_residual(model, seq_W_full)

        # The patch SOURCE for each position comes from the cached hidden states of the correct
        # sample at the corresponding NAMED position (which we have in the cached .pt file).
        hs_path = r_C.get("hidden_states_path")
        if not hs_path or not Path(hs_path).exists():
            continue
        c_hs = torch.load(hs_path, map_location="cpu", weights_only=False)
        # c_hs[pos] is (n_layers+1, hidden)

        for pi, pos_name in enumerate(positions):
            # Target position in the wrong run depends on the named position.
            if pos_name == "prompt_last":
                target_pos = prompt_len_W - 1
                seq_W = prompt_ids_W
            elif pos_name == "answer_first":
                target_pos = a_first_W
                seq_W = seq_W_full
            elif pos_name == "answer_last":
                target_pos = a_last_W
                seq_W = seq_W_full
            elif pos_name == "answer_mean":
                target_pos = a_first_W  # apply mean-residual at answer_first
                seq_W = seq_W_full

            # Baseline: re-run seq_W cleanly, get logits at last position (predicting next token).
            with torch.no_grad():
                logits_baseline = model(seq_W, use_cache=False).logits[0, -1, :].cpu().float()
            ld_baseline = float(logits_baseline[gold_tok].item() - logits_baseline[w_tok].item())

            for li, L in enumerate(test_layers):
                patch_vec = c_hs[pos_name][L + 1].to(model.device)  # (hidden,)
                logits = patch_residual_and_eval(model, seq_W, patch_vec, L, target_pos)
                ld = float(logits[gold_tok].item() - logits[w_tok].item())
                eff[pos_name][i, li] = ld - ld_baseline

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_pairs": int(n_pairs),
        "n_layers_tested": len(test_layers),
        "layers_tested": test_layers,
        "positions": positions,
        "mean_rescue": {pos: eff[pos].mean(0).tolist() for pos in positions},
        "peak_layer_per_position": {pos: int(test_layers[int(np.argmax(eff[pos].mean(0)))])
                                    for pos in positions},
        "peak_effect_per_position": {pos: float(np.max(eff[pos].mean(0))) for pos in positions},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with (tables_dir / "position_patching.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + positions)
        for li, L in enumerate(test_layers):
            w.writerow([L] + [eff[pos].mean(0)[li] for pos in positions])

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"prompt_last": "#1f77b4", "answer_first": "#2ca02c",
              "answer_last": "#d62728", "answer_mean": "#9467bd"}
    for pos in positions:
        means = eff[pos].mean(0)
        sems = eff[pos].std(0) / max(np.sqrt(n_pairs), 1)
        ax.plot(test_layers, means, marker="o", color=colors[pos], label=pos, linewidth=1.6)
        ax.fill_between(test_layers, means - sems, means + sems, alpha=0.12, color=colors[pos])
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ logit-diff (gold − wrong)")
    ax.set_title(f"Position-specific rescue effect — {args.out_tag}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plots_dir / "position_specific_patching.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[pos-patch] wrote {out_dir / 'summary.json'} and plot in {plots_dir}")
    print(f"[pos-patch] peak per position: {summary['peak_layer_per_position']}")
    print(f"[pos-patch] peak effect per position: {summary['peak_effect_per_position']}")


if __name__ == "__main__":
    main()
