"""Experiment 5+6 v2: SAME-QUESTION paired residual patching.

Stronger causal test than 14_patch_residual.py: instead of patching across
different questions (which mixes correctness signal with question semantics),
we use the sampled SimpleQA data to find qids where the same prompt sometimes
produced a correct answer and sometimes a wrong one. We then patch the
correct-trajectory's residual into the wrong-trajectory's forward pass at the
answer-first position.

Setup per (qid, correct-sample C, wrong-sample W):
  seq_W = prompt + W[0]          (forced-feed first wrong answer token)
  seq_C = prompt + C[0]          (forced-feed first correct answer token)
  At each layer L, patch seq_W's residual at position prompt_len with the
  residual seq_C had at the same position+layer.  Then re-run forward and
  measure logit changes at position prompt_len.

Metric: logit(C[0]) − logit(W[0]) at position prompt_len in seq_W's run.
  This is a *next-token* logit at the position holding W's first answer token,
  i.e., it predicts the token AFTER the embedded first answer token. We use
  C[0] vs W[0] as proxy tokens for "correct trajectory continuation" vs "wrong
  trajectory continuation" — if the patch transferred trajectory information
  the logit diff should rise.

Also compares to a control: random-pair patching (correct sample from a
*different* question), which should show much weaker rescue if the signal is
truly same-question correctness rather than generic answer content.
"""

from __future__ import annotations

import argparse
import collections
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
    ap.add_argument("--gen-dir", required=True,
                    help="dir with graded.jsonl that has >=2 gens per qid")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pairs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", default="force")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")
    random.seed(args.seed)
    np.random.seed(args.seed)

    gen_dir = Path(args.gen_dir)
    rows = [r for r in read_jsonl(gen_dir / "graded.jsonl") if not r.get("did_abstain")]
    by_qid = collections.defaultdict(list)
    for r in rows:
        by_qid[r["qid"]].append(r)

    # Build same-question pairs: each qid contributes up to one (C, W) pair.
    same_q_pairs = []
    for qid, samples in by_qid.items():
        corrects = [s for s in samples if s["is_correct"]]
        wrongs = [s for s in samples if not s["is_correct"]]
        if corrects and wrongs:
            same_q_pairs.append((random.choice(corrects), random.choice(wrongs)))

    print(f"[same-q] found {len(same_q_pairs)} same-question (C, W) pairs")
    if len(same_q_pairs) < 5:
        print("[same-q] too few pairs; aborting")
        return

    random.shuffle(same_q_pairs)
    same_q_pairs = same_q_pairs[:args.max_pairs]

    # Build cross-question CONTROL pairs: pick a random correct sample from a DIFFERENT qid.
    all_corrects = [r for r in rows if r["is_correct"]]
    cross_q_pairs = []
    for c, w in same_q_pairs:
        diff_qid_corrects = [r for r in all_corrects if r["qid"] != w["qid"]]
        if diff_qid_corrects:
            cross_q_pairs.append((random.choice(diff_qid_corrects), w))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"[same-q] model has {n_layers} layers")

    def run_patch_set(pairs, desc: str):
        """For a list of (correct, wrong) pairs, return per-layer rescue effects.

        Effect = (logit(C[0]) − logit(W[0])) after patch − (logit(C[0]) − logit(W[0])) baseline.
        Higher = more rescue toward correct trajectory.

        The patch source position is each donor's own first-answer-token position
        (so the patch always comes from "[prompt | first answer token]" at the
        right position, regardless of whether prompts match).
        """
        effects = []
        for r_C, r_W in tqdm(pairs, desc=desc):
            c_tok = first_token_id(tok, r_C["answer_text"])
            w_tok = first_token_id(tok, r_W["answer_text"])
            if c_tok is None or w_tok is None or c_tok == w_tok:
                continue

            # Build seq_W = prompt_W + W[0]
            prompt_text_W = build_prompt(tok, r_W["question"], style=args.prompt_style)
            prompt_ids_W = tok(prompt_text_W, return_tensors="pt").input_ids.to(model.device)
            prompt_len_W = prompt_ids_W.shape[1]
            seq_W = torch.cat(
                [prompt_ids_W, torch.tensor([[w_tok]], device=model.device)], dim=1
            )

            # Build seq_C = prompt_C + C[0]  (prompt_C may equal prompt_W for same-q, or differ)
            prompt_text_C = build_prompt(tok, r_C["question"], style=args.prompt_style)
            prompt_ids_C = tok(prompt_text_C, return_tensors="pt").input_ids.to(model.device)
            prompt_len_C = prompt_ids_C.shape[1]
            seq_C = torch.cat(
                [prompt_ids_C, torch.tensor([[c_tok]], device=model.device)], dim=1
            )

            # Cache full residuals for both.
            hs_W = cache_residual(model, seq_W)       # (L+1, prompt_len_W+1, D)
            hs_C = cache_residual(model, seq_C)       # (L+1, prompt_len_C+1, D)

            # Baseline: logits at position prompt_len_W in seq_W's clean run.
            with torch.no_grad():
                logits_W_baseline = model(seq_W, use_cache=False).logits[0, prompt_len_W, :].cpu().float()
            ld_baseline = float(logits_W_baseline[c_tok].item() - logits_W_baseline[w_tok].item())

            row = []
            for L in range(n_layers):
                # Patch source: C's residual at C's own first-answer-token position.
                patch_vec = hs_C[L + 1, prompt_len_C, :].to(model.device)
                # Patch target: position prompt_len_W in seq_W's run.
                logits = patch_residual_and_eval(model, seq_W, patch_vec, L, prompt_len_W)
                ld = float(logits[c_tok].item() - logits[w_tok].item())
                row.append(ld - ld_baseline)
            effects.append(row)
        return np.array(effects) if effects else np.empty((0, n_layers))

    print("[same-q] running SAME-question patches (C, W same qid)")
    same_eff = run_patch_set(same_q_pairs, desc="same-q")
    print(f"[same-q] effective n same-question pairs (after filtering): {len(same_eff)}")

    print("[same-q] running CROSS-question control patches (C, W different qid)")
    cross_eff = run_patch_set(cross_q_pairs, desc="cross-q ctrl")
    print(f"[same-q] effective n cross-question control pairs: {len(cross_eff)}")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_same_q_pairs": int(len(same_eff)),
        "n_cross_q_pairs": int(len(cross_eff)),
        "n_layers": n_layers,
    }

    def stats(arr):
        if arr.size == 0:
            return [float("nan")] * n_layers, [float("nan")] * n_layers
        return arr.mean(0).tolist(), (arr.std(0) / max(np.sqrt(len(arr)), 1)).tolist()

    same_mean, same_sem = stats(same_eff)
    cross_mean, cross_sem = stats(cross_eff)

    summary["same_q_mean_per_layer"] = same_mean
    summary["same_q_sem_per_layer"] = same_sem
    summary["cross_q_mean_per_layer"] = cross_mean
    summary["cross_q_sem_per_layer"] = cross_sem
    if same_eff.size:
        peak = int(np.argmax(same_mean))
        summary["same_q_peak_layer"] = peak
        summary["same_q_peak_effect"] = same_mean[peak]
    if cross_eff.size:
        summary["cross_q_peak_effect"] = float(np.max(cross_mean))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with (tables_dir / "same_q_patching.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["layer", "same_q_mean", "same_q_sem", "cross_q_mean", "cross_q_sem"])
        for L in range(n_layers):
            w.writerow([L, same_mean[L], same_sem[L], cross_mean[L], cross_sem[L]])

    # Plot: same-question rescue curve vs cross-question control
    fig, ax = plt.subplots(figsize=(7, 4))
    ls = range(n_layers)
    if same_eff.size:
        same_mean_a = np.array(same_mean)
        same_sem_a = np.array(same_sem)
        ax.plot(ls, same_mean_a, color="#2ca02c", linewidth=1.8,
                label=f"same question (n={len(same_eff)})")
        ax.fill_between(ls, same_mean_a - same_sem_a, same_mean_a + same_sem_a,
                        alpha=0.15, color="#2ca02c")
    if cross_eff.size:
        cross_mean_a = np.array(cross_mean)
        cross_sem_a = np.array(cross_sem)
        ax.plot(ls, cross_mean_a, color="#7f7f7f", linewidth=1.6, linestyle="--",
                label=f"cross-question control (n={len(cross_eff)})")
        ax.fill_between(ls, cross_mean_a - cross_sem_a, cross_mean_a + cross_sem_a,
                        alpha=0.10, color="#7f7f7f")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ logit-diff (C[0] − W[0]) after patch")
    ax.set_title(f"Same-question paired patching — {args.out_tag}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plots_dir / "same_question_patching.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[same-q] wrote {out_dir / 'summary.json'} and plots in {plots_dir}")


if __name__ == "__main__":
    main()
