"""Experiment 9 (partial): Logit-lens analysis across layers.

For each saved generation (correct and incorrect), apply the language model's
unembedding matrix to the cached hidden state at every layer and position.
Compute logit_diff = logit(gold_first_token) - logit(model_first_token).

Key question: for INCORRECT generations, at which layer does the model's internal
representation stop 'favouring' the correct answer?  If logit_diff > 0 at early/mid
layers but flips negative in later layers, the model internally represented the
right answer but 'changed its mind'.
"""

from __future__ import annotations

import argparse
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

from src.config import DEFAULT_MODEL, PLOTS_DIR, TABLES_DIR
from src.grading import normalize as norm_str
from src.patching import logit_lens
from src.utils import read_jsonl


def get_first_token(tokenizer, text: str) -> int | None:
    """Return the first token id produced when tokenizing `text` without special tokens."""
    text = text.strip()
    if not text:
        return None
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--positions", nargs="+",
                    default=["prompt_last", "answer_first", "answer_last"])
    ap.add_argument("--max-examples", type=int, default=200,
                    help="Cap per-class to limit memory usage")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    rows = [r for r in read_jsonl(gen_dir / "graded.jsonl")
            if r.get("hidden_states_path") and not r.get("did_abstain")]
    print(f"[lens] {len(rows)} non-abstain rows")

    os_env = __import__("os")
    os_env.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    final_norm = model.model.norm
    lm_head = model.lm_head

    correct_rows = [r for r in rows if r["is_correct"]][:args.max_examples]
    incorrect_rows = [r for r in rows if not r["is_correct"]][:args.max_examples]
    print(f"[lens] correct={len(correct_rows)} incorrect={len(incorrect_rows)}")

    results: dict[str, dict[str, list]] = {
        pos: {"correct": [], "incorrect": []} for pos in args.positions
    }

    for label, subset in [("correct", correct_rows), ("incorrect", incorrect_rows)]:
        for r in tqdm(subset, desc=f"lens-{label}"):
            gold_tok = get_first_token(tok, r["gold_answers"][0] if r["gold_answers"] else "")
            model_tok = get_first_token(tok, r["answer_text"])
            if gold_tok is None or model_tok is None:
                continue
            hs = torch.load(r["hidden_states_path"], map_location="cpu", weights_only=False)
            n_layers = hs[args.positions[0]].shape[0]
            for pos in args.positions:
                diffs = []
                for L in range(n_layers):
                    logits = logit_lens(hs[pos][L], final_norm, lm_head)
                    diff = float(logits[gold_tok].item() - logits[model_tok].item())
                    diffs.append(diff)
                results[pos][label].append(diffs)

    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    import csv
    for pos in args.positions:
        corr = np.array(results[pos]["correct"]) if results[pos]["correct"] else np.empty((0,))
        incorr = np.array(results[pos]["incorrect"]) if results[pos]["incorrect"] else np.empty((0,))
        if corr.ndim < 2 or incorr.ndim < 2:
            continue
        n_layers = corr.shape[1]
        mean_corr = corr.mean(0)
        mean_incorr = incorr.mean(0)
        sem_corr = corr.std(0) / np.sqrt(len(corr))
        sem_incorr = incorr.std(0) / np.sqrt(len(incorr))

        # Find the "flip layer" for incorrect: where does mean_incorr become < 0?
        flip = next((L for L in range(n_layers) if mean_incorr[L] < 0), None)

        # Plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ls = range(n_layers)
        ax.plot(ls, mean_corr, color="#2ca02c", linewidth=1.8, label=f"correct (n={len(corr)})")
        ax.fill_between(ls, mean_corr - sem_corr, mean_corr + sem_corr, alpha=0.15, color="#2ca02c")
        ax.plot(ls, mean_incorr, color="#d62728", linewidth=1.8, label=f"incorrect (n={len(incorr)})")
        ax.fill_between(ls, mean_incorr - sem_incorr, mean_incorr + sem_incorr, alpha=0.15, color="#d62728")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        if flip is not None:
            ax.axvline(flip, color="#d62728", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.text(flip + 0.3, ax.get_ylim()[0] * 0.85, f"flip@L{flip}", fontsize=8, color="#d62728")
        ax.set_xlabel("Layer")
        ax.set_ylabel("logit(gold) − logit(model answer)")
        ax.set_title(f"Logit lens — {args.out_tag}, pos={pos}")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"logit_lens_{pos}.png", dpi=160)
        plt.close(fig)

        # CSV
        with (tables_dir / f"logit_lens_{pos}.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(["layer", "mean_correct", "mean_incorrect", "sem_correct", "sem_incorrect"])
            for L in range(n_layers):
                w.writerow([L, mean_corr[L], mean_incorr[L], sem_corr[L], sem_incorr[L]])

        print(f"[lens] pos={pos} flip_layer={flip} "
              f"early_correct={mean_corr[:5].mean():.2f} early_incorrect={mean_incorr[:5].mean():.2f} "
              f"late_correct={mean_corr[-5:].mean():.2f} late_incorrect={mean_incorr[-5:].mean():.2f}")


if __name__ == "__main__":
    main()
