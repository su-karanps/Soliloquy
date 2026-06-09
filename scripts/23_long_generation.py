"""Longer-generation probe: generate multi-sentence answers and track token-level
output confidence (logprob) as a function of generation step.

Key questions:
  - Do token logprobs drop after the model commits to a wrong answer?
  - Does the model ever add hedges mid-sentence for wrong answers?
  - Is there a "commitment point" where logprob diverges between correct/wrong runs?

NOTE: Tracking the probe direction score during generation is NOT meaningful for
causal (decoder-only) LMs. The hidden state at position `prompt_len` is determined
solely by tokens ≤ that position (causal masking), so it never changes as more
tokens are generated. We therefore track token logprobs and hedge phrases instead.

Protocol:
  - Use the same SimpleQA questions with a full-sentence answer prompt.
  - Generate up to max_new_tokens tokens.
  - Record per-token log-probabilities and detect hedge phrases.
  - Compute the probe score ONCE at the first generated token (answer_first),
    which is its only meaningful application.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DEFAULT_MODEL, PLOTS_DIR, PROBES_DIR, TABLES_DIR
from src.grading import grade, is_abstain
from src.utils import read_jsonl

HEDGE_PHRASES = [
    "i think", "i'm not sure", "i believe", "i'm unsure", "i don't know",
    "i'm uncertain", "not certain", "may be", "might be", "perhaps",
    "probably", "possibly", "unclear", "i cannot", "i can't",
    "i am not sure", "i am uncertain", "it's possible",
]


def has_hedge(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in HEDGE_PHRASES)


def apply_probe(h: np.ndarray, coef: np.ndarray, intercept: float,
                mean: np.ndarray, std: np.ndarray) -> float:
    z = (h - mean) / (std + 1e-8)
    logit = float(np.dot(z, coef) + intercept)
    return 1 / (1 + np.exp(-logit))


LONG_PROMPT = (
    "Answer the following factual question in a complete sentence. "
    "You must give your best answer; do NOT say you don't know.\n"
    "Question: {question}\n"
    "Answer:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, help="existing graded gen dir for question list")
    ap.add_argument("--probe-tag", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-questions", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--probe-stride", type=int, default=4,
                    help="Measure probe every N generated tokens")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import os, random
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")
    random.seed(args.seed)

    # Load probe direction
    probe_data = np.load(
        PROBES_DIR / args.probe_tag / "best_probe_direction.npz", allow_pickle=True
    )
    coef = probe_data["coef"].astype(np.float32)
    intercept = float(probe_data["intercept"])
    mean = probe_data["scaler_mean"].astype(np.float32)
    std = probe_data["scaler_std"].astype(np.float32)
    position_name = str(probe_data["position"][0])
    probe_layer = int(probe_data["layer"][0])
    print(f"[long-gen] probe at L{probe_layer}, pos={position_name}")

    # Get questions from existing gen dir
    rows = [r for r in read_jsonl(Path(args.gen_dir) / "graded.jsonl")
            if not r.get("did_abstain")]
    random.shuffle(rows)
    questions = [r["question"] for r in rows[:args.n_questions]]
    gold_map = {r["question"]: r["gold_answers"] for r in rows}

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    # Probe scores at answer_first (single value per example, not a trajectory)
    probe_scores_correct = []
    probe_scores_wrong = []
    logprob_trajectories_correct = []
    logprob_trajectories_wrong = []
    hedge_examples = []   # qualitative: wrong examples that produce a hedge word
    answer_lengths = {"correct": [], "wrong": []}

    for question in tqdm(questions, desc="long-gen"):
        gold = gold_map.get(question, [])
        prompt_text = LONG_PROMPT.format(question=question)

        # Apply chat template
        messages = [{"role": "user", "content": prompt_text}]
        try:
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

        enc = tok(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = enc.input_ids.shape[1]

        # Generate full answer
        with torch.no_grad():
            out = model.generate(
                enc.input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True, output_scores=True,
            )

        answer_ids = out.sequences[0, prompt_len:]
        answer_text = tok.decode(answer_ids, skip_special_tokens=True).strip()
        if is_abstain(answer_text) or len(answer_ids) == 0:
            continue

        g = grade(answer_text, gold)
        is_correct = g["is_correct"]

        # Per-token logprobs
        token_lps = []
        for step, score in enumerate(out.scores):
            tid = out.sequences[0, prompt_len + step].item()
            lp = float(F.log_softmax(score[0].float(), dim=-1)[tid].item())
            token_lps.append(lp)

        # Probe score at answer_first (only position where probe is valid).
        # NOTE: in a causal LM, hidden state at position p is fixed once tokens
        # 0..p are determined; re-running on longer sequences won't change it.
        # So we compute ONE probe score per example (no trajectory).
        with torch.no_grad():
            fwd = model(
                torch.cat([enc.input_ids, out.sequences[:, prompt_len:prompt_len+1]], dim=1),
                output_hidden_states=True, use_cache=False
            )
        hs_stacked = torch.stack([h[0].cpu().float() for h in fwd.hidden_states], dim=0)
        h_vec = hs_stacked[probe_layer, prompt_len, :].numpy()
        probe_score = apply_probe(h_vec, coef, intercept, mean, std)

        # Detect hedges
        hedged = has_hedge(answer_text)

        # Interpolate logprob to fixed length for averaging
        target_len = args.max_new_tokens
        lp_xs = np.linspace(0, 1, len(token_lps)) if token_lps else np.array([0.0])
        lp_interp = np.interp(np.linspace(0, 1, target_len),
                               lp_xs, token_lps if token_lps else [0.0])

        if is_correct:
            probe_scores_correct.append(probe_score)
            logprob_trajectories_correct.append(lp_interp)
            answer_lengths["correct"].append(len(answer_ids))
        else:
            probe_scores_wrong.append(probe_score)
            logprob_trajectories_wrong.append(lp_interp)
            answer_lengths["wrong"].append(len(answer_ids))
            if hedged and len(hedge_examples) < 20:
                hedge_examples.append({
                    "question": question,
                    "answer": answer_text,
                    "gold": gold[0] if gold else "?",
                    "probe_score": float(probe_score),
                    "mean_logprob": float(np.mean(token_lps)),
                    "n_tokens": len(answer_ids),
                })

    print(f"[long-gen] correct: {len(logprob_trajectories_correct)}, "
          f"wrong: {len(logprob_trajectories_wrong)}")
    print(f"[long-gen] hedge examples found: {len(hedge_examples)}")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    target_len = args.max_new_tokens
    x_steps = np.arange(target_len)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: token logprob trajectory
    def plot_traj(ax, traj_correct, traj_wrong, ylabel, title):
        if traj_correct:
            arr_c = np.stack(traj_correct)
            ax.plot(x_steps, arr_c.mean(0), color="#2ca02c", linewidth=2.0, label=f"correct (n={len(arr_c)})")
            ax.fill_between(x_steps,
                            arr_c.mean(0) - arr_c.std(0) / np.sqrt(len(arr_c)),
                            arr_c.mean(0) + arr_c.std(0) / np.sqrt(len(arr_c)),
                            alpha=0.15, color="#2ca02c")
        if traj_wrong:
            arr_w = np.stack(traj_wrong)
            ax.plot(x_steps, arr_w.mean(0), color="#d62728", linewidth=2.0, label=f"incorrect (n={len(arr_w)})")
            ax.fill_between(x_steps,
                            arr_w.mean(0) - arr_w.std(0) / np.sqrt(len(arr_w)),
                            arr_w.mean(0) + arr_w.std(0) / np.sqrt(len(arr_w)),
                            alpha=0.15, color="#d62728")
        ax.set_xlabel("Generated token index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.18)

    plot_traj(axes[0], logprob_trajectories_correct, logprob_trajectories_wrong,
              "Token log-prob", "Token log-prob during generation")

    # Panel 2: probe score distribution at answer_first (one value per example)
    if probe_scores_correct or probe_scores_wrong:
        axes[1].hist(probe_scores_wrong, bins=20, color="#d62728", alpha=0.65, label=f"incorrect (n={len(probe_scores_wrong)})")
        axes[1].hist(probe_scores_correct, bins=20, color="#2ca02c", alpha=0.65, label=f"correct (n={len(probe_scores_correct)})")
        axes[1].axvline(0.5, color="grey", linewidth=1, linestyle="--")
        axes[1].set_xlabel("Probe p(correct) at answer_first")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Probe score at first token (L{probe_layer})")
        axes[1].legend(frameon=False, fontsize=9)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

    # Panel 3: logprob by answer length bucket
    ax3 = axes[2]
    all_lens = answer_lengths["correct"] + answer_lengths["wrong"]
    if all_lens:
        bins = np.percentile(all_lens, [0, 33, 67, 100])
        bins = sorted(set(int(b) for b in bins))
        for group, color, trajs in [("correct", "#2ca02c", logprob_trajectories_correct),
                                     ("incorrect", "#d62728", logprob_trajectories_wrong)]:
            for traj, length in zip(trajs, answer_lengths[group]):
                pass  # placeholder — just plot mean logprob vs answer length
        mean_lps_c = [float(np.mean(t)) for t in logprob_trajectories_correct]
        mean_lps_w = [float(np.mean(t)) for t in logprob_trajectories_wrong]
        ax3.scatter(answer_lengths["correct"], mean_lps_c, color="#2ca02c", alpha=0.5, s=18, label="correct")
        ax3.scatter(answer_lengths["wrong"], mean_lps_w, color="#d62728", alpha=0.35, s=12, label="incorrect")
        ax3.set_xlabel("Answer length (tokens)")
        ax3.set_ylabel("Mean token log-prob")
        ax3.set_title("Mean logprob vs answer length")
        ax3.legend(frameon=False, fontsize=9)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.grid(alpha=0.18)

    fig.suptitle(f"Long-generation probe trajectory — {args.out_tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "long_gen_trajectory.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_correct": len(logprob_trajectories_correct),
        "n_wrong": len(logprob_trajectories_wrong),
        "n_hedge_examples": len(hedge_examples),
        "probe_layer": probe_layer,
        "probe_position": position_name,
        "mean_len_correct": float(np.mean(answer_lengths["correct"])) if answer_lengths["correct"] else 0,
        "mean_len_wrong": float(np.mean(answer_lengths["wrong"])) if answer_lengths["wrong"] else 0,
        "probe_score_correct_mean": float(np.mean(probe_scores_correct)) if probe_scores_correct else None,
        "probe_score_wrong_mean": float(np.mean(probe_scores_wrong)) if probe_scores_wrong else None,
        "mean_logprob_correct": float(np.mean([np.mean(t) for t in logprob_trajectories_correct])) if logprob_trajectories_correct else None,
        "mean_logprob_wrong": float(np.mean([np.mean(t) for t in logprob_trajectories_wrong])) if logprob_trajectories_wrong else None,
        "note": "probe score is evaluated once at answer_first position (not a trajectory); "
                "causal masking means the hidden state at that position does not change with longer generation",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "hedge_examples.json").write_text(json.dumps(hedge_examples, indent=2))
    print(f"[long-gen] wrote {out_dir / 'summary.json'} and plot in {plots_dir}")
    if summary["probe_score_correct_mean"] is not None:
        print(f"[long-gen] probe@answer_first: correct={summary['probe_score_correct_mean']:.3f} "
              f"wrong={summary['probe_score_wrong_mean']:.3f}")
    print(f"[long-gen] mean logprob: correct={summary['mean_logprob_correct']} "
          f"wrong={summary['mean_logprob_wrong']}")
    if hedge_examples:
        print("[long-gen] sample hedge examples:")
        for ex in hedge_examples[:3]:
            print(f"  Q: {ex['question'][:60]}...")
            print(f"  A: {ex['answer'][:80]}")


if __name__ == "__main__":
    main()
