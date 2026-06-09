"""Measure actual answer flip rate from activation patching at the peak rescue layer.

Complements 14_patch_residual.py which only records logit-diff.
For each (wrong, correct) pair, patches the residual stream at the peak layer
and generates the full answer, then grades it.

Reports:
  - baseline correct rate (should be ~0 since we select wrong examples)
  - flip rate: fraction of wrong examples that now produce a correct answer after patching
  - also reports for MLP-only and attention-only patches at peak layer
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DEFAULT_MODEL, PROBES_DIR
from src.generation import build_prompt
from src.grading import grade
from src.patching import cache_component_outputs, cache_residual
from src.utils import read_jsonl


def get_first_token_id(tok, text):
    ids = tok.encode((text or "").strip(), add_special_tokens=False)
    return ids[0] if ids else None


@torch.no_grad()
def generate_with_patch(model, tok, prompt_ids, hs_donor, layer,
                        target_position, donor_position,
                        patch_type, max_new_tokens=20):
    """Generate answer after patching hidden state at (layer, target_position).

    Reads patch vector from hs_donor at donor_position (donor prompt may differ in length).
    patch_type: 'residual'
    """
    if patch_type == "residual":
        patch_vec = hs_donor[layer + 1, donor_position, :]  # +1 for embedding offset
    else:
        return None, None

    def make_hook(vec, tgt_pos):
        def hook(mod, inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            if h.shape[1] <= tgt_pos:
                return out
            h = h.clone()
            h[:, tgt_pos, :] = vec.to(h.device, h.dtype)
            return (h,) + out[1:] if is_tuple else h
        return hook

    handle = model.model.layers[layer].register_forward_hook(
        make_hook(patch_vec.clone(), target_position)
    )
    try:
        out = model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    finally:
        handle.remove()

    gen_ids = out[0, prompt_ids.shape[1]:]
    answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return answer, gen_ids


@torch.no_grad()
def generate_with_component_patch(model, tok, prompt_ids, comp_cache,
                                   layer, donor_position, target_position,
                                   comp, max_new_tokens=20):
    """Patch just MLP or attention output at (layer, target_position) using donor vector."""
    vec = comp_cache[layer][donor_position].clone()

    def make_hook(target_vec, tgt_pos):
        def hook(mod, inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            if h.shape[1] <= tgt_pos:
                return out
            h = h.clone()
            h[:, tgt_pos, :] = target_vec.to(h.device, h.dtype)
            return (h,) + out[1:] if is_tuple else h
        return hook

    if comp == "mlp":
        module = model.model.layers[layer].mlp
    else:
        module = model.model.layers[layer].self_attn

    handle = module.register_forward_hook(make_hook(vec, target_position))
    try:
        out = model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    finally:
        handle.remove()

    gen_ids = out[0, prompt_ids.shape[1]:]
    answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--peak-layer", type=int, default=35)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pairs", type=int, default=60)
    ap.add_argument("--prompt-style", default="force")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")
    random.seed(args.seed)

    gen_dir = Path(args.gen_dir)
    rows = [r for r in read_jsonl(gen_dir / "graded.jsonl") if not r.get("did_abstain")]
    correct_rows = [r for r in rows if r["is_correct"]]
    wrong_rows = [r for r in rows if not r["is_correct"]]
    random.shuffle(wrong_rows)
    wrong_rows = wrong_rows[:args.max_pairs]
    print(f"[flip] {len(wrong_rows)} wrong examples, {len(correct_rows)} correct donors")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    L = min(args.peak_layer, model.config.num_hidden_layers - 1)
    print(f"[flip] patching at layer L={L}")

    results = []
    for r_W in tqdm(wrong_rows, desc="patch+generate"):
        r_C = random.choice(correct_rows)
        while r_C["qid"] == r_W["qid"]:
            r_C = random.choice(correct_rows)

        prompt_W = build_prompt(tok, r_W["question"], style=args.prompt_style)
        prompt_C = build_prompt(tok, r_C["question"], style=args.prompt_style)
        ids_W = tok(prompt_W, return_tensors="pt").input_ids.to(model.device)
        ids_C = tok(prompt_C, return_tensors="pt").input_ids.to(model.device)
        pos_W = ids_W.shape[1] - 1  # prompt_last for wrong question
        pos_C = ids_C.shape[1] - 1  # prompt_last for correct donor

        # Cache residual from correct run
        hs_C = cache_residual(model, ids_C)
        # Cache component outputs from correct run
        comp_C = cache_component_outputs(model, ids_C, layers=[L])

        # Baseline answer (no patch)
        baseline_out = model.generate(ids_W, max_new_tokens=20, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
        baseline_ans = tok.decode(baseline_out[0, ids_W.shape[1]:],
                                   skip_special_tokens=True).strip()
        baseline_correct = grade(baseline_ans, r_W["gold_answers"])["is_correct"]

        # Full residual patch
        res_ans, _ = generate_with_patch(model, tok, ids_W, hs_C, L,
                                          pos_W, pos_C, "residual")
        res_correct = grade(res_ans, r_W["gold_answers"])["is_correct"] if res_ans else False

        # MLP-only patch
        mlp_ans = generate_with_component_patch(model, tok, ids_W, comp_C["mlp"], L, pos_C, pos_W, "mlp")
        mlp_correct = grade(mlp_ans, r_W["gold_answers"])["is_correct"]

        # Attention-only patch
        attn_ans = generate_with_component_patch(model, tok, ids_W, comp_C["attn"], L, pos_C, pos_W, "attn")
        attn_correct = grade(attn_ans, r_W["gold_answers"])["is_correct"]

        results.append({
            "qid": r_W["qid"],
            "question": r_W["question"],
            "original_answer": r_W["answer_text"],
            "gold": r_W["gold_answers"][0] if r_W["gold_answers"] else "",
            "baseline_answer": baseline_ans,
            "baseline_correct": baseline_correct,
            "residual_patch_answer": res_ans,
            "residual_patch_correct": res_correct,
            "mlp_patch_answer": mlp_ans,
            "mlp_patch_correct": mlp_correct,
            "attn_patch_answer": attn_ans,
            "attn_patch_correct": attn_correct,
        })

    n = len(results)
    baseline_rate = np.mean([r["baseline_correct"] for r in results])
    res_rate = np.mean([r["residual_patch_correct"] for r in results])
    mlp_rate = np.mean([r["mlp_patch_correct"] for r in results])
    attn_rate = np.mean([r["attn_patch_correct"] for r in results])

    print(f"\n[flip] Results at L{L} (n={n}):")
    print(f"  baseline correct rate:       {baseline_rate:.3f}")
    print(f"  residual patch correct rate: {res_rate:.3f}  (flip rate: {res_rate - baseline_rate:+.3f})")
    print(f"  MLP-only patch correct rate: {mlp_rate:.3f}  (flip rate: {mlp_rate - baseline_rate:+.3f})")
    print(f"  attn-only patch correct rate:{attn_rate:.3f}  (flip rate: {attn_rate - baseline_rate:+.3f})")

    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": n, "peak_layer": L,
        "baseline_correct_rate": baseline_rate,
        "residual_patch_correct_rate": res_rate,
        "mlp_patch_correct_rate": mlp_rate,
        "attn_patch_correct_rate": attn_rate,
        "residual_flip_rate": res_rate - baseline_rate,
        "mlp_flip_rate": mlp_rate - baseline_rate,
        "attn_flip_rate": attn_rate - baseline_rate,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "examples.json").write_text(json.dumps(results, indent=2))
    print(f"[flip] wrote {out_dir}")


if __name__ == "__main__":
    main()
