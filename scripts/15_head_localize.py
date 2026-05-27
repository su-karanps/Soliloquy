"""Experiment 7 (continued): Attention-head–level localization at peak rescue layers.

After 14_patch_residual.py identifies the peak rescue layer(s), this script
patches individual attention head outputs z[layer, head] to find which heads
drive the rescue effect.

For efficiency we only test head-level patching at a small window of layers
around the peak.

Output: heatmap of rescue-effect over (layer, head).
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
from src.patching import cache_residual
from src.utils import read_jsonl


def get_first_token_id(tokenizer, text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def patch_head_and_eval(model, corrupted_ids, clean_z_head, layer, head, position, head_dim):
    """Patch the output of a single attention head at (layer, position) and return logits.

    Uses a forward_pre_hook on o_proj so that we modify the input *before* the
    projection is applied (avoids re-calling the module and infinite recursion).
    """
    clean_z = clean_z_head.clone()
    start = head * head_dim
    end = start + head_dim

    def hook(mod, inp):
        if inp[0].shape[1] <= position:   # generation steps with KV cache
            return
        x = list(inp)
        x[0] = x[0].clone()
        x[0][:, position, start:end] = clean_z.to(x[0].device, x[0].dtype)
        return tuple(x)

    o_proj = model.model.layers[layer].self_attn.o_proj
    handle = o_proj.register_forward_pre_hook(hook)
    with torch.no_grad():
        out = model(corrupted_ids, use_cache=False)
    handle.remove()
    return out.logits[0, -1, :].cpu().float()


def cache_head_outputs(model, input_ids, layers, head_dim):
    """For each layer in layers, cache the pre-o_proj output (n_heads*head_dim) at all positions."""
    cache = {}
    handles = []
    for L in layers:
        def make_hook(l):
            def h(mod, inp):
                # pre-hook: inp[0] is (B, seq, n_heads*head_dim) — z-vectors before the output projection
                cache[l] = inp[0][0].detach().cpu().float()
            return h
        handles.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(L)))
    with torch.no_grad():
        model(input_ids, use_cache=False)
    for h in handles:
        h.remove()
    return cache  # {layer: (seq, n_heads*head_dim)}


def logit_diff_baseline(model, input_ids, gold_tok, wrong_tok):
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    logits = out.logits[0, -1, :].cpu().float()
    return float(logits[gold_tok].item() - logits[wrong_tok].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--patch-summary", required=True,
                    help="summary.json from 14_patch_residual.py to get peak layer")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer-window", type=int, default=3,
                    help="Test layers in [peak-window, peak+window]")
    ap.add_argument("--max-pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", default="force")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")
    random.seed(args.seed)

    patch_summary = json.loads(Path(args.patch_summary).read_text())
    peak_layer = patch_summary["peak_rescue_layer"]
    layer_lo = max(0, peak_layer - args.layer_window)
    layer_hi = peak_layer + args.layer_window  # clamped below after model load

    gen_dir = Path(args.gen_dir)
    rows = [r for r in read_jsonl(gen_dir / "graded.jsonl") if not r.get("did_abstain")]
    correct_rows = [r for r in rows if r["is_correct"]]
    incorrect_rows = [r for r in rows if not r["is_correct"]]
    n_pairs = min(args.max_pairs, len(incorrect_rows), len(correct_rows))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    n_layers = model.config.num_hidden_layers
    layer_hi = min(layer_hi, n_layers - 1)   # clamp to valid range
    test_layers = list(range(layer_lo, layer_hi + 1))
    print(f"[head] peak_layer={peak_layer}, testing layers {test_layers}, heads={n_heads}")

    # head_rescue[L][head] = list of effects across pairs
    head_rescue: dict[int, dict[int, list]] = {L: {h: [] for h in range(n_heads)} for L in test_layers}

    wrong_sample = random.sample(incorrect_rows, n_pairs)

    for r_wrong in tqdm(wrong_sample, desc="head-patch"):
        r_right = random.choice(correct_rows)

        gold_tok = get_first_token_id(tok, r_wrong["gold_answers"][0] if r_wrong["gold_answers"] else "")
        model_tok = get_first_token_id(tok, r_wrong["answer_text"])
        if any(t is None for t in [gold_tok, model_tok]) or gold_tok == model_tok:
            continue

        prompt_wrong = build_prompt(tok, r_wrong["question"], style=args.prompt_style)
        prompt_right = build_prompt(tok, r_right["question"], style=args.prompt_style)
        enc_wrong = tok(prompt_wrong, return_tensors="pt").to(model.device)
        enc_right = tok(prompt_right, return_tensors="pt").to(model.device)
        pos_wrong = enc_wrong.input_ids.shape[1] - 1
        pos_right = enc_right.input_ids.shape[1] - 1

        ld_base = logit_diff_baseline(model, enc_wrong.input_ids, gold_tok, model_tok)

        # Cache head z-vectors at test_layers for the right (clean) run
        z_right = cache_head_outputs(model, enc_right.input_ids, test_layers, head_dim)

        for L in test_layers:
            if L not in z_right:
                continue
            z_L = z_right[L]  # (seq_right, n_heads*head_dim)
            for h in range(n_heads):
                start = h * head_dim
                end = start + head_dim
                z_head = z_L[pos_right, start:end].to(model.device)
                logits = patch_head_and_eval(model, enc_wrong.input_ids, z_head, L, h, pos_wrong, head_dim)
                ld = float(logits[gold_tok].item() - logits[model_tok].item())
                head_rescue[L][h].append(ld - ld_base)

    # Aggregate
    heatmap = np.zeros((len(test_layers), n_heads))
    for i, L in enumerate(test_layers):
        for h in range(n_heads):
            vals = head_rescue[L][h]
            heatmap[i, h] = np.mean(vals) if vals else 0.0

    # Save
    out_dir = PROBES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PLOTS_DIR / args.out_tag
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR / args.out_tag
    tables_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "head_rescue_heatmap.npy", heatmap)
    top5 = sorted(
        [(float(heatmap[i, h]), test_layers[i], h) for i in range(len(test_layers)) for h in range(n_heads)],
        reverse=True
    )[:10]
    summary = {
        "peak_rescue_layer_from_residual": peak_layer,
        "test_layers": test_layers,
        "n_pairs": n_pairs,
        "top5_heads": [{"effect": e, "layer": L, "head": hd} for e, L, hd in top5],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[head] top heads: {top5[:5]}")

    # Heatmap plot
    fig, ax = plt.subplots(figsize=(min(1.5 + n_heads * 0.35, 20), 2 + len(test_layers) * 0.5))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn",
                   vmin=-max(0.1, np.abs(heatmap).max()), vmax=max(0.1, np.abs(heatmap).max()))
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(h) for h in range(n_heads)], fontsize=7)
    ax.set_yticks(range(len(test_layers)))
    ax.set_yticklabels([f"L{L}" for L in test_layers])
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Head-level rescue effect (correct→incorrect patching) — {args.out_tag}")
    plt.colorbar(im, ax=ax, label="Δ logit diff")
    fig.tight_layout()
    fig.savefig(plots_dir / "head_rescue_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[head] wrote heatmap to {plots_dir / 'head_rescue_heatmap.png'}")

    with (tables_dir / "head_rescue.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "mean_rescue_effect"])
        for i, L in enumerate(test_layers):
            for h in range(n_heads):
                w.writerow([L, h, heatmap[i, h]])


if __name__ == "__main__":
    main()
