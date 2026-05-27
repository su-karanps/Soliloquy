"""Hook-based activation patching utilities for Qwen2.5-style decoder-only models.

Three patching targets are supported:
  - residual: the full residual-stream output of a decoder layer (attn + MLP combined)
  - attn_out: the attention module's contribution *before* the residual add
  - mlp_out: the MLP module's contribution *before* the residual add

All functions are zero-grad (inference only).
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level hook helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _replace_at(module: nn.Module, position: int, replacement: torch.Tensor, output_idx: int = 0):
    """Context manager that replaces output_idx-th element of module's output
    at sequence position `position` with `replacement` (shape: hidden_dim,)."""
    replacement = replacement.clone()

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            h = out[output_idx]
            if h.shape[1] <= position:
                return out
            h = h.clone()
            h[:, position, :] = replacement.to(h.device, h.dtype)
            return (h,) + out[output_idx + 1:]
        else:
            if out.shape[1] <= position:
                return out
            out = out.clone()
            out[:, position, :] = replacement.to(out.device, out.dtype)
            return out

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _get_layer(model, layer: int) -> nn.Module:
    return model.model.layers[layer]


def _get_attn(model, layer: int) -> nn.Module:
    return model.model.layers[layer].self_attn


def _get_mlp(model, layer: int) -> nn.Module:
    return model.model.layers[layer].mlp


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_residual(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return all residual-stream hidden states: shape (n_layers+1, seq_len, hidden)."""
    out = model(input_ids, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0].cpu().float() for h in out.hidden_states], dim=0)


@torch.no_grad()
def cache_component_outputs(
    model, input_ids: torch.Tensor, layers: list[int]
) -> dict[str, dict[int, torch.Tensor]]:
    """Cache attn_out and mlp_out at specified layers.

    Returns {
        "attn": {layer: (seq_len, hidden)},
        "mlp":  {layer: (seq_len, hidden)},
    }
    These are the module contributions BEFORE being added to the residual.
    """
    attn_cache: dict[int, torch.Tensor] = {}
    mlp_cache: dict[int, torch.Tensor] = {}
    handles = []

    for L in layers:
        def make_attn_hook(l):
            def h(mod, inp, out):
                attn_cache[l] = out[0][0].detach().cpu().float()
                return out
            return h

        def make_mlp_hook(l):
            def h(mod, inp, out):
                if isinstance(out, tuple):
                    mlp_cache[l] = out[0][0].detach().cpu().float()
                else:
                    mlp_cache[l] = out[0].detach().cpu().float()
                return out
            return h

        handles.append(_get_attn(model, L).register_forward_hook(make_attn_hook(L)))
        handles.append(_get_mlp(model, L).register_forward_hook(make_mlp_hook(L)))

    with torch.no_grad():
        model(input_ids, use_cache=False)
    for h in handles:
        h.remove()
    return {"attn": attn_cache, "mlp": mlp_cache}


# ---------------------------------------------------------------------------
# Patching forward passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def patch_residual_and_eval(
    model,
    corrupted_ids: torch.Tensor,
    patch_vec: torch.Tensor,
    layer: int,
    position: int,
) -> torch.Tensor:
    """Patch the residual stream at (layer, position) and return next-token logits.

    patch_vec: (hidden_dim,) tensor to inject.
    Returns logits of shape (vocab_size,) at the final sequence position.
    """
    with _replace_at(_get_layer(model, layer), position, patch_vec):
        out = model(corrupted_ids, use_cache=False)
    return out.logits[0, -1, :].cpu().float()


@torch.no_grad()
def patch_component_and_eval(
    model,
    corrupted_ids: torch.Tensor,
    patch_vec: torch.Tensor,
    layer: int,
    position: int,
    component: str,  # "attn" or "mlp"
) -> torch.Tensor:
    """Patch attn_out or mlp_out at (layer, position) and return next-token logits."""
    mod = _get_attn(model, layer) if component == "attn" else _get_mlp(model, layer)
    with _replace_at(mod, position, patch_vec):
        out = model(corrupted_ids, use_cache=False)
    return out.logits[0, -1, :].cpu().float()


# ---------------------------------------------------------------------------
# Directional steering
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _steer_at(module: nn.Module, position: int, direction: torch.Tensor, alpha: float, output_idx: int = 0):
    """Context manager that adds alpha * direction to output at `position`.

    The hook only fires during the prefill pass (seq_len > position).
    During autoregressive generation steps (seq_len == 1 with KV cache) it
    is a no-op — but that's fine because the steered KV representations are
    already cached from the prefill and propagate forward.
    """
    direction = direction.clone()

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            h = out[output_idx]
            if h.shape[1] <= position:  # generation step with KV cache — skip
                return out
            h = h.clone()
            h[:, position, :] += alpha * direction.to(h.device, h.dtype)
            return (h,) + out[output_idx + 1:]
        else:
            if out.shape[1] <= position:
                return out
            out = out.clone()
            out[:, position, :] += alpha * direction.to(out.device, out.dtype)
            return out

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def steer_and_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
    layer: int,
    position: int,
    max_new_tokens: int = 24,
) -> dict:
    """Generate with a directional steer applied at (layer, position) of the prompt.

    The hook fires on the residual stream after layer `layer`. Because attention is
    causal, applying the steer at `position` affects only the logits at positions
    >= position during subsequent generation.

    Returns a dict with 'answer_text', 'token_logprobs', 'first_token_logits'.
    """
    with _steer_at(_get_layer(model, layer), position, direction, alpha):
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    prompt_len = input_ids.shape[1]
    answer_ids = out.sequences[0, prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    token_logprobs = []
    import torch.nn.functional as F
    for step, score in enumerate(out.scores):
        tok = out.sequences[0, prompt_len + step].item()
        lp = float(F.log_softmax(score[0].float(), dim=-1)[tok].item())
        token_logprobs.append(lp)
    first_logits = out.scores[0][0].cpu().float() if out.scores else None
    return {
        "answer_text": answer_text,
        "answer_token_ids": answer_ids.tolist(),
        "token_logprobs": token_logprobs,
        "first_token_logits": first_logits,
    }


# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------

@torch.no_grad()
def logit_lens(
    hidden: torch.Tensor,
    final_norm: nn.Module,
    lm_head: nn.Module,
) -> torch.Tensor:
    """Apply the unembedding to a (hidden_dim,) tensor. Returns (vocab_size,) logits."""
    h = hidden.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)
    h = final_norm(h.to(next(final_norm.parameters()).device).to(next(final_norm.parameters()).dtype))
    return lm_head(h)[0, 0, :].cpu().float()
