"""Generation + hidden-state capture for Qwen2.5-style chat models.

For each question we record, per generation:

* the chat-templated prompt and prompt token ids
* generated answer text and token ids
* per-answer-token logprobs (of the realised token) and entropies
* margin between top-1 and top-2 logits at each answer step
* residual-stream hidden states at four positions x all layers:
    prompt_last, answer_first, answer_last, answer_mean

Heavy tensors are stored on disk in `<gen_dir>/hidden_states/<qid>__<gen_idx>.pt`.
Light JSONL metadata sits in `<gen_dir>/generations.jsonl`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    ACTIVATIONS_DIR,
    ANSWER_PROMPT,
    CONFIDENCE_PROMPT,
    GENERATIONS_DIR,
    POSITIONS,
    PROMPT_STYLES,
)


@dataclass
class GenConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_new_tokens: int = 32
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0
    seed: int | None = None
    capture_layers: tuple[int, ...] | None = None  # None means all layers (incl. embedding)


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_prompt(tokenizer, question: str, system: str | None = None, style: str = "default") -> str:
    template = PROMPT_STYLES.get(style, ANSWER_PROMPT)
    user = template.format(question=question)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_confidence_prompt(tokenizer, question: str, answer: str) -> str:
    user = CONFIDENCE_PROMPT.format(question=question, answer=answer)
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_model_and_tokenizer(cfg: GenConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=_resolve_dtype(cfg.dtype),
        device_map=cfg.device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def _single_generate(
    model,
    tok,
    prompt_text: str,
    cfg: GenConfig,
    capture_hidden: bool = True,
) -> dict:
    """Generate one answer for one prompt; return everything we need.

    Returns dict with keys:
      prompt_token_ids, answer_token_ids, answer_text,
      token_logprobs, token_entropies, token_margins,
      hidden_states (dict[position -> tensor of shape (L+1, hidden)]) or None
    """
    enc = tok(prompt_text, return_tensors="pt").to(model.device)
    input_ids = enc.input_ids
    attn_mask = enc.attention_mask
    prompt_len = input_ids.shape[1]

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=(cfg.temperature > 0),
        temperature=max(cfg.temperature, 1e-5),
        top_p=cfg.top_p,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    out = model.generate(**gen_kwargs)
    full_ids = out.sequences[0]  # (total_len,)
    answer_ids = full_ids[prompt_len:]
    # Drop a trailing eos for cleaner accounting (still recorded in answer_text).
    eff_answer_ids = answer_ids
    if eff_answer_ids.numel() > 0 and eff_answer_ids[-1].item() in tok.all_special_ids:
        eff_answer_ids = eff_answer_ids[:-1]

    answer_text = tok.decode(answer_ids, skip_special_tokens=True).strip()

    # Per-step logprob/entropy/margin from `out.scores`.
    token_logprobs: list[float] = []
    token_entropies: list[float] = []
    token_margins: list[float] = []
    for step_idx, score in enumerate(out.scores):
        # score shape: (1, vocab)
        logp = F.log_softmax(score[0].float(), dim=-1)
        sampled_id = full_ids[prompt_len + step_idx].item()
        token_logprobs.append(float(logp[sampled_id].item()))
        p = logp.exp()
        token_entropies.append(float(-(p * logp).sum().item()))
        top2 = torch.topk(logp, k=2).values
        token_margins.append(float((top2[0] - top2[1]).item()))

    record = {
        "prompt_text": prompt_text,
        "prompt_len": prompt_len,
        "answer_text": answer_text,
        "answer_token_ids": answer_ids.tolist(),
        "answer_len_eff": int(eff_answer_ids.shape[0]),
        "token_logprobs": token_logprobs,
        "token_entropies": token_entropies,
        "token_margins": token_margins,
        "hidden_states": None,
    }

    if capture_hidden and eff_answer_ids.shape[0] > 0:
        with torch.no_grad():
            fwd = model(
                full_ids[: prompt_len + eff_answer_ids.shape[0]].unsqueeze(0),
                output_hidden_states=True,
                use_cache=False,
            )
        # tuple of (L+1) tensors, each (1, seq_len, hidden)
        hs = fwd.hidden_states
        if cfg.capture_layers is not None:
            hs = tuple(hs[i] for i in cfg.capture_layers)
        # Stack -> (L+1, seq_len, hidden)
        stacked = torch.stack([h[0].to(torch.float32).cpu() for h in hs], dim=0)
        a_first = prompt_len
        a_last = prompt_len + eff_answer_ids.shape[0] - 1
        record["hidden_states"] = {
            "prompt_last": stacked[:, prompt_len - 1, :].clone(),
            "answer_first": stacked[:, a_first, :].clone(),
            "answer_last": stacked[:, a_last, :].clone(),
            "answer_mean": stacked[:, a_first : a_last + 1, :].mean(dim=1).clone(),
        }
    return record


def _hidden_path(out_dir: Path, qid: str, gen_idx: int) -> Path:
    return out_dir / "hidden_states" / f"{qid}__g{gen_idx}.pt"


def run_generation(
    records: Sequence,
    out_dir: Path,
    cfg: GenConfig,
    samples_per_question: int = 1,
    sampled_temperature: float = 0.7,
    capture_hidden: bool = True,
    capture_only_first: bool = True,
    overwrite: bool = False,
    desc: str = "gen",
    prompt_style: str = "default",
) -> Path:
    """Run generation for a list of QARecord-like objects.

    If `samples_per_question > 1`, the first generation is greedy and the rest use
    `sampled_temperature`. Hidden states are captured on each generation unless
    `capture_only_first=True`, in which case they are saved only on the greedy run
    (we still capture them for sampled runs that we may later use as within-question
    paired data).
    """
    out_dir = Path(out_dir)
    (out_dir / "hidden_states").mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "generations.jsonl"

    if overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    done_keys: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_keys.add(f"{obj['qid']}__g{obj['gen_idx']}")
                except Exception:
                    continue

    model, tok = load_model_and_tokenizer(cfg)

    with jsonl_path.open("a") as f_out:
        for rec in tqdm(records, desc=desc):
            qid = rec.qid
            for g in range(samples_per_question):
                key = f"{qid}__g{g}"
                if key in done_keys:
                    continue
                temp = 0.0 if g == 0 and samples_per_question > 1 else (
                    cfg.temperature if samples_per_question == 1 else sampled_temperature
                )
                gen_cfg = GenConfig(
                    model_name=cfg.model_name,
                    dtype=cfg.dtype,
                    device=cfg.device,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=temp,
                    top_p=cfg.top_p,
                    seed=cfg.seed,
                    capture_layers=cfg.capture_layers,
                )
                if gen_cfg.seed is not None:
                    torch.manual_seed(gen_cfg.seed + g)
                prompt_text = build_prompt(tok, rec.question, style=prompt_style)
                want_hidden = capture_hidden and (g == 0 or not capture_only_first)
                result = _single_generate(model, tok, prompt_text, gen_cfg, capture_hidden=want_hidden)

                # Save hidden states to disk and record path.
                hs_path = None
                if result["hidden_states"] is not None:
                    hs_path = _hidden_path(out_dir, qid, g)
                    torch.save(result["hidden_states"], hs_path)

                json_obj = {
                    "qid": qid,
                    "dataset": rec.dataset,
                    "topic": rec.topic,
                    "question": rec.question,
                    "gold_answers": rec.gold_answers,
                    "gen_idx": g,
                    "temperature": temp,
                    "model": cfg.model_name,
                    "prompt_text": result["prompt_text"],
                    "prompt_len": result["prompt_len"],
                    "answer_text": result["answer_text"],
                    "answer_token_ids": result["answer_token_ids"],
                    "answer_len_eff": result["answer_len_eff"],
                    "token_logprobs": result["token_logprobs"],
                    "token_entropies": result["token_entropies"],
                    "token_margins": result["token_margins"],
                    "hidden_states_path": str(hs_path) if hs_path else None,
                }
                f_out.write(json.dumps(json_obj) + "\n")
                f_out.flush()
    return jsonl_path


@torch.no_grad()
def ask_confidence(
    model,
    tok,
    question: str,
    answer: str,
    max_new_tokens: int = 5,
) -> dict:
    """Run a separate verbalised-confidence prompt; return integer 0..100 if parseable."""
    prompt_text = build_confidence_prompt(tok, question, answer)
    enc = tok(prompt_text, return_tensors="pt").to(model.device)
    out = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    val: int | None = None
    import re

    m = re.search(r"\d{1,3}", text)
    if m:
        try:
            v = int(m.group())
            if 0 <= v <= 100:
                val = v
        except Exception:
            val = None
    return {"raw": text, "value": val}


def run_verbal_confidence(
    generations_jsonl: Path,
    out_path: Path,
    cfg: GenConfig,
    overwrite: bool = False,
) -> Path:
    """For each (qid, gen_idx) in generations.jsonl, ask verbalized confidence."""
    out_path = Path(out_path)
    if overwrite and out_path.exists():
        out_path.unlink()
    done: set[tuple[str, int]] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    o = json.loads(line)
                    done.add((o["qid"], o["gen_idx"]))
                except Exception:
                    continue
    model, tok = load_model_and_tokenizer(cfg)
    with generations_jsonl.open() as f_in, out_path.open("a") as f_out:
        rows = [json.loads(line) for line in f_in]
        for row in tqdm(rows, desc="verbal_conf"):
            key = (row["qid"], row["gen_idx"])
            if key in done:
                continue
            if not row["answer_text"]:
                conf = {"raw": "", "value": None}
            else:
                conf = ask_confidence(model, tok, row["question"], row["answer_text"])
            f_out.write(json.dumps({
                "qid": row["qid"],
                "gen_idx": row["gen_idx"],
                "verbal_conf_raw": conf["raw"],
                "verbal_conf": conf["value"],
            }) + "\n")
            f_out.flush()
    return out_path
