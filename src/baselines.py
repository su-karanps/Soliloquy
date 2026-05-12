"""Confidence baselines computed from generation metadata.

Each baseline is a per-example score with the convention that HIGHER = MORE LIKELY CORRECT.
We negate entropies so that higher = more confident.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


def mean_logprob(row: dict) -> float | None:
    lps = row.get("token_logprobs") or []
    if not lps:
        return None
    return float(np.mean(lps))


def min_logprob(row: dict) -> float | None:
    lps = row.get("token_logprobs") or []
    if not lps:
        return None
    return float(np.min(lps))


def first_entropy_neg(row: dict) -> float | None:
    ents = row.get("token_entropies") or []
    if not ents:
        return None
    return -float(ents[0])


def mean_entropy_neg(row: dict) -> float | None:
    ents = row.get("token_entropies") or []
    if not ents:
        return None
    return -float(np.mean(ents))


def mean_margin(row: dict) -> float | None:
    mgs = row.get("token_margins") or []
    if not mgs:
        return None
    return float(np.mean(mgs))


def first_margin(row: dict) -> float | None:
    mgs = row.get("token_margins") or []
    if not mgs:
        return None
    return float(mgs[0])


def self_consistency_score(
    rows: list[dict],
    normalize_fn,
) -> dict[tuple[str, int], float]:
    """For each (qid, gen_idx), the fraction of OTHER generations on the same qid
    whose normalized answer matches this generation. Higher = more consistent."""
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    out: dict[tuple[str, int], float] = {}
    for qid, group in by_q.items():
        if len(group) <= 1:
            for r in group:
                out[(qid, r["gen_idx"])] = 0.0
            continue
        norm = [normalize_fn(r.get("answer_text", "")) for r in group]
        for i, r in enumerate(group):
            same = sum(1 for j, n in enumerate(norm) if j != i and n and n == norm[i])
            out[(qid, r["gen_idx"])] = same / (len(group) - 1)
    return out
