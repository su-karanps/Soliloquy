"""Small utilities: jsonl IO, splitting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def stratified_qid_split(qids: list[str], labels: list[int], test_frac: float, seed: int = 0):
    """Split by question ID so all generations from one qid stay on the same side."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(qids))
    # We want stratification on the *majority* label of each qid (if multiple gens).
    qid_label: dict[str, int] = {}
    label_count: dict[str, dict[int, int]] = {q: {0: 0, 1: 0} for q in unique}
    for q, y in zip(qids, labels):
        label_count[q][int(y)] += 1
    for q in unique:
        c = label_count[q]
        qid_label[q] = 1 if c[1] >= c[0] else 0
    pos = [q for q in unique if qid_label[q] == 1]
    neg = [q for q in unique if qid_label[q] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    test_pos = pos[: max(1, int(round(len(pos) * test_frac)))]
    test_neg = neg[: max(1, int(round(len(neg) * test_frac)))]
    test_set = set(test_pos) | set(test_neg)
    train_idx = [i for i, q in enumerate(qids) if q not in test_set]
    test_idx = [i for i, q in enumerate(qids) if q in test_set]
    return train_idx, test_idx
