"""Dataset loaders that return a uniform list of (question, gold_answers) records.

`gold_answers` is always a list of strings to support datasets with multiple aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from datasets import load_dataset


@dataclass
class QARecord:
    qid: str
    question: str
    gold_answers: list[str]
    topic: str | None = None
    dataset: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _truncate(records: list[QARecord], n: int | None) -> list[QARecord]:
    if n is None or n >= len(records):
        return records
    return records[:n]


def load_simpleqa(n: int | None = None) -> list[QARecord]:
    """SimpleQA (OpenAI) hosted by basicv8vc/SimpleQA. Single-answer fact questions."""
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    out: list[QARecord] = []
    for i, row in enumerate(ds):
        q = row.get("problem") or row.get("question")
        a = row.get("answer") or row.get("solution")
        if not q or not a:
            continue
        meta = row.get("metadata") or {}
        topic = None
        if isinstance(meta, dict):
            topic = meta.get("topic")
        elif isinstance(meta, str):
            # SimpleQA stores metadata as a stringified dict in some snapshots
            try:
                import ast

                topic = ast.literal_eval(meta).get("topic")
            except Exception:
                topic = None
        out.append(
            QARecord(
                qid=f"simpleqa-{i}",
                question=q.strip(),
                gold_answers=[a.strip()],
                topic=topic,
                dataset="simpleqa",
            )
        )
    return _truncate(out, n)


def load_triviaqa(n: int | None = None) -> list[QARecord]:
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    out: list[QARecord] = []
    for i, row in enumerate(ds):
        q = row["question"]
        ans = row.get("answer", {})
        aliases = list(ans.get("aliases", [])) + list(ans.get("normalized_aliases", []))
        if ans.get("value"):
            aliases = [ans["value"]] + aliases
        aliases = [a for a in aliases if a]
        if not aliases:
            continue
        out.append(
            QARecord(
                qid=f"triviaqa-{i}",
                question=q.strip(),
                gold_answers=list(dict.fromkeys(aliases)),
                dataset="triviaqa",
            )
        )
    return _truncate(out, n)


def load_nq_open(n: int | None = None) -> list[QARecord]:
    ds = load_dataset("nq_open", split="validation")
    out = []
    for i, row in enumerate(ds):
        q = row["question"]
        ans = row.get("answer", [])
        if not ans:
            continue
        out.append(
            QARecord(
                qid=f"nqopen-{i}",
                question=q.strip(),
                gold_answers=list(ans),
                dataset="nq_open",
            )
        )
    return _truncate(out, n)


def load_truthfulqa(n: int | None = None) -> list[QARecord]:
    ds = load_dataset("truthful_qa", "generation", split="validation")
    out = []
    for i, row in enumerate(ds):
        q = row["question"]
        gold = [row.get("best_answer", "")] + list(row.get("correct_answers", []))
        gold = [g for g in gold if g]
        if not gold:
            continue
        out.append(
            QARecord(
                qid=f"truthfulqa-{i}",
                question=q.strip(),
                gold_answers=gold,
                topic=row.get("category"),
                dataset="truthfulqa",
            )
        )
    return _truncate(out, n)


def load_popqa(n: int | None = None) -> list[QARecord]:
    ds = load_dataset("akariasai/PopQA", split="test")
    out = []
    for i, row in enumerate(ds):
        q = row.get("question")
        # PopQA stores possible answers as a JSON-encoded list in `possible_answers`
        gold_raw = row.get("possible_answers") or row.get("obj")
        if isinstance(gold_raw, str):
            try:
                import json

                gold = json.loads(gold_raw)
            except Exception:
                gold = [gold_raw]
        elif isinstance(gold_raw, list):
            gold = gold_raw
        else:
            gold = [str(gold_raw)] if gold_raw else []
        gold = [g for g in gold if g]
        if not q or not gold:
            continue
        out.append(
            QARecord(
                qid=f"popqa-{i}",
                question=q.strip(),
                gold_answers=gold,
                topic=row.get("prop"),
                dataset="popqa",
            )
        )
    return _truncate(out, n)


LOADERS = {
    "simpleqa": load_simpleqa,
    "triviaqa": load_triviaqa,
    "nq_open": load_nq_open,
    "truthfulqa": load_truthfulqa,
    "popqa": load_popqa,
}


def load(name: str, n: int | None = None) -> list[QARecord]:
    return LOADERS[name](n=n)
