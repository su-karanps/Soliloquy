"""Automatic correctness grading for short-form QA.

Default grader: aggressive normalization + token-F1 + fuzzy partial-ratio.
Optional LLM judge using the same model can be plugged in later.
"""

from __future__ import annotations

import re
import string
import unicodedata

from rapidfuzz.fuzz import partial_ratio, token_set_ratio

from .config import ABSTAIN_STRINGS

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize(s: str) -> str:
    """SQuAD/TriviaQA-style normalization."""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_abstain(answer: str) -> bool:
    a = normalize(answer)
    for sub in ABSTAIN_STRINGS:
        sub_n = normalize(sub)
        if sub_n and (a == sub_n or sub_n in a):
            return True
    return False


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common: dict[str, int] = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(p)
    recall = n_common / len(g)
    return 2 * precision * recall / (precision + recall)


def grade(
    pred: str,
    gold_answers: list[str],
    fuzzy_threshold: float = 85.0,
    f1_threshold: float = 0.6,
) -> dict:
    """Return a structured grade dict.

    is_correct is True if any of:
      - normalized exact match,
      - normalized pred contains a gold answer (or vice versa) for short golds,
      - token-F1 >= f1_threshold,
      - rapidfuzz partial_ratio >= fuzzy_threshold.
    """
    if not pred:
        return {"is_correct": False, "did_abstain": False, "match_kind": "empty", "best_gold": None}
    if is_abstain(pred):
        return {"is_correct": False, "did_abstain": True, "match_kind": "abstain", "best_gold": None}

    pred_n = normalize(pred)
    best = {"is_correct": False, "did_abstain": False, "match_kind": "none", "best_gold": None}
    for gold in gold_answers:
        if not gold:
            continue
        gold_n = normalize(gold)
        if not gold_n:
            continue
        if pred_n == gold_n:
            return {"is_correct": True, "did_abstain": False, "match_kind": "exact", "best_gold": gold}
        if len(gold_n) >= 3 and gold_n in pred_n:
            best = {"is_correct": True, "did_abstain": False, "match_kind": "contains_gold", "best_gold": gold}
            continue
        if len(pred_n) >= 3 and pred_n in gold_n and len(pred_n.split()) >= len(gold_n.split()) - 1:
            best = {"is_correct": True, "did_abstain": False, "match_kind": "contains_pred", "best_gold": gold}
            continue
        f1 = token_f1(pred, gold)
        if f1 >= f1_threshold:
            best = {"is_correct": True, "did_abstain": False, "match_kind": f"f1={f1:.2f}", "best_gold": gold}
            continue
        fuzz = max(partial_ratio(pred_n, gold_n), token_set_ratio(pred_n, gold_n))
        if fuzz >= fuzzy_threshold:
            best = {"is_correct": True, "did_abstain": False, "match_kind": f"fuzz={fuzz:.0f}", "best_gold": gold}
    return best
