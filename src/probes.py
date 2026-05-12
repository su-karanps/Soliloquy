"""Linear-probe utilities for correctness from cached hidden states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)


def load_hidden_dataset(
    records: list[dict],
    position: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load X, y arrays for a list of metadata rows (each must include hidden_states_path).

    `records` are filtered metadata rows already restricted to those with hidden states.
    """
    X = []
    y = []
    qids = []
    for r in records:
        path = r.get("hidden_states_path")
        if not path:
            continue
        h = torch.load(path, map_location="cpu", weights_only=False)
        vec = h[position][layer].numpy()
        X.append(vec)
        y.append(int(r["is_correct"]))
        qids.append(r["qid"])
    return np.stack(X, axis=0), np.array(y, dtype=np.int64), qids


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    standardize: bool = True,
) -> dict:
    if standardize:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    clf = LogisticRegression(
        penalty="l2", C=C, max_iter=2000, class_weight="balanced", solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]
    pred = (p_test >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_test, p_test)) if len(set(y_test)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "auc": auc,
        "acc": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "logloss": float(log_loss(y_test, p_test, labels=[0, 1])) if len(set(y_test)) > 1 else float("nan"),
        "ece": expected_calibration_error(p_test, y_test),
        "p_test": p_test.tolist(),
        "y_test": y_test.tolist(),
        "coef": clf.coef_.squeeze().tolist(),
        "intercept": float(clf.intercept_.item()),
    }


def baseline_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) <= 1 or len(labels) == 0:
        return float("nan")
    return float(roc_auc_score(labels, scores))
