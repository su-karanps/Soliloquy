"""Project-wide configuration: paths, model names, defaults.

All large artifacts (model weights, activations, generations) live in
SCRATCH_DIR; small results (plots, tables, summaries) live in RESULTS_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"

SCRATCH_DIR = Path(os.environ.get("CS221M_SCRATCH", "/hai/scratch/karanps/CS221M"))
ACTIVATIONS_DIR = SCRATCH_DIR / "activations"
GENERATIONS_DIR = SCRATCH_DIR / "generations"
PROBES_DIR = SCRATCH_DIR / "probes"
LOGS_DIR = SCRATCH_DIR / "logs"

for d in (RESULTS_DIR, PLOTS_DIR, TABLES_DIR, ACTIVATIONS_DIR, GENERATIONS_DIR, PROBES_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", "/hai/scratch/karanps/hf/")

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LARGER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

ANSWER_PROMPT = (
    "Answer the following question in one short phrase. "
    'If you do not know, say "I don\'t know."\n'
    "Question: {question}\n"
    "Answer:"
)

ANSWER_FORCED_PROMPT = (
    "Answer the following factual question with a single short phrase. "
    "You must give your best guess; do NOT say you don't know, do NOT refuse, "
    "do NOT add commentary. Reply with the answer only.\n"
    "Question: {question}\n"
    "Answer:"
)

PROMPT_STYLES = {
    "default": ANSWER_PROMPT,
    "force": ANSWER_FORCED_PROMPT,
}

CONFIDENCE_PROMPT = (
    "Question: {question}\n"
    "Your answer: {answer}\n"
    "How confident are you that your answer is correct? "
    "Reply with a single integer from 0 to 100 and nothing else."
)

ABSTAIN_STRINGS = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "unknown",
    "no answer",
    "cannot answer",
    "can't answer",
    "unsure",
)

POSITIONS = ("prompt_last", "answer_first", "answer_last", "answer_mean")
