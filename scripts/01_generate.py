"""Run generation + activation capture for a dataset (Experiment 0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import data
from src.config import GENERATIONS_DIR, DEFAULT_MODEL
from src.generation import GenConfig, run_generation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(data.LOADERS.keys()))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--samples-per-question", type=int, default=1)
    ap.add_argument("--sampled-temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-tag", default=None,
                    help="Subdirectory name under GENERATIONS_DIR. Defaults to <dataset>_<n>.")
    ap.add_argument("--capture-only-first", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--prompt-style", default="default", choices=["default", "force"])
    args = ap.parse_args()

    tag = args.out_tag or f"{args.dataset}_n{args.n}_s{args.samples_per_question}"
    out_dir = GENERATIONS_DIR / args.model.replace("/", "__") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen] loading dataset {args.dataset} (n={args.n})")
    records = data.load(args.dataset, n=args.n)
    print(f"[gen] {len(records)} records")

    cfg = GenConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        seed=args.seed,
    )
    jsonl_path = run_generation(
        records,
        out_dir,
        cfg,
        samples_per_question=args.samples_per_question,
        sampled_temperature=args.sampled_temperature,
        capture_hidden=True,
        capture_only_first=args.capture_only_first,
        overwrite=args.overwrite,
        desc=f"{args.dataset}",
        prompt_style=args.prompt_style,
    )

    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[gen] wrote {jsonl_path}")


if __name__ == "__main__":
    main()
