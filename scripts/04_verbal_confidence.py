"""Ask the same model for verbalized confidence on each generated answer.

Writes <gen_dir>/verbal_conf.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DEFAULT_MODEL
from src.generation import GenConfig, run_verbal_confidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    gen_dir = Path(args.gen_dir)
    out_path = gen_dir / "verbal_conf.jsonl"
    cfg = GenConfig(model_name=args.model, max_new_tokens=5)
    run_verbal_confidence(gen_dir / "generations.jsonl", out_path, cfg)
    print(f"[verbal] wrote {out_path}")


if __name__ == "__main__":
    main()
