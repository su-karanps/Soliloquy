"""Grade generated answers and write graded.jsonl alongside generations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.grading import grade
from src.utils import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    args = ap.parse_args()
    gen_dir = Path(args.gen_dir)
    in_path = gen_dir / "generations.jsonl"
    out_path = gen_dir / "graded.jsonl"
    rows = read_jsonl(in_path)
    n = 0
    n_correct = 0
    n_abstain = 0
    for r in rows:
        g = grade(r.get("answer_text", ""), r.get("gold_answers", []))
        r.update(g)
        n += 1
        n_correct += int(g["is_correct"])
        n_abstain += int(g["did_abstain"])
    write_jsonl(out_path, rows)
    print(f"[grade] {in_path.name}: n={n} correct={n_correct} ({n_correct/n:.2%}) "
          f"abstain={n_abstain} ({n_abstain/max(n,1):.2%})")
    with (gen_dir / "grade_summary.json").open("w") as f:
        json.dump({"n": n, "correct": n_correct, "abstain": n_abstain}, f, indent=2)


if __name__ == "__main__":
    main()
