"""Aggregate top-level numbers from all experiments into a single report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import PROBES_DIR, RESULTS_DIR, TABLES_DIR


def maybe(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    rep = {}
    rep["per_dataset_layer_probes"] = {}
    rep["per_dataset_grade_summary"] = {}

    gen_root = Path("/hai/scratch/karanps/CS221M/generations/Qwen__Qwen2.5-3B-Instruct")
    for d in sorted(gen_root.glob("*_n*")):
        gs = maybe(d / "grade_summary.json")
        if gs:
            rep["per_dataset_grade_summary"][d.name] = gs

        probe = maybe(PROBES_DIR / d.name / "summary.json")
        if probe:
            rep["per_dataset_layer_probes"][d.name] = {
                "best_probe": probe["best_probe"],
                "baselines": probe["baselines"],
                "n": probe["n"],
                "n_correct": probe["n_correct"],
                "n_incorrect": probe["n_incorrect"],
            }

    rep["within_question"] = maybe(PROBES_DIR / "simpleqa_within_q" / "summary.json")
    rep["transfer"] = maybe(PROBES_DIR / "cross_dataset" / "summary.json")
    rep["verbal_compare"] = maybe(PROBES_DIR / "verbal_compare" / "summary.json")
    rep["qualitative_dissociation"] = maybe(PROBES_DIR / "verbal_compare" / "qualitative_dissociation.json")

    out_path = RESULTS_DIR / "experiments_summary.json"
    out_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[summary] wrote {out_path}")

    # Also dump a human-readable markdown
    md = ["# Experiments summary", ""]
    md.append("Qwen2.5-3B-Instruct on short-form factual QA — see `README.md` for design.")
    md.append("")
    md.append("## Top-line numbers")
    if rep["per_dataset_layer_probes"]:
        for ds, info in rep["per_dataset_layer_probes"].items():
            bp = info["best_probe"]
            best_baseline = max(info["baselines"].items(), key=lambda kv: kv[1])
            delta = bp["auc"] - best_baseline[1]
            md.append(f"- **{ds}** — probe AUC {bp['auc']:.3f} (pos `{bp['position']}`, L{bp['layer']}); "
                      f"best baseline {best_baseline[0]} {best_baseline[1]:.3f}; Δ = {delta:+.3f}")
    if rep["within_question"]:
        w = rep["within_question"]
        md.append(f"- **within-question control**: qid-split AUC "
                  f"{w['best_qid_split']['auc_qid_split']:.3f} vs within-question split AUC "
                  f"{w['best_within_q_split']['auc_within_q_split']:.3f}.")
    md.append("")
    md.append("## Generation totals")
    for ds, gs in rep["per_dataset_grade_summary"].items():
        md.append(f"- **{ds}**: n={gs['n']} correct={gs['correct']} "
                  f"({gs['correct']/max(gs['n'],1):.1%}) abstain={gs['abstain']} "
                  f"({gs['abstain']/max(gs['n'],1):.1%})")
    md.append("")
    md.append("## Experiment 1+4 — best per-dataset probe vs confidence baselines")
    for ds, info in rep["per_dataset_layer_probes"].items():
        bp = info["best_probe"]
        md.append(f"### {ds}  (n_correct={info['n_correct']}, n_incorrect={info['n_incorrect']})")
        md.append(f"- Best probe: pos=`{bp['position']}` L={bp['layer']}  AUC={bp['auc']:.3f}  Acc={bp['acc']:.3f}  ECE={bp['ece']:.3f}")
        for k, v in sorted(info["baselines"].items(), key=lambda kv: -kv[1]):
            md.append(f"- {k}: {v:.3f}")
        md.append("")
    if rep["within_question"]:
        w = rep["within_question"]
        ba = w["best_qid_split"]
        bb = w["best_within_q_split"]
        md.append("## Experiment 2 — within-question paired probe")
        md.append(f"- best qid-split (questions disjoint) AUC: {ba['auc_qid_split']:.3f} @ pos=`{ba['position']}`, L={ba['layer']}")
        md.append(f"- best within-question split   AUC: {bb['auc_within_q_split']:.3f} @ pos=`{bb['position']}`, L={bb['layer']}")
        md.append("")
    if rep["transfer"]:
        t = rep["transfer"]
        md.append(f"## Experiment 3 — cross-dataset transfer (pos=`{t['position']}`, L={t['layer']})")
        md.append("| train \\ test | " + " | ".join(t["datasets"]) + " |")
        md.append("|" + "---|" * (1 + len(t["datasets"])))
        for i, name in enumerate(t["datasets"]):
            cells = []
            for j in range(len(t["datasets"])):
                v = t["auc_matrix"][i][j]
                cells.append(f"{v:.2f}" if v == v else "-")
            md.append(f"| **{name}** | " + " | ".join(cells) + " |")
        md.append("")
    if rep["verbal_compare"]:
        md.append("## Experiment 4 — verbalized confidence vs probe vs baselines")
        for ds, info in rep["verbal_compare"].items():
            md.append(f"### {ds}  (best L={info['best_probe_layer']}, n_test={info['n_test']})")
            for k, v in sorted(info["aucs"].items(), key=lambda kv: -kv[1]):
                md.append(f"- {k}: {v:.3f}")
            md.append("")
    if rep["qualitative_dissociation"]:
        md.append("## Qualitative dissociation examples (high external confidence, low probe correctness, actually wrong)")
        for ex in rep["qualitative_dissociation"][:8]:
            md.append(f"- **[{ex['dataset']}]** Q: {ex['question']}")
            md.append(f"  - model: `{ex['model_answer']}`  | gold: `{ex['gold'][:3]}`")
            md.append(f"  - probe p(correct)={ex['probe_p_correct']:.2f}  "
                      f"mean-logprob={ex['mean_logprob']:.2f}  verbal_conf={ex['verbal_conf']}")
        md.append("")

    md_path = RESULTS_DIR / "experiments_summary.md"
    md_path.write_text("\n".join(md))
    print(f"[summary] wrote {md_path}")


if __name__ == "__main__":
    main()
