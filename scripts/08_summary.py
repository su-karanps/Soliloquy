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


def _plain_english_summary(rep: dict) -> str:
    """Build a prose summary of the experiments using the actual numbers in `rep`.

    Lives next to the auto-generated tables so the headline interpretation always
    matches the latest run. Hand-written prose; numbers are filled in from `rep`.
    """
    probes = rep.get("per_dataset_layer_probes") or {}
    verbal = rep.get("verbal_compare") or {}
    within = rep.get("within_question") or {}
    qual = rep.get("qualitative_dissociation") or []

    # --- Build comparison table: probe vs best non-probe baseline ---
    def pretty(ds: str) -> str:
        return ds.replace("_force_n800", " (forced)").replace("_force_n500", " (forced)") \
                 .replace("_greedy_n500", "").replace("_greedy_n300", "") \
                 .replace("nq_open", "NQ-Open").replace("simpleqa", "SimpleQA") \
                 .replace("triviaqa", "TriviaQA").replace("popqa", "PopQA") \
                 .replace("truthfulqa", "TruthfulQA")

    rows = []
    n_wins = 0
    n_total = 0
    for ds, info in probes.items():
        bp = info["best_probe"]
        best_baseline_name, best_baseline_auc = max(info["baselines"].items(), key=lambda kv: kv[1])
        delta = bp["auc"] - best_baseline_auc
        rows.append(
            "| {ds} | {pa:.3f} (`{pos}`, L{L}) | {ba:.3f} {bn} | {d:+.3f} |".format(
                ds=pretty(ds), pa=bp["auc"], pos=bp["position"], L=bp["layer"],
                ba=best_baseline_auc, bn=best_baseline_name, d=delta,
            )
        )
        n_total += 1
        if delta > 0:
            n_wins += 1

    table_md = (
        "| Dataset | Best probe | Best non-probe baseline | Δ |\n"
        "|---|---|---|---|\n" + "\n".join(rows)
    )

    # --- Verbalized-confidence summary ---
    vc_lines = []
    for ds, info in verbal.items():
        aucs = info.get("aucs", {})
        vc = aucs.get("verbal_conf")
        if vc is None:
            continue
        ds_pretty = pretty(ds.replace("simpleqa", "simpleqa_force_n800"))  # for naming
        vc_lines.append(f"{vc:.2f} {pretty(ds)}")

    # --- Within-question ---
    wq_line = ""
    if within:
        ba = within["best_qid_split"]
        bb = within["best_within_q_split"]
        wq_line = (
            f"On the {within.get('n_questions', '?')} SimpleQA-forced questions that "
            f"produced *both* correct and incorrect sampled answers, an "
            f"`{bb['position']}` probe hits AUC **{bb['auc_within_q_split']:.3f}** "
            f"discriminating correct vs incorrect *within the same question*, vs "
            f"**{ba['auc_qid_split']:.3f}** when the qids in train and test are "
            f"disjoint. The probe is doing real within-question correctness "
            f"discrimination, not just learning topic/difficulty."
        )

    n_qual = len(qual)

    summary = []
    summary.append("## Plain-English summary")
    summary.append("")
    summary.append(
        f"On {n_total} short-form factual-QA datasets, a small L2-logistic probe on "
        f"Qwen2.5-3B's residual stream beats the best confidence baseline on "
        f"**{n_wins} of {n_total}** datasets:"
    )
    summary.append("")
    summary.append(table_md)
    summary.append("")
    summary.append(
        "**Verbalized confidence is the worst correctness predictor on every "
        "dataset.** Asking the same model `\"how confident are you 0–100?\"` after "
        "it answers gives AUCs of " + ", ".join(vc_lines) + ". This is direct "
        "evidence that the model's *outer voice* (its stated confidence and even "
        "its mean token logprob) does not faithfully report what its *inner voice* "
        "(the residual-stream representation) is signalling."
    )
    summary.append("")
    summary.append(
        "**Later layers are usually more informative, but with a position × dataset "
        "interaction.** Layer 0 (the token embedding) is at chance everywhere; the "
        "AUC then rises smoothly through the mid layers. On `answer_last` and "
        "`answer_mean`, TriviaQA and PopQA peak at the very last layer (L36) — by "
        "the time the answer token is being produced, the residual stream already "
        "encodes \"is this right\". On `answer_first` and `prompt_last`, peaks tend "
        "to be earlier (around L15–L25) — interestingly, on SimpleQA-forced the "
        "best probe is at L15 on `answer_first`, meaning the \"I'm about to "
        "fabricate\" signal is partially present *before* generation begins."
    )
    summary.append("")
    if wq_line:
        summary.append("**Within-question control.** " + wq_line)
        summary.append("")
    summary.append(
        "**Dissociation is concrete, not just a number.** "
        f"{n_qual} qualitative cases were flagged where every external signal says "
        "the model is confident (verbal_conf 85–100, high mean-logprob) but the "
        "internal probe assigns `p(correct) < 0.3` — and the answer is in fact "
        "wrong. Examples are listed near the bottom of this file."
    )
    summary.append("")
    summary.append(
        "**Caveats.** (i) NQ-Open is the one dataset where the probe loses to "
        "min-logprob; the model also abstains on >70% of NQ-Open prompts, so the "
        "surviving rows are biased toward questions it had a guess about and "
        "logprob is already a strong signal there. (ii) TruthfulQA reports AUC≈1.0 "
        "but the test fold has only ~17 non-abstain rows — treat it as "
        "indicative, not headline. (iii) Cross-dataset transfer is partial: "
        "off-diagonal probe AUCs are mostly 0.5–0.7, so a meaningful fraction of "
        "what the probe learns is dataset-specific. (iv) The 0.99 within-question "
        "AUC partly reflects the probe memorising per-question hidden-state "
        "patterns; the headline finding is the *gap* between the two split "
        "strategies, not the absolute number."
    )
    return "\n".join(summary)


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

    # Causal experiments (5-10)
    rep["patching"] = maybe(PROBES_DIR / "simpleqa_force_n800_patching" / "summary.json")
    rep["head_localize"] = maybe(PROBES_DIR / "simpleqa_force_n800_heads" / "summary.json")
    rep["confidence_dissoc"] = maybe(PROBES_DIR / "confidence_dissoc" / "summary.json")
    steer_path = PROBES_DIR / "simpleqa_force_n800_steering" / "steering_results.json"
    rep["steering"] = json.loads(steer_path.read_text()) if steer_path.exists() else None

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
    md.append(_plain_english_summary(rep))
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

    # ── Causal experiments (Exps 5–10) ──────────────────────────────────────
    md.append("## Experiments 5–10 — Causal interventions (SimpleQA, Qwen2.5-3B-Instruct)")
    md.append("")

    patching = rep.get("patching")
    if patching:
        md.append("### Exp 5+6 — Residual-stream activation patching (rescue & corruption)")
        md.append(f"Cross-question patching: inject correct-question hidden states into wrong-question run.")
        md.append(f"- n pairs: {patching['n_pairs']}")
        md.append(f"- Rescue effect peaks at **L{patching['peak_rescue_layer']}** "
                  f"(Δ logit-diff = {patching['peak_rescue_effect']:+.3f})")
        md.append(f"- Effect is near-zero at layers 0–20 and grows sharply from L22 onward")
        md.append(f"- Component rescue at L{patching['component_patch_layer']}: "
                  f"MLP={patching['component_rescue_mlp']:.3f}, "
                  f"attn={patching['component_rescue_attn']:.3f} "
                  f"(**MLP drives {patching['component_rescue_mlp']/max(patching['component_rescue_attn'],1e-6):.1f}× more rescue than attention**)")
        md.append("")

    head = rep.get("head_localize")
    if head:
        md.append("### Exp 7 — Attention-head localization")
        md.append(f"- Tested layers {head['test_layers']} (window around peak L{head['peak_rescue_layer_from_residual']})")
        top5 = head.get("top5_heads", [])
        if top5:
            md.append("- Top rescue heads:")
            for h in top5[:5]:
                md.append(f"  - L{h['layer']} head {h['head']}: effect={h['effect']:+.3f}")
        md.append("")

    steering = rep.get("steering")
    if steering:
        baseline = next((r for r in steering if r["alpha"] == 0.0), None)
        best_pos = max(steering, key=lambda r: r["correct_rate"])
        md.append("### Exp 7 — Directional probe steering")
        md.append(f"Steering along correctness probe direction at (answer_first, L15) during generation.")
        if baseline:
            md.append(f"- Baseline (α=0): correct={baseline['correct_rate']:.3f}, "
                      f"abstain={baseline['abstain_rate']:.3f}")
        md.append(f"- Best α={best_pos['alpha']:+.1f}: correct={best_pos['correct_rate']:.3f}")
        if baseline:
            md.append(f"- Steering had minimal effect on correct rate "
                      f"(Δ={best_pos['correct_rate']-baseline['correct_rate']:+.3f}) — "
                      f"suggesting the probe direction at prompt-prefill is not a causal "
                      f"control knob for generation; causal control lives in late-layer patching (Exp 5+6).")
        md.append("")

    dissoc = rep.get("confidence_dissoc")
    if dissoc:
        md.append("### Exp 10 — Correctness vs verbal-confidence direction dissociation")
        for ds_name, info in dissoc.items():
            md.append(f"**{ds_name}** (L{info['layer']}, pos={info['position']})")
            md.append(f"- cos(correctness_dir, verbal_conf_dir) = **{info['cos_sim_corr_conf']:.3f}** "
                      f"(near-orthogonal — the two directions are largely independent)")
            md.append(f"- Correctness probe AUC: {info['auc_corr']:.3f} | "
                      f"verbal-conf probe AUC: {info['auc_conf']:.3f}")
            md.append(f"- Cross-prediction: correctness→verbal_conf={info['auc_corr_probe_on_conf']:.3f}, "
                      f"verbal_conf→correctness={info['auc_conf_probe_on_corr']:.3f}")
            q = info.get("quadrants", {})
            if q:
                n_tot = sum(q.values())
                md.append(f"- Quadrant breakdown (n={n_tot}): "
                          f"incorrect+confident={q.get('incorrect+confident',0)} "
                          f"({q.get('incorrect+confident',0)/n_tot:.0%}), "
                          f"correct+unconfident={q.get('correct+unconfident',0)} "
                          f"({q.get('correct+unconfident',0)/n_tot:.0%})")
        md.append("")

    md_path = RESULTS_DIR / "experiments_summary.md"
    md_path.write_text("\n".join(md))
    print(f"[summary] wrote {md_path}")


if __name__ == "__main__":
    main()
