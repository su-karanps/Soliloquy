#!/usr/bin/env bash
# Experiments 5-10 (causal/mechanistic interventions).
# Run AFTER run_all_analysis.sh has completed (requires graded.jsonl, probe directions, etc.)
set -euo pipefail
export HF_HOME=${HF_HOME:-/hai/scratch/karanps/hf/}
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
SLUG=${MODEL//\//__}
GEN_ROOT=/hai/scratch/karanps/CS221M/generations/$SLUG
LOG=/hai/scratch/karanps/CS221M/logs/run_causal.log
mkdir -p "$(dirname "$LOG")"

# Primary dataset for causal experiments
SQA_DIR=$GEN_ROOT/simpleqa_force_n800
SQA_TAG=simpleqa_force_n800

echo "== Re-running probe scripts to save probe directions ==" | tee -a "$LOG"
for tag in simpleqa_force_n800 triviaqa_force_n500 nq_open_force_n500 popqa_force_n500 truthfulqa_force_n300; do
    gen_dir=$GEN_ROOT/$tag
    if [ -d "$gen_dir" ]; then
        echo "  probe-direction $tag" | tee -a "$LOG"
        python3 scripts/03_probe_layers.py --gen-dir "$gen_dir" --out-tag "$tag" 2>&1 | tee -a "$LOG"
    fi
done

echo "== Exp 9 (partial) — Logit lens analysis ==" | tee -a "$LOG"
if [ -d "$SQA_DIR" ]; then
    python3 scripts/11_logit_lens.py \
        --gen-dir "$SQA_DIR" \
        --out-tag "$SQA_TAG" \
        --model "$MODEL" \
        --max-examples 200 2>&1 | tee -a "$LOG"
fi

echo "== Exp 7 (causal) — Directional probe steering ==" | tee -a "$LOG"
if [ -d "$SQA_DIR" ]; then
    python3 scripts/12_steer_probe.py \
        --gen-dir "$SQA_DIR" \
        --probe-tag "$SQA_TAG" \
        --out-tag "${SQA_TAG}_steering" \
        --model "$MODEL" \
        --max-examples 100 \
        --alphas -5.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 5.0 8.0 2>&1 | tee -a "$LOG"
fi

echo "== Exp 10 — Correctness vs verbal-confidence dissociation ==" | tee -a "$LOG"
python3 scripts/13_confidence_dissoc.py \
    --gen-dirs \
        "simpleqa:$SQA_DIR" \
        "triviaqa:$GEN_ROOT/triviaqa_force_n500" \
        "popqa:$GEN_ROOT/popqa_force_n500" \
    --probe-tags simpleqa_force_n800 triviaqa_force_n500 popqa_force_n500 \
    --out-tag confidence_dissoc 2>&1 | tee -a "$LOG"

echo "== Exp 5+6 — Residual-stream activation patching ==" | tee -a "$LOG"
if [ -d "$SQA_DIR" ]; then
    python3 scripts/14_patch_residual.py \
        --gen-dir "$SQA_DIR" \
        --out-tag "${SQA_TAG}_patching" \
        --model "$MODEL" \
        --max-pairs 60 \
        --prompt-style force 2>&1 | tee -a "$LOG"
fi

echo "== Exp 7 — Head-level localization ==" | tee -a "$LOG"
PATCH_SUMMARY=/hai/scratch/karanps/CS221M/probes/${SQA_TAG}_patching/summary.json
if [ -f "$PATCH_SUMMARY" ]; then
    python3 scripts/15_head_localize.py \
        --gen-dir "$SQA_DIR" \
        --patch-summary "$PATCH_SUMMARY" \
        --out-tag "${SQA_TAG}_heads" \
        --model "$MODEL" \
        --max-pairs 40 \
        --layer-window 3 \
        --prompt-style force 2>&1 | tee -a "$LOG"
else
    echo "  [skip] patch summary not found at $PATCH_SUMMARY" | tee -a "$LOG"
fi

echo "== All causal experiments done ==" | tee -a "$LOG"
