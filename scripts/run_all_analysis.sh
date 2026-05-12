#!/usr/bin/env bash
# Run all analysis steps for Experiments 0-4 after generations have been produced.
set -euo pipefail
export HF_HOME=${HF_HOME:-/hai/scratch/karanps/hf/}
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
SLUG=${MODEL//\//__}
GEN_ROOT=/hai/scratch/karanps/CS221M/generations/$SLUG
LOG=/hai/scratch/karanps/CS221M/logs/run_all_analysis.log
mkdir -p "$(dirname "$LOG")"

DATASETS=(simpleqa_force_n800 triviaqa_greedy_n500 nq_open_greedy_n500 popqa_force_n500 truthfulqa_greedy_n300 simpleqa_force_sampled_n200_k8)

echo "== Grading ==" | tee -a "$LOG"
for tag in "${DATASETS[@]}"; do
    gen_dir=$GEN_ROOT/$tag
    if [ -d "$gen_dir" ]; then
        echo "  grade $tag" | tee -a "$LOG"
        python3 scripts/02_grade.py --gen-dir "$gen_dir" 2>&1 | tee -a "$LOG"
    fi
done

echo "== Verbalized confidence ==" | tee -a "$LOG"
for tag in simpleqa_force_n800 triviaqa_greedy_n500 nq_open_greedy_n500 popqa_force_n500; do
    gen_dir=$GEN_ROOT/$tag
    if [ -d "$gen_dir" ]; then
        echo "  verbal-conf $tag" | tee -a "$LOG"
        python3 scripts/04_verbal_confidence.py --gen-dir "$gen_dir" --model "$MODEL" 2>&1 | tee -a "$LOG"
    fi
done

echo "== Experiment 1+4: layer probes + baselines ==" | tee -a "$LOG"
for tag in simpleqa_force_n800 triviaqa_greedy_n500 nq_open_greedy_n500 popqa_force_n500 truthfulqa_greedy_n300; do
    gen_dir=$GEN_ROOT/$tag
    if [ -d "$gen_dir" ]; then
        echo "  probe $tag" | tee -a "$LOG"
        python3 scripts/03_probe_layers.py --gen-dir "$gen_dir" --out-tag "$tag" 2>&1 | tee -a "$LOG"
    fi
done

echo "== Experiment 2: within-question paired probes ==" | tee -a "$LOG"
if [ -d "$GEN_ROOT/simpleqa_force_sampled_n200_k8" ]; then
    python3 scripts/05_within_question.py \
        --gen-dir "$GEN_ROOT/simpleqa_force_sampled_n200_k8" \
        --out-tag simpleqa_within_q 2>&1 | tee -a "$LOG"
fi

echo "== Experiment 3: cross-dataset transfer ==" | tee -a "$LOG"
python3 scripts/06_transfer.py --out-tag cross_dataset --position answer_last \
    --gen-dirs \
        "simpleqa:$GEN_ROOT/simpleqa_force_n800" \
        "triviaqa:$GEN_ROOT/triviaqa_greedy_n500" \
        "nq_open:$GEN_ROOT/nq_open_greedy_n500" \
        "popqa:$GEN_ROOT/popqa_force_n500" \
        "truthfulqa:$GEN_ROOT/truthfulqa_greedy_n300" 2>&1 | tee -a "$LOG"

echo "== Experiment 4: verbalized-confidence comparison ==" | tee -a "$LOG"
python3 scripts/07_verbal_compare.py --out-tag verbal_compare \
    --gen-dirs \
        "simpleqa:$GEN_ROOT/simpleqa_force_n800" \
        "triviaqa:$GEN_ROOT/triviaqa_greedy_n500" \
        "nq_open:$GEN_ROOT/nq_open_greedy_n500" \
        "popqa:$GEN_ROOT/popqa_force_n500" 2>&1 | tee -a "$LOG"

echo "== Aggregate summary ==" | tee -a "$LOG"
python3 scripts/08_summary.py 2>&1 | tee -a "$LOG"
