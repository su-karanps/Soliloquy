#!/usr/bin/env bash
# Run all generation passes used by Experiments 0-4.
# Designed to be re-runnable: each script skips already-generated (qid, gen_idx) keys.
# All datasets use --prompt-style force so the model always gives an answer
# (rather than abstaining), giving us the "confidently-wrong" examples we need.
set -euo pipefail
export HF_HOME=${HF_HOME:-/hai/scratch/karanps/hf/}
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
LOG=/hai/scratch/karanps/CS221M/logs/run_all_generation.log
mkdir -p "$(dirname "$LOG")"

echo "== SimpleQA forced, n=800 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset simpleqa --n 800 --samples-per-question 1 \
    --max-new-tokens 24 --prompt-style force \
    --out-tag simpleqa_force_n800 --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "== TriviaQA forced, n=500 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset triviaqa --n 500 --samples-per-question 1 \
    --max-new-tokens 24 --prompt-style force \
    --out-tag triviaqa_force_n500 --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "== NQ-Open forced, n=500 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset nq_open --n 500 --samples-per-question 1 \
    --max-new-tokens 24 --prompt-style force \
    --out-tag nq_open_force_n500 --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "== PopQA forced, n=500 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset popqa --n 500 --samples-per-question 1 \
    --max-new-tokens 24 --prompt-style force \
    --out-tag popqa_force_n500 --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "== TruthfulQA forced, n=300 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset truthfulqa --n 300 --samples-per-question 1 \
    --max-new-tokens 32 --prompt-style force \
    --out-tag truthfulqa_force_n300 --model "$MODEL" 2>&1 | tee -a "$LOG"

# Sampled SimpleQA (forced) — for Experiment 2 within-question paired controls.
echo "== SimpleQA forced sampled, n=200, 8 samples ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset simpleqa --n 200 --samples-per-question 8 \
    --sampled-temperature 0.7 --max-new-tokens 24 --prompt-style force \
    --out-tag simpleqa_force_sampled_n200_k8 --model "$MODEL" 2>&1 | tee -a "$LOG"
