#!/usr/bin/env bash
# Run all generation passes used by Experiments 0-4.
# Designed to be re-runnable: each script skips already-generated (qid, gen_idx) keys.
set -euo pipefail
export HF_HOME=${HF_HOME:-/hai/scratch/karanps/hf/}
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
LOG=/hai/scratch/karanps/CS221M/logs/run_all_generation.log
mkdir -p "$(dirname "$LOG")"

# Datasets where Qwen2.5-3B abstains too often under the default prompt
# (almost 100% on SimpleQA, ~60% on PopQA) -> use the forced-answer prompt to
# elicit confidently-wrong answers we can probe.
echo "== SimpleQA forced, n=800 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset simpleqa --n 800 --samples-per-question 1 \
    --max-new-tokens 24 --prompt-style force \
    --out-tag simpleqa_force_n800 --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "== PopQA forced, n=500 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset popqa --n 500 --samples-per-question 1 \
    --max-new-tokens 24 --prompt-style force \
    --out-tag popqa_force_n500 --model "$MODEL" 2>&1 | tee -a "$LOG"

# Datasets where the model already answers a healthy fraction under the default prompt.
echo "== TriviaQA default, n=500 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset triviaqa --n 500 --samples-per-question 1 \
    --max-new-tokens 24 --out-tag triviaqa_greedy_n500 --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "== NQ-Open default, n=500 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset nq_open --n 500 --samples-per-question 1 \
    --max-new-tokens 24 --out-tag nq_open_greedy_n500 --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "== TruthfulQA default, n=300 ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset truthfulqa --n 300 --samples-per-question 1 \
    --max-new-tokens 32 --out-tag truthfulqa_greedy_n300 --model "$MODEL" 2>&1 | tee -a "$LOG"

# Sampled SimpleQA (forced) — for Experiment 2 within-question paired controls.
echo "== SimpleQA forced sampled, n=200, 8 samples ==" | tee -a "$LOG"
python3 scripts/01_generate.py --dataset simpleqa --n 200 --samples-per-question 8 \
    --sampled-temperature 0.7 --max-new-tokens 24 --prompt-style force \
    --out-tag simpleqa_force_sampled_n200_k8 --model "$MODEL" 2>&1 | tee -a "$LOG"
