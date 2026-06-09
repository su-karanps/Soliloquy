# Soliloquy

Soliloquy is an experimental scaffold for studying whether language models encode
an internal signal of factual correctness that is separable from output confidence.
The code generates short-form QA answers, records residual-stream activations,
trains linear probes for answer correctness, compares them against confidence
baselines, and runs causal interventions such as activation patching and steering.

The current experiments focus on:

- Primary model: `Qwen/Qwen2.5-3B-Instruct`.
- Replication models: `Qwen/Qwen2.5-7B-Instruct` and
  `meta-llama/Llama-3.1-8B-Instruct`.
- Datasets: SimpleQA, TriviaQA, NQ-Open, PopQA, and TruthfulQA-generation.
- Main prompt regime: forced-answer prompting, with retained greedy runs for
  comparison where available.
- Tracked outputs: compact CSV, JSON, Markdown, and PNG artifacts under
  `results/`.

Raw generations, hidden states, probe checkpoints, and logs are expected to live
outside the repository under `/hai/scratch/karanps/CS221M/`.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── baselines.py      # token-confidence and self-consistency baselines
│   ├── config.py         # paths, prompts, and default settings
│   ├── data.py           # dataset loaders and QARecord normalization
│   ├── generation.py     # answer generation and hidden-state capture
│   ├── grading.py        # short-answer grading and abstention detection
│   ├── patching.py       # activation patching and steering helpers
│   ├── plotting.py       # plots for probes, baselines, and interventions
│   ├── probes.py         # linear probe training and evaluation
│   └── utils.py
├── scripts/
│   ├── 01_generate.py
│   ├── 02_grade.py
│   ├── 03_probe_layers.py
│   ├── 04_verbal_confidence.py
│   ├── 05_within_question.py
│   ├── 06_transfer.py
│   ├── 07_verbal_compare.py
│   ├── 08_summary.py
│   ├── 09_overview_plot.py
│   ├── 10_minimal_layer_curve.py
│   ├── 11_logit_lens.py
│   ├── 12_steer_probe.py
│   ├── 13_confidence_dissoc.py
│   ├── 14_patch_residual.py
│   ├── 15_head_localize.py
│   ├── 16_same_question_patch.py
│   ├── 17_mlp_steer.py
│   ├── 18_decoda_causa_plot.py
│   ├── 19_position_patch.py
│   ├── 21_probe_failures.py
│   ├── 22_conf_steer.py
│   ├── 23_long_generation.py
│   ├── 24_conf_elicit_steer.py
│   ├── 25_joint_verbal_conf.py
│   ├── 26_patch_flip_rate.py
│   ├── run_all_generation.sh
│   ├── run_all_analysis.sh
│   └── run_causal_experiments.sh
├── results/
│   ├── experiments_summary.json
│   ├── experiments_summary.md
│   ├── plots/
│   └── tables/
```

## Setup

```bash
export HF_HOME=/hai/scratch/karanps/hf/
pip install -r requirements.txt
```

The generation and causal scripts assume access to the configured Hugging Face
models and enough GPU memory for the selected model. The primary Qwen2.5-3B runs
were developed for a single H100 80 GB. Probe and plotting stages are mostly CPU
bound once hidden states have been collected.

## Reproduction

Run the pipeline from the repository root:

```bash
bash scripts/run_all_generation.sh
bash scripts/run_all_analysis.sh
bash scripts/run_causal_experiments.sh
```

The generation script writes raw answers and hidden states under
`/hai/scratch/karanps/CS221M/generations/`. The analysis scripts write tracked
summaries and figures under `results/`.

Useful individual entry points:

- `scripts/01_generate.py`: generate answers and save hidden states.
- `scripts/02_grade.py`: assign correctness labels.
- `scripts/03_probe_layers.py`: train layer-by-position correctness probes.
- `scripts/04_verbal_confidence.py`: collect verbalized confidence scores.
- `scripts/06_transfer.py`: evaluate cross-dataset probe transfer.
- `scripts/14_patch_residual.py`: run residual-stream rescue/corruption patching.
- `scripts/08_summary.py`: refresh `results/experiments_summary.*`.

## Stored Generation Fields

Each generated example records:

- `question`, `gold_answers`, `prompt_text`, and `prompt_len`.
- `answer_text`, generated token ids, and effective answer length.
- token log-probabilities, entropies, and margins.
- `hidden_states_path`, pointing to a tensor bundle with
  `prompt_last`, `answer_first`, `answer_last`, and `answer_mean` states.

`scripts/02_grade.py` adds correctness labels and abstention metadata. Verbalized
confidence is stored separately by `scripts/04_verbal_confidence.py`.

## Experiments

The repository implements the following experiment families:

- Generation and correctness labeling for short-form QA.
- Layer-by-position linear probes for answer correctness.
- Within-question paired controls for sampled SimpleQA generations.
- Cross-dataset transfer of learned correctness probes.
- Probe comparisons against log-probability, entropy, margin, self-consistency,
  and verbalized-confidence baselines.
- Residual-stream rescue and corruption patching.
- Component, head, and token-position localization.
- Logit-lens analysis across layers.
- Probe-direction and MLP-output steering experiments.
- Correctness-versus-verbal-confidence direction comparisons.
- Full-answer flip-rate checks after late-layer patching.

## Current Results

See `results/experiments_summary.md` for the detailed tracked report. The headline
pattern is:

- Correctness is linearly decodable from residual-stream activations on all tested
  datasets.
- Verbalized confidence is a weak correctness predictor, especially under
  forced-answer prompting.
- Decodability and causal influence separate: mid-layer probes can be predictive
  while late-layer MLP activations dominate patching effects.
- Patching can move next-token evidence toward the correct answer, but full-answer
  flips remain rare.
- One-dimensional steering along a probe direction does not reliably control
  correctness or verbal confidence.
