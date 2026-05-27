# Soliloquy — Probing Internal Correctness Signals in LLMs

> **soliloquy**: *a monologue in which a character speaks their thoughts aloud, used
> to reveal the character's inner feelings, motivations, or plans directly to the
> audience.*

The premise: a language model has an "outer voice" (the answer it generates and the
confidence it can be made to verbalise) and, plausibly, an *inner* one — a
representation of whether what it's about to say is actually correct. Soliloquy is
the experimental scaffold for reading that inner voice off the residual stream of a
small instruction-tuned model and asking the harder question: is that inner signal
*causally involved* in answer selection, or merely decodable?

This repo implements all ten experiments of the proposal: establish a robust
internal-correctness signal, show it is not merely answer-confidence/logit-entropy,
characterise where in the network it lives, and then probe its causal status with
activation patching, component localization, directional steering, and a
correctness-vs-confidence direction comparison.

- Model: `Qwen/Qwen2.5-3B-Instruct` (36 transformer layers, hidden_dim 2048).
- Datasets: SimpleQA, TriviaQA (rc.nocontext), NQ-Open, PopQA, TruthfulQA-generation.
- Prompt: unified **forced-answer** style across all datasets ("You must give your
  best guess; do NOT say you don't know") — eliminates abstentions and ensures a
  clean pool of confidently-wrong examples.
- Hardware: single H100 80 GB. Forward passes are bf16; probes run on CPU sklearn.
- Activations & generations live in `/hai/scratch/karanps/CS221M/`; only small CSVs,
JSONs, and PNGs are tracked here under `results/`.

## Repo layout

```
soliloquy/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py          # paths, prompts, defaults
│   ├── data.py            # uniform QARecord loaders for the five datasets
│   ├── generation.py      # gen + per-layer hidden-state capture + verbal-conf prompt
│   ├── grading.py         # normalized-EM/fuzzy/F1 grader + abstain detection
│   ├── probes.py          # logistic-regression probing utilities
│   ├── baselines.py       # mean/min logprob, entropy, margin, self-consistency
│   ├── plotting.py        # layer curves, AUC bars, transfer matrix, calibration
│   ├── patching.py        # hook-based activation patching + directional steering
│   └── utils.py
├── scripts/
│   ├── 01_generate.py            # Experiment 0: gen + activations per (dataset)
│   ├── 02_grade.py               # label correctness, write graded.jsonl
│   ├── 03_probe_layers.py        # Experiment 1+4: layer×position probes + baselines
│   ├── 04_verbal_confidence.py   # ask the model "0..100" verbal confidence
│   ├── 05_within_question.py     # Experiment 2: within-qid paired control
│   ├── 06_transfer.py            # Experiment 3: cross-dataset transfer matrix
│   ├── 07_verbal_compare.py      # Experiment 4: probe vs verbal-conf vs logprob
│   ├── 08_summary.py             # roll up all results into json + md report
│   ├── 09_overview_plot.py       # cross-dataset AUC overview bar chart
│   ├── 10_minimal_layer_curve.py # single-position layer curve (no extras)
│   ├── 11_logit_lens.py          # Experiment 9: logit-lens across layers
│   ├── 12_steer_probe.py         # Experiment 7: directional probe steering
│   ├── 13_confidence_dissoc.py   # Experiment 10: correctness vs verbal-conf directions
│   ├── 14_patch_residual.py      # Experiments 5+6: residual-stream patching
│   ├── 15_head_localize.py       # Experiment 7: attention-head localization
│   ├── run_all_generation.sh
│   ├── run_all_analysis.sh
│   └── run_causal_experiments.sh
└── results/
    ├── plots/                    # per-experiment PNG plots
    ├── tables/                   # per-experiment CSVs
    ├── experiments_summary.json
    └── experiments_summary.md    # human-readable top-level report (see scripts/08)
```

## Reproducing

```bash
export HF_HOME=/hai/scratch/karanps/hf/
cd soliloquy
pip install -r requirements.txt

bash scripts/run_all_generation.sh       # ~10 min on an H100
bash scripts/run_all_analysis.sh         # ~5-10 min, CPU bound
bash scripts/run_causal_experiments.sh   # ~2-3 hr on an H100
```

After this, `results/experiments_summary.md` and the per-experiment plots/CSVs are
written. Large artefacts (per-question hidden states, raw generations) live under
`/hai/scratch/karanps/CS221M/{generations,probes,logs}/`.

## What is collected per generation

For every `(qid, gen_idx)` we store, in `<gen_dir>/generations.jsonl`:

- `question`, `gold_answers`, `prompt_text`, `prompt_len`
- `answer_text`, `answer_token_ids`, `answer_len_eff` (drops trailing EOS)
- `token_logprobs`, `token_entropies`, `token_margins` (per generated token)
- `hidden_states_path` -> a `.pt` dict with four keys
`{prompt_last, answer_first, answer_last, answer_mean}`, each a tensor of shape
`(L+1, hidden_dim)` (layer 0 = token embedding, layers 1..L = transformer blocks).

`02_grade.py` then adds `is_correct`, `did_abstain`, `match_kind`, `best_gold` to a
sibling `graded.jsonl`. Verbalized confidence (a separate forward pass on a 0–100
prompt) lives in `verbal_conf.jsonl`.

## Experiments implemented

### Experiment 0 — Generation & correctness labelling

- Greedy generation under the **force** prompt style ("You must give your best guess;
  do NOT say you don't know") across all five datasets. This produces a uniform pool
  of answered (non-abstaining) examples, including many confidently-wrong ones.
- Grader uses SQuAD-style normalization plus token-F1, partial-ratio fuzzy match,
  and substring containment. Abstention strings are detected before grading.

### Experiment 1 — Layer × position correctness probes

For each `(position, layer)` in `{prompt_last, answer_first, answer_last, answer_mean}` ×
`0..36`, an L2-logistic regression is fit on standardized hidden states to predict
`is_correct` (non-abstain only). We report AUC, accuracy, F1, log-loss, and ECE; the
key figure is the layer×position AUC curve overlaid against confidence baselines.
The best probe direction is saved to `best_probe_direction.npz` for downstream
causal experiments.

### Experiment 2 — Within-question paired control

We use the sampled SimpleQA-forced runs (8 samples / question) and restrict to qids
that produced both at least one correct and one incorrect non-abstain answer. We
then train a probe (a) splitting at the qid level (questions disjoint) and (b)
splitting within qid (each qid contributes to both train and test). Comparing the
two AUC curves tests whether the signal survives once topic/difficulty are held fixed.

### Experiment 3 — Cross-dataset transfer

A single probe is trained at the best position/layer on each dataset and evaluated
on every other dataset's non-abstain generations. We compare the resulting `5×5`
AUC matrix to the matching mean-logprob AUC matrix.

### Experiment 4 — Confidence baselines & verbalized confidence dissociation

On the held-out 30% split of each dataset we compare the best layer probe against:
mean and min token log-probability, first-token entropy, mean entropy, margin
(top-1 − top-2 logit), self-consistency over sampled generations, and verbalized
confidence (a separate 0–100 prompt to the same model).

### Experiment 5+6 — Residual-stream activation patching (rescue & corruption)

For pairs of (correctly-answered question, incorrectly-answered question), we patch
the residual-stream hidden state at each layer from the correct run into the
incorrect run and measure the change in logit-diff =
logit(gold\_first\_token) − logit(model\_wrong\_token). We also run the reverse
(corrupt a correct run with a wrong run's activations). This gives a layer-wise
rescue-effect and corruption-effect curve.

### Experiment 7 — Component & head localization + directional steering

- **Attn/MLP split**: at the peak rescue layer, we patch only attn\_out or mlp\_out
  separately to determine which component carries the causal effect.
- **Head-level**: patch individual attention-head outputs (pre-o\_proj z-vectors) at
  layers within a window of the peak, generating a (layer × head) rescue heatmap.
- **Directional steering**: add α × probe\_direction to the residual stream at the
  best probe (layer, position) and sweep α, measuring correct rate, abstain rate,
  and logit-diff at the first generated token.

### Experiment 9 (partial) — Logit lens

Apply the model's unembedding matrix to the saved hidden states at each layer to
track logit(gold\_first\_token) − logit(model\_answer\_first\_token) as a function of
layer for both correct and incorrect generations.

### Experiment 10 — Correctness vs verbal-confidence direction dissociation

Train two parallel probes at the best (layer, position): one predicting `is_correct`
and one predicting high verbal confidence (≥50). Compute the cosine similarity
between the two weight directions, cross-prediction AUCs, and quadrant counts.

## Headline findings (`results/experiments_summary.md` has full numbers)

- **Internal correctness signal exists in mid–late layers.** Best held-out probe
  AUCs (30% qid-disjoint split): PopQA `0.913`, TriviaQA `0.850`, TruthfulQA
  `0.821`, SimpleQA `0.820`, NQ-Open `0.805`. Peak positions are consistently
  `answer_last` or `answer_mean`, in the last third of the network (layers ~21–36).

- **The probe beats output-confidence baselines on most datasets.** Probe AUC −
  best-non-probe-baseline: PopQA +0.141, SimpleQA +0.036, TriviaQA +0.006, NQ-Open
  −0.028 (only loss). Verbalized confidence is the worst predictor on every dataset
  (0.50–0.79 AUC), well below the internal probe.

- **The signal is not just a topic/difficulty detector.** On 39 SimpleQA questions
  that produced both correct and incorrect sampled answers, a probe reaches AUC
  **0.989** discriminating correct vs incorrect *within the same question*, vs 0.724
  when questions are disjoint.

- **Late-layer causal dominance.** Residual-stream patching rescue effect is
  near-zero at layers 0–20 and grows sharply from L22, **peaking at L35**
  (Δ logit-diff = +2.76). The causally active layer is later than the most
  probe-informative layer (L15–25), suggesting the readable signal and the causal
  bottleneck are distinct.

- **MLP > attention for causal control.** At the peak rescue layer (L35), patching
  the MLP output contributes **4.7× more** rescue than patching the attention output
  (0.97 vs 0.21). The dominant attention head is L35 head 5 (+0.24 effect).

- **Probe-direction steering does not control generation.** Adding α × probe\_dir
  at the prompt level (answer\_first, L15) had negligible effect on correct rate
  across α ∈ {−5, …, +8}. This is a meaningful negative: the direction is readable
  but is not a causal handle at the prompt level — the causal bottleneck lies in the
  late-layer MLP, not where the probe signal first appears.

- **Correctness and verbal confidence are nearly orthogonal directions.**
  cos(correctness\_dir, verbal\_conf\_dir) ≈ **0.07** (SimpleQA), **0.11** (PopQA),
  **0.15** (TriviaQA). On SimpleQA, **61% of responses are incorrect+verbally
  confident** vs just 1% correct+verbally unconfident.

## Notes & caveats

- All datasets use the forced-answer prompt so results are comparable. The greedy
  runs retain old data for reference but the forced runs are the primary analysis.
- All splits are at the qid level except Experiment 2, which deliberately allows
  within-qid splits to isolate topic/difficulty effects.
- Grading is rule-based; for short-form QA this matches TriviaQA/PopQA conventions.
  An LLM-judge grader can be plugged into `src.grading` for future work.
- The patching experiment uses cross-question patching (inject correct-Q hidden
  states into wrong-Q runs at `prompt_last`). Because both runs share the same
  prompt tokens only up to `prompt_last`, the patch represents "correct-question
  context" rather than a surgical same-question intervention; the clean-vs-corrupted
  input counterfactual is left for future work with temperature-sampled pairs.

## Why the name

Soliloquy keeps the proposal's "inner voice" framing but pins it to the more
specific claim the experiments actually support: that the model is, in effect,
muttering one verdict (in the residual stream) while saying another (in its
generated answer and verbalized confidence). Like an aside in a play, the inner
verdict is reliable; the outer one is not. The causal experiments add a twist: the
inner verdict is sometimes not even *wired* to the outer voice — it exists but
doesn't reach the output.
