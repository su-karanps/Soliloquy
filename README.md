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

- **Primary model**: `Qwen/Qwen2.5-3B-Instruct` (36 transformer layers, hidden_dim 2048).
- **Replication models**: `Qwen/Qwen2.5-7B-Instruct` (28 layers), `meta-llama/Llama-3.1-8B-Instruct` (32 layers).
- Datasets: SimpleQA, TriviaQA (rc.nocontext), NQ-Open, PopQA, TruthfulQA-generation.
- Prompt: unified **forced-answer** style across all datasets and models ("You must give your
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
│   ├── 01_generate.py            # Experiment 0: gen + activations per (dataset/model)
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
│   ├── 12_steer_probe.py         # Experiment 7: directional probe steering (prompt-level)
│   ├── 13_confidence_dissoc.py   # Experiment 10: correctness vs verbal-conf directions
│   ├── 14_patch_residual.py      # Experiments 5+6: residual-stream patching
│   ├── 15_head_localize.py       # Experiment 7: attention-head localization
│   ├── 16_same_question_patch.py # Exp 5+6 v2: same-question paired patching (stronger control)
│   ├── 17_mlp_steer.py           # Exp 7 v2: directional steering at the late-MLP output
│   ├── 18_decoda_causa_plot.py   # decodability vs causality overlay figure
│   ├── 19_position_patch.py      # position-specific patching (prompt_last / answer_* positions)
│   ├── 20_paper_figures.py       # assembles poster-ready figures (Figs 1–6)
│   ├── 21_probe_failures.py      # qualitative FP/FN analysis with verbal confidence
│   ├── 22_conf_steer.py          # steer verbal-conf direction at L35 during generation
│   ├── 23_long_generation.py     # logprob trajectory + hedge detection over 80 tokens
│   ├── 24_conf_elicit_steer.py   # steer during confidence-elicitation forward pass
│   ├── 25_joint_verbal_conf.py   # joint answer+confidence single-generation elicitation
│   ├── 26_patch_flip_rate.py     # full-answer generation after L35 patching, graded
│   ├── run_all_generation.sh
│   ├── run_all_analysis.sh
│   └── run_causal_experiments.sh
└── results/
    ├── plots/                    # per-experiment PNG plots
    ├── tables/                   # per-experiment CSVs
    ├── experiments_summary.json
    └── experiments_summary.md    # human-readable top-level report
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
- **Directional steering (prompt-level)**: add α × probe\_direction to the residual
  stream at the best probe (layer, position) during generation, sweeping α.
- **Directional steering (MLP-level, script 17)**: apply ±α × probe\_dir (or
  project-out) to the late-layer MLP output at the causally active layer/position.

### Experiment 9 (partial) — Logit lens

Apply the model's unembedding matrix to the saved hidden states at each layer to
track logit(gold\_first\_token) − logit(model\_answer\_first\_token) as a function of
layer for both correct and incorrect generations.

### Experiment 10 — Correctness vs verbal-confidence direction dissociation

Train two parallel probes at the best (layer, position): one predicting `is_correct`
and one predicting high verbal confidence (≥50). Compute the cosine similarity
between the two weight directions, cross-prediction AUCs, and quadrant counts.

### New experiments — robustness & replication (scripts 16–26)

- **16 — Same-question paired patching**: uses temperature-sampled runs (8 samples/question) to find (correct, wrong) trajectories from the *same question*. Includes a cross-question control for comparison. See caveats below.
- **17 — Late-MLP directional steering**: applies the probe direction as a steering vector directly to the MLP output at the causally active late layer (L35), testing whether 1-D steering at the right component controls generation.
- **18 — Decodability vs causality figure**: overlays per-layer probe AUC (decodability) and per-layer rescue effect (causality) on dual axes, illustrating the layer gap.
- **19 — Position-specific patching**: sweeps the patch position across `{prompt_last, answer_first, answer_last, answer_mean}`.
- **20 — Paper figures**: assembles all key figures for presentation; poster-sized fonts.
- **21 — Probe failure qualitative analysis**: extracts false-positive (probe says correct, model wrong) and false-negative (probe says wrong, model correct) cases with verbal confidence annotation.
- **22 — Confidence-direction steering**: steers the verbal-confidence probe direction (not the correctness direction) at L35 MLP during answer generation; measures whether stated confidence drops.
- **23 — Long generation probe**: generates multi-sentence answers and tracks token log-probability over 80 generated tokens, plus hedge-phrase detection.
- **24 — Confidence elicitation steering**: steers the verbal-confidence direction during the confidence-elicitation forward pass itself (not a separate re-ask), measuring whether the stated number changes.
- **25 — Joint answer+confidence generation**: elicits answer and confidence number in a single unbroken generation (vs. the standard two-pass design), for direct comparison.
- **26 — Patching answer flip rate**: generates the full answer after patching at L35 (residual, MLP-only, attention-only) and grades it, reporting actual answer flip rates rather than logit-diff.

## Headline findings (`results/experiments_summary.md` has full numbers)

- **Internal correctness signal exists in mid–late layers.** Best held-out probe
  AUCs (30% qid-disjoint split): PopQA `0.913`, TriviaQA `0.850`, TruthfulQA
  `0.821`, SimpleQA `0.820`, NQ-Open `0.805`. Consistent across three models.

- **Decodability precedes causal control by ~20 layers (Qwen-3B).** Probe AUC peaks
  at L15 (answer_first), but residual patching rescue peaks at L35. Correctness is
  readable ~20 layers before it becomes causally active — more than half the network
  sits between where the signal is decodable and where it is used.

- **MLP > attention for causal control, consistently.** At the peak rescue layer,
  MLP rescue exceeds attention rescue on every model and dataset: 4.7× (Qwen-3B
  SimpleQA), 8.2× (Qwen-7B SimpleQA), across all five 3B datasets, and across
  TriviaQA, NQ-Open, PopQA, TruthfulQA.

- **Rescue effect is real in logit-diff but rarely flips answers.** Full residual
  patching at L35 shifts the correct token's logit by +2.76 but produces 0% actual
  answer flips (0/60 examples). MLP-only patching gives 5% flip rate. The model is
  highly committed to wrong answers; patching shifts the distribution without
  crossing the argmax threshold.

- **Probe-direction steering does not control generation.** Neither prompt-level nor
  late-MLP steering at L35 changes correct rate meaningfully. The verbal-confidence
  direction is also unresponsive (steering during the confidence-elicitation pass
  produces a ceiling effect: the model outputs ~100 regardless). The causal pathway
  is not accessible via any single linear direction.

- **Self-consistency provides zero benefit on SimpleQA.** Majority vote across 8
  sampled answers gives 8.0% accuracy vs 8.5% greedy — no improvement. 78.5% of
  questions are never answered correctly across all 8 samples. Errors are committed
  hallucinations, not random noise.

- **Correctness and verbal confidence are nearly orthogonal in all models.**
  cos(correctness\_dir, verbal\_conf\_dir) = 0.07 (Qwen-3B), 0.04 (Qwen-7B), 0.02
  (Llama-8B). The dissociation is most extreme on SimpleQA (hard factual recall
  under forced-answer prompting); on TriviaQA and PopQA the incorrect+confident rate
  is lower (31% and 21%).

- **Verbalized confidence is degenerate as an elicited scalar.** Under both
  separate-pass and joint (single-generation) elicitation, the model outputs near-100
  confidence for confidently-wrong examples. Joint elicitation is *worse*: 96%
  high-confidence responses vs 63.5% in separate-pass, with only 17% answer
  agreement between formats. The model's verbal confidence number is essentially a
  linguistic reflex, not a calibrated readout.

- **Pattern replicates across model families.** Qwen-7B: probe L27 / rescue L27 /
  MLP 8.2×. Llama-3.1-8B: probe L28 / rescue L31 / MLP 1.6×. Late-layer, MLP-
  dominated causal bottleneck holds, though the decodability–causality gap shrinks
  in larger models.

## Notes & caveats

- All datasets use the forced-answer prompt so results are comparable across models
  and datasets. The greedy runs retain old data for reference.
- All splits are at the qid level except Experiment 2, which deliberately allows
  within-qid splits to isolate topic/difficulty effects.
- Grading is rule-based; for short-form QA this matches TriviaQA/PopQA conventions.
  An LLM-judge grader can be plugged into `src.grading` for future work.
- **Cross-question patching caveat**: injecting hidden states from a different
  question at `prompt_last` conflates correctness signal with topic/answer content
  from the donor. Same-question patching (script 16) addresses this but has limited
  power (n=25 pairs, SEM≈1.85). The same-question rescue effect at L35 is 0.47 vs
  7.68 for cross-question — this gap is large and interpretively important: much of
  the cross-question rescue may be content injection rather than a pure correctness
  signal. We report both but treat the cross-question logit-diff curves as the
  primary decodability/localization signal rather than a clean causal claim.
- **Logit-diff vs flip rate**: the rescue effect is measured in Δlogit-diff, which
  does not require crossing the argmax threshold. Actual answer flip rates at L35
  are 0% (full residual) and 5% (MLP-only). Logit-diff captures causal influence
  on the output distribution; flip rate is a stricter test that the model largely passes.
- MLP directional steering null result (script 17) should be qualified: the probe
  direction is 1-D; the causal effect likely requires higher-dimensional intervention.
- Verbalized confidence steering experiments (scripts 22, 24) are limited by a
  ceiling effect: the model defaults to ~100 confidence for any answer it committed
  to, regardless of steering direction or magnitude.

## Why the name

Soliloquy keeps the proposal's "inner voice" framing but pins it to the more
specific claim the experiments actually support: that the model is, in effect,
muttering one verdict (in the residual stream) while saying another (in its
generated answer and verbalized confidence). Like an aside in a play, the inner
verdict is reliable; the outer one is not. The causal experiments add a twist: the
inner verdict is sometimes not even *wired* to the outer voice — it exists but
doesn't reach the output.
