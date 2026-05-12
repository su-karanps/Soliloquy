# Soliloquy — Probing Internal Correctness Signals in LLMs

> **soliloquy**: *a monologue in which a character speaks their thoughts aloud, used
> to reveal the character's inner feelings, motivations, or plans directly to the
> audience.*

The premise: a language model has an "outer voice" (the answer it generates and the
confidence it can be made to verbalise) and, plausibly, an *inner* one — a
representation of whether what it's about to say is actually correct. Soliloquy is
the experimental scaffold for reading that inner voice off the residual stream of a
small instruction-tuned model and showing it is not the same thing as the outer
voice.

This repo implements the first four experiments of the project proposal: establish a
robust internal-correctness signal in a small instruction-tuned model, show it is not
merely answer-confidence/logit-entropy, and characterise where in the network it lives.
Causal interventions (Experiments 5–8) are left for a later iteration once these
probing/baseline results are in.

- Model: `Qwen/Qwen2.5-3B-Instruct` (36 transformer layers, hidden_dim 2048).
- Datasets: SimpleQA, TriviaQA (rc.nocontext), NQ-Open, PopQA, TruthfulQA-generation.
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
│   ├── run_all_generation.sh
│   └── run_all_analysis.sh
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
pip install -r requirements.txt          # adds rapidfuzz; everything else already present

bash scripts/run_all_generation.sh       # ~10 min on an H100
bash scripts/run_all_analysis.sh         # ~5-10 min, CPU bound
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

- Greedy generation under two prompt styles:
  - **default** ("If you do not know, say 'I don't know'") — used on TriviaQA,
  NQ-Open, TruthfulQA, where the 3B model already answers a decent fraction.
  - **force** ("You must give your best guess; do NOT say you don't know") — used
  on SimpleQA and PopQA, where the default prompt produced almost 100% abstain
  on a 3B model (so we had zero confidently-wrong examples).
- Grader uses SQuAD-style normalization plus token-F1, partial-ratio fuzzy match,
and substring containment, mirroring TriviaQA/PopQA conventions. Abstention
strings ("I don't know", "I'm not sure", etc.) are detected before grading.

### Experiment 1 — Layer × position correctness probes

For each `(position, layer)` in `{prompt_last, answer_first, answer_last, answer_mean}` ×
`0..36`, an L2-logistic regression is fit on standardized hidden states to predict
`is_correct` (non-abstain only). We report AUC, accuracy, F1, log-loss, and ECE; the
key figure is the layer×position AUC curve overlaid against confidence baselines.

### Experiment 2 — Within-question paired control

We use the sampled SimpleQA-forced runs (8 samples / question) and restrict to qids
that produced both at least one correct and one incorrect non-abstain answer. We
then train a probe (a) splitting at the qid level (questions disjoint) and (b)
splitting within qid (each qid contributes to both train and test). Comparing the
two AUC curves tests whether the signal survives once topic/difficulty are held
fixed.

### Experiment 3 — Cross-dataset transfer

A single probe is trained at the best position/layer on each dataset and evaluated
on every other dataset's non-abstain generations. We compare the resulting `5×5`
AUC matrix to the matching mean-logprob AUC matrix.

### Experiment 4 — Confidence baselines & verbalized confidence dissociation

On the held-out 30% split of each dataset we compare:

- the best layer probe's AUC (and ECE),
- mean and min token log-probability,
- first-token entropy and mean entropy (negated so higher = more confident),
- margin (top-1 − top-2 logit),
- self-consistency over sampled generations,
- verbalized confidence (a separate 0–100 prompt to the same model).

The script also surfaces qualitative dissociation cases — high verbal/log-prob
confidence, low internal-probe correctness probability, actually wrong — into
`results/experiments_summary.md`.

## Headline findings (`results/experiments_summary.md` has the full numbers)

- **Internal correctness signal exists in mid–late layers.** Best held-out per-dataset
probe AUCs (Qwen2.5-3B-Instruct, 30% qid-disjoint split):
TriviaQA `0.929`, PopQA `0.913`, SimpleQA `0.820`, NQ-Open `0.730`,
TruthfulQA `1.000` (n_test=17, indicative). The peak position is consistently
`answer_last` or `answer_mean`, around the last third of the network (layers
~22–36). Plot: `results/plots/<dataset>/layer_curve.png`. Cross-dataset overview:
`results/plots/overview_probe_vs_baselines.png`.
- **The signal beats every output-confidence baseline on most datasets.** Probe
AUC − mean-logprob AUC: SimpleQA +0.107, TriviaQA +0.075, PopQA +0.165,
NQ-Open −0.045 (the only loss). Verbalized confidence is the *worst* baseline
in every dataset (0.49 NQ-Open, 0.60 SimpleQA, 0.69 PopQA, 0.78 TriviaQA) —
direct evidence that the model's stated confidence is poorly aligned with its
internal correctness representation.
- **The probe is not just a topic/difficulty detector.** On 39 SimpleQA questions
where the same prompt produced both correct and incorrect sampled answers
(n=312 generations, 113 correct), an `answer_mean` probe reaches
`AUC 0.989` discriminating correct vs incorrect *within the same question*,
while qid-disjoint AUC is only 0.72. The probe is doing real
within-question correctness discrimination, not just learning "this question
is hard". Plot: `results/plots/simpleqa_within_q/comparison_within_vs_qid.png`.
- **Transfer across datasets is partial.** Training on one dataset and testing on
another at a single shared layer (pos=`answer_last`, L=23) gives off-diagonal
AUCs in `[0.50, 0.75]` — well above chance and worth probing further, but the
signal has a meaningful dataset-specific component. SimpleQA-trained → PopQA
transfers best at 0.71. Plot: `results/plots/cross_dataset/transfer_probe_auc.png`.
- **Concrete dissociation examples.** 160 cases (top examples in
`results/experiments_summary.md`) where the model is confidently wrong by every
external signal (verbal_conf=85–100, mean_logprob > −0.5) and the internal
probe assigns `p(correct) < 0.05`. e.g. "Who won the Gerard P. Kuiper Prize in
2001?" → model: `Robert H. Brown`, gold: `Bruce W. Hapke`, verbal_conf=85,
probe p(correct)=0.00.

## Notes & caveats

- SimpleQA on a 3B model is genuinely difficult; even with the forced-answer prompt
the model answers most questions wrong. That actually makes it a clean source of
*confidently-wrong* examples for the probe.
- We avoid using the abstain class in any probe target so the probe doesn't just
learn to detect the model's "I know I don't know" pathway.
- All splits are at the qid level (no leakage from multiple sampled generations of
one question) except where Experiment 2 deliberately violates that to compare
with the qid-split baseline.
- Grading is rule-based; for short-form QA this matches the standard TriviaQA/PopQA
metrics. A small manual audit of the labelled outputs is recommended before
reading too much into single-dataset numbers. An LLM-judge grader can be plugged
into `src.grading` later.
- Experiments 5–10 from the proposal (activation patching, attention-head
localisation, SAEs, etc.) are intentionally not implemented yet — Experiments 1–4
are what this pass produces.

## Why the name

Soliloquy keeps the proposal's "inner voice" framing but pins it to the more
specific claim the experiments actually support: that the model is, in effect,
muttering one verdict (in the residual stream) while saying another (in its
generated answer and verbalized confidence). Like an aside in a play, the inner
verdict is reliable; the outer one is not.