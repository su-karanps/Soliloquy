# Probe Analysis README

This note explains, in simple terms, how the correctness probes in this repo work
and what the current probe results say.

## Big Picture

This repo studies whether an LLM has an internal signal for whether its own answer
is right or wrong.

The experiment flow is:

1. Ask `Qwen/Qwen2.5-3B-Instruct` factual questions.
2. Save the model's answer.
3. Save hidden-state vectors from the model while it answers.
4. Grade the answer as correct or incorrect.
5. Train small classifiers, called probes, to predict correctness from the hidden
   states.
6. Compare those probes against ordinary confidence signals like log probability,
   entropy, margin, and verbalized confidence.

The key idea is that the probe is not another LLM. It is only a linear logistic
regression classifier. If it performs well, that suggests correctness information
is already present in the model's hidden states in a fairly readable form.

## How The Probe Works

Each generated answer becomes one supervised example:

```text
X = hidden-state vector from the model
y = 1 if the answer was correct, 0 if the answer was wrong
```

For Qwen2.5-3B, each hidden vector has roughly 2048 numbers.

The probe learns a weighted sum:

```text
score = w1*x1 + w2*x2 + ... + w2048*x2048 + bias
```

Then it converts that score into:

```text
p(correct)
```

The implementation is in `src/probes.py`. The training function:

- standardizes hidden vectors with `StandardScaler`
- trains `LogisticRegression`
- reports AUC, accuracy, F1, log loss, and expected calibration error

The main layer/position sweep is in `scripts/03_probe_layers.py`.

## Why These Four Positions Are Tested

The repo saves four hidden-state positions for each answer:

```text
prompt_last   = state at the last prompt token
answer_first  = state at the first answer token
answer_last   = state at the last answer token
answer_mean   = average state across all answer tokens
```

Each position tests a different hypothesis.

`prompt_last` asks whether the model already has a correctness or familiarity
signal after reading the question, before producing the answer.

`answer_first` asks whether the signal appears as soon as the model starts
answering.

`answer_last` asks whether the signal is clearest after the full answer has been
generated.

`answer_mean` asks whether correctness is spread across the whole answer rather
than concentrated at one token.

## How Many Probes Are Trained

The code does not choose one layer ahead of time.

It trains one probe for every combination of:

```text
4 positions x 37 layers = 148 probes
```

Layer `0` is the embedding layer. Layers `1..36` are the transformer block outputs.

For each dataset, the file:

```text
results/tables/<dataset>/layer_probes.csv
```

contains all 148 probe results.

Each row is:

```text
position, layer, auc, acc, f1, logloss, ece
```

The script then picks the best probe by AUC.

Current best probes:

```text
SimpleQA:   answer_first, layer 15, AUC 0.820
TriviaQA:   answer_last,  layer 36, AUC 0.929
NQ-Open:    answer_mean,  layer 31, AUC 0.730
PopQA:      answer_last,  layer 36, AUC 0.913
TruthfulQA: prompt_last,  layer 16, AUC 1.000
```

## Dataset Sizes And Evaluation Counts

The main single-answer runs use:

```text
SimpleQA:     800 questions
TriviaQA:     500 questions
NQ-Open:      500 questions
PopQA:        500 questions
TruthfulQA:   300 questions
```

There is also a sampled SimpleQA control run:

```text
SimpleQA sampled: 200 questions x 8 generations = 1600 generated answers
```

The probe evaluation does not always use the full original dataset size. The
analysis drops abstentions by default, so each dataset has a smaller usable set:

```text
SimpleQA:     800 usable examples -> 240 test examples
TriviaQA:     197 usable examples -> 59 test examples
NQ-Open:      132 usable examples -> 39 test examples
PopQA:        409 usable examples -> 123 test examples
TruthfulQA:    57 usable examples -> 17 test examples
```

TruthfulQA is especially small because the model abstained on most examples:

```text
300 total generations
- 243 abstentions
= 57 non-abstain usable examples
```

That means the TruthfulQA AUC of `1.000` should be treated cautiously because it
is evaluated on only about 17 held-out examples.

## Prompt Conditions

Two prompt conditions were used.

The `default` prompt allows abstention:

```text
If you do not know, say "I don't know."
```

The `force` prompt tells the model not to abstain:

```text
You must give your best guess; do NOT say you don't know.
```

The datasets are split by prompt condition:

```text
Force prompt:
  SimpleQA
  PopQA

Default / abstain-allowed prompt:
  TriviaQA
  NQ-Open
  TruthfulQA
```

This matters. Aggregating all datasets together mixes two different experimental
settings. Aggregate position/layer analysis should therefore be reported both
overall and split by prompt condition.

## Prediction Success By Dataset

The repo reports probe performance per dataset, not as one combined pooled
accuracy.

Main best-probe AUCs:

```text
SimpleQA:   0.820
TriviaQA:   0.929
NQ-Open:    0.730
PopQA:      0.913
TruthfulQA: 1.000
```

There is also a cross-dataset transfer experiment, but that reports a train/test
dataset matrix rather than one global average.

## Correlation Analysis: Position

Using AUC as prediction success, position was analyzed as a categorical factor.
The ranking below uses mean AUC.

Overall average across all datasets:

```text
answer_mean   avg AUC 0.770
answer_last   avg AUC 0.762
answer_first  avg AUC 0.727
prompt_last   avg AUC 0.684
```

By dataset:

```text
nq_open:
  answer_last   0.608
  answer_mean   0.585
  prompt_last   0.573
  answer_first  0.551

popqa:
  answer_mean   0.872
  answer_last   0.870
  answer_first  0.861
  prompt_last   0.642

simpleqa:
  answer_mean   0.743
  answer_first  0.741
  answer_last   0.726
  prompt_last   0.668

triviaqa:
  answer_last   0.782
  answer_mean   0.763
  prompt_last   0.705
  answer_first  0.685

truthfulqa:
  answer_mean   0.884
  prompt_last   0.833
  answer_last   0.826
  answer_first  0.798
```

Main takeaway: `answer_mean` and `answer_last` are the most consistently useful
positions. `prompt_last` is usually weaker, which suggests the best correctness
signal often appears during or after answer generation rather than purely before
the model starts answering.

## Correlation Analysis: Layer

Layer was analyzed by correlating layer index with probe AUC.

Overall across all datasets:

```text
Spearman correlation: +0.810
Pearson correlation:  +0.836
```

This means later layers are strongly associated with better probe performance.

Top average layers across all datasets:

```text
L30  avg AUC 0.799
L29  avg AUC 0.799
L28  avg AUC 0.797
L26  avg AUC 0.786
L23  avg AUC 0.784
L31  avg AUC 0.784
L25  avg AUC 0.784
L24  avg AUC 0.780
L27  avg AUC 0.779
L22  avg AUC 0.776
```

Layer correlation by dataset:

```text
TriviaQA:   Spearman +0.684, Pearson +0.670
TruthfulQA: Spearman +0.522, Pearson +0.581
NQ-Open:    Spearman +0.280, Pearson +0.312
PopQA:      Spearman +0.179, Pearson +0.232
SimpleQA:   Spearman +0.130, Pearson +0.198
```

Main takeaway: the layer effect is clearest in the aggregate. Within individual
datasets, the effect varies, but the best layers are usually mid-to-late or late.

## Prompt-Condition Split Analysis

Because the prompt conditions are different, the aggregate analysis should be
split into forced-answer and abstain-allowed settings.

### Force Prompt: SimpleQA + PopQA

Position ranking:

```text
answer_mean   avg AUC 0.808
answer_first  avg AUC 0.801
answer_last   avg AUC 0.798
prompt_last   avg AUC 0.655
```

Layer correlation with AUC:

```text
Spearman +0.776
Pearson  +0.764
```

Top layers:

```text
L29 0.807
L30 0.803
L28 0.797
L31 0.796
L21 0.788
L34 0.785
L35 0.782
L26 0.781
L23 0.781
L25 0.781
```

### Default / Abstain-Allowed Prompt: TriviaQA + NQ-Open + TruthfulQA

Position ranking:

```text
answer_mean   avg AUC 0.744
answer_last   avg AUC 0.739
prompt_last   avg AUC 0.704
answer_first  avg AUC 0.678
```

Layer correlation with AUC:

```text
Spearman +0.794
Pearson  +0.813
```

Top layers:

```text
L28 0.798
L30 0.796
L29 0.794
L27 0.792
L26 0.789
L23 0.787
L25 0.787
L24 0.781
L22 0.781
L31 0.776
```

The split does not change the broad conclusion. In both prompt regimes,
`answer_mean` and `answer_last` are strong, and later layers correlate with
better probe performance. The biggest difference is that `answer_first` is much
stronger in the forced-answer setting than in the abstain-allowed setting.

## Practical Commands

View all 148 probes for one dataset:

```bash
column -s, -t results/tables/simpleqa_force_n800/layer_probes.csv | less -S
```

Sort one dataset's probes by AUC:

```bash
tail -n +2 results/tables/simpleqa_force_n800/layer_probes.csv \
  | sort -t, -k3,3gr \
  | column -s, -t \
  | less -S
```

Count probe rows:

```bash
wc -l results/tables/*/layer_probes.csv
```

Each file should have `149` lines: one header plus `148` probe rows.

## Bottom Line

The probe results suggest that the model's hidden states contain useful
information about whether its answer is correct. This signal is usually strongest
in answer-token representations, especially `answer_mean` and `answer_last`, and
it tends to become more readable in mid-to-late layers.

The most important caveat is that some datasets used a forced-answer prompt while
others allowed abstention. Any aggregate analysis should therefore be split by
prompt condition, as done above.
