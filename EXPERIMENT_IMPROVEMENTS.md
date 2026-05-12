# Experiment Improvements And Next Steps

This document lists practical improvements for the current probing experiments.
The goal is to make the results easier to interpret, more comparable across
datasets, and more convincing as evidence about internal correctness signals.

## 1. Separate Prompt Conditions In All Aggregate Analyses

The current runs use two different prompt regimes:

```text
Force prompt:
  SimpleQA
  PopQA

Default / abstain-allowed prompt:
  TriviaQA
  NQ-Open
  TruthfulQA
```

This makes one pooled aggregate across all datasets hard to interpret. Forced
answers and abstain-allowed answers are different behavioral settings.

Next steps:

- Report all aggregate position/layer analyses separately by prompt condition.
- Add prompt-condition labels to summary JSON/CSV outputs.
- Optionally rerun every dataset under both prompt conditions.

Best version:

```text
Each dataset x each prompt condition
```

That would let the analysis separate dataset effects from prompt effects.

## 2. Rerun All Datasets Under Both Prompt Styles

Right now, SimpleQA and PopQA use the forced-answer prompt, while TriviaQA,
NQ-Open, and TruthfulQA use the default abstain-allowed prompt.

This creates a confound:

```text
dataset identity is partially entangled with prompt condition
```

Next steps:

- Run SimpleQA with default prompt.
- Run PopQA with default prompt.
- Run TriviaQA with forced prompt.
- Run NQ-Open with forced prompt.
- Run TruthfulQA with forced prompt.

Then compare:

```text
same dataset, different prompt
same prompt, different dataset
```

This would make it much clearer whether probe behavior is driven by the dataset,
the prompt condition, or both.

## 3. Increase Usable Examples For Abstain-Heavy Datasets

TruthfulQA, NQ-Open, and TriviaQA lose many examples because abstentions are
dropped before probe training.

Current usable counts:

```text
TruthfulQA:  57 usable examples
NQ-Open:    132 usable examples
TriviaQA:   197 usable examples
```

TruthfulQA is especially fragile because the best probe is evaluated on only
about 17 held-out examples.

Next steps:

- Increase dataset size where possible.
- Use forced-answer runs to collect more non-abstain examples.
- Report confidence intervals or bootstrap intervals for AUC.
- Avoid over-interpreting very small held-out test sets.

## 4. Add Confidence Intervals

The current reports give point estimates, such as:

```text
AUC = 0.913
```

But small datasets can make point estimates unstable.

Next steps:

- Bootstrap the test set to estimate 95% confidence intervals.
- Report uncertainty for AUC, accuracy, F1, log loss, and ECE.
- Use wider caution notes for datasets with small test sets.

Example desired output:

```text
PopQA answer_last L36: AUC 0.913 [0.861, 0.952]
TruthfulQA prompt_last L16: AUC 1.000 [wide interval due to n=17]
```

## 5. Use Nested Model Selection

The current analysis trains probes for every position/layer and picks the best
one by test AUC.

That is useful for exploration, but it can overstate performance because the test
set is used both to choose the best probe and to report its final score.

Next steps:

- Split into train/validation/test.
- Use validation to choose position and layer.
- Use test only once for final reporting.

Better evaluation structure:

```text
train: fit probe weights
validation: choose position/layer
test: final unbiased metric
```

This is especially important when scanning 148 probes per dataset.

## 6. Compare Position Metrics More Carefully

Position comparisons can be summarized in several ways:

```text
mean AUC across layers
median AUC across layers
best-layer AUC
top-k-layer average
area under the layer curve
```

Each answers a different question.

Next steps:

- Report multiple summaries for each position.
- Use median AUC to reduce sensitivity to outlier layers.
- Use top-5 layer average to measure how strong a position is in its best region.
- Avoid relying only on the single best layer.

## 7. Control For Answer Length

`answer_mean` averages over all generated answer tokens. If correct and incorrect
answers differ in length, answer length may become a confound.

Possible issue:

```text
answer_mean may partly reflect answer length/style, not only correctness
```

Next steps:

- Report answer-length distributions for correct vs incorrect answers.
- Add answer length as a baseline predictor.
- Add a combined model: answer length + logprob + entropy.
- Evaluate whether the probe still helps after controlling for answer length.

## 8. Control For Answer Type

Correctness may correlate with answer type:

```text
person names
dates
locations
numbers
organizations
titles
```

If one answer type is easier than another, probes may partially learn answer
category or format.

Next steps:

- Automatically tag answer types.
- Report probe performance within each answer type.
- Compare correct vs incorrect examples inside the same answer-type bucket.

## 9. Strengthen Within-Question Controls

The SimpleQA sampled run already tests whether probes can distinguish correct and
incorrect answers to the same question. This is one of the strongest current
controls.

Next steps:

- Run within-question sampled experiments on more datasets.
- Increase samples per question.
- Ensure each retained question has both correct and incorrect non-abstain
  generations.
- Report within-question results split by prompt condition.

This helps answer:

```text
Is the probe detecting correctness, or just question difficulty/topic?
```

## 10. Add Harder Baselines

The current baselines include:

```text
mean logprob
min logprob
entropy
margin
self-consistency
verbalized confidence
```

Useful additional baselines:

- answer length
- prompt length
- answer type
- dataset/question topic
- final-token logprob
- first-token logprob
- normalized logprob per character
- embedding-based answer similarity to training answers
- combined confidence baseline using logistic regression over metadata features

The combined metadata baseline is especially important. It asks whether hidden
states still help after combining all easy external signals.

## 11. Evaluate Calibration More Directly

AUC measures ranking quality, not whether probabilities are trustworthy.

The repo already computes ECE, but calibration deserves more focused reporting.

Next steps:

- Plot reliability curves for every dataset.
- Report Brier score.
- Compare calibrated vs uncalibrated probe probabilities.
- Try Platt scaling or isotonic regression on validation data.

This matters if `p(correct)` will be interpreted as an actual probability.

## 12. Test Earlier Causal Interpretations Carefully

`answer_mean` and `answer_last` are strong, but they are measured after or during
answer generation. They show that the completed answer representation contains
correctness-related information.

They do not prove the model knew the answer was wrong before saying it.

Next steps:

- Emphasize `prompt_last` and `answer_first` when asking pre-answer questions.
- Compare early-token probes against later-token probes.
- Test whether early probes can predict correctness before the answer content is
  fully available.

Clean interpretation:

```text
prompt_last: possible pre-answer knowledge/familiarity signal
answer_first: early commitment signal
answer_last/answer_mean: completed-answer correctness signal
```

## 13. Add Causal Interventions

Current probes are correlational. They show that correctness is readable from
hidden states, but not that the probed direction causes the model's behavior.

Next steps:

- Activation patching.
- Directional steering along probe directions.
- Ablating or adding the correctness direction.
- Layer-specific interventions.
- Test whether interventions change answer correctness, abstention, or confidence.

Important caution:

```text
Probe accuracy alone does not prove causal control.
```

## 14. Test Cross-Model Generalization

Current results use `Qwen/Qwen2.5-3B-Instruct`.

Next steps:

- Repeat on `Qwen/Qwen2.5-7B-Instruct`.
- Test another model family.
- Compare layer locations after normalizing by relative depth.
- Check whether `answer_mean`/late-layer dominance is model-specific.

This would show whether the signal is a general LLM phenomenon or specific to one
model and setup.

## 15. Improve Grading Reliability

The current grader is rule-based. That is reasonable for short-form QA, but it
can make mistakes.

Next steps:

- Manually audit a random sample of correct/incorrect labels.
- Add an LLM judge for ambiguous cases.
- Track which match rule fired: exact, substring, F1, fuzzy.
- Report results separately for exact-match labels only as a stricter subset.

This helps make sure the probe is learning answer correctness rather than quirks
of the automatic grader.

## 16. Save Analysis Scripts For Reproducibility

Some aggregate analyses were run interactively from the existing CSV files.

Next steps:

- Add a script such as `scripts/10_position_layer_analysis.py`.
- Write outputs to `results/tables/position_layer_analysis/`.
- Include prompt-condition split summaries by default.
- Make the script regenerate all numbers in `PROBE_ANALYSIS_README.md`.

This would prevent the analysis from living only in notes.

## Suggested Priority Order

Highest priority:

1. Split aggregate analysis by prompt condition.
2. Add train/validation/test selection for the best probe.
3. Add confidence intervals.
4. Rerun all datasets under both prompt styles.
5. Add answer length and combined metadata baselines.

Medium priority:

1. Add within-question sampled controls for more datasets.
2. Improve calibration reporting.
3. Audit grading quality.
4. Add answer-type controls.

Longer-term:

1. Run causal interventions.
2. Test larger and different model families.
3. Analyze whether probe directions transfer across models.

## Bottom Line

The current experiments are a good first pass: they show that correctness is
readable from hidden states, especially in answer-token representations and
mid-to-late layers.

The next step is to make the result harder to explain away. The most important
improvements are controlling for prompt condition, avoiding best-of-148 test-set
selection bias, adding uncertainty intervals, and testing whether the signal
survives stronger metadata and within-question controls.
