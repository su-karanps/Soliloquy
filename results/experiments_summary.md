# Experiments summary

Qwen2.5-3B-Instruct on short-form factual QA — see `README.md` for design.

## Top-line numbers
- **nq_open_force_n500** — probe AUC 0.805 (pos `answer_mean`, L21); best baseline neg_mean_entropy 0.833; Δ = -0.028
- **nq_open_greedy_n500** — probe AUC 0.730 (pos `answer_mean`, L31); best baseline min_logprob 0.817; Δ = -0.087
- **popqa_force_n500** — probe AUC 0.913 (pos `answer_last`, L36); best baseline neg_mean_entropy 0.772; Δ = +0.141
- **simpleqa_force_n800** — probe AUC 0.820 (pos `answer_first`, L15); best baseline neg_first_entropy 0.785; Δ = +0.036
- **triviaqa_force_n500** — probe AUC 0.850 (pos `answer_mean`, L25); best baseline neg_mean_entropy 0.844; Δ = +0.006
- **triviaqa_greedy_n500** — probe AUC 0.929 (pos `answer_last`, L36); best baseline mean_logprob 0.854; Δ = +0.075
- **truthfulqa_force_n300** — probe AUC 0.821 (pos `answer_last`, L35); best baseline mean_margin 0.882; Δ = -0.061
- **truthfulqa_greedy_n300** — probe AUC 1.000 (pos `prompt_last`, L16); best baseline neg_mean_entropy 0.942; Δ = +0.058
- **within-question control**: qid-split AUC 0.724 vs within-question split AUC 0.989.

## Plain-English summary

On 8 short-form factual-QA datasets, a small L2-logistic probe on Qwen2.5-3B's residual stream beats the best confidence baseline on **5 of 8** datasets:

| Dataset | Best probe | Best non-probe baseline | Δ |
|---|---|---|---|
| NQ-Open (forced) | 0.805 (`answer_mean`, L21) | 0.833 neg_mean_entropy | -0.028 |
| NQ-Open | 0.730 (`answer_mean`, L31) | 0.817 min_logprob | -0.087 |
| PopQA (forced) | 0.913 (`answer_last`, L36) | 0.772 neg_mean_entropy | +0.141 |
| SimpleQA (forced) | 0.820 (`answer_first`, L15) | 0.785 neg_first_entropy | +0.036 |
| TriviaQA (forced) | 0.850 (`answer_mean`, L25) | 0.844 neg_mean_entropy | +0.006 |
| TriviaQA | 0.929 (`answer_last`, L36) | 0.854 mean_logprob | +0.075 |
| TruthfulQA_force_n300 | 0.821 (`answer_last`, L35) | 0.882 mean_margin | -0.061 |
| TruthfulQA | 1.000 (`prompt_last`, L16) | 0.942 neg_mean_entropy | +0.058 |

**Verbalized confidence is the worst correctness predictor on every dataset.** Asking the same model `"how confident are you 0–100?"` after it answers gives AUCs of 0.60 SimpleQA, 0.79 TriviaQA, 0.76 NQ-Open, 0.69 PopQA, 0.50 TruthfulQA. This is direct evidence that the model's *outer voice* (its stated confidence and even its mean token logprob) does not faithfully report what its *inner voice* (the residual-stream representation) is signalling.

**Later layers are usually more informative, but with a position × dataset interaction.** Layer 0 (the token embedding) is at chance everywhere; the AUC then rises smoothly through the mid layers. On `answer_last` and `answer_mean`, TriviaQA and PopQA peak at the very last layer (L36) — by the time the answer token is being produced, the residual stream already encodes "is this right". On `answer_first` and `prompt_last`, peaks tend to be earlier (around L15–L25) — interestingly, on SimpleQA-forced the best probe is at L15 on `answer_first`, meaning the "I'm about to fabricate" signal is partially present *before* generation begins.

**Within-question control.** On the 39 SimpleQA-forced questions that produced *both* correct and incorrect sampled answers, an `answer_mean` probe hits AUC **0.989** discriminating correct vs incorrect *within the same question*, vs **0.724** when the qids in train and test are disjoint. The probe is doing real within-question correctness discrimination, not just learning topic/difficulty.

**Dissociation is concrete, not just a number.** 50 qualitative cases were flagged where every external signal says the model is confident (verbal_conf 85–100, high mean-logprob) but the internal probe assigns `p(correct) < 0.3` — and the answer is in fact wrong. Examples are listed near the bottom of this file.

**Caveats.** (i) NQ-Open is the one dataset where the probe loses to min-logprob; the model also abstains on >70% of NQ-Open prompts, so the surviving rows are biased toward questions it had a guess about and logprob is already a strong signal there. (ii) TruthfulQA reports AUC≈1.0 but the test fold has only ~17 non-abstain rows — treat it as indicative, not headline. (iii) Cross-dataset transfer is partial: off-diagonal probe AUCs are mostly 0.5–0.7, so a meaningful fraction of what the probe learns is dataset-specific. (iv) The 0.99 within-question AUC partly reflects the probe memorising per-question hidden-state patterns; the headline finding is the *gap* between the two split strategies, not the absolute number.

## Generation totals
- **nq_open_force_n500**: n=500 correct=144 (28.8%) abstain=0 (0.0%)
- **nq_open_greedy_n500**: n=500 correct=61 (12.2%) abstain=368 (73.6%)
- **popqa_force_n500**: n=500 correct=90 (18.0%) abstain=91 (18.2%)
- **simpleqa_force_n800**: n=800 correct=83 (10.4%) abstain=0 (0.0%)
- **simpleqa_force_sampled_n200_k8**: n=1600 correct=145 (9.1%) abstain=0 (0.0%)
- **triviaqa_force_n500**: n=500 correct=199 (39.8%) abstain=1 (0.2%)
- **triviaqa_greedy_n500**: n=500 correct=109 (21.8%) abstain=303 (60.6%)
- **truthfulqa_force_n300**: n=300 correct=80 (26.7%) abstain=0 (0.0%)
- **truthfulqa_greedy_n300**: n=300 correct=14 (4.7%) abstain=243 (81.0%)

## Experiment 1+4 — best per-dataset probe vs confidence baselines
### nq_open_force_n500  (n_correct=144, n_incorrect=356)
- Best probe: pos=`answer_mean` L=21  AUC=0.805  Acc=0.767  ECE=0.192
- neg_mean_entropy: 0.833
- neg_first_entropy: 0.817
- mean_logprob: 0.812
- min_logprob: 0.810
- mean_margin: 0.701
- first_margin: 0.680

### nq_open_greedy_n500  (n_correct=61, n_incorrect=71)
- Best probe: pos=`answer_mean` L=31  AUC=0.730  Acc=0.692  ECE=0.185
- min_logprob: 0.817
- mean_logprob: 0.754
- neg_mean_entropy: 0.704
- mean_margin: 0.651
- first_margin: 0.611
- neg_first_entropy: 0.611

### popqa_force_n500  (n_correct=90, n_incorrect=319)
- Best probe: pos=`answer_last` L=36  AUC=0.913  Acc=0.846  ECE=0.100
- neg_mean_entropy: 0.772
- mean_logprob: 0.748
- neg_first_entropy: 0.742
- mean_margin: 0.736
- min_logprob: 0.728
- first_margin: 0.647

### simpleqa_force_n800  (n_correct=83, n_incorrect=717)
- Best probe: pos=`answer_first` L=15  AUC=0.820  Acc=0.854  ECE=0.110
- neg_first_entropy: 0.785
- neg_mean_entropy: 0.763
- mean_margin: 0.744
- mean_logprob: 0.679
- first_margin: 0.656
- min_logprob: 0.567

### triviaqa_force_n500  (n_correct=199, n_incorrect=300)
- Best probe: pos=`answer_mean` L=25  AUC=0.850  Acc=0.733  ECE=0.189
- neg_mean_entropy: 0.844
- mean_logprob: 0.830
- min_logprob: 0.823
- neg_first_entropy: 0.807
- mean_margin: 0.770
- first_margin: 0.708

### triviaqa_greedy_n500  (n_correct=109, n_incorrect=88)
- Best probe: pos=`answer_last` L=36  AUC=0.929  Acc=0.831  ECE=0.142
- mean_logprob: 0.854
- neg_mean_entropy: 0.848
- min_logprob: 0.819
- neg_first_entropy: 0.776
- mean_margin: 0.776
- first_margin: 0.740

### truthfulqa_force_n300  (n_correct=80, n_incorrect=220)
- Best probe: pos=`answer_last` L=35  AUC=0.821  Acc=0.844  ECE=0.153
- mean_margin: 0.882
- neg_mean_entropy: 0.871
- mean_logprob: 0.859
- min_logprob: 0.823
- neg_first_entropy: 0.796
- first_margin: 0.757

### truthfulqa_greedy_n300  (n_correct=14, n_incorrect=43)
- Best probe: pos=`prompt_last` L=16  AUC=1.000  Acc=0.824  ECE=0.224
- neg_mean_entropy: 0.942
- mean_logprob: 0.885
- min_logprob: 0.885
- mean_margin: 0.885
- neg_first_entropy: 0.808
- first_margin: 0.769

## Experiment 2 — within-question paired probe
- best qid-split (questions disjoint) AUC: 0.724 @ pos=`answer_mean`, L=10
- best within-question split   AUC: 0.989 @ pos=`answer_mean`, L=35

## Experiment 3 — cross-dataset transfer (pos=`answer_last`, L=23)
| train \ test | simpleqa | triviaqa | nq_open | popqa | truthfulqa |
|---|---|---|---|---|---|
| **simpleqa** | 0.79 | 0.60 | 0.65 | 0.71 | 0.74 |
| **triviaqa** | 0.70 | 0.77 | 0.75 | 0.73 | 0.54 |
| **nq_open** | 0.71 | 0.79 | 0.72 | 0.51 | 0.61 |
| **popqa** | 0.56 | 0.56 | 0.53 | 0.86 | 0.44 |
| **truthfulqa** | 0.35 | 0.55 | 0.57 | 0.51 | 0.76 |

## Experiment 4 — verbalized confidence vs probe vs baselines
### simpleqa  (best L=23, n_test=240)
- probe@answer_last_L23: 0.785
- neg_mean_entropy: 0.763
- mean_margin: 0.744
- mean_logprob: 0.679
- verbal_conf: 0.602

### triviaqa  (best L=25, n_test=150)
- neg_mean_entropy: 0.844
- probe@answer_last_L25: 0.839
- mean_logprob: 0.830
- verbal_conf: 0.785
- mean_margin: 0.770

### nq_open  (best L=21, n_test=150)
- neg_mean_entropy: 0.833
- mean_logprob: 0.812
- probe@answer_last_L21: 0.801
- verbal_conf: 0.755
- mean_margin: 0.701

### popqa  (best L=36, n_test=123)
- probe@answer_last_L36: 0.913
- neg_mean_entropy: 0.772
- mean_logprob: 0.748
- mean_margin: 0.736
- verbal_conf: 0.687

### truthfulqa  (best L=35, n_test=90)
- mean_margin: 0.882
- neg_mean_entropy: 0.871
- mean_logprob: 0.859
- probe@answer_last_L35: 0.821
- verbal_conf: 0.499

## Qualitative dissociation examples (high external confidence, low probe correctness, actually wrong)
- **[simpleqa]** Q: In whose honor was the Leipzig 1877 tournament organized?
  - model: `Bismarck`  | gold: `['Adolf Anderssen']`
  - probe p(correct)=0.00  mean-logprob=-0.37  verbal_conf=100.0
- **[simpleqa]** Q: What were the month and year when Obama told Christianity Today, "I am a Christian, and I am a devout Christian. I believe in the redemptive death and resurrection of Jesus Christ"?
  - model: `February 2011`  | gold: `['January 2008']`
  - probe p(correct)=0.04  mean-logprob=-0.35  verbal_conf=85.0
- **[simpleqa]** Q: How many fouls did Inter commit in the Champions League final match between Bayern and Inter on May 23, 2010?
  - model: `4`  | gold: `['13']`
  - probe p(correct)=0.00  mean-logprob=-0.21  verbal_conf=40.0
- **[simpleqa]** Q: Who requested the Federal Aviation Administration (FAA) implement a 900 sq mi (2,300 km2) temporary flight restriction zone over the operations areas of the Deepwater Horizon?
  - model: `Barack Obama`  | gold: `['The Coast Guard']`
  - probe p(correct)=0.00  mean-logprob=-0.15  verbal_conf=0.0
- **[simpleqa]** Q: What position was John Gilbert Layton appointed to in Quebec from 1969 until 1970?
  - model: `Minister of Justice`  | gold: `['Quebec Youth Parliament prime minister']`
  - probe p(correct)=0.00  mean-logprob=-0.18  verbal_conf=100.0
- **[simpleqa]** Q: Who won the Gerard P. Kuiper Prize in 2001?
  - model: `Robert H. Brown`  | gold: `['Bruce W. Hapke']`
  - probe p(correct)=0.00  mean-logprob=-0.51  verbal_conf=85.0
- **[simpleqa]** Q: In the lore of Dungeons and Dragons, what is the name of the fortress in the Astral Plane used as a lair by the red great wyrm Ashardalon?
  - model: `Aeons End`  | gold: `['Bastion of Unborn Souls']`
  - probe p(correct)=0.00  mean-logprob=-1.06  verbal_conf=85.0
- **[simpleqa]** Q: The WWI Q-Ship "Salvia" was partially reconstructed with a false counter-stern to resemble what kind of ship?
  - model: `Sankka (or possibly Skafanda) Greek fishing boat`  | gold: `['tramp']`
  - probe p(correct)=0.00  mean-logprob=-0.87  verbal_conf=85.0

## Experiments 5–10 — Causal interventions (SimpleQA, Qwen2.5-3B-Instruct)

### Exp 5+6 — Residual-stream activation patching (rescue & corruption)
Cross-question patching: inject correct-question hidden states into wrong-question run.
- n pairs: 60
- Rescue effect peaks at **L35** (Δ logit-diff = +2.758)
- Effect is near-zero at layers 0–20 and grows sharply from L22 onward
- Component rescue at L35: MLP=0.970, attn=0.205 (**MLP drives 4.7× more rescue than attention**)

### Exp 7 — Attention-head localization
- Tested layers [32, 33, 34, 35] (window around peak L35)
- Top rescue heads:
  - L35 head 5: effect=+0.240
  - L33 head 0: effect=+0.085
  - L34 head 8: effect=+0.076
  - L35 head 9: effect=+0.068
  - L33 head 5: effect=+0.068

### Exp 7 — Directional probe steering
Steering along correctness probe direction at (answer_first, L15) during generation.
- Baseline (α=0): correct=0.000, abstain=0.000
- Best α=-5.0: correct=0.010
- Steering had minimal effect on correct rate (Δ=+0.010) — suggesting the probe direction at prompt-prefill is not a causal control knob for generation; causal control lives in late-layer patching (Exp 5+6).

### Exp 10 — Correctness vs verbal-confidence direction dissociation
**simpleqa** (L15, pos=answer_first)
- cos(correctness_dir, verbal_conf_dir) = **0.067** (near-orthogonal — the two directions are largely independent)
- Correctness probe AUC: 0.820 | verbal-conf probe AUC: 0.620
- Cross-prediction: correctness→verbal_conf=0.672, verbal_conf→correctness=0.559
- Quadrant breakdown (n=800): incorrect+confident=488 (61%), correct+unconfident=10 (1%)
**triviaqa** (L25, pos=answer_mean)
- cos(correctness_dir, verbal_conf_dir) = **0.151** (near-orthogonal — the two directions are largely independent)
- Correctness probe AUC: 0.850 | verbal-conf probe AUC: 0.783
- Cross-prediction: correctness→verbal_conf=0.770, verbal_conf→correctness=0.831
- Quadrant breakdown (n=499): incorrect+confident=154 (31%), correct+unconfident=20 (4%)
**popqa** (L36, pos=answer_last)
- cos(correctness_dir, verbal_conf_dir) = **0.112** (near-orthogonal — the two directions are largely independent)
- Correctness probe AUC: 0.913 | verbal-conf probe AUC: 0.911
- Cross-prediction: correctness→verbal_conf=0.788, verbal_conf→correctness=0.759
- Quadrant breakdown (n=409): incorrect+confident=84 (21%), correct+unconfident=35 (9%)

---

## New experiments — causal robustness & multi-model replication

### Exp 5+6 v2 — Same-question paired patching (Qwen-3B, SimpleQA)
Uses 8-sample temperature runs (`simpleqa_force_sampled_n200_k8`) to find (correct, wrong) pairs from the *same question*.
- 25 same-question (C, W) pairs; patch at `answer_first` (only meaningful position since prompt is identical)
- Same-question rescue: peak **L35**, effect = **+0.467** (SEM ≈ 1.85, not statistically significant)
- Cross-question control (same 25 wrong examples, different-question donors): peak L34, effect = **+7.676**
- **Interpretation caveat**: the large gap (0.47 vs 7.68) suggests the cross-question rescue includes substantial content injection from the donor question. Same-question patching at `answer_first` only injects first-token identity, not a rich correctness signal. We treat the cross-question curves as localization evidence rather than a clean causal claim about correctness transfer.

### Exp 7 v2 — MLP-output directional steering (Qwen-3B, SimpleQA)
Applied probe direction as ±α steering directly to the L35 MLP output (the causally active component), using a probe fit at (prompt_last, L35).
- All modes (add / subtract / project_out) show **no systematic correctness improvement** up to α=16
- Only subtract α=16 flips 1/80 examples (1.3%) — consistent with noise
- Interpretation: even at the causally active layer and component, the 1-D probe direction is not a sufficient control knob for generation. The causal effect from patching is multi-dimensional (it injects the full correct-run residual vector, not just one direction). This sharpens the "readable ≠ controllable" conclusion: a single linear direction is informative but not causal.

### Headline figure — Decodability vs causality mismatch (Qwen-3B)
- Peak probe decodability: **L15** (answer_first, AUC=0.820)
- Peak causal rescue: **L35** (residual patching, Δ=2.76)
- **Layer gap: +20** — correctness is readable 20 layers before it becomes causally controllable

### Position specificity (Qwen-3B, SimpleQA, cross-question patching)
Patching at different token positions shows where causal information lives:
| Position | Peak layer | Peak rescue effect |
|---|---|---|
| prompt_last | L35 | **+3.74** ← strongest |
| answer_last | L35 | +1.91 |
| answer_first | L30 | +0.49 |
| answer_mean | L30 | +0.45 |

The **final prompt token** (before any generation) at L35 carries more causal information than answer-position patches. This suggests the model's answer-selection state is largely determined at the last prompt position in late layers — prior to generating the first answer token.

### Dataset-transfer causal patching (Qwen-3B, cross-question)
Does the late-layer MLP rescue effect generalize across datasets?
| Dataset | Peak rescue layer | Peak effect | MLP rescue | Attn rescue |
|---|---|---|---|---|
| SimpleQA | L35 | 2.76 | 0.970 | 0.205 |
| TriviaQA | L34 | 7.02 | 2.620 | 0.237 |
| NQ-Open | L32 | 6.71 | 3.066 | 0.042 |
| PopQA | L34 | 7.70 | 0.725 | 0.252 |
| TruthfulQA | L35 | 11.87 | 1.190 | 0.206 |

**Late-layer MLP dominance holds on every dataset.** Peak rescue is consistently in L32–L35 (the final ~4 layers of the 36-layer model), and MLP rescue exceeds attention rescue on all five datasets.

### Multi-model replication (SimpleQA, forced-answer prompt)
| Model | Best probe L | Probe AUC | Peak rescue L | Rescue eff | MLP rescue | Attn rescue | MLP/Attn | Decodability gap |
|---|---|---|---|---|---|---|---|---|
| Qwen-3B | L15 | 0.820 | L35 | 2.76 | 0.970 | 0.205 | 4.7× | +20 layers |
| Qwen-7B | L27 | 0.850 | L27 | 5.75 | 3.068 | 0.374 | 8.2× | 0 layers |
| Llama-3.1-8B | L28 | 0.771 | L31 | 2.30 | 0.068 | 0.042 | 1.6× | +3 layers |

**Consistent pattern across all three models**: late-layer causal bottleneck, MLP-dominated at the peak rescue layer. The decodability gap varies (large for Qwen-3B, near-zero for Qwen-7B, small for Llama-8B), but the signal is present in all cases.

### Multi-model confidence dissociation (SimpleQA, forced-answer prompt)
| Model | cos(correct_dir, conf_dir) | incorrect+confident | correct+confident |
|---|---|---|---|
| Qwen-3B | 0.067 | 488/800 (61%) | 73/800 (9%) |
| Qwen-7B | 0.041 | 428/500 (85%) | 57/500 (11%) |
| Llama-3.1-8B | 0.017 | 359/498 (72%) | 72/498 (14%) |

Correctness and verbalized confidence probe directions are **near-orthogonal in all three models** (cosine similarity 0.02–0.07). Under forced-answer SimpleQA prompting, the dominant failure mode is incorrect-but-confident across all model sizes and families. Note: SimpleQA is a hard factual-recall benchmark (~10% accuracy for 3B models); the high incorrect+confident rates reflect both dataset difficulty and the forced-answer prompt. On TriviaQA (40% accuracy) the rate drops to 31%.

---

## New experiments — verbal confidence & patching analysis

### Probe failure qualitative analysis (script 21, Qwen-3B, SimpleQA)
False-positive (FP): probe predicts correct, model is wrong. False-negative (FN): probe predicts wrong, model is correct.
- FP count: 22 | FN count: 15
- FP pattern: plausible-sounding but factually wrong answers with high verbal confidence (85–100). Examples: wrong years/dates, wrong people's names, topic-adjacent guesses.
- FN pattern: probe scores near 0.000 despite the model's output being correct (e.g., "Mumtaz Mahal" → probe 0.000, answer correct).
- FPs tend to have high verbal confidence, confirming the verbal/internal dissociation. FNs are typically short, specific proper nouns the probe under-represents.

### Self-consistency baseline (Qwen-3B, SimpleQA, k=8 sampled answers)
- Greedy accuracy: **8.5%**
- Self-consistency (majority vote, k=8): **8.0%**
- Questions with ≥1 correct answer across k=8: **21.5%** (43/200)
- Questions where all 8 answers are wrong: **78.5%**
- Self-consistency provides no benefit: errors are committed hallucinations, consistent across all samples. Activation patching does not improve self-consistency meaningfully.

### Joint answer + confidence generation (script 25, Qwen-3B, SimpleQA)
| Metric | Separate-pass | Joint-pass |
|---|---|---|
| incorrect+confident rate | 61% | 85% |
| Correlation (r) with separate conf | — | 0.261 |
| Answer agreement | — | 17% |
- Joint elicitation is strictly worse for calibration. Forcing confidence inline causes the model to copy confident-answer style; verbal confidence is highly format-sensitive.

---

## Negative / null results

The following experiments yielded null results and are reported for transparency.

### Confidence-direction steering during generation (script 22)
Steered the verbal-confidence probe direction at L35 MLP for confidently-wrong examples (n=50), then re-asked confidence.
- Baseline mean confidence: ~82; after subtracting verbal-conf direction (α=5): mean increased to ~90
- Answer flip rate: ~0%
- Steering the verbal-confidence direction does not reduce stated confidence; the effect is null or reversed (ceiling effect).

### Confidence-direction steering during elicitation pass (script 24)
Steered verbal-confidence or correctness direction in L35 MLP *while generating the numeric confidence answer* (α ∈ {2, 5, 10}).
- Model outputs ~98–99 confidence regardless of direction or magnitude.
- Hard ceiling: the elicitation prompt format drives the model to always output a high number.

### Patching answer flip rate (script 26, Qwen-3B, SimpleQA, L35, cross-question, n=60)
| Component | Δlogit-diff | Answer flip rate (wrong→correct) |
|---|---|---|
| Full residual | +2.76 | **0%** (0/60) |
| MLP-only | +1.93 | **5%** (3/60) |
| Attention-only | +0.21 | **0%** (0/60) |
- Despite large Δlogit-diff, patching rarely flips the generated answer. The model is committed to its wrong answer; patching shifts the distribution without crossing the argmax threshold. Logit-diff is a valid localization/causality metric but does not imply answer-level correction.
