# Experiments summary

Qwen2.5-3B-Instruct on short-form factual QA — see `README.md` for design.

## Top-line numbers
- **nq_open_greedy_n500** — probe AUC 0.730 (pos `answer_mean`, L31); best baseline min_logprob 0.817; Δ = -0.087
- **popqa_force_n500** — probe AUC 0.913 (pos `answer_last`, L36); best baseline neg_mean_entropy 0.772; Δ = +0.141
- **simpleqa_force_n800** — probe AUC 0.820 (pos `answer_first`, L15); best baseline neg_first_entropy 0.785; Δ = +0.036
- **triviaqa_greedy_n500** — probe AUC 0.929 (pos `answer_last`, L36); best baseline mean_logprob 0.854; Δ = +0.075
- **truthfulqa_greedy_n300** — probe AUC 1.000 (pos `prompt_last`, L16); best baseline neg_mean_entropy 0.942; Δ = +0.058
- **within-question control**: qid-split AUC 0.724 vs within-question split AUC 0.989.

## Generation totals
- **nq_open_greedy_n500**: n=500 correct=61 (12.2%) abstain=368 (73.6%)
- **popqa_force_n500**: n=500 correct=90 (18.0%) abstain=91 (18.2%)
- **simpleqa_force_n800**: n=800 correct=83 (10.4%) abstain=0 (0.0%)
- **simpleqa_force_sampled_n200_k8**: n=1600 correct=145 (9.1%) abstain=0 (0.0%)
- **triviaqa_greedy_n500**: n=500 correct=109 (21.8%) abstain=303 (60.6%)
- **truthfulqa_greedy_n300**: n=300 correct=14 (4.7%) abstain=243 (81.0%)

## Experiment 1+4 — best per-dataset probe vs confidence baselines
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

### triviaqa_greedy_n500  (n_correct=109, n_incorrect=88)
- Best probe: pos=`answer_last` L=36  AUC=0.929  Acc=0.831  ECE=0.142
- mean_logprob: 0.854
- neg_mean_entropy: 0.848
- min_logprob: 0.819
- neg_first_entropy: 0.776
- mean_margin: 0.776
- first_margin: 0.740

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
| **simpleqa** | 0.79 | 0.54 | 0.64 | 0.71 | 0.71 |
| **triviaqa** | 0.71 | 0.88 | 0.63 | 0.64 | 0.68 |
| **nq_open** | 0.69 | 0.75 | 0.68 | 0.60 | 0.64 |
| **popqa** | 0.56 | 0.54 | 0.56 | 0.86 | 0.50 |
| **truthfulqa** | 0.56 | 0.69 | 0.59 | 0.62 | 0.96 |

## Experiment 4 — verbalized confidence vs probe vs baselines
### simpleqa  (best L=23, n_test=240)
- probe@answer_last_L23: 0.785
- neg_mean_entropy: 0.763
- mean_margin: 0.744
- mean_logprob: 0.679
- verbal_conf: 0.602

### triviaqa  (best L=36, n_test=59)
- probe@answer_last_L36: 0.929
- mean_logprob: 0.854
- neg_mean_entropy: 0.848
- verbal_conf: 0.779
- mean_margin: 0.776

### nq_open  (best L=24, n_test=39)
- mean_logprob: 0.754
- probe@answer_last_L24: 0.709
- neg_mean_entropy: 0.704
- mean_margin: 0.651
- verbal_conf: 0.492

### popqa  (best L=36, n_test=123)
- probe@answer_last_L36: 0.913
- neg_mean_entropy: 0.772
- mean_logprob: 0.748
- mean_margin: 0.736
- verbal_conf: 0.687

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
