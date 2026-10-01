---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
base_model_relation: finetune
tags:
- decision-model
- classification
- system-one
- safetensors
---

![DEV2.0-0.6B owl banner](assets/DEV2.0-0.6B-owl-banner.png)

# DEV2.0-0.6B

A compact decision model for routing choices, yes/no checks and rubric scores over one state: native System One Choice, Noul and Score answers with probabilities, no chat API.

[Decision 2.0 collection](https://huggingface.co/collections/llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00) · [Download](#download-and-decide)

| Type | Use it for | Output |
| --- | --- | --- |
| **Choice** | Route a request or choose among 2–255 supplied options. | Selected ID + distribution |
| **Noul** | Check a condition against the supplied evidence. | P(yes) |
| **Score** | Apply 2–10 ordered rubric levels. | Expected level + distribution |

Ask one or many named questions about one state; options and rubrics are supplied at request time. Answers are typed decisions with probabilities, not generated text.

## Measured decisions

Post-key same-panel results.

On the same 8,147-item JevArena v3 panel, DEV2.0-0.6B scores **48.64** versus **35.94** for Decision 1.0 Kai (+12.70; paired 95% interval [+10.00, +17.50]). On the separate 231 public JevBench questions it answers **152/231** correctly versus **114/231**. It ranks 1 of 5 models shown on JevArena v3.

> **Independent sealed confirmation, same input limit (previous revision)** — JevArena-C1 v1.2 (2,840 human-labeled items from 8 sources published after the relevant cutoffs, never used for training or development; scored once): the previous DEV2.0-0.6B weights score **33.02** vs Decision 1.0 Kai at 8,192 tokens 22.14 (+10.89, 95% CI [+8.88, +12.64]). These are the predictions sealed at event 2 from revision `e61b2b44` (model files identical to the previous revision `99c4e799`), rescored on v1.2; Kai 1.0 at its published 1,024-token limit scores 17.79 on v1.2. The current weights (released as revision `b2131337`) were not part of any sealed event.
>
> **JevArena-C1 v1.2, post-key (not an independent validation):** 36.92 on this revision vs 33.02 for the previous revision (+3.89, 95% CI [+2.16, +5.60]; 2,840 items, paired by source group). The independent sealed confirmation above measured the previous revision.

| Rank | Model | Parameters | JevArena v3 ↑ | T | H | Choice | Noul | Score | JevBench public 231 (easy / standard / hard) ↑ | mlx-diag non-English Choice / Noul ↑ | Typed Brier / ECE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **DEV2.0-0.6B** | 0.60B | **48.64** | 0.515 | 0.459 | 389/800 | 517/800 | 137/400 | 152/231 (47 / 61 / 44) | 70.2 / 52.8 | 0.321 / 0.155 |
| 2 | GLiNER2.5-Decide | 0.49B | 42.52 | 0.410 | 0.441 | 313/800 | 502/800 | 88/400 | 116/231 (48 / 46 / 22) | 57.7 / 52.5 | 0.336 / 0.128 |
| 3 | Bosun v3.1 0.6B | 0.61B | 38.52 | 0.434 | 0.342 | 365/800 | 541/800 | 83/400 | 133/231 (47 / 50 / 36) | 63.2 / 61.2 | 0.302 / 0.090 |
| 4 | Decision 1.0 Kai | 0.57B | 35.94 | 0.362 | 0.357 | 277/800 | 404/800 | 98/400 | 114/231 (45 / 39 / 30) | 49.1 / 47.7 | 0.390 / 0.209 |
| 5 | Decision 1.0 Lex | 0.57B | 31.02 | 0.362 | 0.266 | 235/800 | 402/800 | 107/400 | 113/231 (42 / 42 / 29) | 26.5 / 51.0 | 0.336 / 0.156 |

![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)

![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)

![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)

Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and over-budget answers count as failures. JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not a blind test. T is typed-decision accuracy (four-family macro), H the median macro-F1 over 15 human-labeled transfer tasks, and v3 = 100 × sqrt(T × H). JevBench public 231: public-only rerun (about a third of the official Intelligence inputs), not the official JevBench score; easy tier at ceiling; totals within about 10 items are not distinguishable. mlx-diag is a multilingual development diagnostic (public test splits in seven languages, English instructions over target-language states), not a release score; its Score part is not shown. Ranks include only the models shown. GLiNER2.5-Decide declares Apache-2.0 in its model card metadata, but its repository has no LICENSE file. [Methods and per-model results](evaluation/EVALUATION.md)

### Tradeoffs versus Decision 1.0 Kai

Results below Decision 1.0 Kai on this panel:

| Result | DEV2.0-0.6B | Decision 1.0 Kai |
| --- | ---: | ---: |
| Transfer: mrf (macro-F1) | 55.5% | 56.6% |

## Download and decide

```bash
hf download llm-semantic-router/DEV2.0-0.6B --local-dir DEV2.0-0.6B
```

```python
import json
import os
import sys

sys.path.insert(0, os.path.abspath("DEV2.0-0.6B"))
from decision2 import Decision2

model = Decision2.from_pretrained("DEV2.0-0.6B")  # cuda:0 if a GPU is visible, else CPU
result = model.system_one(
    state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults"
            }
        },
        "receipt": {
            "type": "noul",
            "instructions": "Does the customer have a receipt?"
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": [
                "Routine",
                "Soon",
                "Today"
            ]
        }
    },
)
print(json.dumps(result["answers"], indent=2))
```

The download includes a small local runtime (`decision2/`); it runs on CPU or one CUDA/ROCm GPU and does not start a hosted endpoint. Tested with Python 3.12, PyTorch 2.12 and Transformers 5.17 on one ROCm GPU (BF16 backbone, FP32 head), where answers match the evaluated predictions, and on CPU (FP32), where probabilities can differ from the GPU results by a few thousandths. Transformers 5.17 prints a Mistral tokenizer-regex warning when it finds the package's root `config.json`; it does not change tokenization and can be ignored.

## Model details

- **Architecture:** The 28-layer Qwen3-0.6B text backbone (causal attention) reads the state, the question and every supplied candidate once; a shared candidate head scores the candidates against a global query and returns probabilities without generating text.
- **Parameters:** 597,103,104 loaded (backbone 596,049,920; decision head 1,053,184).
- **Direct weight origin:** [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Every weight of the official Qwen3-0.6B-Base (Apache-2.0) was fine-tuned (nothing frozen, no adapter) together with a new candidate head trained from random initialization; the release is the uniform average of six such fine-tunes (two training recipes × three seeds), plus fixed per-level offsets on five-level Score logits. It contains no Decision 1.0 Kai or Lex weights: Decision 1.0 Kai is the comparison model, and our own Decision 1.0 Lux-9B supplied soft training targets only.
- **Input limit:** 8,192 tokens for the complete state, question and candidates.
- **Calibration:** Raw probabilities at T = 1. Five-level Score answers first add the fixed per-level logit offsets in `score_bias.json` (see Limitations). Per-type temperatures fitted on the clean 698-item calibration partition (Choice 0.584, Noul 0.554, Score 0.450) were evaluated and not adopted because they worsened calibration on out-of-distribution development data.
- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 of every file and the runtime verifies them before loading.
- **Weights:** The uniform average of six full fine-tunes of Qwen3-0.6B-Base (two training recipes × three seeds; 30M tokens per seed, one epoch), with a new candidate head. Decision 1.0 Lux-9B probabilities were soft training targets on every row; none of its weights are included.
- **This revision** replaces `99c4e799` (a different three-seed average): post-key JevArena v3 48.64 against 43.54, a paired difference of +5.09 (95% interval [+2.65, +9.30]). Only five-level Score answers depend on the offsets; Choice, Noul and other Score scales come from the weights alone.
- **Storage:** the backbone's Linear projection matrices are stored in BF16 exactly as BF16 autocast rounds them, every other tensor in FP32; on every evaluated prompt the answers and probabilities equal those of the evaluated FP32 weights.
- **Runtime update:** BF16-resident weights; answers unchanged; latency p50 19.2 → 16.6 ms, p95 19.4 → 16.9 ms; peak GPU memory 2.3 → 1.5 GiB (400 single requests, 331 input tokens on average, one AMD MI325X GPU).

### Training

- **Recipe:** each fine-tune is one epoch on its own 30M-token mixture (logical batch 16; backbone learning rate 1e-5 with 10% warm-up and cosine decay, head 2e-4) with cross-entropy plus 0.5 × Brier and a KL term (weight 1.0) toward soft targets from our own [Decision 1.0 Lux-9B](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) on every row. No other teacher, and no outputs of Jev or of any third-party decision model.
- **Data:** together the six mixtures hold 232,754 distinct decision rows: Choice 88,676, Noul 75,701 and Score 68,377; English 130,105, Chinese 46,699, Hinglish 12,269, Spanish 8,791, Japanese 6,537, Korean 5,884 and 26 other languages.
- **Program-generated tasks (89,771 rows):** Decision 1.0 stage 1–4 generators and Decision 2.0 programmatic and verifiable generators, labeled by program oracles (no third-party text).
- **Human-labeled and public data (142,983 rows)** from 37 public datasets that keep their own licences ([attributions](ATTRIBUTIONS.md)). The training mixtures contain no MASSIVE, PAWS-X or XNLI rows (the sources of mlx-diag), and every group the evaluation-overlap rescreen excluded was removed before training.
- **Score offsets:** fitted afterwards on 400 fresh draws of the benchmark's own typed Score generator (see Limitations); no weight was changed.

### Limits

- **Score offsets:** Five-level Score answers add fixed per-level offsets to the model's logits before the softmax (`score_bias.json`: +0.039, +0.203, +0.079, −0.152, −0.170 for levels 0–4). Choice, Noul and other Score scales are unchanged. The offsets were fitted on 400 fresh draws of the benchmark's own typed Score generator (`resource_ledger`, the family behind the typed panel's Score items). These draws are disjoint from the panel's items up to event ids and row order, but about three quarters share an answer structure with them. The offsets were chosen on a separate 400-item check half and on held-out human 5-level ratings. Without them the same weights give the top level on 94.75% of the typed Score items.
- **Typed Score accuracy stays near always-majority:** On the typed Score family the model is right on 34.3% of items, against 32.0% for always answering the most common level (difference +2.3 points, 95% CI −1.5 to +6.3). The offsets fix level usage (62.0% of answers now on the top level), not Score skill on this family, and level 3 is never predicted there.
- **Human 5-level ratings:** On 1,809 held-out human-rated 5-level items the offsets change accuracy by −1.1 points (34.1% → 33.0%; 95% CI −2.8 to +0.7).
- **Against the previous revision:** The human-transfer axis is .459 against .480 (−.021, 95% CI −.064 to +.061). CSS15 losses are wiki politeness −3.4, Indian English dialect −3.4 and FLUTE −1.6 accuracy points. Public-231 easy is 47 against 48. Typed-panel calibration is weaker at T = 1 (ECE .155 against .118; Brier .321 against .309).
- **Multilingual (mlx-diag, card-eligible parts):** Choice: 75.4% English, 70.2% other languages (weakest Arabic, 62.7%). Noul: 53.0% English, 52.8% other languages, near chance (weakest Spanish, 50.0%). The Choice + Noul macro is +0.4 points against the previous revision (95% CI −0.6 to +1.4).
- **Post-key C1 declines against the previous revision** (JevArena-C1 v1.2, post-key; task macro-F1 × 100, language accuracy): star_rating 21.3 vs 30.5, moment_type 53.2 vs 60.1, is_rapport 43.6 vs 49.0 and likelihood 10.2 vs 13.1; Russian .263 vs .353 and Finnish .426 vs .500.
- **Public 231:** public-only rerun (about a third of the official Intelligence inputs), not the official JevBench score; easy tier at ceiling; totals within about 10 items are not distinguishable. It answers 152 (easy 47, standard 61, hard 44); paired differences: previous revision +10 items (95% interval [0, +20], p .087), Bosun v3.1 0.6B +19 ([+7, +31]), GLiNER2.5-Decide +36 ([+23, +49]), Decision 1.0 Kai +38 ([+23, +53]; +25 [+10, +41] against Kai at 8,192 tokens) and Decision 1.0 Lex +39 ([+24, +55]). No peer is significantly ahead on it.
- **Same-limit comparison:** the table shows Decision 1.0 Kai at its published 1,024-token input limit. Run at this model's 8,192-token limit, Kai scores 35.97 on JevArena v3 and 127/231 on JevBench public, and the paired v3 difference is +12.67 (95% interval [+9.74, +17.42]).
- **Overlap:** None of the evaluation-overlap rescreen's excluded training groups is in the training mixtures.
- JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not an untouched blind test.
- The v3 and public panels are almost entirely English; they do not measure multilingual ability.
- Complete inputs above 8,192 tokens are rejected, never truncated; 15 evaluation answers were invalid (including over-budget inputs) and counted as failures.
- It decides from the supplied state only and does not retrieve missing facts. Probabilities are estimates; review consequential decisions against the evidence.

[License](LICENSE) · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)
