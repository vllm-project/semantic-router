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

On the same 8,147-item JevArena v3 panel, DEV2.0-0.6B scores **43.54** versus **35.94** for Decision 1.0 Kai (+7.60; paired 95% interval [+4.70, +10.76]). On the separate 231 public JevBench questions it answers **142/231** correctly versus **114/231**. It ranks 1 of 5 models shown on JevArena v3.

> **Independent sealed confirmation** — JevArena-C1 v1.1 (2,874 human-labelled items from 8 sources published after the relevant cutoffs, never used for training or development; scored once): DEV2.0-0.6B **33.21** vs Decision 1.0 Kai 17.77 (+15.44, 95% CI [+13.51, +17.21]); Bosun v3.1 0.6B 35.03 (−1.82, 95% CI [−4.05, +0.35]); GLiNER2.5-Decide 22.82; Decision 1.0 Lex 20.82. Kai and Lex run at their published 1,024-token input limit and GLiNER2.5-Decide at its own; items they cannot take (525 and 935 of 2,874) count as wrong.

| Rank | Model | Parameters | JevArena v3 ↑ | T | H | Choice | Noul | Score | JevBench public 231 (easy / standard / hard) ↑ | mlx-diag non-English Choice / Noul ↑ | Typed Brier / ECE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **DEV2.0-0.6B** | 0.60B | **43.54** | 0.395 | 0.480 | 255/800 | 442/800 | 133/400 | 142/231 (48 / 54 / 40) | 70.2 / 51.8 | 0.309 / 0.118 |
| 2 | GLiNER2.5-Decide | 0.49B | 42.52 | 0.410 | 0.441 | 313/800 | 502/800 | 88/400 | 116/231 (48 / 46 / 22) | 57.7 / 52.5 | 0.336 / 0.128 |
| 3 | Bosun v3.1 0.6B | 0.61B | 38.52 | 0.434 | 0.342 | 365/800 | 541/800 | 83/400 | 133/231 (47 / 50 / 36) | 63.2 / 61.2 | 0.302 / 0.090 |
| 4 | Decision 1.0 Kai | 0.57B | 35.94 | 0.362 | 0.357 | 277/800 | 404/800 | 98/400 | 114/231 (45 / 39 / 30) | 49.1 / 47.7 | 0.390 / 0.209 |
| 5 | Decision 1.0 Lex | 0.57B | 31.02 | 0.362 | 0.266 | 235/800 | 402/800 | 107/400 | 113/231 (42 / 42 / 29) | 26.5 / 51.0 | 0.336 / 0.156 |

![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)

![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)

![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)

Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and over-budget answers count as failures. JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not a blind test. T is typed-decision accuracy (four-family macro), H the median macro-F1 over 15 human-labeled transfer tasks, and v3 = 100 × sqrt(T × H). JevBench public 231 is our rerun of the 231 public questions; it is not the official sealed JevBench rank. mlx-diag is a multilingual development diagnostic (public test splits in seven languages, English instructions over target-language states), not a release score; its Score part is not shown. Ranks include only the models shown. GLiNER2.5-Decide declares Apache-2.0 in its model card metadata, but its repository has no LICENSE file. [Methods and per-model results](evaluation/EVALUATION.md)

### Tradeoffs versus Decision 1.0 Kai

Results below Decision 1.0 Kai on this panel:

| Result | DEV2.0-0.6B | Decision 1.0 Kai |
| --- | ---: | ---: |
| Typed Choice (correct) | 255/800 | 277/800 |
| Transfer: mrf (macro-F1) | 46.0% | 56.6% |

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
- **Direct weight origin:** [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Every weight of the official Qwen3-0.6B-Base (Apache-2.0) was fine-tuned (nothing frozen, no adapter) together with a new candidate head trained from random initialization, and the release is the uniform average of three seeds of that fine-tune. It contains no Decision 1.0 Kai or Lex weights: Decision 1.0 Kai is the comparison model, and our own Decision 1.0 Lux-9B supplied soft training targets only.
- **Input limit:** 8,192 tokens for the complete state, question and candidates.
- **Calibration:** raw model probabilities, exactly as evaluated (temperature 1). Per-type temperatures fitted on the clean 698-item calibration partition (Choice 0.592, Noul 0.588, Score 0.345) were evaluated and not adopted because they worsened calibration on out-of-distribution development data.
- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 of every file and the runtime verifies them before loading.
- **Weights:** the uniform average of three full fine-tunes of Qwen3-0.6B-Base with the same recipe, start and head initialization; the seeds differ only in data order, and each seed's checkpoint was chosen on a held-out selection set (all three at update 1,433 of 1,638).

### Training

- **Recipe:** one epoch of full fine-tuning (1,638 updates of 16 rows; backbone learning rate 1e-5 with 10% warm-up and cosine decay, head 2e-4) with cross-entropy plus 0.5 × Brier, and a KL term (weight 1.0) toward soft targets from our own [Decision 1.0 Lux-9B](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) on the 6,430 rows whose exact input it had labeled. No other teacher, and no outputs of Jev or of any third-party decision model.
- **Data:** 26,203 decision rows (20.0M tokens): Choice 13,422, Noul 5,565 and Score 7,216 (2 to 10 levels); English 17,498, Chinese 6,722, Japanese 860, Korean 853 and German 270.
- **Program-generated tasks (14,912 rows):** Decision 1.0 stage 1–4 generators and Decision 2.0 verifiable rule, counterfactual and multi-level Score generators, labeled by program oracles (no third-party text).
- **Human labels (11,291 rows):** GoEmotions (CC BY 4.0), SNLI (CC BY-SA 4.0), MultiNLI non-fiction genres (OANC terms), CLINC150 (CC BY 3.0), BANKING77 (CC BY 4.0), KLUE STS and JGLUE JSTS (CC BY-SA 4.0), IBM ArgQ-30k (CC BY-SA 3.0) and SAF (CC BY 4.0).
- **Not used:** two shortcut-prone reading families (Cosmos QA, SQuAD 2.0 answerability). Option keys that revealed the answer through their construction order were renumbered by position, and the training data were screened for overlap with the evaluation panels. Datasets keep their own licences ([attributions](ATTRIBUTIONS.md)).

### Limits

- **Typed Choice is below Decision 1.0 Kai:** 255 versus 277 of 800 JevArena v3 typed Choice questions (−22). The overall gain comes from Noul (442 vs 404), Score (133 vs 98) and human transfer (H 0.480 vs 0.357); the tradeoffs table lists every result below Kai.
- **Same-limit comparison:** the table shows Decision 1.0 Kai at its published 1,024-token input limit. Run at this model's 8,192-token limit, Kai scores 35.97 on JevArena v3 and 127/231 on JevBench public, and the paired v3 difference is +7.57 (95% interval [+4.43, +10.59]).
- **Not clearly ahead of the strongest 0.6B peer:** against GLiNER2.5-Decide the v3 difference is +1.02 with a paired 95% interval of [−1.77, +7.80], which includes zero; its typed accuracy T is higher (0.410 vs 0.395).
- **Seed dependence:** the three single fine-tunes differed markedly on our development panels. Only their uniform average was evaluated and is released, so retraining the same recipe with another seed can give a noticeably weaker model.
- **Multilingual yes/no checks:** on the mlx-diag diagnostic, non-English Choice (intent routing) is 70.2%, but non-English Noul (paraphrase detection) is 51.8%, below Bosun v3.1 0.6B (61.2%); treat non-English Noul answers with care.
- **Typed accuracy is below Bosun v3.1 0.6B:** T 0.395 versus 0.434 (paired difference −0.038, 95% interval [−0.068, −0.010]); the v3 gain over Bosun comes from human transfer.
- **Noul and middle Score levels are weak:** typed Noul is only narrowly above chance (.552 against .50; 95% interval from .518, with 78% of answers on one side), and typed Score leans on the lowest and highest of five levels (recall .54 and .77 there, .03 to .13 on the three middle levels).
- **On the sealed C1 set** it is below Decision 1.0 Kai on Arabic HalluTruthQA (hallucination and find-the-truth), narrative event causality and Arabic overall, and Noul is its weakest type; versus Bosun v3.1 0.6B it is weaker on Choice and Noul and stronger on Score.
- JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not an untouched blind test.
- The v3 and public panels are almost entirely English; they do not measure multilingual ability.
- Complete inputs above 8,192 tokens are rejected, never truncated; 15 evaluation answers were invalid (including over-budget inputs) and counted as failures.
- It decides from the supplied state only and does not retrieve missing facts. Probabilities are estimates; review consequential decisions against the evidence.

[License](LICENSE) · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)
