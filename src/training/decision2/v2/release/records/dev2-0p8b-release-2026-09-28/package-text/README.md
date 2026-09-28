---
license: apache-2.0
base_model: llm-semantic-router/Decision-1.0-Eos-0.8B
base_model_relation: finetune
tags:
- decision-model
- classification
- system-one
- safetensors
---

![DEV2.0-0.8B owl banner](assets/DEV2.0-0.8B-owl-banner.png)

# DEV2.0-0.8B

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

On the same 8,147-item JevArena v3 panel, DEV2.0-0.8B scores **50.24** versus **42.55** for Decision 1.0 Eos (+7.69; paired 95% interval [+3.65, +13.32]). On the separate 231 public JevBench questions it answers **156/231** correctly versus **142/231**. It ranks 1 of 4 models shown on JevArena v3.

> **Independent sealed confirmation (JevArena-C1): PLACEHOLDER, result pending. The coordinator fills in this line after the one-shot C1 scoring event.**

| Rank | Model | Parameters | JevArena v3 ↑ | T | H | Choice | Noul | Score | JevBench public 231 (easy / standard / hard) ↑ | mlx-diag non-English Choice / Noul ↑ | Typed Brier / ECE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **DEV2.0-0.8B** | 0.75B | **50.24** | 0.573 | 0.440 | 529/800 | 611/800 | 107/400 | 156/231 (48 / 63 / 45) | 68.7 / 54.2 | 0.277 / 0.150 |
| 2 | Intern-Decision-0.8B | 0.85B | 43.54 | 0.496 | 0.382 | 495/800 | 431/800 | 88/400 | 164/231 (47 / 57 / 60) | 64.2 / 62.7 | 0.297 / 0.093 |
| 3 | Kev-0.8B | 0.75B | 43.22 | 0.479 | 0.390 | 489/800 | 407/800 | 82/400 | 147/231 (48 / 58 / 41) | 61.5 / 58.7 | 0.303 / 0.066 |
| 4 | Decision 1.0 Eos | 0.75B | 42.55 | 0.392 | 0.461 | 315/800 | 410/800 | 120/400 | 142/231 (48 / 55 / 39) | 68.5 / 59.2 | 0.333 / 0.165 |

![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)

![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)

![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)

Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and over-budget answers count as failures. JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not a blind test. T is typed-decision accuracy (four-family macro), H the median macro-F1 over 15 human-labeled transfer tasks, and v3 = 100 × sqrt(T × H). JevBench public 231 is our rerun of the 231 public questions; it is not the official sealed JevBench rank. mlx-diag is a multilingual development diagnostic (public test splits in seven languages, English instructions over target-language states), not a release score; its Score part is not shown. Ranks include only the models shown. [Methods and per-model results](evaluation/EVALUATION.md)

### Tradeoffs versus Decision 1.0 Eos

Results below Decision 1.0 Eos on this panel:

| Result | DEV2.0-0.8B | Decision 1.0 Eos |
| --- | ---: | ---: |
| Typed Score (correct) | 107/400 | 120/400 |
| Human transfer H (median task macro-F1) | 0.440 | 0.461 |
| mlx-diag non-English Noul (accuracy) | 54.2% | 59.2% |
| Transfer: mrf (macro-F1) | 45.6% | 62.5% |
| Transfer: conv go awry (macro-F1) | 43.0% | 57.1% |
| Transfer: flute (macro-F1) | 38.7% | 46.1% |
| Transfer: media ideology (macro-F1) | 30.2% | 36.6% |
| Transfer: wiki corpus (macro-F1) | 44.0% | 49.2% |
| Transfer: tempowic (macro-F1) | 53.5% | 58.2% |
| Transfer: ibc (macro-F1) | 32.2% | 35.2% |
| Transfer: persuasion (macro-F1) | 49.7% | 50.1% |

## Download and decide

```bash
hf download llm-semantic-router/DEV2.0-0.8B --local-dir DEV2.0-0.8B
```

```python
import json
import os
import sys

sys.path.insert(0, os.path.abspath("DEV2.0-0.8B"))
from decision2 import Decision2

model = Decision2.from_pretrained("DEV2.0-0.8B")  # cuda:0 if a GPU is visible, else CPU
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

The download includes a small local runtime (`decision2/`) for one CUDA or ROCm GPU; it does not start a hosted endpoint. Tested on one ROCm GPU with Python 3.12, PyTorch 2.12 and Transformers 5.17 (BF16 backbone, FP32 head). GPU answers match the evaluated predictions with the flash-linear-attention 0.5.2 and causal-conv1d 1.7.0 kernels and a persisted Triton autotune cache; without those kernels Transformers uses its reference implementation of the same operations, which is slower and differs slightly in the last digits.

## Model details

- **Architecture:** The 24-layer Qwen3.5 text backbone inherited from Decision 1.0 Eos (18 gated-delta and 6 attention layers) reads the state, the question and every supplied candidate once; a shared candidate head scores the candidates against a global query and returns probabilities without generating text.
- **Parameters:** 753,446,208 loaded (backbone 752,393,024; decision head 1,053,184).
- **Direct weight origin:** [llm-semantic-router/Decision-1.0-Eos-0.8B](https://huggingface.co/llm-semantic-router/Decision-1.0-Eos-0.8B) at `363c4a5e56afc115b1c78c837633956d0bbb63ab`. Every weight of Decision 1.0 Eos was fine-tuned (nothing frozen, no adapter), and the release is the uniform average of three seeds of that fine-tune. Decision 1.0 Eos is itself a text-only fine-tune of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) at `2fc06364715b967f1860aea9cf38778875588b17` (Apache-2.0), whose text backbone and tokenizer this model inherits; the Qwen3.5 vision tower is not part of it.
- **Input limit:** 16,384 tokens for the complete state, question and candidates.
- **Calibration:** per-type temperatures fitted for this exact checkpoint on the clean 698-item calibration partition at the 16,384-token limit (Choice 1.123, Noul 1.071, Score 0.395; `calibration.json`).
- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 of every file and the runtime verifies them before loading.

### Training

- **Recipe:** one epoch of full fine-tuning (backbone learning rate 1e-5, head 1e-4; cross-entropy plus 0.5 × Brier) on 162,777 decision rows (138.4M tokens) with seeds 20260926, 20260927 and 20260928, each seed's checkpoint chosen on a held-out selection set, then a uniform weight average of the three. Hard labels only: no teacher, distillation or soft targets.
- **Own Decision 1.0 training corpora (130,157 rows):** program-generated English and Chinese decision tasks labeled by program oracles (no model-written states or labels), plus the human labels of CLINC150 (CC BY 3.0), BANKING77 (CC BY 4.0), MultiNLI non-fiction genres (OANC terms) and SNLI (CC BY-SA 4.0).
- **Decision 2.0 data arms (32,620 rows):** project-generated verifiable rules, counterfactuals, hard negatives and multi-level Score tasks, and the human labels of GoEmotions (CC BY 4.0), ABCD (MIT), Schema-Guided Dialogue (CC BY-SA 4.0), MuSiQue (CC BY 4.0; Wikipedia text CC BY-SA), KLUE YNAT and STS and JGLUE JCommonsenseQA, JNLI and JSTS (CC BY-SA 4.0), IBM ArgQ-30k (CC BY-SA 3.0) and SAF (CC BY 4.0), in English, Chinese, Japanese, Korean and German.
- **Not used:** outputs of Jev or of any third-party decision model, and two shortcut-prone reading families (Cosmos QA, SQuAD 2.0 answerability). Training data were screened for overlap with the evaluation panels. Datasets keep their own licences ([attributions](ATTRIBUTIONS.md)).

### Limits

- **Seed dependence.** The three single fine-tunes scored 49.78, 47.36 and 41.08 on JevArena v3 (8,192-token runs); only their uniform average is released (50.24 at 16,384 tokens). A retrained seed of the same recipe can land well below this model.
- **Where it improves and where it does not.** The gain over Decision 1.0 Eos is typed reasoning (Choice 529 vs 315, Noul 611 vs 410 of 800). Typed Score is lower (107 vs 120 of 400) and human transfer is slightly lower (H 0.440 vs 0.461); the tradeoffs table lists every task below Eos.
- **Multilingual.** On the mlx-diag diagnostic, non-English Noul (paraphrase pairs) is about 5 points below Decision 1.0 Eos (54.2% vs 59.2%), while non-English Choice is level (68.7% vs 68.5%).
- **CPU.** CPU inference was not verified for this release: where the GPU-only causal-conv1d package is installed, Transformers routes the convolution to that kernel and a CPU run fails.
- JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not an untouched blind test.
- The v3 and public panels are almost entirely English; they do not measure multilingual ability.
- Complete inputs above 16,384 tokens are rejected, never truncated; 4 evaluation answers were invalid (including over-budget inputs) and counted as failures.
- It decides from the supplied state only and does not retrieve missing facts. Probabilities are estimates; review consequential decisions against the evidence.

[License](LICENSE) · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)
