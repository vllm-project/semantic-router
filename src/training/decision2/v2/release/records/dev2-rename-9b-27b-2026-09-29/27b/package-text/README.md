---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
tags:
- decision-model
- classification
- system-one
- safetensors
---

![DEV2.0-27B owl banner](assets/DEV2.0-27B-owl-banner.png)

# DEV2.0-27B

A decision model for routing choices, yes/no checks and rubric scores over one state: native System One Choice, Noul and Score answers with probabilities, no chat API.

[Decision 2.0 collection](https://huggingface.co/collections/llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00) · [Download](#download-and-decide)

| Type | Use it for | Output |
| --- | --- | --- |
| **Choice** | Route a request or choose among 2–255 supplied options. | Selected ID + distribution |
| **Noul** | Check a condition against the supplied evidence. | P(yes) |
| **Score** | Apply 2–10 ordered rubric levels. | Expected level + distribution |

Ask one or many named questions about one state; options and rubrics are supplied at request time. Answers are typed decisions with probabilities, not generated text.

## Measured decisions

Post-key same-panel results.

On the same 8,147-item JevArena v3 panel, DEV2.0-27B scores **67.21**. There is no Decision 1.0 model at this size; AutoJev-27B, the strongest same-size model shown, scores **72.13** (-4.92; paired 95% interval [-6.82, -0.43]). On the separate 231 public JevBench questions it answers **198/231** correctly versus **201/231** for AutoJev-27B. It ranks 3 of 4 models shown on JevArena v3.

> **Independent sealed confirmation (JevArena-C1): PLACEHOLDER — to be added after the final sealed scoring event.**

| Rank | Model | Parameters | JevArena v3 ↑ | T | H | Choice | Noul | Score | JevBench public 231 (easy / standard / hard) ↑ | mlx-diag non-English Choice / Noul ↑ | Typed Brier / ECE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | AutoJev-27B | 26.09B | 72.13 | 0.887 | 0.587 | 800/800 | 737/800 | 282/400 | 201/231 (48 / 72 / 81) | 80.6 / 88.7 | 0.064 / 0.082 |
| 2 | Eikos-27B | 27.78B | 69.29 | 0.818 | 0.587 | 783/800 | 589/800 | 336/400 | 212/231 (48 / 72 / 92) | 78.4 / 85.0 | 0.107 / 0.015 |
| 3 | **DEV2.0-27B** | 25.75B | **67.21** | 0.787 | 0.574 | 628/800 | 715/800 | 317/400 | 198/231 (48 / 71 / 79) | 79.1 / 85.5 | 0.126 / 0.048 |
| 4 | Jebadiah-27B | 26.90B | 65.47 | 0.742 | 0.578 | 617/800 | 720/800 | 250/400 | 176/231 (48 / 70 / 58) | 77.6 / 85.3 | 0.143 / 0.062 |

![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)

![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)

![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)

Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and over-budget answers count as failures. JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not a blind test. T is typed-decision accuracy (four-family macro), H the median macro-F1 over 15 human-labeled transfer tasks, and v3 = 100 × sqrt(T × H). JevBench public 231 is our rerun of the 231 public questions; it is not the official sealed JevBench rank. mlx-diag is a multilingual development diagnostic (public test splits in seven languages, English instructions over target-language states), not a release score; its Score part is not shown. Ranks include only the models shown. The three 27B peers were rerun on the same kernel image as this model. Eikos-27B is shown from its BF16 weights (`caiovicentino1/Eikos-27B`), the sibling of the FP8 build listed on the Decision Index; its card declares MIT for the authors' contributions on the Apache-2.0 base. AutoJev-27B declares Apache-2.0 weights and MIT code, Jebadiah-27B Apache-2.0. Invalid answers count as failures: none for this model; 13, 4 and 153 transfer answers for AutoJev-27B, Eikos-27B and Jebadiah-27B, and 36 public-231 answers for Jebadiah-27B. [Methods and per-model results](evaluation/EVALUATION.md)

### Tradeoffs versus AutoJev-27B

There is no Decision 1.0 model at this size, so this compares with AutoJev-27B. Results below AutoJev-27B on this panel:

| Result | DEV2.0-27B | AutoJev-27B |
| --- | ---: | ---: |
| Typed Choice (correct) | 628/800 | 800/800 |
| Typed Noul (correct) | 715/800 | 737/800 |
| Human transfer H (median task macro-F1) | 0.574 | 0.587 |
| mlx-diag non-English Choice (accuracy) | 79.1% | 80.6% |
| mlx-diag non-English Noul (accuracy) | 85.5% | 88.7% |
| Transfer: wiki politeness (macro-F1) | 44.8% | 52.9% |
| Transfer: persuasion (macro-F1) | 56.9% | 61.1% |
| Transfer: tropes (macro-F1) | 14.9% | 18.7% |
| Transfer: flute (macro-F1) | 86.0% | 88.6% |
| Transfer: raop (macro-F1) | 57.4% | 59.4% |
| Transfer: media ideology (macro-F1) | 62.8% | 63.3% |
| JevBench public standard (correct) | 71/72 | 72/72 |
| JevBench public hard (correct) | 79/111 | 81/111 |

## Download and decide

```bash
hf download llm-semantic-router/DEV2.0-27B --local-dir DEV2.0-27B
```

```python
import json
import os
import sys

sys.path.insert(0, os.path.abspath("DEV2.0-27B"))
from decision2 import Decision2

model = Decision2.from_pretrained("DEV2.0-27B")  # cuda:0 if a GPU is visible, else CPU
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

The download includes a small local runtime (`decision2/`) for one CUDA or ROCm GPU; it fetches the pinned base and does not start a hosted endpoint. Tested on one AMD MI325X GPU (ROCm 7.2) with Python 3.12, PyTorch 2.12, Transformers 5.17 and PEFT 0.21 (BF16 backbone, FP32 head). The evaluation peaked at 108 GB of allocated GPU memory (120 GB reserved) on its longest input, 18,261 tokens; no evaluated input reached the 32,768-token limit. The runtime downloads the pinned Qwen3.8-27B base files (or takes `base_path`) and checks every file's SHA-256 before loading. GPU answers match the evaluated predictions with the flash-linear-attention 0.5.2 and causal-conv1d 1.7.0 kernels and a persisted Triton autotune cache; without those kernels Transformers uses its reference implementation of the same operations, which is slower and differs slightly in the last digits.

## Model details

- **Architecture:** A rank-16 LoRA adapter on the frozen 64-layer Qwen3.8-27B text backbone (48 gated-delta and 16 attention layers) reads the state, the question and every supplied candidate once; a shared candidate head scores the candidates against a global query and returns probabilities without generating text.
- **Parameters:** 25,746,591,744 loaded (pinned base text backbone 25,624,600,064, not redistributed here; LoRA adapter 116,727,808; decision head 5,263,872).
- **Name:** Named after its base model ([Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)); it loads 25,746,591,744 parameters.
- **Direct weight origin:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. A PEFT LoRA adapter (rank 16, alpha 32, on all 496 text projections of the attention, gated-delta and MLP blocks) and a native candidate head trained on the frozen Qwen3.8-27B text backbone at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0), which is not redistributed. The adapter is the exact uniform soup of two seeds of one rank-8 recipe (their LoRA factors stacked along the rank with the up-projections halved, so the mean of the two seeds' weight updates is reproduced exactly), with the two heads averaged.
- **Input limit:** 32,768 tokens for the complete state, question and candidates.
- **Calibration:** raw model probabilities (temperature 1). Per-type temperatures fitted on the clean 698-item calibration partition (Choice 0.538, Noul 0.428, Score 0.567) were evaluated and not adopted because they worsened calibration on out-of-distribution development data.
- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 of every file and the runtime verifies them before loading.
- **Weights:** the exact soup of two rank-8 LoRA adapters of the same recipe and start (seeds 20260926 and 20260928; selected updates 1,614 and 1,414), stored as one standard PEFT rank-16 adapter, plus the averaged decision head.
- **Loading:** `Decision2.from_pretrained` downloads the 28 pinned files of Qwen/Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` into the Hugging Face cache (or takes `base_path=` for a local copy of that revision) and checks each file's SHA-256 before loading the adapter on it.

### Training

- **Recipe:** a rank-8 LoRA (alpha 16, dropout 0.05) on all 496 text projections of the frozen Qwen3.8-27B base plus a new candidate head, one pass of cross-entropy plus 0.5 × Brier over 10.0M tokens of decision rows (inputs up to 4,096 tokens), learning rates LoRA 2e-5 / head 1e-4, seeds 20260926 and 20260928; each seed's checkpoint chosen on a held-out selection set, then the exact uniform soup of the two adapters (one rank-16 adapter whose weight update is the mean of the two) and heads.
- **Data (25,822 rows, 10.0M tokens):** the strict Decision 2.0 base (6,416 rows: program-generated decision tasks plus the human labels of GoEmotions, SNLI, CLINC150 and BANKING77; the Cosmos QA and SQuAD 2.0 answerability families removed), the A6 Score arms (4,651 rows: project-generated Score tasks with 2 to 10 levels in English and Chinese, and the graded human ratings of KLUE STS, JGLUE JSTS, IBM ArgQ-30k and SAF in Korean, Japanese, English and German), and a 5.0M-token slice of the A7 typed curricula (14,755 rows: program-generated Stage 1-4 decision tasks plus the BANKING77 and CLINC150 intent labels, with an equal token quota per family). By tokens the data is 58% English, 38% Chinese and about 4% Korean, Japanese and German together. The human sources are CC BY (3.0 or 4.0) or CC BY-SA (3.0 or 4.0).
- **What the typed curricula did:** Against a control trained on the same data and number of tokens without them, the Stage 1–4 typed curricula (about 5M tokens, 50% of training) raised typed-reasoning accuracy on unseen formal families by 9.9 points (95% CI +8.1 to +11.8). Constraint competition rose from .38 to .57 and exception stacks from .59 to .79. The JevArena v3 composite rose 2.7 points (not significant), and human-transfer accuracy was 3.0 points lower (not significant).
- **Not used:** outputs of Jev or of any third-party decision model; every row trains on its gold label. Datasets keep their own licences ([attributions](ATTRIBUTIONS.md)).

### Limits

- **Below AutoJev-27B.** JevArena v3 is significantly lower than AutoJev-27B (−4.92, 95% CI [−6.82, −0.43]). Typed accuracy is the gap (T 0.7875 vs 0.8869, −0.099, 95% CI [−0.123, −0.075]; constraint competition 0.570 vs 1.000, exception stack 0.788 vs 0.843; resource ledger is higher, 0.793 vs 0.705), while human transfer is level with all three 27B models shown (H 0.574 against 0.587, 0.587 and 0.578; every paired interval includes 0, e.g. −0.013, 95% CI [−0.040, +0.056] against AutoJev-27B).
- **Against the other 27B models** the JevArena v3 margins are not significant: Eikos-27B −2.08 (95% CI [−5.04, +2.62]) and Jebadiah-27B +1.74 (95% CI [−0.97, +4.83]).
- **JevBench public 231:** 198 correct (48 / 71 / 79): level with AutoJev-27B's 201 (−3, 95% CI [−10, +5]) and above Jebadiah-27B's 176 (+22, 95% CI [+11, +33]), but significantly below Eikos-27B's 212 (−14, 95% CI [−22, −6]; hard tier 79 vs 92).
- **Score levels.** On typed Score it uses all five levels but over-predicts the top level (182 of 400 answers versus 128 in the gold labels); recall at levels 1, 2 and 3 is 0.72, 0.61 and 0.65 (level 0: 0.91).
- **Transfer tasks.** Wiki politeness is lower than all three 27B models shown (macro-F1 −0.081, −0.142 and −0.087 against AutoJev-27B, Eikos-27B and Jebadiah-27B); persuasion trails AutoJev-27B and Jebadiah-27B, and flute trails Eikos-27B.
- **Long inputs** (over 4,000 characters): transfer macro-F1 0.574, level with AutoJev-27B (0.576) and Eikos-27B (0.583); public 231 answers 26 of 37 (AutoJev-27B 28, Eikos-27B 29).
- **Multilingual.** On the mlx-diag diagnostic (public test splits in seven languages, English instructions over target-language states), non-English Choice is level with English (79.1% vs 79.4%) and non-English Noul (paraphrase pairs) is 1.5 points lower (85.5% vs 87.0%). The weakest languages are Korean Noul (79%, 100 items) and Spanish and Arabic Choice (76.2% and 77.8%, 126 items each). The three 27B peers ran the same diagnostic (Choice and Noul parts; no paired test): non-English Choice is 80.6% for AutoJev-27B, 78.4% for Eikos-27B and 77.6% for Jebadiah-27B (79.1% here), and non-English Noul is 88.7%, 85.0% and 85.3% (85.5% here). Korean Noul is 85%, 84% and 80% for the peers (79% here) and Arabic Choice 78.6%, 75.4% and 75.4% (77.8% here).
- **Calibration.** The package returns raw probabilities (temperature 1); per-type temperatures fitted on the calibration partition were rejected because they worsened calibration on development data.
- **Input limit.** 32,768 tokens for the complete state, question and candidates; longer inputs return an over-budget error instead of being truncated. Training inputs were at most 4,096 tokens.
- **Seed soup.** Only the uniform soup of two seeds was evaluated on JevArena v3 and is released; a retrained seed of the same recipe can land below it.
- **Base download and memory.** The base weights are not in this repository; loading needs the pinned Qwen3.8-27B revision (about 55 GB of files) and one GPU with at least 120 GB of memory (the evaluation's peak on an 18,261-token input; longer inputs were not measured). CPU inference was not verified.
- **Evaluation familiarity.** A later screen of the program's training pools found topical-phrase overlap (no exact duplicates) between some training groups and 84 items of the reported evaluation panels. This model's training data contains none of those groups. Rescored without the 84 items, post-key v3 stays 67.21, the release bar (90% of AutoJev-27B) moves from 64.92 to 64.89, and none of this card's v3, human-transfer or public-231 comparisons change.
- JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not an untouched blind test.
- The v3 and public panels are almost entirely English; they do not measure multilingual ability.
- Complete inputs above 32,768 tokens are rejected, never truncated.
- It decides from the supplied state only and does not retrieve missing facts. Probabilities are estimates; review consequential decisions against the evidence.

[License](LICENSE) · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)
