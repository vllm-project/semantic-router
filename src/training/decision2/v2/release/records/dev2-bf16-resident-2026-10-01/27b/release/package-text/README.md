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

On the same 8,147-item JevArena v3 panel, DEV2.0-27B scores **72.36**. There is no Decision 1.0 model at this size; AutoJev-27B, the strongest other same-size model shown, scores **72.13** (+0.23; paired 95% interval [-1.60, +4.74]). On the separate 231 public JevBench questions it answers **203/231** correctly versus **201/231** for AutoJev-27B. It ranks 1 of 4 models shown on JevArena v3.

> **Independent sealed confirmation** — JevArena-C1 v1.2 (2,840 human-labeled items from 8 sources published after the relevant cutoffs, never used for training or development; scored once) measured the previous revision (release package `5683c6f0`, weights identity `b7fd44e3`): 57.33 vs AutoJev-27B 58.17 (−0.84, 95% CI [−2.50, +0.69]) and Eikos-27B 59.37 (−2.04, 95% CI [−3.66, −0.44]); it has not been repeated for this revision. No Decision 1.0 model exists at this size.

**JevArena-C1 v1.2, post-key (not an independent validation):** 57.56 on this revision vs 57.33 for the previous revision (+0.23, 95% CI [−1.22, +1.70]; 2,840 items, paired by source group). The independent sealed confirmation above measured the previous revision.

| Rank | Model | Parameters | JevArena v3 ↑ | T | H | Choice | Noul | Score | JevBench public 231 (easy / standard / hard) ↑ | mlx-diag non-English Choice / Noul ↑ | Typed Brier / ECE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **DEV2.0-27B** | 26.10B | **72.36** | 0.896 | 0.584 | 753/800 | 737/800 | 344/400 | 203/231 (48 / 72 / 83) | 81.0 / 86.0 | 0.059 / 0.013 |
| 2 | AutoJev-27B | 26.09B | 72.13 | 0.887 | 0.587 | 800/800 | 737/800 | 282/400 | 201/231 (48 / 72 / 81) | 80.6 / 88.7 | 0.064 / 0.082 |
| 3 | Eikos-27B | 27.78B | 69.29 | 0.818 | 0.587 | 783/800 | 589/800 | 336/400 | 212/231 (48 / 72 / 92) | 78.4 / 85.0 | 0.107 / 0.015 |
| 4 | Jebadiah-27B | 26.90B | 65.47 | 0.742 | 0.578 | 617/800 | 720/800 | 250/400 | 176/231 (48 / 70 / 58) | 77.6 / 85.3 | 0.143 / 0.062 |

![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)

![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)

![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)

Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and over-budget answers count as failures. JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not a blind test. T is typed-decision accuracy (four-family macro), H the median macro-F1 over 15 human-labeled transfer tasks, and v3 = 100 × sqrt(T × H). JevBench public 231: public-only rerun (about a third of the official Intelligence inputs), not the official JevBench score; easy tier at ceiling; totals within about 10 items are not distinguishable. mlx-diag is a multilingual development diagnostic (public test splits in seven languages, English instructions over target-language states), not a release score; its Score part is not shown. Ranks include only the models shown. The three 27B peers were rerun on the same kernel image as this model. Eikos-27B is shown from its BF16 weights (`caiovicentino1/Eikos-27B`), the sibling of the FP8 build listed on the Decision Index; its card declares MIT for the authors' contributions on the Apache-2.0 base. AutoJev-27B declares Apache-2.0 weights and MIT code, Jebadiah-27B Apache-2.0. Invalid answers count as failures: none for this model; 13, 4 and 153 transfer answers for AutoJev-27B, Eikos-27B and Jebadiah-27B, and 36 public-231 answers for Jebadiah-27B. [Methods and per-model results](evaluation/EVALUATION.md)

### Tradeoffs versus AutoJev-27B

There is no Decision 1.0 model at this size, so this compares with AutoJev-27B. Results below AutoJev-27B on this panel:

| Result | DEV2.0-27B | AutoJev-27B |
| --- | ---: | ---: |
| Typed Choice (correct) | 753/800 | 800/800 |
| Human transfer H (median task macro-F1) | 0.584 | 0.587 |
| mlx-diag non-English Noul (accuracy) | 86.0% | 88.7% |
| Transfer: wiki politeness (macro-F1) | 49.0% | 52.9% |
| Transfer: reddit humor (macro-F1) | 56.9% | 60.7% |
| Transfer: tempowic (macro-F1) | 71.3% | 73.6% |
| Transfer: media ideology (macro-F1) | 61.3% | 63.3% |
| Transfer: persuasion (macro-F1) | 59.4% | 61.1% |
| Transfer: flute (macro-F1) | 87.0% | 88.6% |
| Transfer: mrf (macro-F1) | 77.2% | 77.5% |

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

The download includes a small local runtime (`decision2/`) for one CUDA or ROCm GPU; it fetches the pinned base and does not start a hosted endpoint. Tested on one AMD MI325X GPU (ROCm 7.2) with Python 3.12, PyTorch 2.12, Transformers 5.17 and PEFT 0.21 (BF16 backbone, FP32 head). The evaluation peaked at 110 GB of allocated GPU memory (121 GB reserved) on its longest input, 18,261 tokens; no evaluated input reached the 32,768-token limit. The runtime downloads the pinned Qwen3.8-27B base files (or takes `base_path`) and checks every file's SHA-256 before loading. GPU answers match the evaluated predictions with the flash-linear-attention 0.5.2 and causal-conv1d 1.7.0 kernels and a persisted Triton autotune cache; without those kernels Transformers uses its reference implementation of the same operations, which is slower and differs slightly in the last digits.

## Model details

- **Architecture:** A rank-64 LoRA adapter on the frozen 64-layer Qwen3.8-27B text backbone (48 gated-delta and 16 attention layers) reads the state, the question and every supplied candidate once; a shared candidate head scores the candidates against a global query and returns probabilities without generating text.
- **Parameters:** 26,096,775,168 loaded (pinned base text backbone 25,624,600,064, not redistributed here; LoRA adapter 466,911,232; decision head 5,263,872).
- **Name:** Named after its base model ([Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)); it loads 26,096,775,168 parameters.
- **Direct weight origin:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. A PEFT LoRA adapter (rank 64, alpha 128, on all 496 text projections of the attention, gated-delta and MLP blocks) and a native candidate head trained on the frozen Qwen3.8-27B text backbone at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0), which is not redistributed. The adapter is the exact uniform soup of two seeds of one rank-32 recipe (their LoRA factors stacked along the rank with the up-projections halved, so the mean of the two seeds' weight updates is reproduced exactly), with the two heads averaged.
- **Input limit:** 32,768 tokens for the complete state, question and candidates.
- **Calibration:** raw model probabilities (temperature 1). Per-type temperatures fitted on the clean 698-item calibration partition (Choice 0.639, Noul 0.545, Score 0.05) were evaluated and not adopted because they worsened calibration on out-of-distribution development data.
- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 of every file and the runtime verifies them before loading.
- **Weights:** the exact soup of two rank-32 LoRA adapters of the same recipe and start (seeds 20260926 and 20260928; selected updates 3,561 and 2,230), stored as one standard PEFT rank-64 adapter (alpha 128), plus the averaged decision head.
- **Loading:** `Decision2.from_pretrained` downloads the 28 pinned files of Qwen/Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` into the Hugging Face cache (or takes `base_path=` for a local copy of that revision) and checks each file's SHA-256 before loading the adapter on it.
- **Runtime update:** BF16-resident weights; answers unchanged; latency p50 122.9 → 93.3 ms, p95 127.5 → 96.8 ms; peak GPU memory 97.6 → 52.1 GiB (400 single requests, 346 input tokens on average, one AMD MI325X GPU).

### Training

- **Recipe:** a rank-32 LoRA (alpha 64, dropout 0.05) on all 496 text projections of the frozen Qwen3.8-27B base plus a new candidate head, one pass of cross-entropy plus 0.5 × Brier over 25.0M tokens of decision rows (inputs up to 4,096 tokens), learning rates LoRA 2e-5 / head 1e-4, seeds 20260926 and 20260928; each seed's checkpoint chosen on a held-out selection set (updates 3,561 and 2,230 of 3,561), then the exact uniform soup of the two adapters (one rank-64 adapter whose weight update is the mean of the two) and heads.
- **Data (56,969 rows, 25.0M tokens):** the strict Decision 2.0 base (6,416 rows: program-generated decision tasks plus the human labels of GoEmotions, SNLI, CLINC150 and BANKING77; the Cosmos QA and SQuAD 2.0 answerability families removed), the A6 Score arms (4,651 rows: project-generated Score tasks with 2 to 10 levels in English and Chinese, and the graded human ratings of KLUE STS, JGLUE JSTS, IBM ArgQ-30k and SAF in Korean, Japanese, English and German), and a 20.0M-token slice of the A7 typed curricula (45,902 rows, 80% of the training tokens, an equal token quota for each of 39 families): program-generated Decision 1.0 Stage 1–4 decision tasks (36,105 rows) plus the BANKING77 (5,115 rows) and CLINC150 (4,682 rows) intent labels. By tokens the data is 60% English, 38% Chinese and about 2% Korean, Japanese and German together; 66% Choice, 19% Noul and 14% Score. The human sources are CC BY (3.0 or 4.0) or CC BY-SA (3.0 or 4.0).
- **Where the A7 dose comes from:** the A7 typed curricula are the project's own Decision 1.0 decoder-family training corpora (the shared Choice and Score training data of the Sol, Nox, Eos, Lux and Kai 1.0 models), re-audited into separable parts. The program-generated tasks carry program-oracle labels that the 1.0 builders verified independently; BANKING77 (CC BY 4.0) and CLINC150 (CC BY 3.0) keep their publishers' labels. Whole groups that touched a protected evaluation inventory were set aside, and no formal evaluation panel, JevArena-C1 source or mlx-diag test split is in the training data. The previous revision trained on a 5.0M-token slice of the same curricula (14,755 rows); this one adds 31,147 rows.
- **What the A7 dose did:** against a control trained on the same number of tokens that repeats the previous revision's data, the new A7 content raised typed-reasoning accuracy on unseen formal families by 4.6 points (95% CI +3.1 to +6.1) and the JevArena v3 composite by 2.4 points (95% CI +0.1 to +5.9); it cannot be separated from the Choice-heavier type mix (66% Choice tokens against 54%). At that dose, rank 32 per seed instead of rank 8 added 5.8 typed points (95% CI +4.0 to +7.6); its v3 gain (+1.9, 95% CI −0.9 to +5.0) is not significant on its own. More updates on the previous revision's data alone added nothing (v3 +0.8, 95% CI −1.9 to +2.0).
- **Not used:** outputs of Jev or of any third-party decision model; every row trains on its gold label. Datasets keep their own licences ([attributions](ATTRIBUTIONS.md)).

### Limits

- **Level with AutoJev-27B.** Post-key v3 72.36 vs AutoJev-27B 72.13 (+0.23; 95% CI lower bound −1.60): not a significant difference, so the higher point estimate is not a lead. Typed accuracy is level (T 0.896 vs 0.887) and so is human transfer (H 0.584 vs 0.587; −0.002, 95% CI [−0.030, +0.066], not significant). Typed Choice is lower (753 vs 800 of 800 answers correct) and constraint competition is 0.882 vs 1.000; exception stack is level (0.843) and resource ledger is higher (0.860 vs 0.705).
- **Against the other 27B models** JevArena v3 is significantly higher than Eikos-27B (+3.07, 95% CI [+0.20, +7.75]) and Jebadiah-27B (+6.89, 95% CI [+4.33, +10.54]); human transfer is level with both (−0.003, 95% CI [−0.047, +0.071], and +0.006, 95% CI [−0.031, +0.065]).
- **Against the previous revision** (`c0dba600`, the rank-16 M3-A soup): post-key v3 +5.15 (95% CI [+2.19, +8.02]), typed accuracy +0.109 (95% CI [+0.085, +0.133]), human transfer +0.011 (95% CI [−0.037, +0.056], level), public 231 +5 (95% CI [−1, +11], level) and card-eligible mlx-diag +1.4 points (95% CI +0.3 to +2.5). Transfer macro-F1 is lower on TempoWiC (0.713 vs 0.769) and Reddit humor (0.569 vs 0.631).
- **JevBench public 231:** 203 correct (48 / 72 / 83): level with AutoJev-27B's 201 (+2, 95% CI [−6, +10]) and above Jebadiah-27B's 176 (+27, 95% CI [+16, +38]), but significantly below Eikos-27B's 212 (−9, 95% CI [−16, −2]; hard tier 83 vs 92), mostly in two hard-tier skills that neither JevArena nor this model's training covers: applying long policy documents with amendments and precedence (11 vs 15 of the 19 such items) and checking a quoted person's plausible but wrong conclusion against the evidence instead of adopting it (33 vs 38 of 46).
- **JevArena-C1 (human-labeled, post-key).** The JevArena v3 gain over the previous revision does not carry over to C1: 57.56 vs 57.33 (+0.23, 95% CI [−1.22, +1.70]), with Choice +1.79 (95% CI [−0.18, +3.78]), Noul −1.56 (95% CI [−3.65, +0.53]) and Score −1.16 (95% CI [−4.73, +2.32]), none significant. C1 tasks below the previous revision (macro-F1 × 100): setting_temporal_grounding 38.6 vs 45.4, span_is_event 49.2 vs 51.6, star_rating 47.8 vs 49.3, is_rapport 83.5 vs 84.9, find_truth 85.3 vs 86.3 and hallucination 84.1 vs 84.9; languages below it (accuracy, at least 20 items): Russian .477 vs .507 (300 items) and Arabic .852 vs .859 (555 items). The previous revision's sealed comparison with AutoJev-27B (level) and Eikos-27B (significantly ahead) has not been repeated for this revision.
- **Score levels.** On typed Score it uses all five levels (51 / 66 / 67 / 72 / 150 answers at levels 0–4 against 35 / 79 / 83 / 75 / 128 in the gold labels): the top level is still over-predicted, less than in the previous revision (182), and levels 1 and 2 are under-predicted.
- **Transfer tasks.** Wiki politeness, TempoWiC and Reddit humor are lower than all three 27B models shown (macro-F1 0.490 against 0.529–0.591, 0.713 against 0.736–0.798 and 0.569 against 0.607–0.645), and MRF by less than 0.01; flute and media ideology trail AutoJev-27B and Eikos-27B, and persuasion trails AutoJev-27B and Jebadiah-27B.
- **Long inputs** (over 4,000 characters): transfer accuracy 0.578, level with AutoJev-27B (0.576) and Eikos-27B (0.583); public 231 answers 26 of 37 (AutoJev-27B 28, Eikos-27B 29).
- **Multilingual.** On the mlx-diag diagnostic (public test splits in seven languages, English instructions over target-language states), non-English Choice is level with English (81.0% vs 81.7%) and non-English Noul (paraphrase pairs) is 4.0 points lower (86.0% vs 90.0%). The weakest are Korean Noul (80%, 100 items) and Spanish Choice (77.8%, 126 items). The three 27B peers ran the same diagnostic (Choice and Noul parts; no paired test): non-English Choice is 80.6% for AutoJev-27B, 78.4% for Eikos-27B and 77.6% for Jebadiah-27B (81.0% here), and non-English Noul is 88.7%, 85.0% and 85.3% (86.0% here); Korean Noul is 85%, 84% and 80% (80% here). The diagnostic's Score part, not shown because its source data are non-commercial, is weakest outside English.
- **Calibration.** The package returns raw probabilities (temperature 1); per-type temperatures fitted on the calibration partition were rejected because they worsened calibration on development data.
- **Input limit.** 32,768 tokens for the complete state, question and candidates; longer inputs return an over-budget error instead of being truncated. Training inputs were at most 4,096 tokens.
- **Seed soup.** Only the uniform soup of two seeds was evaluated on JevArena v3 and is released; a retrained seed of the same recipe can land below it.
- **Base download and memory.** The base weights are not in this repository; loading needs the pinned Qwen3.8-27B revision (about 55 GB of files) and one GPU with at least 122 GB of memory (the evaluation's peak on an 18,261-token input: 110 GB allocated, 121 GB reserved; longer inputs were not measured). CPU inference was not verified.
- **Evaluation familiarity.** A later screen of the program's training pools found topical-phrase overlap (no exact duplicates) between some training groups and 84 items of the reported evaluation panels. This model's training data contains none of those groups. Rescored without the 84 items, post-key v3 stays 72.36, the comparison bar (90% of AutoJev-27B) moves from 64.92 to 64.89, and none of this card's v3, human-transfer or public-231 comparisons change.
- JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not an untouched blind test.
- The v3 and public panels are almost entirely English; they do not measure multilingual ability.
- Complete inputs above 32,768 tokens are rejected, never truncated.
- It decides from the supplied state only and does not retrieve missing facts. Probabilities are estimates; review consequential decisions against the evidence.

[License](LICENSE) · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)
