---
license: apache-2.0
base_model: llm-semantic-router/Decision-1.0-Sol-2B
base_model_relation: finetune
tags:
- decision-model
- classification
- system-one
- safetensors
---

![DEV2.0-2B owl banner](assets/DEV2.0-2B-owl-banner.png)

# DEV2.0-2B

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

On the same 8,147-item JevArena v3 panel, DEV2.0-2B scores **53.44** versus **45.78** for Decision 1.0 Sol (+7.66; paired 95% interval [+3.26, +10.81]). On the separate 231 public JevBench questions it answers **171/231** correctly versus **160/231**. It ranks 1 of 5 models shown on JevArena v3.

> **Independent sealed confirmation (JevArena-C1): PLACEHOLDER — to be added after the final sealed scoring event.**

| Rank | Model | Parameters | JevArena v3 ↑ | T | H | Choice | Noul | Score | JevBench public 231 (easy / standard / hard) ↑ | mlx-diag non-English Choice / Noul ↑ | Typed Brier / ECE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **DEV2.0-2B** | 1.88B | **53.44** | 0.543 | 0.525 | 445/800 | 567/800 | 175/400 | 171/231 (48 / 66 / 57) | 66.5 / 65.8 | 0.271 / 0.103 |
| 2 | Decider 2B | 1.88B | 49.50 | 0.583 | 0.420 | 545/800 | 497/800 | 133/400 | 175/231 (48 / 64 / 63) | 75.4 / 66.0 | 0.259 / 0.063 |
| 3 | This-That 1.2 | 1.88B | 46.11 | 0.525 | 0.405 | 529/800 | 431/800 | 79/400 | 147/231 (48 / 64 / 35) | 76.2 / 66.5 | 0.344 / 0.260 |
| 4 | Decision 1.0 Sol | 1.88B | 45.78 | 0.425 | 0.493 | 374/800 | 438/800 | 155/400 | 160/231 (48 / 66 / 46) | 67.1 / 68.0 | 0.345 / 0.249 |
| 5 | Bosun v3.1 1.7B | 1.74B | 42.12 | 0.467 | 0.380 | 415/800 | 493/800 | 99/400 | 151/231 (46 / 62 / 43) | 70.0 / 70.2 | 0.312 / 0.127 |

![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)

![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)

![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)

Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and over-budget answers count as failures. JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not a blind test. T is typed-decision accuracy (four-family macro), H the median macro-F1 over 15 human-labeled transfer tasks, and v3 = 100 × sqrt(T × H). JevBench public 231 is our rerun of the 231 public questions; it is not the official sealed JevBench rank. mlx-diag is a multilingual development diagnostic (public test splits in seven languages, English instructions over target-language states), not a release score; its Score part is not shown. Ranks include only the models shown. Decision 1.0 Sol is shown from its stricter same-renderer run at this model's 16,384-token limit (JevArena v3 45.78; its adopted run scores 45.58). Decider 2B (Apache-2.0) and This-That 1.2 (MIT) declare their licences in model card metadata only; their repositories have no LICENSE file. [Methods and per-model results](evaluation/EVALUATION.md)

### Tradeoffs versus Decision 1.0 Sol

Results below Decision 1.0 Sol on this panel:

| Result | DEV2.0-2B | Decision 1.0 Sol |
| --- | ---: | ---: |
| mlx-diag non-English Choice (accuracy) | 66.5% | 67.1% |
| mlx-diag non-English Noul (accuracy) | 65.8% | 68.0% |
| Transfer: mrf (macro-F1) | 53.7% | 63.1% |
| Transfer: wiki corpus (macro-F1) | 45.0% | 49.5% |
| Transfer: flute (macro-F1) | 62.8% | 66.1% |
| Transfer: ibc (macro-F1) | 40.1% | 42.7% |
| Transfer: wiki politeness (macro-F1) | 55.0% | 57.4% |
| Transfer: persuasion (macro-F1) | 57.2% | 58.0% |
| Transfer: talklife (macro-F1) | 31.9% | 32.0% |
| Transfer: conv go awry (macro-F1) | 44.1% | 44.3% |

## Download and decide

```bash
hf download llm-semantic-router/DEV2.0-2B --local-dir DEV2.0-2B
```

```python
import json
import os
import sys

sys.path.insert(0, os.path.abspath("DEV2.0-2B"))
from decision2 import Decision2

model = Decision2.from_pretrained("DEV2.0-2B")  # cuda:0 if a GPU is visible, else CPU
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

- **Architecture:** The 24-layer Qwen3.5 text backbone inherited from Decision 1.0 Sol (18 gated-delta and 6 attention layers) reads the state, the question and every supplied candidate once; a shared candidate head scores the candidates against a global query and returns probabilities without generating text.
- **Parameters:** 1,883,930,944 loaded (backbone 1,881,825,088; decision head 2,105,856).
- **Direct weight origin:** [llm-semantic-router/Decision-1.0-Sol-2B](https://huggingface.co/llm-semantic-router/Decision-1.0-Sol-2B) at `ce0c018a28de16d6639b1cd203b761bf643b89e6`. Every weight of Decision 1.0 Sol was fine-tuned (nothing frozen, no adapter), with Decision 1.0 Sol's own answer probabilities as soft targets, and the release is the uniform average of three seeds of that fine-tune. Decision 1.0 Sol is itself a text-only fine-tune of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) at `15852e8c16360a2fea060d615a32b45270f8a8fc` (Apache-2.0), whose text backbone and tokenizer this model inherits; the Qwen3.5 vision tower is not part of it.
- **Input limit:** 16,384 tokens for the complete state, question and candidates.
- **Calibration:** raw model probabilities (temperature 1). Per-type temperatures fitted on the clean 698-item calibration partition (Choice 0.742, Noul 0.816, Score 0.154) were evaluated and not adopted because they worsened calibration on out-of-distribution development data.
- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 of every file and the runtime verifies them before loading.
- **Weights:** the uniform average of three full fine-tunes of Decision 1.0 Sol with the same recipe and start; the seeds differ only in data order, and each seed's checkpoint was chosen on a held-out selection set (updates 366, 741 and 638).
- **Comparator licences:** Decider 2B's model card declares Apache-2.0 and This-That 1.2's declares MIT (it is adapted from Decider 2B); neither repository has a LICENSE file. Bosun v3.1 1.7B ships Apache-2.0 LICENSE and NOTICE files.

### Training

- **Recipe:** one epoch of full fine-tuning of every weight of Decision 1.0 Sol (backbone learning rate 5e-6, head 5e-5; cross-entropy plus 0.5 × Brier plus 0.5 × KL divergence to Decision 1.0 Sol's own answer probabilities on every row, as a trust region) on 56,198 decision rows (29.2M tokens, inputs up to 8,192 tokens) with seeds 20260926, 20260927 and 20260928, each seed's checkpoint chosen on a held-out selection set, then a uniform weight average of the three.
- **Teacher:** Decision 1.0 Sol only; its own probabilities are the soft targets. No outputs of Jev or of any third-party decision model were used.
- **Own Decision 1.0 training corpora (8,276 rows, a retention replay):** program-generated English and Chinese decision tasks labeled by program oracles (no model-written states or labels), plus the human labels of CLINC150 (CC BY 3.0), BANKING77 (CC BY 4.0) and MultiNLI non-fiction genres (OANC terms).
- **Decision 2.0 data (47,922 rows, 20 languages):** project-generated verifiable rules, counterfactuals, hard negatives and multi-level Score tasks, plus the human labels of public datasets: reading comprehension and multi-hop evidence (TyDi QA, SQuAD 2.0, QuAC, HotpotQA, 2WikiMultihopQA, MuSiQue, DRCD, CMRC 2018, JSQuAD, KLUE MRC, SQAC, GermanQuAD, PIAF), commonsense and math (WinoGrande, CommonsenseQA, GSM8K, SciTail, ROPES, QuaRTz), dialogue and intent (MultiWOZ 2.2, Taskmaster-2, MTOP, CLINC150, BANKING77), topic, emotion and inference labels (DBpedia-14, GoEmotions, SNLI), and graded similarity and quality ratings (KLUE STS, JGLUE JSTS, MLQE-PE, IBM ArgQ-30k, SAF, OneStopEnglish). Most reading sources are CC BY-SA (3.0 or 4.0, or Wikipedia text under CC BY-SA); the others are CC BY, MIT, Apache-2.0 or CC0.
- **Not used:** outputs of Jev or of any third-party decision model; the Cosmos QA and SQuAD 2.0 answerability labels that Decision 1.0 Sol was trained on (two shortcut-prone reading families; SQuAD 2.0 enters only through project-built evidence-removal pairs); and the MASSIVE, PAWS-X and XNLI sources behind the multilingual diagnostic. Datasets keep their own licences ([attributions](ATTRIBUTIONS.md)).

### Limits

- **Seed dependence.** The three single fine-tunes differ on the development panels (development proxy 46.2, 42.6 and 48.1; 47.1 for their weight average); only the uniform average was evaluated on JevArena v3 and is released. A retrained seed of the same recipe can land below this model.
- **Where it improves over Decision 1.0 Sol and where it does not.** The gain is typed reasoning (T 0.543 vs 0.425; Choice 445 vs 374 of 800, Noul 567 vs 438 of 800, Score 175 vs 155 of 400); human transfer is level (H 0.525 vs 0.493, +0.033, 95% CI [−0.044, +0.092]), and several transfer tasks are lower, most of all mrf (−0.094), wiki_corpus (−0.045) and flute (−0.032); the tradeoffs table lists every task below Decision 1.0 Sol.
- **Against Decider 2B** the JevArena v3 margin is not significant (+3.94, 95% CI [−2.48, +5.67]). It comes from human transfer (+0.105, 95% CI [−0.008, +0.132], itself not significant), while typed accuracy is significantly lower (−0.040, 95% CI [−0.070, −0.009]). Choice is the weakest type relative to the 2B peers (typed Choice accuracy 0.556 vs 0.681 for Decider 2B and 0.661 for This-That 1.2), and the hard tier of JevBench public 231 is lower (57 vs 63).
- **Score levels.** On typed Score it almost never predicts the lowest level (6 of 400 predictions; recall 0.03 on level 0), which Decision 1.0 Sol does predict (recall 0.29).
- **Multilingual.** On the mlx-diag diagnostic, non-English Choice is level with Decision 1.0 Sol (66.5% vs 67.1%), non-English Noul (paraphrase pairs) is about 2 points lower (65.8% vs 68.0%), and Korean Noul is the weakest language (59% vs 63%, 100 items).
- **Evaluation familiarity.** A stricter overlap screen run after training (covering 98% of the training rows; the rest are program-generated) found 45 training rows in 18 reading-passage groups (from MuSiQue, SQuAD 2.0, QuAC, HotpotQA and CommonsenseQA) that share long word sequences with items of the 15-task human-transfer panel; current data builds exclude such rows. No training row matched the typed panels, JevBench public 231 or the mlx-diag diagnostic. Those rows touch 19 of the panel's items (18 media_ideology, 1 tropes); rescored without them, or counting all 19 as errors, post-key v3 stays 53.44 and none of this card's comparisons change.
- **CPU.** CPU inference was not verified for this release: where the GPU-only causal-conv1d package is installed, Transformers routes the convolution to that kernel and a CPU run fails.
- JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not an untouched blind test.
- The v3 and public panels are almost entirely English; they do not measure multilingual ability.
- Complete inputs above 16,384 tokens are rejected, never truncated; 4 evaluation answers were invalid (including over-budget inputs) and counted as failures.
- It decides from the supplied state only and does not retrieve missing facts. Probabilities are estimates; review consequential decisions against the evidence.

[License](LICENSE) · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)
