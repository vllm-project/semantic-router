---
license: apache-2.0
base_model: llm-semantic-router/Decision-1.0-Nox-4B
base_model_relation: finetune
tags:
- decision-model
- classification
- system-one
- safetensors
---

![DEV2.0-4B owl banner](assets/DEV2.0-4B-owl-banner.png)

# DEV2.0-4B

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

On the same 8,147-item JevArena v3 panel, DEV2.0-4B scores **63.15** versus **56.47** for Decision 1.0 Nox (+6.68; paired 95% interval [+0.99, +9.64]). On the separate 231 public JevBench questions it answers **171/231** correctly versus **173/231**. It ranks 1 of 4 models shown on JevArena v3.

> **Independent sealed confirmation (JevArena-C1): PLACEHOLDER — to be added after the final sealed scoring event.**

| Rank | Model | Parameters | JevArena v3 ↑ | T | H | Choice | Noul | Score | JevBench public 231 (easy / standard / hard) ↑ | mlx-diag non-English Choice / Noul ↑ | Typed Brier / ECE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **DEV2.0-4B** | 4.21B | **63.15** | 0.688 | 0.580 | 582/800 | 734/800 | 185/400 | 171/231 (48 / 67 / 56) | 75.1 / 72.7 | 0.175 / 0.116 |
| 2 | Decider 4B | 4.21B | 61.88 | 0.689 | 0.555 | 750/800 | 623/800 | 114/400 | 192/231 (48 / 71 / 73) | 79.4 / 75.3 | 0.154 / 0.056 |
| 3 | Jet v6.2 | 4.21B | 60.37 | 0.678 | 0.538 | 720/800 | 640/800 | 125/400 | 174/231 (48 / 69 / 57) | 77.5 / 77.8 | 0.171 / 0.077 |
| 4 | Decision 1.0 Nox | 4.21B | 56.47 | 0.614 | 0.519 | 552/800 | 653/800 | 178/400 | 173/231 (48 / 66 / 59) | 75.9 / 80.0 | 0.205 / 0.092 |

![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)

![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)

![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)

Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and over-budget answers count as failures. JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not a blind test. T is typed-decision accuracy (four-family macro), H the median macro-F1 over 15 human-labeled transfer tasks, and v3 = 100 × sqrt(T × H). JevBench public 231 is our rerun of the 231 public questions; it is not the official sealed JevBench rank. mlx-diag is a multilingual development diagnostic (public test splits in seven languages, English instructions over target-language states), not a release score; its Score part is not shown. Ranks include only the models shown. Decision 1.0 Nox is shown from its adopted run (JevArena v3 56.47), the comparator of this size's release bar; its same-renderer run at this model's 16,384-token limit scores 55.69. Decider 4B (Apache-2.0) declares its licence in model card metadata only; its repository has no LICENSE file. [Methods and per-model results](evaluation/EVALUATION.md)

### Tradeoffs versus Decision 1.0 Nox

Results below Decision 1.0 Nox on this panel:

| Result | DEV2.0-4B | Decision 1.0 Nox |
| --- | ---: | ---: |
| mlx-diag non-English Choice (accuracy) | 75.1% | 75.9% |
| mlx-diag non-English Noul (accuracy) | 72.7% | 80.0% |
| Transfer: wiki corpus (macro-F1) | 45.4% | 51.9% |
| Transfer: mrf (macro-F1) | 71.5% | 76.1% |
| Transfer: media ideology (macro-F1) | 42.6% | 44.3% |
| Transfer: talklife (macro-F1) | 29.5% | 29.6% |
| JevBench public hard (correct) | 56/111 | 59/111 |

## Download and decide

```bash
hf download llm-semantic-router/DEV2.0-4B --local-dir DEV2.0-4B
```

```python
import json
import os
import sys

sys.path.insert(0, os.path.abspath("DEV2.0-4B"))
from decision2 import Decision2

model = Decision2.from_pretrained("DEV2.0-4B")  # cuda:0 if a GPU is visible, else CPU
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

- **Architecture:** The 32-layer Qwen3.5 text backbone inherited from Decision 1.0 Nox (24 gated-delta and 8 attention layers) reads the state, the question and every supplied candidate once; a shared candidate head scores the candidates against a global query and returns probabilities without generating text.
- **Parameters:** 4,208,383,488 loaded (backbone 4,205,751,296; decision head 2,632,192).
- **Direct weight origin:** [llm-semantic-router/Decision-1.0-Nox-4B](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B) at `cde2a68dbaa557ea65dc458104d410a0802ee259`. Every weight of Decision 1.0 Nox was fine-tuned (nothing frozen, no adapter), with the answer probabilities of our own Decision 1.0 Lux as soft targets, and the release is the uniform average of three seeds of that fine-tune. Decision 1.0 Nox is itself a text-only fine-tune of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (Apache-2.0), whose text backbone and tokenizer this model inherits; the Qwen3.5 vision tower is not part of it.
- **Input limit:** 16,384 tokens for the complete state, question and candidates.
- **Calibration:** raw model probabilities (temperature 1). Per-type temperatures fitted on the clean 698-item calibration partition (Choice 0.667, Noul 0.496, Score 0.212) were evaluated on out-of-distribution development data and not adopted: they lowered the Brier score there but raised the calibration error of the human-transfer development pilot, and temperatures are adopted only if no development calibration measure worsens.
- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 of every file and the runtime verifies them before loading.
- **Weights:** the uniform average of three full fine-tunes of Decision 1.0 Nox that share the recipe and the start and differ in seed; each seed's checkpoint was chosen on a held-out selection set (updates 690, 676 and 492).
- **Comparator licences:** Decider 4B's model card declares Apache-2.0 and its repository has no LICENSE file; Jet v6.2 ships an Apache-2.0 LICENSE file and was measured with its bundled runtime on ROCm, a platform its card does not list.

### Training

- **Recipe:** one epoch of full fine-tuning of every weight of Decision 1.0 Nox (backbone learning rate 5e-6, head 5e-5; AdamW, weight decay 0.01, clip 1.0, 5% warmup then cosine to 10%; cross-entropy plus 0.5 × Brier plus 1.0 × KL divergence to Decision 1.0 Lux's answer probabilities on every row except the 3,759 gold-only long-evidence and multilingual rows) on 58,742 decision rows (29.4M tokens, inputs up to 8,192 tokens) with seeds 20260926, 20260927 and 20260928, each seed's checkpoint chosen on a held-out selection set, then a uniform weight average of the three.
- **Teacher:** Decision 1.0 Lux-9B only (our own model, [`llm-semantic-router/Decision-1.0-Lux-9B`](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) at `bd45a30a`); its answer probabilities, computed on a runtime without the causal-conv1d kernel, are the soft targets on 54,983 rows. The 3,759 long-evidence and multilingual rows (HoVer, Natural Questions, TyDi QA, MIRACL, JCommonsenseQA, SentiMix Spanglish) train on their gold labels only. No outputs of Jev or of any third-party decision model were used.
- **Own Decision 1.0 training corpora (17,663 rows, 25 languages):** program-generated English and Chinese decision tasks labeled by program oracles (8,038 rows; no model-written states or labels), plus the human labels of CLINC150 (CC BY 3.0), BANKING77 (CC BY 4.0), MultiNLI non-fiction genres (OANC terms) and SNLI (CC BY-SA 4.0), and, rebuilt from their upstream files with human labels only, OpenAssistant OASST1 reply ratings in 23 languages (Apache-2.0), KLUE STS and JGLUE JSTS similarity (CC BY-SA 4.0), and SentiMix Hinglish and AfriSenti Swahili sentiment (CC BY 4.0).
- **Decision 2.0 data (41,079 rows, 23 languages):** project-generated verifiable rules, counterfactuals, hard negatives and multi-level Score tasks, plus the human labels of public datasets: reading comprehension, retrieval and multi-hop evidence (TyDi QA, MIRACL, SQuAD 2.0, QuAC, Natural Questions, HotpotQA, 2WikiMultihopQA, MuSiQue, HoVer, DRCD, CMRC 2018, JSQuAD, KLUE MRC, SQAC, GermanQuAD, PIAF), commonsense and math (WinoGrande, CommonsenseQA, JCommonsenseQA, GSM8K, SciTail, ROPES, QuaRTz), dialogue and intent (MultiWOZ 2.2, Taskmaster-2, Schema-Guided Dialogue, ABCD, MTOP, CLINC150, BANKING77), topic, emotion, sentiment and inference labels (DBpedia-14, KLUE YNAT, GoEmotions, SentiMix Spanglish, SNLI, JNLI), and graded similarity and quality ratings (KLUE STS, JGLUE JSTS, MLQE-PE, IBM ArgQ-30k, SAF, OneStopEnglish). Most reading sources are CC BY-SA (3.0 or 4.0, or Wikipedia text under CC BY-SA); the others are CC BY, MIT, Apache-2.0 or CC0.
- **Not used:** outputs of Jev or of any third-party decision model; the Cosmos QA and SQuAD 2.0 answerability labels that Decision 1.0 Nox was trained on (two shortcut-prone reading families; SQuAD 2.0 enters only through project-built evidence-removal pairs); and the MASSIVE, PAWS-X and XNLI sources behind the multilingual diagnostic. Datasets keep their own licences ([attributions](ATTRIBUTIONS.md)).

### Limits

- **Seed dependence.** The three single fine-tunes differ on the development panels (development proxy 62.6, 60.2 and 63.3; 62.9 for their weight average); only the uniform average was evaluated on JevArena v3 and is released. A retrained seed of the same recipe can land below this model.
- **Where it improves over Decision 1.0 Nox and where it does not.** The gain is typed reasoning (T 0.688 vs 0.614, +0.074, 95% CI [+0.051, +0.096]; Choice 582 vs 552 of 800, Noul 734 vs 653 of 800, Score 185 vs 178 of 400). Human transfer is higher but not significantly (H 0.580 vs 0.519, +0.061, 95% CI [−0.037, +0.113]; 11 of 15 tasks up), and four transfer tasks are lower: wiki_corpus (−0.065), mrf (−0.045), media_ideology (−0.016) and talklife (−0.001), as the tradeoffs table lists. JevBench public 231 is level (171 vs 173; hard tier 56 vs 59), and so are inputs over 4,000 characters (public 231: 16 vs 17 of 37; human-transfer panel: 41.1% vs 41.5%).
- **Against the 4B peers** the JevArena v3 margins are not significant (Decider 4B +1.27, 95% CI [−5.71, +4.31]; Jet v6.2 +2.78, 95% CI [−3.33, +5.94]); typed accuracy and human transfer are level with Decider 4B (T −0.001, 95% CI [−0.030, +0.029]; H +0.024, 95% CI [−0.095, +0.073]). Choice is the weakest type relative to the peers (typed Choice accuracy 0.728 vs 0.938 for Decider 4B and 0.900 for Jet v6.2). JevBench public 231 is significantly below Decider 4B (171 vs 192; −21 items, 95% CI [−32, −10]; hard tier 56 vs 73) and level with Jet v6.2 (174). Long inputs are weaker than both peers (inputs over 4,000 characters: public 231 16 of 37 vs 25 and 19; human-transfer panel 41.1% vs 52.3% and 45.4%).
- **Score levels.** Like Decision 1.0 Nox, it rarely predicts the lowest typed Score level (4 of 400 predictions; recall 0.11 on level 0, Nox 0.14), and its level-2 recall is lower (0.19 vs 0.30).
- **Multilingual.** On the mlx-diag diagnostic, non-English Choice is level with Decision 1.0 Nox (75.1% vs 75.9%), while non-English Noul (paraphrase pairs) is 7.3 points lower (72.7% vs 80.0%) and the lowest of the four models shown (Decider 4B 75.3%, Jet v6.2 77.8%); Korean and Japanese Noul are the weakest languages (63% vs 69% and 67% vs 77%, 100 items each).
- **Calibration.** At temperature 1 the typed-panel calibration error (ECE 0.116) is higher than Decision 1.0 Nox's (0.092) and the peers' (0.056 and 0.077), although its Brier score is lower than Nox's (0.175 vs 0.205).
- **Evaluation familiarity.** A later screen of the program's training pools found topical-phrase overlap (no exact duplicates) between some training groups and 84 items of the reported evaluation panels. This model's training data was built after that screen and contains none of those groups. Rescored without the 84 items, post-key v3 stays 63.15 and none of this card's v3 or human-transfer comparisons change; the media_ideology gap to Decision 1.0 Nox moves from −0.016 to +0.006, both within noise.
- **CPU.** CPU inference was not verified for this release: where the GPU-only causal-conv1d package is installed, Transformers routes the convolution to that kernel and a CPU run fails.
- JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not an untouched blind test.
- The v3 and public panels are almost entirely English; they do not measure multilingual ability.
- Complete inputs above 16,384 tokens are rejected, never truncated; 4 evaluation answers were invalid (including over-budget inputs) and counted as failures.
- It decides from the supplied state only and does not retrieve missing facts. Probabilities are estimates; review consequential decisions against the evidence.

[License](LICENSE) · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)
