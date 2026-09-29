---
license: apache-2.0
base_model: llm-semantic-router/Decision-1.0-Lux-9B
base_model_relation: finetune
tags:
- decision-model
- classification
- system-one
- safetensors
---

![DEV2.0-9B owl banner](assets/DEV2.0-9B-owl-banner.png)

# DEV2.0-9B

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

On the same 8,147-item JevArena v3 panel, DEV2.0-9B scores **67.74** versus **65.81** for Decision 1.0 Lux (+1.93; paired 95% interval [+0.61, +4.14]). On the separate 231 public JevBench questions it answers **178/231** correctly versus **183/231**. It ranks 1 of 3 models shown on JevArena v3.

> **Independent sealed confirmation** — JevArena-C1 v1.2 (2,840 human-labeled items from 8 sources published after the relevant cutoffs, never used for training or development; scored once): DEV2.0-9B **53.77** vs Decision 1.0 Lux 51.97 (+1.80, 95% CI [+0.55, +3.07]); Nimble v2 52.77 (+1.00, 95% CI [−0.69, +2.75]). Measured on the frozen package `53bac735` (weights identical to this revision); items Nimble v2 cannot answer at its 8,192-token limit (58 of 2,840) count as wrong. It is significantly above Decision 1.0 Lux and level with Nimble v2.

| Rank | Model | Parameters | JevArena v3 ↑ | T | H | Choice | Noul | Score | JevBench public 231 (easy / standard / hard) ↑ | mlx-diag non-English Choice / Noul ↑ | Typed Brier / ECE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **DEV2.0-9B** | 7.94B | **67.74** | 0.811 | 0.566 | 736/800 | 715/800 | 246/400 | 178/231 (48 / 66 / 64) | 79.4 / 80.8 | 0.100 / 0.016 |
| 2 | Decision 1.0 Lux | 7.94B | 65.81 | 0.776 | 0.558 | 711/800 | 704/800 | 227/400 | 183/231 (48 / 67 / 68) | 78.3 / 83.3 | 0.119 / 0.017 |
| 3 | Nimble v2 | 9.45B | 62.06 | 0.719 | 0.536 | 784/800 | 654/800 | 107/400 | 185/231 (48 / 68 / 69) | 75.0 / 73.5 | 0.138 / 0.061 |

![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)

![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)

![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)

Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and over-budget answers count as failures. JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not a blind test. T is typed-decision accuracy (four-family macro), H the median macro-F1 over 15 human-labeled transfer tasks, and v3 = 100 × sqrt(T × H). JevBench public 231: public-only rerun (about a third of the official Intelligence inputs), not the official JevBench score; easy tier at ceiling; totals within about 10 items are not distinguishable. mlx-diag is a multilingual development diagnostic (public test splits in seven languages, English instructions over target-language states), not a release score; its Score part is not shown. Ranks include only the models shown. Decision 1.0 Lux is shown from its native run at this model's 16,384-token limit (JevArena v3 65.81), the comparator of this size's release bar; run through this model's own renderer at the same limit it scores 65.23. Nimble v2 (a LoRA over Qwen3.5-9B; Apache-2.0 with a LICENSE file) ran with its bundled scorer on ROCm, a platform its card does not list, at its native 8,192-token limit (21 human-transfer answers over that limit count as failures); its parameter count includes its vision tower and the unmerged LoRA. [Methods and per-model results](evaluation/EVALUATION.md)

### Tradeoffs versus Decision 1.0 Lux

Results below Decision 1.0 Lux on this panel:

| Result | DEV2.0-9B | Decision 1.0 Lux |
| --- | ---: | ---: |
| mlx-diag non-English Noul (accuracy) | 80.8% | 83.3% |
| Transfer: talklife (macro-F1) | 24.3% | 26.6% |
| Transfer: wiki corpus (macro-F1) | 56.3% | 58.4% |
| Transfer: tropes (macro-F1) | 11.8% | 12.8% |
| Transfer: conv go awry (macro-F1) | 55.1% | 55.2% |
| JevBench public standard (correct) | 66/72 | 67/72 |
| JevBench public hard (correct) | 64/111 | 68/111 |

## Download and decide

```bash
hf download llm-semantic-router/DEV2.0-9B --local-dir DEV2.0-9B
```

```python
import json
import os
import sys

sys.path.insert(0, os.path.abspath("DEV2.0-9B"))
from decision2 import Decision2

model = Decision2.from_pretrained("DEV2.0-9B")  # cuda:0 if a GPU is visible, else CPU
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

- **Architecture:** The 32-layer Qwen3.5 text backbone inherited from Decision 1.0 Lux (24 gated-delta and 8 attention layers) reads the state, the question and every supplied candidate once; a shared candidate head scores the candidates against a global query and returns probabilities without generating text.
- **Parameters:** 7,940,895,744 loaded (backbone 7,936,684,544; decision head 4,211,200).
- **Name:** Named after its base model ([Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)); it loads 7,940,895,744 parameters.
- **Direct weight origin:** [llm-semantic-router/Decision-1.0-Lux-9B](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) at `bd45a30aee8c84032791c245c70f86dee5389cc8`. Every weight of Decision 1.0 Lux was fine-tuned (nothing frozen, no adapter), with Decision 1.0 Lux's own answer probabilities as soft targets on every training row; three seeds of that fine-tune were averaged, and the release is one third of that average plus two thirds of the original Decision 1.0 Lux weights (a weight interpolation back toward Decision 1.0 Lux). Decision 1.0 Lux is itself a text-only decision adaptation of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Apache-2.0), whose text backbone and tokenizer this model inherits; the Qwen3.5 vision tower is not part of it.
- **Input limit:** 16,384 tokens for the complete state, question and candidates.
- **Calibration:** raw model probabilities (temperature 1). Per-type temperatures fitted on the clean 698-item calibration partition (Choice 1.420, Noul 1.037, Score 0.563) were evaluated on out-of-distribution development data and not adopted: they lowered the calibration error there but raised the Brier score of both development panels, and temperatures are adopted only if no development calibration measure worsens.
- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 of every file and the runtime verifies them before loading.
- **Size:** the model is the Qwen3.5 text backbone with a decision head; the Qwen3.5 vision tower is not part of it.
- **Weights:** one third of the uniform average of three full fine-tunes of Decision 1.0 Lux that share the recipe and the start and differ in seed (each seed's checkpoint chosen on a held-out selection set: updates 1,624, 1,420 and 1,424) plus two thirds of Decision 1.0 Lux.
- **Storage:** the 248 projection matrices are stored in BF16, the precision the GPU runtime multiplies them in, and every other tensor (embedding table, norms, gated-delta parameters, convolution filters, decision head) in FP32; on every evaluated prompt the answers equal those of the evaluated all-FP32 checkpoint (largest probability difference below 1e-15).

### Training

- **Recipe:** one epoch of full fine-tuning of every weight of Decision 1.0 Lux (backbone learning rate 1e-5, head 1e-4; weight decay 0.01, 5% warmup; cross-entropy plus 0.5 × Brier plus 1.0 × KL divergence to Decision 1.0 Lux's own answer probabilities on every row) on 122,651 decision rows (60.2M tokens, 35 languages, English 55% of rows, inputs up to 8,192 tokens) with seeds 20260926, 1 and 2, each seed's checkpoint chosen on a held-out selection set; the three fine-tunes were averaged uniformly, and the released weights are one third of that average plus two thirds of Decision 1.0 Lux, the interpolation weight chosen by a preregistered rule on development panels.
- **Teacher:** Decision 1.0 Lux-9B only, the model this one starts from (our own model, [`llm-semantic-router/Decision-1.0-Lux-9B`](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) at `bd45a30a`); its answer probabilities, mostly computed on a runtime without the causal-conv1d kernel, are the soft targets on all 122,651 rows, including the long-evidence and multilingual human-label rows. No outputs of Jev or of any third-party decision model were used.
- **Own Decision 1.0 training corpora (42,511 rows):** program-generated English and Chinese decision tasks labeled by program oracles (18,431 rows; no model-written states or labels), plus the human labels of CLINC150 (CC BY 3.0), BANKING77 (CC BY 4.0), MultiNLI non-fiction genres (OANC terms) and SNLI (CC BY-SA 4.0), and, rebuilt from their upstream files with human labels only, OpenAssistant OASST1 reply ratings (Apache-2.0), KLUE STS and JGLUE JSTS similarity (CC BY-SA 4.0), and SentiMix Hinglish and AfriSenti Swahili sentiment (CC BY 4.0).
- **Decision 2.0 data (80,140 rows):** project-generated verifiable rules, counterfactuals, hard negatives and multi-level Score tasks, plus the human labels of public datasets: reading comprehension, retrieval and multi-hop evidence (TyDi QA, MIRACL, SQuAD 2.0, QuAC, Natural Questions, HotpotQA, 2WikiMultihopQA, MuSiQue, HoVer, DRCD, CMRC 2018, JSQuAD, KLUE MRC, SQAC, GermanQuAD, PIAF), commonsense and math (WinoGrande, CommonsenseQA, JCommonsenseQA, GSM8K, SciTail, ROPES, QuaRTz), dialogue and intent (MultiWOZ 2.2, Taskmaster-2, Schema-Guided Dialogue, ABCD, MTOP, CLINC150, BANKING77), topic, emotion, sentiment and inference labels (DBpedia-14, KLUE YNAT, GoEmotions, SentiMix Spanglish, SNLI, JNLI), and graded similarity and quality ratings (KLUE STS, JGLUE JSTS, MLQE-PE, IBM ArgQ-30k, SAF, OneStopEnglish). Most reading sources are CC BY-SA (3.0 or 4.0, or Wikipedia text under CC BY-SA); the others are CC BY, MIT, Apache-2.0 or CC0. The human-label arms among them are cross-domain short tasks (WinoGrande, GSM8K, CommonsenseQA, SciTail, MultiWOZ 2.2, DBpedia-14, Taskmaster-2, ROPES, QuaRTz), long evidence (HoVer, Natural Questions page windows) and multilingual tasks (JCommonsenseQA, MIRACL, SentiMix Spanglish, TyDi QA); like every other row they carry both the gold label and Decision 1.0 Lux's probabilities.
- **Not used:** outputs of Jev or of any third-party decision model; the Cosmos QA and SQuAD 2.0 answerability labels that Decision 1.0 Lux was trained on (two shortcut-prone reading families; SQuAD 2.0 enters only through project-built evidence-removal pairs); and the MASSIVE, PAWS-X and XNLI sources behind the multilingual diagnostic. Datasets keep their own licences ([attributions](ATTRIBUTIONS.md)).

### Limits

- **Where it improves over Decision 1.0 Lux and where it does not.** The gain is typed reasoning (T 0.811 vs 0.776, +0.034, 95% CI [+0.015, +0.054]; Choice 736 vs 711 of 800, Noul 715 vs 704 of 800, Score 246 vs 227 of 400). Human transfer is level (H 0.566 vs 0.558, +0.008, 95% CI [−0.011, +0.042]; 11 of 15 tasks up), and four transfer tasks are lower: talklife (−0.022), wiki_corpus (−0.021), tropes (−0.011) and conv_go_awry (−0.001), as the tradeoffs table lists.
- **JevBench public 231 and long inputs.** JevBench public 231 is 5 items below Decision 1.0 Lux, within noise (178 vs 183; 95% CI [−11, +1]; hard tier 64 vs 68 of 111, standard tier 66 vs 67 of 72), and so are its inputs over 4,000 characters (19 vs 22 of 37; −3, 95% CI [−7, +1]); on the human-transfer panel long inputs are level (265 vs 262 of 509; +3, 95% CI [−10, +16]).
- **Against Nimble v2** JevArena v3 is higher (+5.68, 95% CI [+3.19, +10.09]) through typed accuracy (T +0.092, 95% CI [+0.063, +0.121]); human transfer is not significantly different (H +0.030, 95% CI [−0.007, +0.102]). Typed Choice is lower (0.920 vs 0.980), and JevBench public 231 is level (178 vs 185; −7 items, 95% CI [−18, +4]; inputs over 4,000 characters 19 vs 23 of 37).
- **Weight origin and seeds.** Each full fine-tune on its own scored below Decision 1.0 Lux on the development panels (development proxy 65.8, 65.1 and 61.5 vs 70.8; a development-only measure, not comparable with JevArena v3), and so did their uniform average (67.2); the gain comes from interpolating one third of that average back into Decision 1.0 Lux (73.2). Only this interpolation of the recipe was evaluated on JevArena v3; a retrained seed can land elsewhere.
- **Multilingual.** On the mlx-diag diagnostic, non-English Choice is level with Decision 1.0 Lux (79.4% vs 78.3%), while non-English Noul (paraphrase pairs) is 2.5 points lower (80.8% vs 83.3%; −2.5, 95% CI [−4.7, −0.3]); Korean Noul is the weakest language and the largest drop (70% vs 77%, 100 items), followed by Spanish (84% vs 88%), and English Noul is also lower (84% vs 88%).
- **Score levels.** It uses all five typed Score levels but predicts the lowest less often than Decision 1.0 Lux (42 vs 63 of 400 predictions; recall 0.51 vs 0.69 on the 35 level-0 items) and under-predicts the highest (104 predictions for 128 level-4 items); recall is higher on levels 1–4.
- **Calibration.** At temperature 1 the typed-panel Brier score is lower than Decision 1.0 Lux's (0.100 vs 0.119) and the calibration error is level with it (ECE 0.016 vs 0.017); on the human-transfer panel the median calibration error (0.114) is below Decision 1.0 Lux's (0.137) and above Nimble v2's (0.082).
- **Evaluation familiarity.** A later screen of the program's training pools found topical-phrase overlap (no exact duplicates) between some training groups and 84 items of the reported evaluation panels. This model's fine-tuning data was built after that screen and contains none of those groups; the screen does not cover the data behind the Decision 1.0 Lux weights it builds on. Rescored without the 84 items, post-key v3 stays 67.74 and none of this card's v3 or human-transfer comparisons change; the margin over Decision 1.0 Lux moves from +1.93 to +2.06.
- **CPU.** CPU inference was not verified for this release: where the GPU-only causal-conv1d package is installed, Transformers routes the convolution to that kernel and a CPU run fails.
- JevArena v3 answers were available during development, so these are post-key same-panel comparisons, not an untouched blind test.
- The v3 and public panels are almost entirely English; they do not measure multilingual ability.
- Complete inputs above 16,384 tokens are rejected, never truncated; 4 evaluation answers were invalid (including over-budget inputs) and counted as failures.
- It decides from the supplied state only and does not retrieve missing facts. Probabilities are estimates; review consequential decisions against the evidence.

[License](LICENSE) · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)
