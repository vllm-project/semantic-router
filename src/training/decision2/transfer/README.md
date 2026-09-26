# CSS human-label transfer panel

This panel mirrors the 18 closed-choice test tasks in [Ibrahim and Zaki
(2026)](https://arxiv.org/html/2609.24574v2), using the same released test
splits, instruction text, and author-checked option-to-gold mapping. The three
paper pilot tasks (stance, implicit hate, discourse) are separated from the 15
evaluation tasks. It contains 7,977 human-labelled items: 1,430 pilot and
6,547 evaluation. Full evaluation-task labels must stay out of training,
model selection, prompt tuning, and calibration.

The source snapshots are pinned:

| Source | Revision | Role |
| --- | --- | --- |
| [Authors' replication package](https://github.com/hazemibrahim97/decision-models-css) | `311956c2d1096cabc0f09a1f248e7dc0e1c0c41e` | Task map and prompt conversion contract |
| [SALT-NLP/LLMs_for_CSS](https://github.com/SALT-NLP/LLMs_for_CSS) | `55183a64d7faaf6d5fc23eddf2bfc48ece4cacad` | Released `css_data/*/test*.json` with human gold |

The authors' replication **code** is MIT licensed. The SALT repository has no
top-level license file at the pinned revision; it aggregates datasets from
different original creators. This panel's source text and labels should stay
on the experiment host unless each dataset's redistribution terms have been
reviewed. The generated `css-manifest.json` records every task's source path,
source file SHA-256, row and class counts, snapshot revision, and license
status. The `*.gold.jsonl` files also contain source IDs, input hashes, raw
context hashes, and normalized context hashes for train/test overlap audits.
The normalized hash is SHA-256 of
`" ".join(context.casefold().split()).encode("utf-8")`.

## Build on the experiment host

Download both repositories **on the experiment host** at the pinned revisions
above. Use a clean checkout; the builder rejects changed tracked files. Set
the two paths and a fresh output directory:

```bash
export CSS_REPLICATION_ROOT=/path/to/decision-models-css
export CSS_DATA_ROOT=/path/to/LLMs_for_CSS
export CSS_PANEL_DIR=/path/to/private/css-transfer-v1
PYTHONPATH=/path/to/decision-2-research python3 -m transfer.build \
  --replication-root "$CSS_REPLICATION_ROOT" \
  --data-root "$CSS_DATA_ROOT" \
  --output-dir "$CSS_PANEL_DIR"
```

The files are:

```text
css-pilot.prompts.jsonl       1,430 gold-free model inputs
css-pilot.gold.jsonl          1,430 private human labels/provenance
css-evaluation.prompts.jsonl  6,547 gold-free model inputs
css-evaluation.gold.jsonl     6,547 private human labels/provenance
css-manifest.json             source revisions, SHA-256, task counts
```

The prompt files contain only `id`, `state`, and `questions`. All are native
Jev-style typed `choice` questions. The 114 character-trope labels, including composites,
are retained as complete labels; smaller letter-readout models unable to score
that many choices should report this task as unsupported rather than silently
drop options. `flute` and `mrf` use `test-classification.json`; the other 16
tasks use `test.json`. The source package's two variable-length extraction
tasks are excluded, following the paper.

## Evaluate

Collect predictions from each model using only the relevant
`*.prompts.jsonl`. The inference collectors in this repository write the
standard `id`/`answers`/`source_input_sha256` contract. For the official Jev
collector, run separate pilot and evaluation calls and normalize its receipts
to the same contract. The CSS normalizer verifies the original API-body hash,
which includes the model ID, then records the model-independent prompt hash
used by the gold manifest:

```bash
PYTHONPATH=/path/to/decision-2-research python3 -m transfer.normalize_jev \
  --prompts "$CSS_PANEL_DIR/css-evaluation.prompts.jsonl" \
  --receipts /path/to/jev-css-evaluation.receipts.jsonl \
  --output /path/to/jev-css-evaluation.predictions.jsonl \
  --expected-model jev-1.13.0
```

Score each role on the experiment host:

```bash
PYTHONPATH=/path/to/decision-2-research python3 -m transfer.score \
  --gold "$CSS_PANEL_DIR/css-evaluation.gold.jsonl" \
  --predictions /path/to/model-css-evaluation.predictions.jsonl \
  --output /path/to/model-css-evaluation.score.json
```

The `css-transfer-score/2` report gives per-task accuracy, macro-F1,
multiclass Brier sum, NLL, and 15-bin ECE. Valid option maps are divided by
their original probability sum for Brier, NLL, and top-option ECE; the
original map still controls validity and the returned choice controls
accuracy/F1. Each task and role records the original absolute sum deviation
distribution in `option_probability_sum_abs_delta`. It reports both
top-option-probability ECE and native-confidence ECE. The former is comparable
across typed model interfaces;
the latter mirrors the paper's native-confidence analysis but the confidence
fields differ by model. Invalid or missing predictions count as misses in the
all-item accuracy and F1 and are listed by reason. The role summary reports
the median across the 15 evaluation tasks, separate from the three pilots.

## Paired model comparison

After both prediction files are complete, compare them with a fixed-seed
paired item bootstrap. Each replicate samples item IDs with replacement
**within each task** and applies the same sampled IDs to both models.
Missing or invalid predictions count as misses, matching `transfer.score`.
The 95% intervals use the 2.5th and 97.5th percentiles of 5,000 replicates.
The headline compares the two models' **medians of task scores** across all
15 evaluation tasks; the three pilot tasks never enter that median.

```bash
PYTHONPATH=/path/to/decision-2-research python3 -m transfer.compare \
  --gold "$CSS_PANEL_DIR/css-evaluation.gold.jsonl" \
  --predictions-a /path/to/decision2-css-evaluation.predictions.jsonl \
  --predictions-b /path/to/nox4b-css-evaluation.predictions.jsonl \
  --model-a Decision-2.0 --model-b Decision-1.0-Nox-4B \
  --replicates 5000 --seed 20260926 \
  --output /path/to/decision2-vs-nox-css.compare.json
```

The JSON report includes each task's macro-F1 and accuracy for both models,
their paired difference and 95% interval, plus intervals for each model's
15-task median and the difference of those medians. It records the exact gold,
prediction, scorer, and comparison-code SHA-256 hashes. These intervals hold
the 15 task identities fixed and express item-sampling uncertainty within
those tasks; they do not account for choosing different tasks or potential
pretraining overlap. Keep model selection and calibration confined to the
pilot tasks or a separate development set before inspecting evaluation results.

## Validation and limits

We converted the authors' released historical Jev receipts to this prediction
contract for a **panel validation only**. All 7,977 source IDs and gold labels
matched; all responses were valid; per-task macro-F1, Brier, and native ECE
reproduced the authors' published replication CSV to numerical precision.
Those historical receipts are not a fresh comparison of current APIs.

The test data and many annotation corpora are public, so independence from a
model's pretraining cannot be assumed. Report exact train/test overlap audits
and distinguish this external human-labelled panel from the fresh
programmatically verifiable benchmark. Human labels are adjudicated dataset
answers; some social constructs can have substantial annotator disagreement.
