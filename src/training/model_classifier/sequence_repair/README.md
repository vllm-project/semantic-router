# Reproducible sequence-classifier repair

This module initializes, continues, evaluates, and freezes standard Hugging Face
`ModernBertForSequenceClassification` checkpoints. Domain, FactCheck, and Modality
keep their existing label IDs. Safety, Feedback, and PromptGuard can reuse the
single-label runner; multi-label Hazard uses its own masked-loss training loop.
The shared loader and exporter preserve the contract's `problem_type`.

The Vela family name uses the shared encoder scale, **307M**. Export receipts also
record the actual unique parameter count including each task head. A configured
32,768-position capacity is a limit, not evidence of long-document accuracy.

## Data and provenance

The [source manifest](data/source-files.json) pins public dataset revisions,
individual files, and SHA-256 values. It downloads metadata and text only, not
DiffusionDB images. Prepare one source directory:

```bash
python -m src.training.model_classifier.sequence_repair.fetch_sources \
  --output artifacts/vela/sources
```

The reviewed-label sidecars contain source identifiers and task judgments rather
than copies of source requests. Hydration verifies the complete source file and
each request's text hash; Aya also verifies its contributor group. For example:

```bash
python -m src.training.model_classifier.sequence_repair.hydrate_annotations \
  --sidecar src/training/model_classifier/sequence_repair/data/aya-reviewed-labels-v1.json \
  --source artifacts/vela/sources/CohereLabs--aya_dataset/data/train-00000-of-00001.parquet \
  --output artifacts/vela/annotations/aya-reviewed-labels-v1.jsonl
```

The same command reconstructs `aya-fact-reviewed-v2.json` and
`dolly-fact-reviewed-v2.json` using their specified source file. New task labels
were individually reviewed by an assistant before model predictions. Source
requests were human-authored; the new labels must not be described as expert
human annotations. Ambiguous requests and identified duplicate source/template
families retain their exclusion reasons. See [data provenance](data/README.md).

Each task recipe writes `train.jsonl`, `dev.jsonl`, `test.jsonl`, `contract.json`,
and `manifest.json`. Rows contain `id`, `text`, `label`, and `group_id`; reported
metadata includes `source`, `language`, `length_bucket`, and `position`. IDs are
unique. Normalized duplicate text or source groups cannot cross partitions.
Missing/conflicting labels cause an error. Similarity screening and source-family
policy belong to the task recipe, not an inferred guarantee of this generic
validator. Old-weight and base-pretraining exposure may remain unknown.

## Initialize and train

The isolated reference environment used Torch 2.10 with ROCm 7,
Transformers 4.57.6, PEFT 0.18.1, and Accelerate 1.10.1. Data preparation additionally
uses PyArrow and scikit-learn. Pin the complete environment for a reproduced run;
do not install into a running inference service environment.

For a fresh task head and LoRA adapter:

```bash
python -m src.training.model_classifier.sequence_repair.initialize \
  --base artifacts/vela/base \
  --base-id llm-semantic-router/mmbert-32k-yarn \
  --base-revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de \
  --contract artifacts/vela/factcheck/contract.json \
  --output artifacts/vela/factcheck-initial
```

Initialization makes the standard head and classifier trainable in addition to
rank-32 LoRA on attention and MLP projections. For continuation, supply the selected
adapter explicitly instead. The published base configuration is retained; this
workflow does not infer or inject a missing RoPE-scaling setting from a model name.
An optional `classifier_pooling` contract field selects the standard HF `mean`
or `cls` mode. Omit it to retain the base's setting. Treat a pooling change as a
new trained candidate and compare short and long development quality; changing
the configuration alone does not make a mean-trained head a valid CLS model.

```bash
python -m src.training.model_classifier.sequence_repair.train \
  --base artifacts/vela/base \
  --base-revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de \
  --adapter artifacts/vela/factcheck-initial \
  --contract artifacts/vela/factcheck/contract.json \
  --train artifacts/vela/factcheck/train.jsonl \
  --dev artifacts/vela/factcheck/dev.jsonl \
  --output artifacts/vela/runs/factcheck-short \
  --steps 600 --batch-size 8 --accumulate 2 --max-length 2048 \
  --learning-rate 0.00002 --eval-every 100 \
  --source-balanced-sampling --balanced-sampling --selection source-macro-f1
```

The manual loop keeps parameters and cross-entropy accumulation in FP32, uses
BF16 autocast and SDPA, and enables non-reentrant gradient checkpointing. It
checks real losses and gradients for finiteness. Each equally sized microbatch
contributes mean CE divided by the number of accumulation steps. The optional
Torch test compares this gradient with a full batch; this avoids relying on
Trainer loss-kwargs behavior for a model that does not consume its normalization
argument.

Selection evaluates the initial checkpoint as a quality floor, then uses only
development data. It writes `run.json`, per-step development metrics,
`selection.json`, `best-adapter`, and `last-adapter`. Existing runs are not
overwritten. The default score is macro F1 across the complete label ontology.
`source-macro-f1` averages that score equally across sources; `length-macro-f1`
averages it across length buckets. A source lacking a class still has that class
in the macro denominator, so inspect per-source accuracy and support as well.
The explicit `source-present-macro-f1` option instead averages only labels with
gold support within each source, then gives sources equal weight. It is useful
when an expanded taxonomy has sources that observe different label subsets.
Errors on observed labels still count, and all complete-ontology metrics remain
in the report. Existing selection modes and sampling are unchanged.

For an explicitly binary task, add `--selection binary-fp-budget-recall
--positive-label unsafe --selection-false-positive-budget 0.1` to select by
positive recall while at most 10% of negative development examples trigger.
Both labels must have development support. Each checkpoint fits a single
threshold using `p(unsafe) >= threshold` within `[0, 1]`; tied scores stay
together. Ties prefer fewer false positives, then a higher threshold, then the
earlier checkpoint. The runner saves per-step probabilities and an operating
point with the threshold, support, confusion counts, feasibility, and evidence
hash. `selection.json` also contains the selected operating point. Argmax F1
remains in the report. A saturated negative probability of 1 can make the
budget infeasible; such a checkpoint is not selected, and a run with no
feasible checkpoint fails after retaining its last adapter and diagnostics.
This is empirical development calibration, not a guarantee of deployment FP
rate. Freeze the selected threshold with the model before separate test
evaluation; this option neither reads test data nor changes runtime defaults.
Use `--evaluation-dtype float32` when selecting for FP32 deployment. The default
`bfloat16` retains the existing development autocast behavior; CPU evaluation
remains FP32. `run.json` records this choice and per-step metrics record actual
forward precision. This flag affects evaluation only; training still uses BF16
autocast with FP32 parameters and loss accumulation. Compare checkpoints and
baselines using the same evaluation precision before freezing a threshold.

Source balancing samples a source first, optional length balancing then samples a
length, and optional label balancing then samples a label. These flags do not
copy rows or move groups between splits. An explicit `--source-weights`
JSON object instead controls source probabilities; it must name every source
remaining after context-budget filtering with a finite positive weight. It
implies source sampling and still permits length/label balance within each
source. For example, `--source-weights '{"natural": 9, "authored": 1}'` draws
approximately 90% from `natural`. Without this flag, original sampling and RNG
behavior are preserved. `sampling-step-*.json` and `sampling-final.json` record
per-source draws, unique-row coverage, and label draws. These receipts expose
repeated sampling of small sources; they are not quality metrics.

### Optional token-budget microbatches

`--microbatch-token-budget 32768` first samples the same
`batch-size × accumulate` examples using the existing RNG, stably orders those
examples by their measured length, and groups them so each padded microbatch
fits the token budget. Every example contributes CE sum divided by the original
optimizer-step example count. A 32K example therefore runs alone; shorter
examples can share a forward pass. Optimizer steps, learning-rate schedule,
sampling probabilities, and example coverage remain unchanged. The budget must
fit the longest eligible training example; no training row is silently dropped.

This explicit option changes batching and the dropout random-number trajectory;
it is a new reproducible run, not byte-identical continuation of a prior run.
Dropout-disabled FP32 ModernBERT tests check gradient equivalence against one
complete batch. Steps record actual microbatch counts and maximum padded tokens;
measure speed on the intended workload. Without the option, the original
microbatch ordering and accumulation path remain unchanged.

## Measure long context

`prepare_context` creates exact attended-token budgets around frozen task
payloads. Backgrounds differ across train/dev/test and payload variants retain
their original groups. The supplied backgrounds are repeated, authored prose;
results are **context stress tests**, not estimates on natural long documents.
The label refers to the explicitly marked requested task. Report paired
length/position variants together rather than counting them as independent
natural requests.

```bash
python -m src.training.model_classifier.sequence_repair.prepare_context \
  --corpus artifacts/vela/factcheck \
  --tokenizer artifacts/vela/base \
  --backgrounds src/training/model_classifier/sequence_repair/data/task-context-backgrounds.json \
  --output artifacts/vela/factcheck-context \
  --budgets 4096 8192 16384 32768
```

Use a one-step probe on one actual long input before raising the training budget.
A long curriculum can pass both short and context files to `--train` and `--dev`,
with batch size 1, accumulation, and `--length-balanced-sampling`. Development
must expose short-quality regression as well as long-position failures. Inputs
over the explicit budget are never silently truncated; training rejections are
recorded and evaluation rejects an oversized input.

## Freeze, then evaluate the independent test

Some historical adapters save only `classifier`, leaving the sequence `head`
inherited from their original base. Before comparing that adapter on another
encoder base, preserve its original task head in a separate adapter artifact:

```bash
python -m src.training.model_classifier.sequence_repair.complete_adapter \
  --base artifacts/vela/original-base \
  --adapter artifacts/vela/runs/domain-context/best-adapter \
  --contract artifacts/vela/domain/contract.json \
  --output artifacts/vela/domain-complete-head
```

This operation copies the existing head without training and checks CPU FP32
logits before and after completion. It refuses adapters that already own `head`.
Legacy adapters retain their inherited frozen head until this explicit operation;
completion does not demonstrate a quality improvement. Training receipts record
the effective head tensors and their trainable scope. Each saved adapter checks
that its trainable head tensors survived serialization with identical bytes.
Use the same completed adapter for both sides of a base-migration comparison;
otherwise changing the base may also replace the task head. Select a base using
short and long development retention, and record the migration separately from
the original training receipt. A family name alone does not establish lineage.

Export merges the selected adapter in FP32, checks short-input numerical
agreement before/after merging, preserves the standard architecture and label
mapping, and writes a `candidate-lock.json` with hashes. Supply the selected run's
receipt. The base revision must match that receipt and test selection must be
explicitly false.
Export also verifies the effective head tensors before and after merge and after
reloading the serialized model, including inherited frozen head tensors.

```bash
python -m src.training.model_classifier.sequence_repair.export \
  --runtime-task fact-check \
  --base artifacts/vela/base \
  --base-revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de \
  --adapter artifacts/vela/runs/factcheck-short/best-adapter \
  --run-manifest artifacts/vela/runs/factcheck-short/run.json \
  --contract artifacts/vela/factcheck/contract.json \
  --output artifacts/vela/frozen/factcheck

python -m src.training.model_classifier.sequence_repair.evaluate \
  --base artifacts/vela/frozen/factcheck \
  --contract artifacts/vela/factcheck/contract.json \
  --data artifacts/vela/factcheck/test.jsonl \
  --output artifacts/vela/evidence/factcheck-final.json \
  --max-length 32768 --dtype bfloat16
```

The required `--runtime-task` writes the router's actual sidecar schema alongside
the HF label mapping, before hashing the frozen candidate. Domain uses
`category_mapping.json`; FactCheck uses `fact_check_mapping.json`; PromptGuard
uses `jailbreak_type_mapping.json`; Modality also receives `modality_mapping.json`
and must retain the native `AR=0`, `DIFFUSION=1`, `BOTH=2` indices. Feedback,
Safety, and Hazard retain the complete named label set in `config.json` and
`label_mapping.json`. Every sidecar derives from the same classifier label IDs.

Test is an explicit separate invocation, never run by training or export. Compare
baseline and candidate on the same rows, precision, and decoding policy; report
source/language/length breakdowns, uncertainty, and regressions. `--dtype
float32` disables GPU autocast for a precision-controlled comparison. Forward
batch timings exclude tokenization and HTTP serving overhead and are not request
latency. Native and ONNX execution need their own parity and serving validation.
A short merge-equivalence check alone is not a quality or 32K-runtime test.
