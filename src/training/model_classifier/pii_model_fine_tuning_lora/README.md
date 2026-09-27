# Train a Vela PII token classifier

Derive a new PII model from the qualified
`llm-semantic-router/Vela-1.0-Encoder-307M` Base at revision
`fe9ccc074b781bc0e2e13c2c8d26f2640410636a`. This workflow initializes a fresh
complete token-classification head and trains the full encoder, preserving 17
entity types and 35 BIO labels. It validates character spans, supervises every
entity subword, and evaluates exact entity boundaries. These instructions produce
a candidate; they do not establish that a new PII model has passed release
qualification.

The scripts use Unicode codepoint offsets. Native binding APIs may use UTF-8 byte
offsets; convert offsets at that boundary rather than comparing the integers
directly. Decoding accepts an orphan `I-*` as a new entity and trims boundary
whitespace without joining different entity types.

## Prepare fixed source revisions

Use an isolated environment with a platform-compatible PyTorch build,
Transformers, and the Hugging Face CLI. The full training path has been exercised
with PyTorch 2.10 and Transformers 4.57.6. PEFT is only needed for historical
adapter continuation. Data-contract tests use Python's standard library; the
tokenizer-measured long-data generator additionally requires Transformers.

From the repository root:

```bash
SCRIPT=src/training/model_classifier/pii_model_fine_tuning_lora
WORK=work/vela-pii
BASE_MODEL_ID=llm-semantic-router/Vela-1.0-Encoder-307M
BASE_REVISION=fe9ccc074b781bc0e2e13c2c8d26f2640410636a
mkdir -p "$WORK/sources"
hf download "$BASE_MODEL_ID" --revision "$BASE_REVISION" --local-dir "$WORK/base"
hf download llm-semantic-router/Vela-1.0-Encoder-307M-PII config.json \
  --revision 6d3300c4bd7975f30a664503f6c725cf1fbbad48 --local-dir "$WORK/reference"
curl --fail --location \
  https://raw.githubusercontent.com/microsoft/presidio-research/f3ff907eba57b8d380711ce7ca82a42696cd0490/data/synth_dataset_v2.json \
  --output "$WORK/sources/presidio.json"
python "$SCRIPT/generate_repair_data.py" \
  --config "$WORK/reference/config.json" --presidio "$WORK/sources/presidio.json" \
  --output "$WORK/corpus"
python "$SCRIPT/generate_long_data.py" --corpus "$WORK/corpus" \
  --tokenizer "$WORK/base" --output "$WORK/long"
cat "$WORK/corpus/train.jsonl" \
  "$WORK/long"/train-{4096,8192,16384,32768}.jsonl > "$WORK/train-mixed.jsonl"
cat "$WORK/corpus/dev.jsonl" \
  "$WORK/long"/dev-{4096,8192,16384,32768}.jsonl > "$WORK/dev-mixed.jsonl"
```

The PII `config.json` download supplies only the existing 35-label schema. Its
task weights are not downloaded or used for initialization. The fresh model
retains the Base's encoder and positional configuration.

The pinned Presidio source has SHA-256
`ec08a771ba8135314cafb60752b2295212222ba3a4cd75d73811839c699e0012` and is covered by
the [Presidio Research MIT license](https://github.com/microsoft/presidio-research/blob/f3ff907eba57b8d380711ce7ca82a42696cd0490/LICENSE).
This workflow uses that source and newly generated synthetic data. It does not
download AI4Privacy datasets.

The generator creates separate train, development, and held-out test files.
Generated splits have disjoint template families, source groups, texts, and
complete entity values. Exact duplicate texts within each split are removed.
Presidio is deduplicated into a separate **seen retention** file: historical PII
checkpoints may already have trained on it. It cannot establish generalization.
Generation refuses to overwrite a nonempty corpus directory and writes hashes,
label mappings, and generation settings in `manifest.json`. Freeze these manifests
before evaluating a candidate; never select checkpoints from test results.

The generated corpus covers English, Chinese, Spanish, French, German, and
Japanese prompts. It remains synthetic: localized field labels, limited value
families, and template coverage do not constitute broad multilingual validation.
Long examples use repeated neutral filler or packed synthetic records, with
head/middle/tail email needles and separate negative examples. Every long example
is measured with the real tokenizer at 4,096, 8,192, 16,384, or 32,768 attended
tokens. These are length/position stress tests, not natural-document benchmarks.

## Probe the full model with explicit budgets

The driver keeps encoder parameters in FP32, uses BF16 autocast and SDPA, and
enables non-reentrant gradient checkpointing. It does not use inference-only ORT
kernels for backward. Select a device with the platform's normal device visibility
environment variables; one process trains on one visible GPU.

```bash
python "$SCRIPT/train_repair.py" --method full --fresh-head \
  --base "$WORK/base" --base-id "$BASE_MODEL_ID" --base-revision "$BASE_REVISION" \
  --config "$WORK/reference/config.json" \
  --train "$WORK/long/train-32768.jsonl" --dev "$WORK/corpus/dev.jsonl" \
  --output "$WORK/probe" --probe-only --steps 1 --batch-size 1 --accumulate 1 \
  --max-length 32768 --learning-rate 1e-5 --head-learning-rate 1e-4 \
  --loss-normalization document_mean --evaluation-dtype float32 --seed 20260917
```

`--max-length` is a ceiling, not evidence of the lengths actually trained.
`steps.jsonl` records measured token lengths, finite gradients, forward/backward
step time, and allocated/reserved memory. The probe above performs a real long
optimizer step but saves no candidate. Discard that process and initialize the
formal run from the same frozen Base. Rows that exceed the selected budget or
cannot represent an exact span with this tokenizer
are recorded as rejections. Invalid annotations and overlapping entities fail;
they cannot silently overwrite labels.

For a long-context run, concatenate only frozen training files into a
training mixture, and only development files into a development mixture. Keep
short training and replay to check short-input and source retention. Use
`--selection-metric length-macro-f1` for the mixed development set so densely packed
examples in one length bucket cannot dominate other lengths. The appropriate
steps, batch size, replay rate, and curriculum require measured development and
memory results; a 32K checkpoint capacity does not establish a deployable budget.

## Train the complete token classifier

`--method full` trains every encoder parameter, including token embeddings, and
the complete `head.dense`, `head.norm`, and `classifier`. Use `--fresh-head` when
deriving a PII model from a frozen ModernBERT base: it constructs and initializes
the token head with the architecture's native initialization without changing
encoder tensors. An existing native 35-label token checkpoint can be continued
by omitting this flag. Full mode rejects adapters and requires explicit
`--base-id` and `--base-revision`. The CLI retains a LoRA default for compatibility;
new Vela derivations must explicitly select full mode and a fresh head.

```bash
python "$SCRIPT/train_repair.py" --method full --fresh-head \
  --base "$WORK/base" --base-id "$BASE_MODEL_ID" \
  --base-revision "$BASE_REVISION" --config "$WORK/reference/config.json" \
  --train "$WORK/train-mixed.jsonl" --replay "$WORK/corpus/replay.jsonl" \
  --dev "$WORK/dev-mixed.jsonl" \
  --output "$WORK/full-run" --learning-rate 1e-5 --head-learning-rate 1e-4 \
  --loss-normalization document_mean --selection-metric length-macro-f1 \
  --evaluation-dtype float32 --steps 400 --batch-size 1 --accumulate 8 \
  --max-length 32768 --seed 20260917
```

The learning rates and budget above are explicit example settings, not a
qualified training recipe. Freeze the actual sampling, budget, development gates,
and selection rule before a run. Both methods default to
`--loss-normalization token_mean`, the existing mean cross-entropy over nonignored
token labels within each microbatch,
including `O` labels. Optional `--loss-normalization document_mean` first averages
the attended, nonignored token losses within each document, then gives every
document equal weight across the logical batch. It uses FP32 cross-entropy and
rejects a document with no supervised tokens. The trainer uses equally sized
microbatches and divides each loss by the accumulation count. Padding and special
tokens have ignored labels; document mode also excludes masked positions even
if their labels are populated. The selected reduction is recorded in `run.json`.
Full mode checks that every trainable parameter receives a finite gradient.
For sparse entity supervision, `--loss-normalization entity_document_mean`
gives half of a positive document's weight to the mean of its individual entity
losses and half to its `O`-token mean. Each entity loss averages all of that
entity's subwords, so longer values do not outweigh shorter values. Fully
negative documents retain their full `O`-token mean; documents with no `O`
tokens use the entity mean alone. Documents then receive equal weight, without
inverse-frequency class weights. Explicit annotation-span IDs are padded only
for this loss and never passed to the model. Special, ignored, and masked tokens
enter neither term. This is an optional training method; it changes no BIO
labels, decoding, evaluation gates, or default reduction.
Optional `--head-learning-rate` gives the entire prediction head a separate rate
with the same schedule. Evaluation uses
`--evaluation-dtype` independently of training autocast; the default remains
BF16. `--device cpu` provides a FP32 engineering path for small fixtures.

Full checkpoints are `best-model` and `last-model`. Each contains native
safetensors, tokenizer/config files, and `full-checkpoint.json`, which binds
every saved tensor to its shape, dtype, exact bytes, and initial-artifact/data
receipt. Saving verifies the complete encoder and head. Missing encoder weights
cannot be replaced by random initialization, and a native checkpoint must retain
the exact label order. Neither full training nor evaluation requires PEFT.
`best-model` is the highest-ranked development checkpoint, not an automatic
qualification result: `length-macro-f1` does not enforce per-type, language, email,
or negative-document retention gates.

Export the chosen full checkpoint using the same initial model identity:

```bash
python "$SCRIPT/export_repair.py" --method full \
  --base "$WORK/base" --base-id "$BASE_MODEL_ID" \
  --base-revision "$BASE_REVISION" --config "$WORK/reference/config.json" \
  --checkpoint "$WORK/full-run/best-model" \
  --run-manifest "$WORK/full-run/run.json" --output "$WORK/full-export"
```

The exporter checks checkpoint provenance and tokenization bytes, writes the
runtime label mappings, and requires exact FP32 short-probe logits after native
save/reload. It does not combine the trained encoder with its old base. The tiny
CPU tests exercise real full training, native export, and malformed-checkpoint
rejection; they do not establish model quality or a 32K memory budget.

## Qualify the candidate on frozen data

```bash
python "$SCRIPT/evaluate.py" --base "$WORK/full-export" \
  --config "$WORK/full-export/config.json" --dtype float32 \
  --data "$WORK/dev-mixed.jsonl" --output "$WORK/results/development.json"
make test-training-contracts
```

Evaluate the original checkpoint and final candidate on the same frozen files,
precision, and decoder. Report per-type exact entity precision/recall/F1, complete
email recall, negative-document false positives, and language/length/position
breakdowns. The evaluator writes per-record entity predictions and fails on
non-finite logits. These model-quality scores use argmax BIO labels with no
confidence threshold; serving-policy thresholds require a separate comparison.
Its forward batch timings exclude HTTP, tokenization, and
router scheduling; they are not service latency. Development selection, held-out
synthetic results, seen retention, and previously failing diagnostics must remain
separate in a report. Only after development gates pass and the candidate and
serving policy are frozen should the held-out files be evaluated. Do not retune
or select checkpoints using those results; preserve failures and predictions.

For a controlled full-sequence versus scanning comparison, repeat evaluation of
the same frozen weights and rows with `--window-tokens 512 --window-stride 64`
and a separate output file. This uses the HF tokenizer's overlapping token
windows, maps offsets onto the full original text, and removes only duplicate
entities with identical type and boundaries. Different boundary fragments remain
false positives. The report records the full document length, actual forward
window limit, and number of forward sequences. This is a tokenizer-window
reference, not evidence of parity with a native serving policy; compare the
actual serving configuration separately. Without these flags, evaluation
uses the entire sequence without truncation.

The exporter preserves full native FP32 weights and the label contract, and
checks short-input numerical equivalence before writing checkpoint provenance.
This probe is not a quality benchmark. Validate long inputs and the exported
ONNX/native/serve path separately before publication. The training scripts do not
upload weights or change router context policy.

## Historical adapter recovery

For reproducing an existing LoRA run, use `--method lora --adapter` with its exact
original Base, explicit `--base-id`/`--base-revision`, label contract, and run
manifest. This requires PEFT and preserves the adapter's trained head. Do not
attach a historical adapter to the new Vela Base or call that operation new-Base
training. The separate `pii_bert_finetuning_lora.py` entrypoint remains a
historical workflow.
