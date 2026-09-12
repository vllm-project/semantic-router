# PII adapter repair and evaluation

This workflow continues an existing mmBERT PII adapter while preserving its 17
entity types and 35 BIO labels. It is separate from the historical
`pii_bert_finetuning_lora.py` training entrypoint. It validates character spans,
supervises every entity subword, evaluates exact entity boundaries, and records
explicit context and training budgets.

The scripts use Unicode codepoint offsets. Native binding APIs may use UTF-8 byte
offsets; convert offsets at that boundary rather than comparing the integers
directly. Decoding accepts an orphan `I-*` as a new entity and trims boundary
whitespace without joining different entity types.

## Prepare fixed source revisions

Use a separate environment with a platform-compatible PyTorch build. The repair
driver was exercised with PyTorch 2.10, Transformers 4.57.6, PEFT 0.18.1, and BF16
autocast. Install `transformers==4.57.6`, `peft==0.18.1`, `accelerate==1.10.1`, and
`huggingface_hub==0.36.2` alongside that PyTorch build. The data-contract tests need
only Python's standard library. The tokenizer-measured long-data generator also
requires Transformers and the checkpoint tokenizer.

From the repository root:

```bash
SCRIPT=src/training/model_classifier/pii_model_fine_tuning_lora
WORK=work/pii-repair
mkdir -p "$WORK/sources"
hf download llm-semantic-router/mmbert-32k-yarn \
  --revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de --local-dir "$WORK/base"
hf download llm-semantic-router/mmbert32k-pii-detector-lora \
  --revision 58ecd71088283cd5ee660d5f2b155d32985a93b2 --local-dir "$WORK/original-adapter"
hf download llm-semantic-router/mmbert32k-pii-detector-merged config.json \
  --revision d22c818cf9f2a8bfbbed8508cb417dc16a1ba3ea --local-dir "$WORK/reference"
curl --fail --location \
  https://raw.githubusercontent.com/microsoft/presidio-research/f3ff907eba57b8d380711ce7ca82a42696cd0490/data/synth_dataset_v2.json \
  --output "$WORK/sources/presidio.json"
python "$SCRIPT/generate_repair_data.py" \
  --config "$WORK/reference/config.json" --presidio "$WORK/sources/presidio.json" \
  --output "$WORK/corpus"
python "$SCRIPT/generate_long_data.py" --corpus "$WORK/corpus" \
  --tokenizer "$WORK/base" --output "$WORK/long"
```

The pinned Presidio source has SHA-256
`ec08a771ba8135314cafb60752b2295212222ba3a4cd75d73811839c699e0012` and is covered by
the [Presidio Research MIT license](https://github.com/microsoft/presidio-research/blob/f3ff907eba57b8d380711ce7ca82a42696cd0490/LICENSE).
This workflow uses that source and newly generated synthetic data. It does not
download AI4Privacy datasets.

The generator creates separate train, development, and held-out test files.
Generated splits have disjoint template families, source groups, texts, and
complete entity values. Exact duplicate texts within each split are removed.
Presidio is deduplicated into a separate **seen retention** file: the original
checkpoint may already have trained on it. It cannot establish generalization.
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

## Continue the adapter with explicit budgets

The driver keeps encoder parameters in FP32, uses BF16 autocast and SDPA, and
enables non-reentrant gradient checkpointing. It does not use inference-only ORT
kernels for backward. Select a device with the platform's normal device visibility
environment variables; one process trains on one visible GPU.

```bash
python "$SCRIPT/train_repair.py" \
  --base "$WORK/base" --base-revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de \
  --adapter "$WORK/original-adapter" --config "$WORK/reference/config.json" \
  --train "$WORK/corpus/train.jsonl" --replay "$WORK/corpus/replay.jsonl" \
  --dev "$WORK/corpus/dev.jsonl" --output "$WORK/short-run" \
  --steps 400 --batch-size 8 --accumulate 2 --max-length 2048
```

`--max-length` is a ceiling, not evidence of the lengths actually trained.
`steps.jsonl` records measured token lengths, finite gradients, forward/backward
step time, and allocated/reserved memory. Use `--probe-only --steps 1 --batch-size
1 --accumulate 1` with one long training file before raising a hardware budget.
Probe runs perform a real optimizer step but do not save a candidate. Rows that
exceed the selected budget or cannot represent an exact span with this tokenizer
are recorded as rejections. Invalid annotations and overlapping entities fail;
they cannot silently overwrite labels.

For long-context continuation, concatenate only frozen training files into a
training mixture, and only development files into a development mixture. Keep
short training/retention replay to check preservation of the original tasks. Use
`--selection-metric length-macro-f1` for the mixed development set so densely packed
examples in one length bucket cannot dominate other lengths. The appropriate
steps, batch size, replay rate, and curriculum require measured development and
memory results; a 32K checkpoint capacity does not establish a deployable budget.

## Evaluate and export a selected checkpoint

```bash
python "$SCRIPT/export_repair.py" --base "$WORK/base" \
  --base-revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de \
  --adapter "$WORK/short-run/best-adapter" --config "$WORK/reference/config.json" \
  --run-manifest "$WORK/short-run/run.json" \
  --output "$WORK/merged"
python "$SCRIPT/evaluate.py" --base "$WORK/merged" \
  --config "$WORK/merged/config.json" \
  --data "$WORK/corpus/test.jsonl" --output "$WORK/results/test.json"
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
separate in a report. Preserve test predictions even when results disappoint.

For a controlled full-sequence versus scanning comparison, repeat evaluation of
the same frozen weights and rows with `--window-tokens 512 --window-stride 64`
and a separate output file. This uses the HF tokenizer's overlapping token
windows, maps offsets onto the full original text, and removes only duplicate
entities with identical type and boundaries. Different boundary fragments remain
false positives. The report records the full document length, actual forward
window limit, and number of forward sequences. This is a tokenizer-window
reference, **not exact parity with the Go router's rune-budget chunking policy**;
native serving remains a separate comparison. Without these flags, evaluation
uses the entire sequence without truncation.

The exporter merges into FP32 weights, preserves the label contract, and checks
short-input numerical equivalence before writing a checkpoint and provenance.
This probe is not a quality benchmark. Validate long inputs and the exported
ONNX/native/serve path separately before publication. The training scripts do not
upload weights or change router context policy.
