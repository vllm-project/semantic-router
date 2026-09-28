# Vela Safety and Hazard training

Safety predicts `safe` / `unsafe`. Hazard predicts twelve independent content-risk
scores using the [fixed rubric](configs/hazard-rubric-v1.json), including explicit
safe negatives and unknown-label masks. See the [application recipes](../vela-applications.md)
for source admission, initialization, training and independent evaluation.

New runs start from `llm-semantic-router/Vela-1.0-Encoder-307M` revision
`fe9ccc074b781bc0e2e13c2c8d26f2640410636a`. Bind the downloaded encoder,
configuration and tokenizer bytes in each run. A shared model name alone does
not establish common training lineage.

Use `sequence_repair.train` for binary Safety and `train_vela_hazard` for Hazard.
Both accept explicit TRAIN and DEV files, source/group provenance and fixed
budgets. `--method full --fresh-head` trains the complete encoder and a newly
initialized task head. Adapter training uses an explicitly initialized adapter
with its actual parent revision. Full checkpoints and adapters have separate
export contracts; neither is a qualified release without the task's independent
evaluation.

A 32K model capacity does not establish 32K task accuracy. Measure actual input
tokens, retain rejected rows, and evaluate short retention, long inputs and
safe quotations separately. Freeze checkpoint selection, thresholds and input
policy before scoring independent evaluation data.

## Vela independent Hazard thresholds

The Vela `train_vela_hazard` entrypoint trains independent sigmoid labels with
masked BCE. Unknown dimensions contribute no loss. Its
`--selection fp-budget-macro-f1` mode fits one shared threshold; the explicit
`--selection joint-fp-budget-macro-f1` mode fits a threshold for each label under
joint safe-input false-positive budgets. Neither mode changes the label masks
or establishes that a checkpoint is qualified for release.

For joint selection, an optional `--selection-safe-groups groups.json` adds
constraints for named subsets of fully observed safe DEV examples:

```json
{
  "clean": ["dev-safe-001", "dev-safe-002"],
  "boundary": ["dev-boundary-001"]
}
```

Every ID must belong to the supplied DEV data. All safe DEV rows are always
constrained together, and each named group must also meet
`--selection-false-positive-budget` (default `0.05`, rounded down to a whole
number of false alarms). A false alarm means any output label fires; separate
per-label budgets cannot replace this union constraint. The training receipt
records the sidecar digest, and each evaluation records fitted thresholds and
group limits.

Joint selection uses deterministic FP32 candidates and two starting points,
then at most 72 improving coordinate updates. It preserves all-taxonomy macro
F1 with `--selection-minimum-support 1`; unsupported labels remain in the
false-alarm rule and are explicitly reported as uncertified. A result with no
feasible threshold is recorded as infeasible. Fit thresholds only on DEV and
retain independent evaluation and the task's existing quality gates. See the
[application recipes](../vela-applications.md) for full-encoder and adapter
training options.

### Replay a fixed training order

For a controlled training comparison, pass `--training-order order.json` instead
of source/length sampling flags or source weights. The version 1 JSON contains
`train_files` (the existing `file_receipts` filename/SHA256 list),
`eligible_ids_sha256`, `steps`, `global_batch`, the complete draw `ids`, and
`ids_sha256`. Both ID hashes use UTF-8 JSON with `ensure_ascii=False` and
`separators=(",", ":")`; `ordered_ids_sha256` in
`sequence_repair/training_order.py` implements this encoding. Eligible IDs follow
TRAIN file/row order after tokenization.

The loader rejects changed TRAIN bytes, unknown or filtered IDs, changed budgets,
and missing or extra draws. Repeated draws are allowed; every TRAIN row need not
be sampled. A fixed order must contain exactly `steps × batch_size × accumulate`
IDs. Token-budget batching may reorder examples within that optimizer step;
its multiplicities and global loss denominator stay unchanged.

`run.json` binds the order file and ID hashes. `actual-training-order.jsonl`
records each completed step's planned IDs and actual microbatch IDs, and
`training-order-completed.json` verifies complete consumption and hashes the
trace. Omitting this option preserves the existing random sampler. A replay
controls sampled examples, not equivalence between different precision, dropout,
or optimization methods.

## Validation

Run the retained data, admission, loss, sampling and training tests from the
repository root in the supported training environment:

```bash
python -m unittest discover \
  -s src/training/model_classifier/safety_classifier/tests \
  -p 'test_*.py' -v
```

The full-training tests require PyTorch and Transformers; a dependency-based skip
is not evidence that a model path was tested. Record the actual CPU, CUDA or ROCm
build separately from dataset and model qualifications.

## Existing v1 compatibility

The explicit [`training-v1.json`](configs/training-v1.json) contract and its
`data`, `train`, `evaluate`, `export` and `release` commands remain available for
existing binary and `legacy-9-v1` artifacts. They retain their pinned parent,
512-token policy and existing label IDs. They do not define the current Vela
twelve-label Hazard recipe. Load an existing adapter with the parent declared in
its `adapter_config.json`.
