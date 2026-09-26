# JevArena release bundle gate

`publication.bundle_arena` assembles a Decision 2.0 Hugging Face model
directory only after the six-axis release artifacts and a separate release
review have passed. It does not train, score, call a GPU, upload to Hugging
Face, or replace an unqualified checkpoint with a release model. The earlier
`publication.bundle` remains a four-family Qwen3.5 audit artifact.

## Inputs

Run `python3 -m publication.generate_arena` first. Give the packager a
**functional, standalone** native model directory and these local files:

| Input | Required binding |
| --- | --- |
| `artifacts` | Generator `/3` manifest, score table and six hash-matched SVGs. |
| `arena_rank`, `public_rank` | Same-panel release rankings whose bytes match the generator manifest. |
| `package_record` | Reviewed model ID, selected revision, architecture, immutable base revision, actual parameter inventory, TRAIN/SELECT/CAL counts, source terms and limitations. |
| `score_inputs` | For each of `synthetic`, `css`, `public`, `dbv4`, `authored`: the scorer report, exact scored prediction file and native prediction manifest. |
| `parity_receipt` | Gold-free native package check on 1,600 DEV and 1,430 transfer pilot items, with zero changed categorical answers, p99 probability drift ≤0.005 and maximum drift ≤0.02. |
| `release_gate` | Passed pretest freeze, authored editorial, overlap, same-panel evaluation, rights, parity and performance reviews, each with an evidence hash. |
| `provenance_inputs`, `freeze_manifest`, `gate_evidence` | The original local files matching every declared training, freeze and review evidence hash. They are checked but never copied. |

The record and gate schemas are `decision2-release-package-record/1` and
`decision2-jevarena-release-gate/1`. The test fixture in
`publication/tests/test_bundle_arena.py` gives a complete small example; its
model size is mocked only to keep the CPU test fast. A real package counts
every active safetensors tensor from its header, subtracting only explicitly
named buffers. Every other safetensors file must be classified as a support
weight file. The count must agree exactly with the record and JevArena roster
and be within 25% of the nominal model name.

The packager accepts these functional layouts:

- `qwen3.5-decision-head` and `qwen3.8-decision-head`: full merged
  `model/backbone/`, decision head, tokenizer, CAL, materialization receipt,
  and all eight portable `decision2/` runtime modules. It constructs the
  existing native `MODEL_MANIFEST.json` inside the copied `native/` directory
  and runs that runtime's CPU byte/lineage verifier before publication.
- `qwen3.5-semif`: standalone merged SemIf model with original `serve.py`,
  tokenizer, CAL, notices and exact `SHA256SUMS` coverage.
- `encoder-decision`: native `model.safetensors`, encoder configuration,
  tokenizer, configuration and runtime entrypoint. Duplicate source encoder
  weights can be marked as support files; a checkpoint explicitly marked
  `research_only` is refused unless a new receipt independently marks it
  release-qualified.

All layouts need an external gold-free parity receipt for the **exact copied
model files**. An encoder or SemIf layout check does not prove the runtime can
actually import and execute on the release device. Generate its parity receipt
by running the standalone native package on the authorized evaluation system.

## Invocation

The configuration is JSON. Paths may be absolute or relative to the config
file. Its fields are `model_dir`, `artifacts`, `arena_rank`, `public_rank`,
`package_record`, `parity_receipt`, `release_gate`, `provenance_inputs`,
`freeze_manifest`, `gate_evidence`, `score_key`, and `score_inputs`. Each of
the five `score_inputs` entries has `score`, `predictions`, and
`native_manifest` paths. `provenance_inputs` names `data_manifest`,
`run_provenance`, and `training_code`; `gate_evidence` names the seven
reviews listed in `bundle_arena.GATE_CHECKS`.

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.bundle_arena \
  --config release-package-config.json --output new-model-directory
```

The output must not exist. A successful assembly copies the native model to
`native/`, writes the new model card with the same-panel table and figures,
copies reviewed release receipts, and records every public file hash in
`PACKAGE_MANIFEST.json`. A failed check leaves no output directory. A copied
bundle can be checked again with `publication.bundle_arena.verify(path)`.

## Interpretation and limits

The five scorer reports are matched to the ranking by their SHA-256 hashes.
Their prediction hashes must match both the raw native predictions and the
corresponding native manifests. The manifests must identify exactly the
published model, selected revision, adapter version and CAL. The 231-item
rank remains an independent public subset rerun; the card does not claim an
official closed-set JevBench rank.

The packager validates a reviewed rights **declaration**; it does not itself
decide third-party rights. Restricted upstream source text is never copied.
The release gate is a signed-off process receipt, not cryptographic proof of
editorial judgment or GPU numerical parity. Its evidence files and the frozen
prediction corpus remain outside the public package. If the authored set,
training/evaluation overlap audit, parity, full evaluation, or threshold
review is missing or blocked, leave the model unpublished.
