# JevArena release bundle gate

`publication.bundle_arena` assembles a Decision 2.0 Hugging Face model
directory only after the six-axis release artifacts and a separate release
review have passed. It does not train, score, call a GPU, upload to Hugging
Face, or replace an unqualified checkpoint with a release model. The earlier
`publication.bundle` remains a four-family Qwen3.5 audit artifact.

## Inputs

Run `python3 -m publication.generate_arena` first. Give the packager a
**functional native** model directory and these local files:

| Input | Required binding |
| --- | --- |
| `artifacts` | Generator `/3` manifest, score table and six hash-matched SVGs. |
| `arena_rank`, `public_rank` | Same-panel release rankings whose bytes match the generator manifest. |
| `package_record` | Reviewed model ID, selected revision, architecture, immutable base revision, actual parameter inventory, TRAIN/SELECT/CAL counts, TRAIN language counts, evaluation language scope, source terms and limitations. |
| `score_inputs` | For each of `synthetic`, `css`, `public`, `dbv4`, `authored`: the scorer report, exact scored prediction file and native prediction manifest. |
| `parity_receipt` | Gold-free native package check on 1,600 DEV and 1,430 transfer pilot items, with zero changed categorical answers, p99 probability drift ≤0.005 and maximum drift ≤0.02. |
| `release_gate` | Passed pretest freeze, authored editorial, overlap, same-panel evaluation, rights, parity and performance reviews, each with an evidence hash. The packager rechecks the copied gate against the copied record, parity and artifact manifest. |
| `provenance_inputs`, `freeze_manifest`, `gate_evidence` | The original local files matching every declared training, freeze and review evidence hash. They are checked but never copied. |
| `base_source`, `adapter_source_parity_receipt` | Required only for `qwen-external-base-peft`: a local snapshot of the exact pinned upstream commit and the separate BF16 scored-source/package parity receipt. Neither input is copied. |

The record and gate schemas are `decision2-release-package-record/2` and
`decision2-jevarena-release-gate/1`. The test fixture in
`publication/tests/test_bundle_arena.py` gives a complete small example; its
model size and editorial judgments are synthetic CPU fixtures, never release
evidence. A real package counts
every active safetensors tensor from its header, subtracting only explicitly
named buffers. Every other safetensors file must be classified as a support
weight file. The count must agree exactly with the record and JevArena roster
and be within 25% of the nominal model name. The record's per-language TRAIN
counts must sum to the exact TRAIN row count; the card states the evaluation
language coverage so English-heavy panels are not read as multilingual proof.

`python3 -m publication.panel_parity` builds the private full-panel
`parity_receipt` from already sealed, **gold-free** source and packaged
predictions for the exact DEV1,600 and CSS pilot1,430 prompt rosters. Supply
the inner package manifest, each panel's prompt file and both prediction
files, plus a new private 0700 output directory. Each prediction file must
have its companion `.manifest.json` binding the prompt/model/CAL hashes; the
package prediction manifest must also bind the inner package SHA. The tool
compares the scorer's actual Choice, Noul and Score point decisions, includes
invalid and over-budget outputs, computes p99 over individual probability and
Score scalars, and writes two detailed private reports plus
`native-parity.json`. It refuses incomplete rows, changed input order or
unbound prediction files. The package gate accepts this receipt only if both
panels pass the **unchanged** zero-decision-change, p99 ≤0.005 and max ≤0.02
thresholds. The comparison does not prove that the two prediction files were
run on comparable devices; retain the external runtime and chronological
attestation separately. Never copy the detailed reports or raw predictions
into a public model repository.

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.panel_parity \
  --package-manifest /private/package/MODEL_MANIFEST.json \
  --dev-prompts /private/dev.prompts.jsonl \
  --dev-source /private/dev.source.predictions.jsonl \
  --dev-package /private/dev.package.predictions.jsonl \
  --css-prompts /private/css-pilot.prompts.jsonl \
  --css-source /private/css-pilot.source.predictions.jsonl \
  --css-package /private/css-pilot.package.predictions.jsonl \
  --output-dir /private/new-0700-parity-directory
```

The pretest freeze uses `decision2-jevarena-pretest-freeze/1`. It names the
selected model revision, native model, model-file and calibration digests,
native adapter version and SHA-256, the complete candidate model/revision
roster digest, the authored selection-lock digest, and the `jevarena-ranking/2`
formula with its exact source SHA-256. It records a UTC freeze time. The
private `candidate_freeze` audit uses
`decision2-candidate-freeze-audit/1` and binds the freeze, selected candidate,
formula, and the exact native-manifest and prediction digests for the
protected typed, transfer and authored FINAL panels. Its declared sequence
must satisfy **freeze < every protected prediction seal < first protected
label access**. The packager checks the identities and this chronology; an
independent release owner must inspect the separate append-only timestamp log
to establish that the asserted times are genuine. The current prediction
collector does not itself provide a trusted UTC timestamp, so a self-filled
audit JSON cannot qualify a release without that external attestation.

The private `authored_editorial` evidence must use
`decision2-authored-editorial-receipt/1`. A bare `passed` flag and file hash do
not qualify. Its 1,200–1,480 rows identify distinct independent originals,
native type, source family, author, domain, operation, document form, template
and length band. Original and paired views each declare their frozen native
answer vocabulary, allowing any valid ordered Score rubric length and checking
Choice labels against the actual criteria. Each row contains two original blind reviews, one paired
blind review, a second paired review for at least 15% of each type, and a
separate adjudication. Each blind review records a direct native answer,
citations to both sources, paragraph inspection and explicit ambiguity,
realism, shortcut and rights judgments. The adjudication records agreement
with the independently held oracle, semantic independence, overlap,
provenance and rights decisions. The gate requires at least 360 accepted
originals per native type, the preregistered domain/operation/template caps
and length mix, no unresolved material errors, distinct opaque identities per
case, and review seals before key access followed by adjudication. The
receipt's prompt, target and pretest-freeze hashes must match the frozen
release panel and the separate freeze file. The freeze and editorial receipt
also commit to the same SHA-256 of the sorted, unique original-ID roster and
the canonical per-row native answer contracts; the packager recomputes both
digests from every row. Reviewer answers, citations,
identity commitments and the private oracle stay outside the public package.
**The program can check receipt structure, declared identities and chronology;
it cannot establish that these identities are real humans, that the reviewers
were blind, or that their evidence citations are correct. An independent
release owner must verify those facts and retain the signed private records.**

The packager accepts these functional layouts:

- `qwen3.5-decision-head` and `qwen3.8-decision-head`: full merged
  `model/backbone/`, decision head, tokenizer, CAL, materialization receipt,
  and all eight portable `decision2/` runtime modules. It constructs the
  existing native `MODEL_MANIFEST.json` inside the copied `native/` directory
  and runs that runtime's CPU byte/lineage verifier before publication.
- `qwen-external-base-peft`: the unmerged output of
  `publication.adapter_bundle` with its own exact `MODEL_MANIFEST.json`,
  LoRA adapter, Decision 2.0 head, tokenizer, CAL and pinned native loader.
  Supply `base_source` and `adapter_source_parity_receipt`. The packager
  verifies every local base snapshot file against the inner manifest and
  requires its repository ID and 40-character commit to match the reviewed
  package record. The scorer manifests for all five score families must bind
  the same inner manifest with `package_manifest_sha256`. The separate
  gold-free BF16 source/package receipt must report all three native types,
  zero missing or categorical differences and maximum numeric drift ≤`1e-4`.
  It is distinct from the 1,600 DEV plus 1,430 CSS pilot package parity gate.
  The output does **not** contain base weights or merged weights. Its count
  includes the externally loaded text backbone, LoRA and custom head;
  `active_weight_files` must list exactly the adapter and head, with empty
  support and excluded-buffer lists. Its `native_identity.scheme` is
  `external-peft-checkpoint-fingerprint` with a null `file` and the scored
  PEFT model digest. Ordinary `AutoModel` loading does not run the head; use
  `decision2.Decision2.from_pretrained` from the `native/` directory and the
  pinned base.
- `qwen3.5-semif`: standalone merged SemIf model with original `serve.py`,
  tokenizer, CAL, notices and exact `SHA256SUMS` coverage.
- `encoder-decision`: native `model.safetensors`, encoder configuration,
  tokenizer, configuration and runtime entrypoint. Duplicate source encoder
  weights can be marked as support files; a checkpoint explicitly marked
  `research_only` is refused unless a new receipt independently marks it
  release-qualified.

All layouts need an external gold-free parity receipt for the **exact copied
model files**. The external-base profile additionally needs the adapter
source/package parity receipt. A pinned local byte inventory alone cannot
prove which HF repository owns the snapshot: independently review the HF
commit and license, and retain that evidence in the rights/provenance gate.
An encoder or SemIf layout check does not prove the runtime can
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
For the external-base profile, add `base_source` and
`adapter_source_parity_receipt` to the JSON config. The source parity receipt
is checked locally and only its SHA-256 is copied into
`PACKAGE_MANIFEST.json`; the possibly environment-specific receipt itself
remains private. The outer manifest also records the immutable upstream
reference and its per-file hashes. Verification of a downloaded publication
bundle without `base_source` checks internal bytes and the immutable reference;
pass `base_source` to recheck the complete external dependency bytes offline.

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.bundle_arena \
  --config release-package-config.json --output new-model-directory
```

The output must not exist. A successful assembly copies the native model to
`native/`, writes the new model card with the same-panel table and figures,
copies reviewed release receipts, and records every public file hash in
`PACKAGE_MANIFEST.json`. A failed check leaves no output directory. A copied
bundle can be checked again with `publication.bundle_arena.verify(path)` or,
for an external-base model,
`publication.bundle_arena.verify(path, base_source=pinned_snapshot)`.
The record, parity, gate, freeze and artifact manifest are each parsed and
hashed from a single byte snapshot. Copied record, parity, gate and artifact
manifest files use exactly those validated bytes. Model and artifact copies
are checked against the prevalidated inventory, and input receipts and
external evidence are rechecked before the package is atomically renamed.
`verify()` checks the internal gate/record/parity/artifact cross-bind and the
release declarations again, even when an attacker recomputes the outer file
inventory. External private review files are intentionally absent and require
the separate release audit at publication time.

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
