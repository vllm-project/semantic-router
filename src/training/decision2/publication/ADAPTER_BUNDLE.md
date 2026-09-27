# Unmerged PEFT publication candidate

The 27B English Score v6 pilot showed why a CPU FP32 LoRA merge cannot inherit
the selected adapter's BF16 scores: the merged model failed native 32-row
parity. `publication.adapter_bundle` stages a **separate, unmerged candidate**.
It does not change the full-materialization builders, the JevArena release
gate, or the failed pilot result.

## Contract

The candidate contains the scored Decision 2.0 checkpoint's LoRA tensors,
head and tokenizer, its exact CAL file, and the native PEFT inference modules.
It does not copy the upstream base weights. `MODEL_MANIFEST.json` pins a
40-character upstream HF commit and SHA-256 for every source snapshot file,
every packaged model and loader file, and supplied exact Python and package
versions. A normal Python import may create `decision2/__pycache__`;
verification ignores only bytecode for a pinned source module in that
directory. Every other unlisted file still fails the inventory gate. The
staged loader uses `DecisionModel.from_checkpoint` with the
original PEFT adapter and refuses a missing or changed base. When no local
source is supplied, it may download **only** the enumerated files from that
commit and verifies all resulting bytes. Base, adapter and head tensors are
counted separately from safetensors headers. The headline count includes the
entire loaded text backbone, even though its weights are an external
dependency; the discarded upstream vision and language-generation head are
not attributed to the Decision 2.0 inference model.

This prototype supports Qwen `base` and `posttrained` PEFT source kinds with
safetensors weights. A Decision 1.0 or Decision 2.0 full-checkpoint source
needs a separate portable dependency contract. Source revision identity is a
declared HF repository and commit plus byte inventory; before public release,
independently confirm that the local snapshot came from that HF commit. An
offline file hash cannot prove repository ownership.

Prepare an exact runtime lock from the **scored** environment (Python,
Torch, Transformers, PEFT, safetensors and Hugging Face Hub). The current
scored prediction manifest verifies only Torch and PEFT versions; the other
supplied lock versions need an independently sealed runtime receipt before
release. The native scored prediction manifest must match the checkpoint
fingerprint, model-file map, inference source hashes, PEFT and Torch versions,
CAL hash, and context limit. The builder checks only these identities; it does
not verify score quality. Copied text and safetensors metadata are screened for
credential-like text, private paths and IP addresses before staging.

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.adapter_bundle \
  --checkpoint CHECKPOINT --source PINNED_BASE_SNAPSHOT \
  --calibration CAL_REPORT --scored-manifest NATIVE_PREDICTION_MANIFEST \
  --dependency-lock SCORED_RUNTIME_VERSIONS \
  --base-repo-id OWNER/BASE --base-revision FORTY_HEX_COMMIT \
  --model-id llm-semantic-router/dev-2.0-27b \
  --output NEW_ADAPTER_PACKAGE
```

This creates a local artifact suitable for later HF upload preparation; it
does **not** upload or publish anything. The output must not exist. A failed
build leaves no output package. Its generated README makes no score claim.
Only copy the reviewed artifact after every gate below passes.

## Native parity before any score or release claim

On the authorized GPU runtime, use a gold-free roster with Choice, Noul and
Score. The script loads the original scored checkpoint with the same native
BF16 backbone/FP32 head path, then the staged package's copied loader with
the same pinned source and CAL. It compares all answers, option argmax and
continuous values. The fixed gate requires no invalid/missing answers, zero
categorical mismatches and maximum absolute probability/Score drift at most
`1e-4`. The private receipt contains only counts, hashes and aggregate drift.

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.adapter_parity \
  --package NEW_ADAPTER_PACKAGE --checkpoint CHECKPOINT \
  --source PINNED_BASE_SNAPSHOT --calibration CAL_REPORT \
  --scored-manifest NATIVE_PREDICTION_MANIFEST \
  --prompts-path GOLD_FREE_THREE_TYPE_ROSTER --output NEW_PRIVATE_RECEIPT
```

The parity roster, checkpoint, scored prediction manifest, CAL and receipt
remain private. Do not use this command with final labels. A failed parity
receipt is evidence of a block, not an invitation to relax the threshold.
The package API supplies default Noul criteria when callers omit them, while
the scored native collector rejects that form. Exact-output parity applies to
the canonical explicit-criteria roster; test that convenience form separately
before documenting its behavior for callers.

## Package-native JevArena predictions

`publication.package_native_arena` executes the **copied package loader** and
its copied question validator. It does not import a current development
`training.model` module for inference. The CLI requires a local immutable base,
the frozen package-manifest SHA-256, a model ID and an immutable model revision.
Until a model repository commit exists, use the explicit
`package-sha256:MANIFEST_DIGEST` development revision; a release run uses its
actual 40-character model commit. The package supplies CAL, context limit and
temperature by type; the CLI offers no overrides.
Create the output directory with mode `0700` before running; the collector
requires that mode and creates prediction and manifest files with mode `0600`.

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.package_native_arena \
  --package NEW_ADAPTER_PACKAGE --source PINNED_BASE_SNAPSHOT \
  --expected-package-sha256 MANIFEST_DIGEST \
  --input GOLD_FREE_PANEL.jsonl --output NEW_PREDICTIONS.jsonl \
  --model-id llm-semantic-router/dev-2.0-27b \
  --model-revision package-sha256:MANIFEST_DIGEST --device cuda:0
```

The input contains only `id`, `state` and `questions`; answer fields are
rejected before GPU loading. Invalid questions are passed through the copied
native validator and counted as failures. Missing, malformed, tied Choice and
over-budget answers remain explicit invalid rows. The output has the same
row-level input hashes and gold-free JSONL/companion-manifest shape as the
native scored collector, including `adapter_version` set to
`decision2-peft-package-native-v1`, `package_manifest_sha256`, full model and
base identities, CAL hash and per-question counts. Frozen input and package
bytes are checked again after inference, before the atomic output appears.
The scorer still checks panel identity, invalidity and answer correctness;
the collector does not read gold or assign accuracy. This entrypoint does not
by itself satisfy the authored panel, transfer, parity, runtime, rights or
release gate.

## Remaining release integration

`publication.bundle_arena` currently accepts only a self-contained Qwen
checkpoint for Decision 2.0, so it will reject this external-base artifact.
To publish an adapter-preserving model, extend that release packager and its
package record with a separately reviewed external-base profile, verify the
upstream HF revision and rights, import the exact package runtime in the
release environment, bind a passing gold-free parity receipt to the package
manifest, and rerun the complete same-panel release evaluation and candidate
gate. Frozen candidates, independent authored questions, multilingual review,
overlap audit, threshold review and model-card evidence remain required.
Freeze the final sticker, card and chart files in a publication inventory
before uploading; adding them to this prototype changes its exact file
roster. Verify a downloaded HF snapshot and ordinary package import after
upload. Neither a syntactically valid 40-character revision nor matching
local bytes alone proves the claimed upstream repository owns that snapshot.
Neither the merged pilot's failed scores nor the source adapter's existing
development scores transfer to this package without native package parity.
