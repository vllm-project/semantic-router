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
every packaged model and loader file, and the exact Python and package
versions. The staged loader uses `DecisionModel.from_checkpoint` with the
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
Torch, Transformers, PEFT, safetensors and Hugging Face Hub). The native scored
prediction manifest must match the checkpoint fingerprint, model-file map,
inference source hashes, PEFT and Torch versions, CAL hash, and context limit.
The builder checks only these identities; it does not verify score quality.

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
Neither the merged pilot's failed scores nor the source adapter's existing
development scores transfer to this package without native package parity.
