# Eikos clean-v2 4B uppercase-ID requalification

This is a prospective **technical** rerun of the unchanged checkpoint and
standalone package after the public repository ID changed to
`llm-semantic-router/DEV2.0-4B`. The earlier lowercase-ID execution receipts
remain historical and must not be relabeled. No typed FINAL or CSS evaluation
gold is used here; DEV 1,600, CSS **pilot** 1,430 and public JevBench 231 are
the only panels. A pass permits a new candidate lock and pretest roster; it is
not a JevArena v3 score or release approval.

## Inputs and execution

Use `scripts.requalify_eikos4b_v3` from an **exact mirror** of the signed local
source commit, inside the previously qualified immutable ROCm image, on one
reserved physical GPU. Its owner-private JSON config has these required keys:

| Key | Meaning |
| --- | --- |
| `model_id`, `checkpoint` | Exactly `llm-semantic-router/DEV2.0-4B`, `checkpoint-0232` |
| `source_commit`, `source_root` | Signed 40-digit local commit and the mirrored `src/training/decision2` directory |
| `runtime_image_id`, `physical_gpu`, `device` | Immutable image digest, `{ "guid": "…", "index": N }`, and `cuda:0` |
| `source_model`, `source_release_manifest`, `run`, `package` | Pinned Eikos source, release manifest, completed clean-v2 LoRA run, standalone merged package |
| `rights_attestation` | Exact private rights receipt if required by the package; omit for a clean package |
| `dev_prompts`, `css_pilot_prompts`, `public_panel` | Complete original-order prompt files and public panel directory |
| `dev_gold`, `css_pilot_gold` | Previously open diagnostic labels only; public targets come from `public_panel` |

All file hashes, item counts and the source adapter hashes are hard-coded from
the signed full-comparison preregistration. The runner hashes the executing
modules and records the exact source commit, model ID, image ID and physical
GPU in owner-private `frozen.inputs.json`. Compare those module hashes with
the local checkout before accepting the new candidate lock. Inject any
credentials through the private environment; the config and logs must remain
outside the repository and public gist. Select exactly one reserved ROCm GPU
with `ROCR_VISIBLE_DEVICES` and expose it as `cuda:0` in the container.

Run once with absent, owner-private output paths:

```bash
python -m scripts.requalify_eikos4b_v3 \
  --config '<owner-private-config.json>' --preflight-only

python -m scripts.requalify_eikos4b_v3 \
  --config '<owner-private-config.json>' \
  --output-root '<new-owner-private-output-directory>'
```

The runner first executes two independent fresh `published_infer` processes
over all 1,430 CSS pilot prompts and audits zero categorical changes plus
maximum option-probability drift at most `1e-6`. It then runs same-process
selected-LoRA/package parity on all DEV 1,600 and CSS pilot 1,430 prompts,
requiring zero category changes, p99 drift at most `0.005` and worst drift at
most `0.02` on **each** panel. Only after these gold-free checks pass does it
run package-only DEV, CSS pilot and public 231 inference and their respective
designated scorers. Every prediction must be present, original-order,
input-digest and uppercase-ID bound; the public score additionally requires
231 strictly valid, zero-renormalized answers. No score threshold is tuned or
applied. All GPU processes use deterministic algorithms and the attested
Qwen3.5 PyTorch-reference gated-delta implementation.

The immutable outputs are `repeat/execution.receipt.json`,
`full/execution.receipt.json`, and `full/parity-dev-css.receipt.json`, plus
their hashed predictions, manifests, parity reports, scores and private logs.
Both execution receipts include GPU-hours. A failed stage writes a failed
receipt and stops before the next stage; do not retry under the same result
name, relax a threshold, change a prompt, or select a different checkpoint.
After a pass, verify the three receipt hashes through
`scripts.eikos_stable_runtime_v3.verified_stable_runtime` and the normal v3
candidate-lock checker before preparing any sealed predictions. Preserve the
old receipts and all failed new receipts for audit.
