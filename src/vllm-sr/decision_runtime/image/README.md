# Decision runtime candidate images

These images package the repository's Decision service separately from the
Router. They contain no model weights, downloaded artifacts, credentials, or
model-repository Python. `vllm-sr drun` verifies a data-only artifact on the host
and mounts it read-only at `/opt/vllm-sr/decision-artifact`.

The image contains both Python paths selected by the current `drun` catalog
bridge:

| Model family | Image Python | Transformers |
| --- | --- | --- |
| Vela (Kai, Lex) | `/opt/vllm-sr/venvs/vela/bin/python` | 4.57.6 |
| Qwen3.5 (Eos, Sol, Nox, Lux) | `/opt/vllm-sr/venvs/qwen35/bin/python` | 5.17.0 |

The environments inherit one base Torch installation. They keep incompatible
Hugging Face Hub and tokenizers major versions separate. The ROCm Qwen
environment also contains FLA 0.5.2. The installed runtime profile selects
native Torch GatedDeltaNet and gated norm functions for Eos on its own model
instance. Sol, Nox, and Lux retain the accelerated path and its strict ROCm FLA
launch-profile checks. The Eos choice follows its stable runtime profile across
model revisions; model-file versions and hashes come from the selected
artifact's verified manifest. A successful image build does not establish ROCm
semantic parity or performance. Static inspection of the installed Transformers
source confirms the intended kernel dispatch but is not a substitute for live
GPU semantic and performance qualification of the exact candidate image and
source commit.

The Eos native Torch ROCm profile separately caps `--max-batch` at 8 physical
rows, the current strict-semantic envelope. A larger launch batch is rejected
before model loading; raising that profile limit requires new old-reference
semantic and performance evidence for the exact candidate, not a model-file
revision allowlist. Sol, Nox, and Lux keep their independent strict FLA guard.

The six packaged profiles use stable model names and carry implementation
policy. Their historical manifest digest, size, and file inventory fields are
retained for synthetic tests; production artifact resolution observes the
selected snapshot's manifest and selects files from that inventory.

## Build

Use a digest-qualified OCI base image with Python 3.12 and pip. CPU and CUDA
builds install PyTorch 2.12.0 from the corresponding official wheel index. A
ROCm base must already contain the intended Torch/HIP/Triton stack, and its
Python executable must import that Torch build. All builds currently target
Linux amd64.

```bash
image_dir=src/vllm-sr/decision_runtime/image
"$image_dir/build-image.sh" cpu \
  'python@sha256:<verified-base-digest>' decision-runtime-cpu:validation
"$image_dir/build-image.sh" cuda \
  'python@sha256:<verified-base-digest>' decision-runtime-cuda:validation
"$image_dir/build-image.sh" rocm \
  'vllm/vllm-openai-rocm@sha256:<verified-base-digest>' \
  decision-runtime-rocm:validation python3
```

Replace each placeholder with the selected base manifest digest. The ROCm
Python path depends on the selected base image. The helper rejects mutable base
references and `latest` output tags. It labels the source commit, dirty state,
and backend for review. The build checks the packaged model catalog and profiles,
both `drun` Python paths, exact Transformers versions, and the shared Torch
identity. It does not download a model.

The wheel-builder stages the canonical built-in catalog from
`config/recipes/built-in` before creating the wheel. Build release candidates
from a fresh clean checkout and verify the wheel's catalog resources and image
smoke; `source-state=clean` alone does not attest ignored developer-side
`cli/model_assets` files.

The CLI accepts only an image reference of the form
`registry/repository@sha256:<manifest-digest>`. A local `docker build` image ID
is not a registry manifest digest. Publish a validated candidate through the
release workflow, record its registry digest, then pass that digest to
`vllm-sr drun run MODEL --backend BACKEND --image REFERENCE`. Source checkouts
deliberately have no default image inventory. Release packaging may inject
`cli/decision_runtime/decision-images.lock.json` only after backend, model,
performance, and supply-chain qualification of the exact image digests.

## Publication gates

The CPU image is part of the normal immutable CI image handoff. Changes to the
Decision image build select `decision-runtime-cpu` for PR builds. The main,
nightly, and release planners also build the CPU image when applicable. The
image uses a digest-qualified Python base, records the source commit and
backend in its OCI configuration, and is sealed and verified before upload.
The trusted publisher copies the tested OCI artifact without rebuilding and
uploads `ci-published-digest-decision-runtime-cpu` with the registry digest.
Until model-backed qualification of that exact digest, it publishes only the
immutable source-commit tag, even when the general publisher requests `latest`
or a release alias.
Image construction runs import and catalog smoke checks; this does not replace
real Kai, Lex, and Eos CPU serving tests before making the image a CLI default.

The ROCm image is approximately 46 GiB in virtual layers. It is intentionally
absent from ordinary hosted CI build and artifact matrices. Build it from the
same clean source commit on a capacity-appropriate ROCm host, push the candidate
to `ghcr.io/<owner>/semantic-router/decision-runtime-rocm-staging` by digest,
then run all six Decision models from that exact digest on a ROCm device. Keep
one JSON evidence file per model. Each file must identify the candidate digest,
source commit, immutable selected model revision and artifact content ID, ROCm
execution, successful health, Noul/Choice/Score, and a mixed two-state batch.
The qualification receipt lists the six canonical model IDs, evidence file
paths relative to the receipt, and the SHA-256 of each evidence file. The
selected model revisions come from the artifact resolver; no version or hash
allowlist is compiled into the runtime.

The top-level receipt has `schema: 1`, `image: decision-runtime-rocm`, the
current `source_sha`, the pinned `base_image`, `backend: rocm`,
`platform: linux/amd64`, and a digest-qualified `candidate_ref`. Its `models`
array has exactly one row for each of Kai, Lex, Eos, Sol, Nox, and Lux. Each row
provides `id`, the selected 40-character `revision`, an
`artifact_content_id` of the form `sha256:<64 hex>`, a relative
`evidence_file`, and that file's `evidence_sha256`. The evidence JSON repeats
`image_ref`, `source_sha`, `model_id`, `revision`, and
`artifact_content_id`, records `backend: rocm`, `device: rocm`, and
`result: passed`, and includes these boolean checks:
`artifact.identity`, `health`, `single.noul`, `single.choice`,
`single.score`, `batch.mixed_two_states`, and `rocm.execution`. Keep the
underlying HTTP and runtime logs with the private receipt for review.

After the six-model and performance review, validate the receipt and registry
configuration from the matching source commit:

```bash
python3 tools/ci/decision_rocm_promotion.py \
  --receipt /path/to/private/qualification.json --owner vllm-project
```

The validator rejects mutable tags, missing or changed model evidence, a
different source or base, and a candidate whose registry digest or OCI labels
disagree with the receipt. On a protected `main` run with registry write
credentials, `--promote` copies that exact digest to the official ROCm package
under its source-commit tag; it does not rebuild. The ROCm promotion is still
pending a capacity-appropriate build host, real six-model hardware and
performance receipts, and registry write access. A standard GitHub hosted
runner is not assumed to provide these.

Once each backend has a published, tested registry digest, the protected
release packager generates the image lock for the exact source commit and
builds the wheel with that resource. The lock maps only qualified backends to
digest-qualified OCI references. Verify the installed wheel's `drun` behavior
and lock before publishing it. Until then an explicit `--image` remains
required; no source-code change or second PR is needed to add a qualified
release digest.

For this rollout, CPU qualification is scoped to Kai, Lex, and Eos. CUDA image
construction is a candidate path; model qualification remains deferred. ROCm
backend capability checks and verified artifact manifests decide whether a
selected immutable revision can launch on a particular device.
