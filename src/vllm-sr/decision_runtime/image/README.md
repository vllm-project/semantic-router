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
environment also contains FLA 0.5.2. The qualified artifact's strict runtime
profile still checks exact Torch, HIP, Triton, FLA, and source identities at
startup; a successful image build is not hardware qualification.

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
  'rocm/pytorch@sha256:<verified-base-digest>' decision-runtime-rocm:validation \
  /opt/venv/bin/python
```

Replace each placeholder with the selected base manifest digest. The ROCm
Python path depends on the selected base image. The helper rejects mutable base
references and `latest` output tags. It labels the source commit, dirty state,
and backend for review. The build checks the packaged model catalog and profiles,
both `drun` Python paths, exact Transformers versions, and the shared Torch
identity. It does not download a model.

The CLI accepts only an image reference of the form
`registry/repository@sha256:<manifest-digest>`. A local `docker build` image ID
is not a registry manifest digest. Publish a validated candidate through the
release workflow, record its registry digest, then pass that digest to
`vllm-sr drun run MODEL --backend BACKEND --image REFERENCE`. Do not populate
`PACKAGED_DECISION_RUNTIME_IMAGES` before backend, model, and supply-chain
qualification is complete.

For this rollout, CPU qualification is scoped to Kai, Lex, and Eos. CUDA image
construction is a candidate path; model qualification remains deferred. ROCm
model profiles and their strict runtime contracts decide which revisions may
launch on a particular device.
