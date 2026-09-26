# Build Decision Runtime images

This is a contributor guide for building the container used by
`vllm-sr decision serve`. An image contains the Decision service and its Python
dependencies, but no model weights or credentials. The CLI resolves a selected
model artifact on the host and mounts it read-only into the container.

One image contains two isolated Python environments because the six models use
different Transformers versions:

| Model family | Models | Python environment |
| --- | --- | --- |
| Vela | Kai, Lex | `/opt/vllm-sr/venvs/vela` |
| Qwen3.5 | Eos, Sol, Nox, Lux | `/opt/vllm-sr/venvs/qwen35` |

The environments share the base image's PyTorch installation. Build an image
for the target device; building alone does not prove that a model runs there.

## Build a local candidate

Use a base image pinned to a digest. CPU and CUDA builds install their
corresponding PyTorch wheels. A ROCm base must already contain a compatible
Torch/HIP/Triton stack. These recipes currently target Linux amd64.

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

Replace each placeholder with the base image's actual digest. The ROCm Python
path depends on that base image. The build verifies both Python environments,
the model catalog, and the runtime imports; it does not download models.

For a source-checkout smoke test with Docker, inspect the built image to get
its full local ID, then pass that exact ID with `--image` and
`--image-pull-policy never` to `vllm-sr decision serve`. A local image ID is
only a development artifact; it is not a published registry reference.

## Release requirement

Before a CLI release offers a default CPU or ROCm image, publish and validate
that backend's image against the models it claims to serve. The release
package then records the published image digest for that backend, so users of
the installed CLI can omit `--image`. CUDA and MLX should be listed only after
their executors and device tests are in place. Until publication is complete,
keep default-image claims out of user-facing quickstarts.
