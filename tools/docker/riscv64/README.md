# Native RISC-V standalone Router

This build runs the Router on RISC-V Linux with model inference configured
separately. It does not build Envoy, Dashboard, the Operator or a local model
runtime. The default bundled image target and published platform inventory are
unchanged.

## Requirements

- Native `riscv64` Linux, Python 3.10+, Docker Engine with BuildKit/buildx's
  `docker` driver. The local SDK image must be visible to the image builder.
- A glibc-compatible host and the standard RISC-V GNU user ISA required by the
  upstream Go/Rust binaries; no vendor extensions are enabled.
- Network access to Docker Hub, Debian packages, Go/Rust downloads, Go modules,
  GitHub and crates.io. The optional smoke fixture also needs PyPI.
- Sufficient disk space and compilation time. Go/Cargo concurrency is limited
  to two to keep the native SDK usable on memory-constrained machines.

The Debian RISC-V manifest is pinned by digest. Go 1.25.14 and Rust 1.90.0 release
archives are pinned by URL and SHA256 in `toolchains.json`. GLIDE source and its
Rust dependency lock are documented in [valkey-glide](../valkey-glide/README.md).
Debian package repositories remain live; this is a reproducible procedure,
not a promise of byte-identical images on every date or support for every board.

The SDK uses BusyBox tar to avoid GNU tar's `fchmodat2` dependency with older
container seccomp profiles. Extraction permissions are checked without relaxing
the host's sandbox. See the [Docker seccomp update](https://github.com/moby/moby/pull/47341).

## Build

Run at repository root on the native machine:

```bash
python3 tools/docker/riscv64/build.py --plan
python3 tools/docker/riscv64/build.py --component router --tag public \
  --output /tmp/vsr-router-build
```

Use a fresh output directory. The runner builds `semantic-router-native-sdk:public`
and `semantic-router-router:public`, recording commands, logs, exit codes and the
final image ID. Both builds use `--no-cache`, without importing module caches or
host toolchains. `--component sdk` builds only the SDK.

`--base-image REGISTRY/debian@sha256:DIGEST` selects a pinned mirror of the RISC-V
base, and `--go-proxy URL` selects a module mirror. Configure registry access and
HTTP/HTTPS proxies outside the repository; standard Docker proxy build arguments
are forwarded from the invocation environment. `--network host` is available
for operator-managed proxies. Do not prune shared host images to test cold builds.

## Run

Copy `standalone.yaml` outside the checkout. Set `provider_model_id` to the
backend's served model ID and `base_url` to its OpenAI-compatible API base.
The backend must be reachable from inside the container; `localhost` refers to
the container. `openai_compat_gateway` accepts response extensions such as
llama.cpp timing fields. Use HTTPS or an access-controlled trusted network.

```bash
mkdir -p "$HOME/.local/share/vsr-riscv"
cp -i tools/docker/riscv64/standalone.yaml "$HOME/.local/share/vsr-riscv/config.yaml"
# Edit the copied configuration before running this command.
docker run --detach --name vsr-riscv --pull never \
  --publish 127.0.0.1:8899:8899 \
  --volume "$HOME/.local/share/vsr-riscv/config.yaml:/app/config/config.yaml:ro" \
  semantic-router-router:public -gateway=standalone -listener-address=0.0.0.0
curl --fail --max-time 10 http://127.0.0.1:8899/ready
curl --fail-with-body --max-time 120 http://127.0.0.1:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Say hello."}],"max_tokens":64,"stream":false}'
```

For a protected backend, export `VSR_BACKEND_API_KEY` locally and add
`--env VSR_BACKEND_API_KEY` to `docker run`; the example contains only its variable
name. Docker administrators can read container environment values. Unauthenticated
backends need no key. The loopback publication protects host access independently
of the container's bind address; do not expose management ports publicly.

Set `stream` to true and add curl `--no-buffer` to check SSE. Readiness alone does
not test backend connectivity, credentials or generation. Inspect logs with
`docker logs vsr-riscv`, restart with `docker restart vsr-riscv`, and stop with
`docker stop vsr-riscv`. These commands do not supervise the external model service.

## Minimal native smoke

This test directly starts the Router image; it does not require an installed CLI.
Prepare Python 3.11+ and a separate environment for the maintained provider fixture:

```bash
VSR_SMOKE_WORK=$(mktemp -d)
python3 -m venv "$VSR_SMOKE_WORK/provider"
"$VSR_SMOKE_WORK/provider/bin/pip" install \
  -r tools/test/services/provider-mocker/requirements.txt
python3 tools/test/smoke/riscv64-standalone.py \
  --image semantic-router-router:public \
  --provider-python "$VSR_SMOKE_WORK/provider/bin/python" \
  --output "$VSR_SMOKE_WORK/results"
```

Use the same native Docker daemon as the build. The test checks the actual image
ID, keyword/default routing, buffered responses and SSE content/termination, then
restarts the container and repeats the four requests. It uses unique resource
names and loopback host ports; only its own container, provider process and
network are removed. Results and logs remain in the output directory.

The deterministic fixture tests transport and restart, not model inference,
classifier quality, CLI lifecycle, configuration persistence or incremental SSE
timing. Those remain separate integration checks.
