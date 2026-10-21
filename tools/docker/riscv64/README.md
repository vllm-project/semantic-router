# Native RISC-V standalone Router

This build runs the Router on RISC-V Linux with model inference configured
separately. It does not build Envoy, Dashboard, the Operator or a local model
runtime. The default bundled image target and published platform inventory are
unchanged.

## Requirements

- Native `riscv64` Linux, Python 3.10+, Docker Engine with BuildKit/buildx.
  The `docker` buildx driver is required for local SDK and Router-binary image
  handoffs, including when reusing an SDK with `--sdk-image`.
- A glibc-compatible host and the standard RISC-V GNU user ISA required by the
  upstream Go/Rust binaries; no vendor extensions are enabled.
- Network access to Docker Hub, Debian packages, Go/Rust downloads, Go modules,
  GitHub and crates.io. The optional smoke fixture also needs PyPI.
- Sufficient disk space and compilation time. New SDKs default to two Go/Cargo
  workers; `--jobs N` changes this according to the machine's available memory.

The Debian RISC-V manifest is pinned by digest. Go 1.25.14 and Rust 1.90.0 release
archives are pinned by URL and SHA256 in `toolchains.json`. GLIDE source and its
Rust dependency lock are documented in [valkey-glide](valkey-glide/README.md).
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
then the binary image `semantic-router-native-router:public` using this
directory's `Dockerfile`, and the runtime `semantic-router-router:public` using
`../Dockerfile.attached`. Commands, logs, exit codes and the
final image ID are recorded. Normal Docker layer caching is enabled. Public archives remain
checksummed, and no private host toolchain is copied into the build.
`--component sdk` builds only the SDK. For a fresh-layer qualification, explicitly
pass `--no-cache`; use an isolated builder for an empty-cache test, rather than
pruning a shared host.

Reuse a previously verified SDK when rebuilding Router code:

```bash
python3 tools/docker/riscv64/build.py --sdk-image semantic-router-native-sdk:public \
  --tag local --output /tmp/vsr-router-local-build
```

This skips the SDK build. A local image must exist in the selected builder; a
remote SDK should be pinned by digest. It must provide native RISC-V Go, Rust,
Python and Debian-compatible build tools. `--jobs` applies to a newly built SDK,
not an existing image. Override tool versions through the checked input manifest,
updating URLs, SHA256 and installer filenames together.

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
  --user 65532:65532 --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,mode=1777 \
  --tmpfs /app/config:rw,noexec,nosuid,uid=65532,gid=65532,mode=0700 \
  --tmpfs /app/models:rw,noexec,nosuid,uid=65532,gid=65532,mode=0700 \
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
"$VSR_SMOKE_WORK/provider/bin/pip" install --upgrade pip
"$VSR_SMOKE_WORK/provider/bin/pip" install \
  -r tools/test/services/provider-mocker/requirements.txt
python3 tools/test/smoke/riscv64-standalone.py \
  --image semantic-router-router:public \
  --provider-python "$VSR_SMOKE_WORK/provider/bin/python" \
  --output "$VSR_SMOKE_WORK/results"
```

Use the same native Docker daemon as the build. The test checks the actual image
ID, non-root UID, read-only root filesystem, keyword/default routing and buffered/
SSE content/termination, then
restarts the container and repeats the four requests. It uses unique resource
names and loopback host ports; only its own container, provider process and
network are removed. Results and logs remain in the output directory.

The deterministic fixture tests transport and restart, not model inference,
classifier quality, CLI lifecycle, configuration persistence or incremental SSE
timing. Those remain separate integration checks.

The writable tmpfs directories hold temporary files, configuration history and
the unused model-cache mount; they are not persistent storage. The mounted input
configuration remains read-only. This does not qualify every possible plugin
or model under a read-only deployment.

## CI artifacts

The `vllm-sr-riscv64` artifact uses the native build runner. Install `skopeo`
alongside Docker/buildx to export its result in the shared CI format:

```bash
python3 tools/docker/riscv64/build.py --tag local \
  --output /tmp/vsr-native-build \
  --artifact-directory /tmp/vsr-native-artifact
python3 tools/ci/image_artifacts.py verify --image vllm-sr-riscv64 \
  --directory /tmp/vsr-native-artifact
```

Use fresh directories. This builds the SDK, Router binary and attached runtime,
then exports `image.tar` and `manifest.json`. The shared artifact verifier checks
the source revision, archive checksum and actual Linux/RISC-V64 platform.
Artifact export requires a clean Git checkout; keep build outputs outside it.
The source revision is checked before the build and again before sealing.
Artifact export requires the default Docker context. To select another local
daemon, set `DOCKER_CONTEXT=default` and `DOCKER_HOST` so Docker and Skopeo use
the same endpoint.
`--artifact-mode` selects `pr` (default), `main`, `nightly` or `release`; for a
release, `--tag` must match the release version.

The generic buildx workflow and hosted AMD64 artifact loader reject this native
build strategy. A native worker
can build and qualify the artifact before it enters the project's publication
process. Official registry publication remains to be coordinated with maintainers.

## Dependency updates

The GLIDE lock file is necessary because the pinned release omits the FFI lock
and the Go module omits the RISC-V static archive. Its Go version, source hash
and Rust lock must be updated together as described in the dependency README.
A future upstream RISC-V archive can replace this source fallback after its
compatibility is verified; the fallback adds no stub or feature-disable path.
