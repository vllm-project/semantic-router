# D3-edge hardware-qualified readiness and NVIDIA L4 validation

On 2026-10-12, D3-edge's native `exact` worker matched the pinned package server
on NVIDIA L4 for 21 requests, including its text and image goldens. Two further
native starts, each with an independent empty Triton cache, reproduced the
captured official responses exactly. All repeats were stable; the largest
numeric difference was **0.0**. Each restart reported **5/5 matched goldens**.

An AVX2/emulated-BF16 CPU worker also served all **21/21** requests with exact
answers and usage, stable repeats and **5/5 matched goldens**, resolving the
reproduced CPU readiness failure.

The managed `vllm-sr serve --engine` Docker/Router path also passed on an
AVX-512/native-BF16 CPU: **21/21** requests matched the same-host package server
exactly, and a complete stack stop/start reproduced that panel. Managed CUDA
container serving remains untested.

## Package and environment

- Router base: `1f0195e87eff4e7d277677aa2d3e59bbbe1b6254`.
- Package: [`vllm-sr/d3-edge` v3.1.0](https://huggingface.co/vllm-sr/d3-edge/tree/18d61df78684675463af9ae732872478845c5a79),
  revision `18d61df78684675463af9ae732872478845c5a79`.
- Model SHA-256: `6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8`.
- Manifest SHA-256: `2eb3066c44c11555eef253c7f4f3c749e4e6f88863b2dcd1268b276c7b89611c`.
- Official `d3_server.py` SHA-256: `f9b8f2481461a074f469b4f58da3b0d7004824096528366d5b1f23187dd6e692`.
- 594,582,720 parameters; BF16 backbone and FP32 readout.
- Ephemeral Modal containers, four CPU threads; NVIDIA L4, `sm_89`,
  22.03 GiB reported device memory.
- PyTorch `2.10.0+cu128`, CUDA 12.8, torchvision `0.25.0+cu128`,
  Transformers `5.17.0`, Triton `3.6.0`, FLA core `0.5.2`.
- OpenCV `5.0.0.93`, tokenizers `0.23.2`, NumPy `2.5.3`, Pillow `12.3.0`.
- CPU oracle and worker use PyTorch `2.10.0+cpu`, torchvision `0.25.0+cpu`
  and the same Transformers/media versions.

These CUDA dependencies align with the model runtime Docker CUDA stage.
The existing [D3 parity record](decision3-parity.md) covers CPU and AMD MI325X.
This adds checkpoint/device-specific NVIDIA evidence. CUDA is validated in
[`cuda-h200.md`](cuda-h200.md).

## Reproduced failure and reference policy

Before this change, the worker selected one generic CPU reference and had no
CUDA golden reference. On a CPU exposing AVX2 without native BF16, startup failed:

```text
ERROR vllm_srun: loading model failed: golden answers differ from the reference
ready: false
status: failed
golden: {status: failed, checked: 3, matched: 1, reference: cpu}
```

A diagnostic observed the original checker without suppressing its failure.
The native and official package text-golden answers were identical and stable
on that host, but differed from the stored generic CPU reference:

| Value | Generic CPU reference | Same-host native and package | Absolute difference |
| --- | --- | --- | --- |
| Domain confidence | 0.7191210067989227 | 0.7121953255333247 | 0.006925681265598005 |
| Domain `code` probability | 0.9077280006989817 | 0.9043661358077931 | 0.003361864891188615 |
| Difficulty score | 0.19180114924537295 | 0.19351467706465733 | 0.0017135278192843828 |

The recorded reference came from a different CPU execution environment. This
identifies a cross-host reference mismatch, without establishing ISA as its
only cause. A separate AVX-512/native-BF16 diagnostic matched both text and image
answers exactly against the package and all five generic CPU goldens.

Readiness now prefers a recorded hardware key and falls back to the existing
device-class key only when that hardware key is absent:

- The optional accelerator hook defaults to `accelerator:arch`.
- CPU adds PyTorch's selected CPU capability and the accelerator's native/emulated
  BF16 capability: `cpu:x86_64:avx2:bf16_emulated` for the new CPU record.
- D3-edge records five official answers under `cuda:sm_89` and pins that oracle
  run's six forward FLA kernels, with all captured tuning keys, at FLA 0.5.2 /
  Triton 3.6.0.
- Existing generic CPU/ROCm answers and AMD kernel choices remain intact.
- CPU tolerance stays `0.001`; GPU tolerance stays `0.02`. A mismatch still
  blocks readiness; the checker never tries another reference after failure.
- `GET /v1/models` exposes the selected key as `golden.reference`.

CPU selection uses PyTorch's capability probes, rather than inferring native
BF16 solely from `/proc/cpuinfo`: the cloud sandbox's exposed flags and PyTorch's
probes were not always identical. Hardware keys describe reference selection;
matching goldens still determine whether an individual worker is ready.

## After the change

CUDA capture compared both live servers using the same persisted reference
Triton cache. It supplied the recorded golden answers and kernel choices.
Qualification then used a new container with no reference server and two
native processes with distinct, initially absent cache directories. Each was
compared against the frozen official responses, requiring identical request
hashes, answers, probabilities and usage:

```text
CUDA reference capture: 21/21 exact, max_numeric_delta=0.0
CUDA independent start 1: 21/21 exact, max_numeric_delta=0.0
CUDA independent start 2: 21/21 exact, max_numeric_delta=0.0
GET /health: 200, ready
GET /v1/models:
  golden: {status: matched, checked: 5, matched: 5, reference: cuda:sm_89}
  accelerator_validated: false
```

The AVX2/emulated-BF16 CPU qualification ran both live servers in separate
processes. The new official text/image answers exactly equalled the registered
CPU hardware reference, and the whole panel matched with stable repeats:

```text
CPU live reference vs native: 21/21 exact, max_numeric_delta=0.0
GET /health: 200, ready
GET /v1/models:
  golden: {status: matched, checked: 5, matched: 5,
           reference: cpu:x86_64:avx2:bf16_emulated}
```

The AVX-512/native-BF16 diagnostic retained the generic `cpu` reference and
matched all five goldens. It is a golden diagnostic, not another full-panel
CPU qualification. Other CPU classes retain their existing fallback behavior.

## Coverage and checks

[`tools/decision3_parity.py`](../../tools/decision3_parity.py) builds 68 questions
across 19 synthetic requests:

| Requests | Cases |
| --- | --- |
| Text: 11 | Code, math, greeting, multilingual Unicode, JSON, literal special tokens, long text; batches of 1, 8, 9 and 12 questions |
| Images: 5 | PNG, JPEG, WebP, two images and a wide image |
| Videos: 3 | One synthetic MP4, two clips and an image with a clip |

The model-backed harness adds the two family golden requests: 21 requests /
73 questions total. Each request is sent twice per process. Choice, Noul and
Score are covered, including question-batch splitting. The comparator requires
HTTP 200, every expected answer without question errors, stable repeats and
matching request hashes. Matching failures or missing answers never count as
parity.

Both servers reject malformed images/videos and unsupported Span questions.
Their error envelopes differ: official FastAPI HTTP 422 versus native HTTP 400
`invalid_request`. Valid decision parity does not imply identical error envelopes.

Checks:

- Fresh pinned Linux CPU environment: **983 passed, 3 skipped, 24 deselected**;
  mypy clean for all **112 source files**.
- Focused local readiness, real HTTP startup, D3 family and comparator tests:
  **73 passed, 10 skipped**. The startup regression proves a matching hardware
  reference makes HTTP serving available; a wrong hardware reference keeps
  health and inference at HTTP 503 even when the generic reference would match.
- Formatting, architecture/structure checks, Python/Go lint, documentation
  coverage, generated-client and configuration-schema checks passed.
- Full `make check` stopped on unrelated router config-history audit/temporary
  directory cleanup failures. Those paths were not changed.

The restarted CUDA worker showed about 1,590 MiB process GPU memory after the
panel and 6,851–6,854 MiB host RSS. These are snapshots, not peak memory results.
Single second-call timings are not a latency or throughput benchmark.

## Recorder regression follow-up — 2026-10-14

Review identified that the recorder emitted only a qualified key, leaving no
generic baseline when starting a new revision. It now records both keys, keeps
the first generic baseline within a revision, and updates qualified entries.
Revision changes still discard all old references; record the intended baseline
device first for the new revision. The accelerator hook docstring and the
parity tool's explanation of discrete Score levels were also clarified.

Seven recorder cases cover an actual offline tiny-model recording with and
without a qualified key, CPU/CUDA/ROCm merges, revision replacement and unrelated
model preservation. The merge cases exercise the real readiness checker to
prove that an unrecorded hardware class still uses the generic drift gate.

```text
Recorder regression before fix: 5 failed, 2 passed
Focused recorder/readiness/startup/D3/parity checks: 80 passed, 11 skipped
Fresh Linux CPU suite: 1004 passed, 3 skipped, 24 deselected
Mypy: no issues found in 112 source files
```

Formatting, structure/architecture checks and generated-client verification
passed. This follow-up does not change inference numerics, golden data, kernel
choices or tolerances. The model-backed CPU/L4 measurements above remain the
2026-10-12 qualification; no new GPU run was needed for the recorder correction.

## Upstream merge verification

The merge with upstream `a61d0e9ab109abc6cf1ab3c2201f5dbced2fe8fb` preserves
qualified reference selection and the recorder baseline policy alongside the
NPU run-to-run tolerance and validation of both responses against the reference.
Two regression cases cover a qualified NPU reference, including second-run drift
that would pass the generic reference but must fail the selected qualified one.

Focused checks passed: **91 passed, 12 skipped**. Mypy checked **113 source
files** cleanly; formatting, structure/architecture and generated-client checks
passed. The merged Linux suite had **1,127 passed, 3 skipped, 31 deselected** and
one failure in `test_repeated_media_are_processed_once_and_answer_alike`
(`decoded["images"]` was 0 rather than 1). Unmodified upstream on the same Linux
dependencies reproduced the identical assertion: **1,098 passed** and that one
failure. The upstream test is unchanged in this PR.

## Managed CPU container qualification

A separate ephemeral Modal VM ran a real Docker daemon. The Router image was
built with `tools/docker/Dockerfile.extproc` from signed source commit
`ab8ac7b8681c4bb76399777603326ce5ae942f9f`, tagged
`d3-router-local:ab8ac7b86`; its image ID was
`sha256:a47b30cc8eeaeddcbc7a14022057435445aeb078c65b00f0166d124e1873325f`.
The CLI was built from the same source, including its staged built-in catalog.
No published Router image substituted for the local build.

The VM exposed AVX-512/native BF16, running PyTorch `2.10.0+cpu` and the media
versions above. Both the managed worker and separate reference process used
12 threads. The worker's hardware key was `cpu:x86_64:avx512:bf16_native`;
because that key has no dedicated entry, it retained the generic `cpu` reference.
Its native model card confirmed the checkpoint revision and model/manifest
hashes above, `profile: exact`, `ready: true`, and **5/5 matched goldens**.

`vllm-sr serve` started the actual Router frontend, its managed worker and Redis
with `--engine --gateway standalone --minimal --platform cpu`. Docker published
the inference and management ports only on loopback. The isolated test oracle
used a derivative image with torchvision, Transformers and FastAPI added;
package Python executed only in that oracle container, never in the Router.

| Assertion | First start | After complete CLI stop/start |
| --- | --- | --- |
| Public `/v1/systemone`: 21 requests / 73 questions | 21/21 exact | 21/21 exact |
| Answer probabilities and usage | Maximum difference 0.0 | Maximum difference 0.0 |
| Two calls per request | Stable throughout | Stable throughout |
| Managed worker startup goldens | 5/5, reference `cpu` | 5/5, reference `cpu` |
| `/v1/systemone/models` | Only the granted D3-edge model | Same grant |
| `/v1/decisions` alias | Text answers/usage identical | Image answers/usage identical |
| Ungranted model | HTTP 403 | HTTP 403 |

The first run also required HTTP 400 for malformed images, malformed videos,
unsupported Span questions and requests missing the public model name. The
restart used frozen same-host reference responses and required matching request
hashes; it did not regenerate expected answers from the worker.

```text
Router is ready (after 10s, 6 checks)
WORKER_CARD_PASS {'status': 'matched', 'checked': 5, 'matched': 5, 'reference': 'cpu'}
MANAGED_STACK_PASS: 21/21 exact, max_numeric_delta=0.0
MANAGED_RESTART_PASS: 21/21 exact, max_numeric_delta=0.0
```

This closes the CPU CLI/container gap on AVX-512/native BF16. It does not add
a managed AVX2 or CUDA container qualification, nor test Dashboard, Envoy or
Chat recipe routing. Minimal stacks intentionally omit the Dashboard-dependent
instance controller; worker readiness was checked through its actual Unix-socket
model inventory. All test containers were stopped and the VM terminated.

### Run the managed CPU smoke test

The measurements above used the 2026-10-12 `exact` profile. Upstream has since
replaced profiles with engine arguments; the commands below use the current
CLI and published precision (`dtype: auto`). The historical measurements do
not qualify the updated serving defaults on a new model-backed run.

Build the local image from this branch, then save this complete declaration as
`d3-config.yaml` in a dedicated test directory:

```yaml
version: v0.3
listeners:
  - name: http-8899
    address: 127.0.0.1
    port: 8899
    timeout: 300s
    systemone:
      models: [vllm-sr/d3-edge]
global:
  router:
    enabled: false
  model_catalog:
    system:
      decision_model: {deployment: primary}
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        device: auto
        engine_args:
          dtype: auto
        revision: 18d61df78684675463af9ae732872478845c5a79
```

Use the same stack identity and state directory for every command:

```bash
docker build --platform linux/amd64 --build-arg ACCELERATOR=cpu \
  --build-arg TARGETARCH=amd64 -f tools/docker/Dockerfile.extproc \
  -t d3-router-local:ab8ac7b86 .
export VLLM_SR_STACK_NAME=d3-managed-probe
export VLLM_SR_STATE_ROOT_DIR=/absolute/path/to/dedicated-test-directory
vllm-sr serve vllm-sr/d3-edge --engine \
  --revision 18d61df78684675463af9ae732872478845c5a79 \
  --dtype auto --config "$VLLM_SR_STATE_ROOT_DIR/d3-config.yaml" \
  --image d3-router-local:ab8ac7b86 --image-pull-policy never \
  --platform cpu --gateway standalone --minimal --startup-timeout 600
curl -fsS http://127.0.0.1:8899/v1/systemone/models
curl -fsS http://127.0.0.1:8899/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-sr/d3-edge","state":"Explain why Python returns None.","questions":{"reasoning":{"type":"noul","instructions":"Does this need reasoning?"}}}'
vllm-sr stop
```

For full public parity, start the isolated CPU oracle described below on the
same host, then add the required public model field to both request panels:

```bash
PYTHONPATH=src/model-runtime/tools python - <<'PY'
from decision3_parity import collect, compare, panel

cases = panel()
for case in cases:
    case["body"]["model"] = "vllm-sr/d3-edge"
reference = collect("http://127.0.0.1:8101", cases)
actual = collect("http://127.0.0.1:8899", cases)
result = compare(reference, actual)
print(result)
assert result["passed"] == result["total"] == 19
PY
```

This standard panel has 19 requests; the recorded run additionally included
the two family golden requests. Keep the managed stack running until the
comparison finishes, and stop it afterwards.

## Direct worker reproduction

Install the runtime and its multimodal dependencies with the pinned versions
above. Download the checkpoint into an ignored directory:

```bash
hf download vllm-sr/d3-edge \
  --revision 18d61df78684675463af9ae732872478845c5a79 \
  --local-dir models/d3-edge-v3.1.0
```

Run the reviewed package server as a separate oracle on loopback:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
TRITON_CACHE_DIR=/tmp/d3-reference TRITON_CACHE_AUTOTUNING=1 \
python models/d3-edge-v3.1.0/d3_server.py \
  --model models/d3-edge-v3.1.0 --device cuda:0 \
  --verify full --no-warmup --host 127.0.0.1 --port 8101
```

The production worker never imports this package's Python code. Start it in a
separate terminal with its own initially empty cache:

```bash
vllm-srun serve models/d3-edge-v3.1.0 --device cuda:0 \
  --threads 4 --host 127.0.0.1 --port 8102 \
  --autotune-cache /tmp/d3-native-1 --result-cache-entries 0
curl -fsS http://127.0.0.1:8102/v1/models
python src/model-runtime/tools/decision3_parity.py \
  --reference-url http://127.0.0.1:8101 \
  --runtime-url http://127.0.0.1:8102 \
  --output .agent-harness/d3-edge/parity.json
```

A freshly autotuned oracle may select different kernel configurations. Exact
restart qualification uses responses from the same oracle run that supplied
the registry's pinned choices. Preserve that output; stop the reference and
native processes, restart the native worker with another unused cache directory
(`/tmp/d3-native-2`), then compare with the frozen output:

```bash
python src/model-runtime/tools/decision3_parity.py \
  --reference-file .agent-harness/d3-edge/parity.json \
  --runtime-url http://127.0.0.1:8102 \
  --output .agent-harness/d3-edge/parity-restart.json
```

Use `--device cpu` on both servers for a CPU run. Inspect `golden.reference`,
require `status: matched`, and check health before sending the panel. Never
bypass the gate to claim a public serving pass.

Raw responses, source hashes and bounded harness logs remain ignored under
`.agent-harness/d3-edge/`. The jobs are ephemeral and deploy no public endpoint
or persistent volume. Synthetic parity measures implementation agreement,
not decision quality; other checkpoints/devices, concurrency and production
video sizes need separate qualification.
