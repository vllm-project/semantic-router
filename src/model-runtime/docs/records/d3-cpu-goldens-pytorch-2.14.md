# Decision 3.0 CPU golden references with PyTorch 2.14.1

Every Decision 3.0 model failed its CPU golden check on an AMD EPYC 9575F
(AVX-512 with BF16) with the PyTorch the CPU images install, 2.14.1: d3-edge
matched 1 of 3 golden answers. The CPU references had been recorded with
PyTorch 2.10.0, which the images installed until #5555. The one operation
whose result changed is PyTorch's CPU flash attention: for the same BF16
inputs, 2.14.1 rounds 90 of the golden requests' 13.8 million attention
outputs one BF16 step differently, and D3's BF16 backbone carries that into
answers up to 0.012 from the references. No decision changes. The CPU
references are re-recorded with 2.14.1, each with the PyTorch it was recorded
with, and a test fails when they and the images' PyTorch differ.

- **Date:** 2026-11-10.
- **Host:** an AMD EPYC 9575F (Zen 5), two sockets in a virtual machine,
  AVX-512 with BF16, no AMX. This is the host class of the earlier `cpu`
  references, which [Decision 3.0 parity](decision3-parity.md) recorded on
  16 cores of an AMD EPYC.
- **Image:** `src/model-runtime/Dockerfile` of this change: Python 3.12,
  PyTorch 2.14.1+cpu (MKL 2024.2, oneDNN 3.12.0) and the lock. The 2.10.0
  image differs only in `TORCH_VERSION` (MKL 2024.2, oneDNN 3.7.1).
- **Packages:** the revisions `registry/tables/decision3.py` pins.
- **Threads:** 16 pinned cores for the diagnostics, 32 and 48 for the
  recordings, which equal the diagnostics' answers. 1, 8, 16 and 32 threads
  give d3-edge identical answers.

## The failure

`vllm-srun serve vllm-sr/d3-edge --device cpu` looks up the key
`cpu:x86_64:avx512:bf16_native`. d3-edge has no reference under it, so
readiness compares with the generic `cpu` reference and stops loading with
`golden answers differ from the reference`. With the same runtime, host and
threads, PyTorch 2.10.0 reproduces the generic references:

| Model | 2.14.1 golden check | 2.14.1 vs `cpu` | 2.10.0 vs `cpu` |
| --- | --- | ---: | ---: |
| d3-edge | 1 of 3 matched | 0.0077 | 0 |
| d3-lite | 2 of 3 matched | 0.0072 | 0 |
| d3-nano | 1 of 3 matched | 0.0042 | 0 |
| d3-mini | 2 of 3 matched | 0.0116 | 0 |
| d3-flash | 2 of 3 matched | 0.0038 | 0 |
| d3 | 4 of 5 matched | 0.0032 | 0 |
| d3-turbo | 4 of 5 matched | 0.0012 | 0.0002 |
| d3-ultra | 4 of 5 matched | 0.0011 | 0.0006 |

The differences are the largest over the compared values (probabilities and
Noul values) of the five golden questions, text and image. No choice, Score
top level or side of 0.5 changes. d3, d3-turbo and d3-ultra are checked
against their `cpu:x86_64:avx512:bf16_native` entries. d3's came from a Xeon
Platinum 8568Y+ (#5185) and is 0.0033 from this host's answers with 2.10.0
and 0.0010 with 2.14.1. Decision 2.0 Kai-0.6B and vela3-fast, which compute
in FP32 on CPU, still match on this host with 2.14.1.

## Cause

An ATen-level log of d3-edge's text golden request under both versions
(1,125 operations, views left out) records a digest of every input and
output. Every operation before the first full-attention layer has the same
inputs and outputs in both, the BF16 linears, the depthwise convolution and
the FP32 gated delta rule included. The first operation with the same inputs
and a different output is `aten.scaled_dot_product_attention` (BF16
`[3, 8, 114, 256]` with a boolean key mask), which runs PyTorch's CPU flash
attention in both versions.

The golden requests make 36 such calls (text, image and the vision tower).
Replaying the inputs captured with 2.10.0, 2.14.1 changes 90 of their
13,787,136 outputs, each by one BF16 step, in 30 calls; 2.10.0 reproduces all
of them. Running d3-edge with 2.14.1 and the 36 outputs of 2.10.0 substituted
gives the generic references exactly, so attention is the only cause. A BF16
backbone rounds every layer's output, so one-step changes compound over its
layers; FP32 models keep such changes in their last FP32 bits.

The released runtime runs this kernel too, through Transformers' SDPA, and
the golden check's forwards stay byte-identical to the released runtime. So
the runtime keeps the kernel, and the references follow the PyTorch the
images install.

## The AVX2 reference

d3-edge's `cpu:x86_64:avx2:bf16_emulated` reference came from a cloud host
that reports AVX2 without native BF16 ([D3-edge validation](d3-edge-cuda-validation.md)).
With PyTorch 2.10.0, this host run with `ATEN_CPU_CAPABILITY=avx2
ONEDNN_MAX_CPU_ISA=AVX2` reproduces it exactly, all 12 values. Readiness did
not select it: the CPU key read native BF16 from the hardware and looked up
`cpu:x86_64:avx2:bf16_native`, then fell back to `cpu`. The key now reports
emulated BF16 when oneDNN is capped below AVX-512 BF16. With 2.14.1, the AVX2
answers are up to 0.0047 from that reference too.

## Re-recorded references

`tools/golden_answers.py --family decision3 --device cpu --record` ran in the
image above, first natively and then with `ATEN_CPU_CAPABILITY=avx2
ONEDNN_MAX_CPU_ISA=AVX2`:

- `cpu` and `cpu:x86_64:avx512:bf16_native`: this host, PyTorch 2.14.1.
- `cpu:x86_64:avx2:bf16_emulated`: this host's AVX2 kernels, PyTorch 2.14.1.

Each model's entry lists the PyTorch each key was recorded with under
`torch`. Before this change the recorder could not replace a reference that
the model no longer matched: loading stopped at the readiness check. It now
loads without comparing, reports how many answers each old reference still
matches (`previous`), and a new PyTorch starts a new generic baseline.
`confidence` values, which readiness does not compare, came from an older
formula in the old references and are current in the new ones.

The new `cpu` references differ from the old ones as the failure table shows.
The AVX2 class needs its own: its answers differ from the AVX-512 ones by more
than the tolerance, with no decision changed. d3-edge's new AVX2 reference is
up to 0.0047 from its old one.

| Model | AVX2 vs AVX-512 BF16, both 2.14.1 |
| --- | ---: |
| d3-edge | 0.0074 |
| d3-lite | 0.0062 |
| d3-nano | 0.0267 |
| d3-mini | 0.0138 |
| d3-flash | 0.0038 |
| d3 | 0.0026 |
| d3-turbo | 0.0036 |
| d3-ultra | 0.0005 |

The GPU and NPU references are unchanged; their images keep their PyTorch.

## Checks

- A second host of this class, with its own downloads and 16 threads: d3-edge,
  d3-lite, d3-nano and d3-mini match 5 of 5 golden answers exactly, natively
  against `cpu:x86_64:avx512:bf16_native` and with `ATEN_CPU_CAPABILITY=avx2
  ONEDNN_MAX_CPU_ISA=AVX2` against `cpu:x86_64:avx2:bf16_emulated`. On the
  recording host with 16 threads, d3, d3-turbo, d3-ultra, d3-flash and d3-edge
  match 5 of 5 exactly.
- `vllm-srun serve vllm-sr/d3-edge --device cpu` on the second host is healthy
  after 6 s; its card reports `golden: {status: matched, checked: 5, matched: 5,
  reference: cpu:x86_64:avx512:bf16_native}` and `POST /v1/systemone` answers.
- `test_cpu_references_are_recorded_with_the_pinned_pytorch` fails on the old
  references, every D3 CPU key without a recorded PyTorch, as it would have
  on #5555. A recorder test reproduces the stale-reference failure on a tiny
  D3 package and records over it. The CPU key tests cover the oneDNN cap.
- Model runtime tests on CPU: 1,744 passed, 2 skipped; mypy reports no issues.

## Not covered

- Intel hosts with AVX-512 BF16 or AMX, such as the Xeon of #5273, answer
  differently from this host class even with PyTorch 2.10.0. They need their
  own references recorded with 2.14.1 on such a host (#5274 keys them by
  vendor). d3's Xeon entry from #5185 was a 2.10.0 recording; this host's
  answers replace it under `cpu:x86_64:avx512:bf16_native`.
- Other CPU classes, such as AVX-512 without BF16 or Arm, still fall back to
  the generic reference.
- The Vela 3.0 d3 sizes, vela3-lite among them, record no golden references
  ([Vela 3.0 d3-size parity](vela3-d3-parity.md)); they load `unverified`.

## Reproduce

Build the image and record on an AMD host with AVX-512 BF16, natively first,
so that the generic baseline is this host class's:

```bash
docker build -t vllm-srun-cpu src/model-runtime
record() {
  docker run --rm --user "$(id -u)" "$@" \
    -v "$PWD/src/model-runtime:/mr" -w /mr -e PYTHONPATH=/mr \
    -v "$HOME/.cache/huggingface:/hf" -e HF_HOME=/hf \
    --entrypoint python vllm-srun-cpu tools/golden_answers.py \
    --family decision3 --device cpu \
    --record vllm_srun/registry/golden_answers_decision3.json
}
record
record -e ATEN_CPU_CAPABILITY=avx2 -e ONEDNN_MAX_CPU_ISA=AVX2
```

Then serve a model and check that `GET /v1/models` reports
`golden: {status: matched}` with the expected `reference`.
