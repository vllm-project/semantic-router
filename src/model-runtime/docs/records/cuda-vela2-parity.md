# Vela 2.0 parity on NVIDIA H200

The `vela2` family answers exactly as each Vela 2.0 package's own engine
(`vela2_inference.py`) on the exact profile, on one NVIDIA H200 (`sm_90`),
for all four sizes. This is the public generator panel
(`tools/vela2_parity.py --generate 300`), not the unpublished historical
ROCm scored panel; those coverages are not interchangeable. Machine
readable summary: [`cuda-vela2-parity.json`](cuda-vela2-parity.json).

The scored request panel used on ROCm is unpublished, so this record uses
the public generator (`tools/vela2_parity.py --generate 300`). The ROCm
record is [`vela2-parity.md`](vela2-parity.md).

- **Date:** 2026-10-12.
- **Hardware:** NVIDIA H200 (`sm_90`, 143771 MiB).
- **Head:** `61677a464c2a1b94ef34cd9e22595bab87981b97`.
- **Image:** `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest`, digest
  `sha256:2f05113e5c7399874ef2b1651ecb3745a5dafeff7ef2f9725acd01b643c06730`.
  Inside the image: Debian 12, Python 3.12.15, torch 2.10.0+cu128, Triton
  3.6.0, fla-core 0.5.2. `transformers==5.17.0` was installed in the pod so
  the package engine could import (`modeling_vela2.py`).
- **Packages:** Vela-2.0-0.3B `a3209a50`, Vela-2.0-0.8B `a778eb2a`,
  Vela-2.0-4B `c1e64d4f`, Vela-2.0-9B `bc876163` (the pins in
  `registry/tables/vela2.py`).
- **Reference:** the package's engine, imported from the package directory by
  the tool only (the runtime never imports package code), on `cuda:0`.
- **Runtime side:** the family through `NativeEngine` and the CUDA
  accelerator, each request alone on the exact profile.
- **Kernel choices:** autotuned in process on this run (`kernel_choices`:
  `autotuned in process`). The `cuda:sm_90` pins in
  `registry/kernel_choices.json` were recorded later, on a separate H200
  ([`cuda-fused-kernels.md`](cuda-fused-kernels.md)).
- **Requests:** `--generate 300` (tool default seed 0): router-style
  questions, every answer type (Choice, Noul, Score, Set, Span), the `pii`,
  `halu` and `relevance` presets, typed JSON states with `over`, thresholds,
  open span labels, long parts that are windowed or cut, and Unicode.

## Exact profile

A request is identical when both responses are byte-identical JSON (Score
legends compared as text, the runtime's `abstain_probability` set aside) and
both sides render the same sequences.

| Model | CUDA H200, published image | Max \|Δp\| | Too-long rows |
| --- | --- | ---: | ---: |
| Vela-2.0-0.3B | 300 comparison cases passed | 0.0 | 13 |
| Vela-2.0-0.8B | 300 comparison cases passed | 0.0 | 0 |
| Vela-2.0-4B | 300 comparison cases passed | 0.0 | 0 |
| Vela-2.0-9B | 300 comparison cases passed | 0.0 | 0 |

300 comparison cases passed, including 13 oversized 0.3B cases compared
after excluding rejected questions. Do not describe all 300 original 0.3B
responses as byte-identical: the package engine rejects those 13 whole
requests (`too_long`); the runtime answers the valid rows and returns
`max_length_exceeded` for the oversized question. The comparator
deliberately accommodates that.

Load / reference / runtime wall time (seconds), one process per model:

| Model | Load | Package engine | Runtime |
| --- | ---: | ---: | ---: |
| Vela-2.0-0.3B | 14.2 | 70.48 | 49.55 |
| Vela-2.0-0.8B | 11.3 | 46.11 | 111.02 |
| Vela-2.0-4B | 28.2 | 98.25 | 161.86 |
| Vela-2.0-9B | 36.4 | 131.26 | 124.72 |

The 0.8B and 4B runtime times include graph capture on first use of a shape.
They are not the performance record; that is a later `vela2_bench.py` run.

## Reproduce

Inside the published CUDA image, with the package directory on disk and
`transformers` 5.17 available for the package engine:

```bash
python3 tools/vela2_parity.py --package PACKAGE_DIR --device cuda:0 \
  --generate 300 --output OUT.json --answers OUT.jsonl
```
