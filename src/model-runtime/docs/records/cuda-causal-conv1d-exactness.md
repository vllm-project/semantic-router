# CUDA causal-conv1d exactness

Directly comparing `_wrap_conv(causal_conv1d_fn)` with `causal_conv1d_ref`
produces deterministic bit differences, supporting `exact=False` on CUDA.
The [A/B model experiment](cuda-gdn-prep-without-causal-conv1d.md) tests a
separate question: both variants already apply this classification.

## Reproduce

From `src/model-runtime`, with the runtime and a matching causal-conv1d 1.7.0
wheel installed:

```bash
python tools/causal_conv1d_parity.py --device cuda:0 --output /tmp/conv-parity.json
```

Choose a new output file. No models are downloaded. Both implementations run
directly on identical tensors; no kernel selector or eager fallback is used.
Each is called three times to check bitwise determinism. Exit 1 means nonfinite
or nondeterministic results; a numerical difference alone is an observation.

The [raw JSON](cuda-causal-conv1d-exactness.json) preserves versions, backend
flags, source hashes, case statistics and differing samples. Small cases include
all input, weight and output bits; larger cases include tensor hashes and use
the fixed generation recipe below. Bits are unsigned IEEE encodings in logical
row-major order; SHA256 uses the byte order recorded in the JSON.

## Environment and cases

- **2026-11-02:** GB10, sm_121, aarch64, driver 580.173.02; PyTorch 2.10.0+cu128,
  CUDA 12.8, cuDNN 91002, causal-conv1d 1.7.0 (CUDA 12 / torch 2.10 / cp312 wheel).
- **cuDNN:** TF32 allowed; benchmark and deterministic flags off. No flags overridden.
- **Inputs:** CPU float32 normal samples, seed 17, then dtype cast/device copy.
  Weights use the same generator, scaled by 0.02. Bias is absent.
- **Shapes `(B,C,T,KW)`:** `(1,8,16,4)`, `(1,6144,16,4)`, `(2,6144,65,4)` and
  `(1,8192,129,4)`. Eos has `(2*16+16)*128=6144` projection channels; Nox has
  `(2*16+32)*128=8192`. These are random tensors, not released model weights.
- **Layouts:** contiguous `[B,C,T]` and `[B,T,C].transpose(1,2)`, as in the native
  Qwen3.5 projection.
- **Modes:** BF16 input / FP32 weight with BF16 autocast; BF16 input/weight
  without autocast; FP32 input/weight without autocast. Each runs with SiLU
  (the native GDN activation) and without activation (diagnostic).

## Results

All 48 cases are finite and deterministic. Each row aggregates four shapes
and two layouts:

| Mode | Activation | Bit mismatches / elements | Max absolute difference | Max ULP distance |
| --- | --- | ---: | ---: | ---: |
| BF16 autocast, FP32 weight | none | 1445186 / 3907840 | 0.001953125 | 29344 |
| BF16 autocast, FP32 weight | SiLU | 1735508 / 3907840 | 0.0009765625 | 29088 |
| BF16 resident | none | 0 / 3907840 | 0 | 0 |
| BF16 resident | SiLU | 992952 / 3907840 | 0.0009765625 | 1 |
| FP32 | none | 0 / 3907840 | 0 | 0 |
| FP32 | SiLU | 1280970 / 3907840 | 2.9802322387695312e-08 | 4 |

ULP distance uses monotonic IEEE encodings, with negative/positive zero
adjacent. It can be large near zero or across sign changes despite small
absolute differences. The one-ULP bound holds only for these BF16-resident
SiLU cases, not the autocast/FP32-weight mode. Differences are not attributed
to a single cause.

### Complete small counterexample

```bash
python tools/causal_conv1d_parity.py --shape 1,8,16,4 --mode bf16-resident \
  --layout transpose --activation silu --output /tmp/conv-witness.json
```

At seed 17, **29/128 outputs differ**; maximum absolute difference is
**0.000244140625**, maximum ULP distance **1**. Logical flat output index 2:

| Implementation | Value | BF16 bits |
| --- | ---: | --- |
| causal-conv1d | -0.00194549560546875 | 47871 |
| torch reference | -0.001953125 | 47872 |

The JSON includes all input, weight and output bits for this case.

## Selection impact

A deterministic counterexample rejects an unconditional `exact=True` for the
slot, even though some tested modes are exact. Approximate classification
keeps the optional kernel behind consent. For hosts with causal-conv1d installed:

| Selection | Before PR | After PR |
| --- | --- | --- |
| approximate allowed | causal-conv1d | causal-conv1d |
| exact-only or family does not consent | causal-conv1d | torch reference |

The standard CUDA image has no causal-conv1d and already uses reference for
exact/non-consenting models. The A/B record compares reference against reference
there; it does not establish unchanged pre-PR outputs on installed hosts.
These results cover the recorded GB10 stack, not every GPU/version or a full
model-accuracy evaluation.
