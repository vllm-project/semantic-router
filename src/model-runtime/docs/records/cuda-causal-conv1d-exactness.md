# CUDA causal-conv1d exactness

Directly comparing `_wrap_conv(causal_conv1d_fn)` with `causal_conv1d_ref`
produces deterministic bit differences. This is evidence about numerical
implementations, not a change to the released-runtime selection policy.
The [A/B model experiment](cuda-gdn-prep-without-causal-conv1d.md) tests a
separate question: both historical variants classified causal-conv1d as approximate.

## Reproduce

Check out source commit `25b423e60f3ea0de2c1f26dd3bf13f367e02205a`.
The fresh run used an immutable `git archive` of that commit mounted read-only
in local CUDA image
`sha256:a852d0e2d1119bbf17ae8d17e11e16d3951849cabcb20b656b6ad4170e7a0d66`
(a local Docker image ID, not a registry manifest digest).
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

- **2026-11-06 rerun:** GB10, sm_121, aarch64, driver 580.173.02; PyTorch 2.10.0+cu128,
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

The 2026-11-06 rerun reproduces all 48 case records from 2026-11-02 exactly,
including tensor hashes, bits and comparisons. The JSON now records this
fresh run with its actual commit and image; the original run did not record
that provenance. No model A/B results were rerun or relabeled.

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

The probe disproves unconditional bit equality between these two implementations
on the recorded stack. It does not establish which implementation each released
model runtime uses. Decision 1.0 Qwen and BF16 Decision 3.0 use causal-conv1d on
CUDA hosts where it is installed; changing them to torch reference would move
away from that runtime. This PR therefore retains the existing `exact=True`
registration and installed-package preference; changing that classification is
outside its scope.

| Selection | Before PR | After PR |
| --- | --- | --- |
| causal-conv1d installed, with or without approximate consent | causal-conv1d | causal-conv1d |
| causal-conv1d absent | torch reference | torch reference |

The standard CUDA image has no causal-conv1d. The historical A/B record applied
an approximate classification in both variants, so its non-consenting models
compared reference against reference. It does not validate the final policy on
installed hosts. These results cover the recorded GB10 stack, not every
GPU/version or a full model-accuracy evaluation.
