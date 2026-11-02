# CUDA causal-conv1d exactness: direct convolution comparison

This is the numerical evidence for classifying optional causal-conv1d kernels
as approximate on CUDA. It compares the actual external convolution with the
torch-reference convolution on the same inputs, independently of model-family
consent and kernel selection. The separate
[A/B model record](cuda-gdn-prep-without-causal-conv1d.md) does not prove this
classification: both its variants already apply it.

## Reproduce

With the runtime installed and causal-conv1d 1.7.0 available for the matching
PyTorch/CUDA build, run from `src/model-runtime`:

```bash
python tools/causal_conv1d_parity.py --device cuda:0 --output /tmp/conv-parity.json
```

The output must not already exist. This needs no downloaded models. The
probe generates inputs and weights with a CPU generator at seed 17, then
casts and copies them to the selected CUDA device. It calls
`_wrap_conv(causal_conv1d_fn)` and `causal_conv1d_ref` directly. No
`CUDAAccelerator`, `KernelSet`, exactness flag, or family selector participates.
Exceptions propagate; no eager fallback substitutes for the external call.

The [raw record](cuda-causal-conv1d-exactness.json) contains package/device
versions, backend flags, source hashes, shapes, dtypes, strides, seeds,
bit-mismatch counts, maximum absolute differences, ULP distances and differing
output samples. Small cases store the complete input, weight and both output
tensors as unsigned IEEE bit encodings in logical row-major order. Large
cases store tensor hashes and sampled differing outputs; the fixed generation
recipe reconstructs their inputs. SHA256 uses native byte order, recorded
explicitly in the JSON. The probe's source hash identifies the exact runner.

A non-exact comparison is an observation, not a failed probe. Exit 1 means
nonfinite or nondeterministic results. Each implementation is repeated three
times on identical inputs and checked independently for bitwise determinism.

## Measured environment and cases

- **Date:** 2026-11-02.
- **Device:** NVIDIA GB10, sm_121, aarch64, driver 580.173.02.
- **Stack:** PyTorch 2.10.0+cu128, CUDA 12.8, cuDNN 91002,
  causal-conv1d 1.7.0 (CUDA 12 / torch 2.10 / CPython 3.12 wheel).
- **Backend flags:** cuDNN TF32 allowed; benchmark and deterministic flags off,
  as reported by the probe. No backend flags are overridden.
- **Weight generation:** CPU float32 normal samples scaled by 0.02, then cast
  to the case's weight dtype. Bias is absent, as in the native GDN convolution.
- **Shapes `(B,C,T,KW)`:** `(1,8,16,4)` (complete small witness),
  `(1,6144,16,4)`, `(2,6144,65,4)` and `(1,8192,129,4)`.
  Eos has `(2*16 + 16)*128 = 6144` projection channels; Nox has
  `(2*16 + 32)*128 = 8192`. These are random tensors at those dimensions,
  not released model weights.
- **Layouts:** contiguous `[B,C,T]` and a contiguous `[B,T,C]` projection
  transposed to `[B,C,T]`, as the native Qwen3.5 forward does.
- **Modes:** BF16 input / FP32 weight with CUDA BF16 autocast (Decision 2.0's
  default path); BF16 input and weight without autocast (resident BF16);
  FP32 input and weight without autocast. SiLU and no activation are measured
  separately. The no-activation cases are diagnostics; native GDN uses SiLU.

## Results

All 48 cases are finite and deterministic across three calls to each
implementation. The table aggregates four shapes and two layouts per row:

| Mode | Activation | Bit mismatches / elements | Max absolute difference | Max ULP distance |
| --- | --- | ---: | ---: | ---: |
| BF16 autocast, FP32 weight | none | 1445186 / 3907840 | 0.001953125 | 29344 |
| BF16 autocast, FP32 weight | SiLU | 1735508 / 3907840 | 0.0009765625 | 29088 |
| BF16 resident | none | 0 / 3907840 | 0 | 0 |
| BF16 resident | SiLU | 992952 / 3907840 | 0.0009765625 | 1 |
| FP32 | none | 0 / 3907840 | 0 | 0 |
| FP32 | SiLU | 1280970 / 3907840 | 2.9802322387695312e-08 | 4 |

ULP distance is the distance between monotonic IEEE encodings, treating
negative and positive zero as adjacent. It can be large near zero and across
sign changes despite a small absolute difference; both metrics are retained.
The results do not support a universal claim of "one BF16 ulp": that bound was
observed only for the measured BF16-resident SiLU cases. In particular, the
autocast/FP32-weight mode is not limited to one ULP. This record does not assign
every difference to a single cause such as activation or weight conversion.

### Complete small counterexample

For `(1,8,16,4)`, BF16 input/weight, transpose layout, SiLU and seed 17:
29 of 128 output elements differ, maximum absolute difference is
0.000244140625 and maximum ULP distance is 1.

This complete small case can be run alone:

```bash
python tools/causal_conv1d_parity.py --shape 1,8,16,4 --mode bf16-resident \
  --layout transpose --activation silu --output /tmp/conv-witness.json
```

At logical flat output index 2:

| Implementation | Value | BF16 bits |
| --- | ---: | --- |
| causal-conv1d | -0.00194549560546875 | 47871 |
| torch reference | -0.001953125 | 47872 |

The JSON stores all input, weight and output bits for this case, not just the
two numbers above. The larger Eos-shaped `(2,6144,65,4)` transpose/SiLU case
also differs: 203080 of 798720 outputs for resident BF16, and 355187 of 798720
outputs for BF16 autocast with FP32 weights.

## Selection policy and compatibility

A finite, deterministic counterexample is sufficient to reject an
unconditional `exact=True` classification for the CUDA slot. Some tested
modes are exact; that does not establish exactness for the other supported
modes. The conservative approximate classification keeps this optional kernel
behind family consent. This is a GB10 result for the recorded versions, not
a measured error bound or hardware qualification for every CUDA GPU/version.

For hosts that already have causal-conv1d installed, the PR intentionally
changes selection:

| Selection | Before PR | After PR |
| --- | --- | --- |
| approximate allowed | causal-conv1d | causal-conv1d |
| exact-only or family does not consent | causal-conv1d | torch reference |

The standard CUDA image has no causal-conv1d, so its exact/non-consenting path
already selects torch reference and stays on it. The A/B model experiment
compares that reference path in both variants; it does not demonstrate that
an installed host's pre-PR model outputs are unchanged. This record supports
the kernel classification, not a full model-accuracy evaluation.
