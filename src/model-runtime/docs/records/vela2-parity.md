# Vela 2.0 parity

The `vela2` family answers exactly as the Vela 2.0 packages' own engine
(`vela2_inference.py`) on the exact profile, on CPU and ROCm, for all three
sizes. The opt-in approximate profiles keep every answer within the design's
GPU bar (section 17) except span probabilities, which move by up to 0.12 on
long windowed documents and change one or two span sets in 60 requests.

- **Date:** 2026-10-04.
- **Packages:** the revisions `registry/tables/vela2.py` pins:
  Vela-2.0-0.3B `13e85201`, Vela-2.0-4B `c50cba67`, Vela-2.0-9B `2c90057d`
  (private preview).
- **Devices:** one AMD Instinct MI325X (gfx942) per run, in the Decision 2.0
  release image (PyTorch ROCm, Triton, FLA). CPU runs use PyTorch 2.10's CPU
  build with MKL on 16–32 cores of an AMD EPYC, for both sides: the release
  image's ROCm PyTorch has no MKL.
- **Reference:** the package's engine, which `tools/vela2_parity.py` imports
  from the package directory (the runtime never imports package code), in the
  same process and on the same device: FP32 on CPU; on GPUs its defaults (BF16
  autocast for the 4B / 9B backbones, FP32 for the 0.3B encoder).
- **Runtime side:** the family through `NativeEngine` and the accelerator,
  each request alone. On GPUs the fast path's fused kernels are on; the 4B /
  9B FLA kernel choices are pinned from `registry/kernel_choices.json` (the
  tool pins them before FLA is imported, so the reference runs them too).
- **Requests:** `--generate 60 --seed 1`: router-style questions, every answer
  type (Choice, Noul, Score, Set, Span), the `pii`, `halu` and `relevance`
  presets, typed JSON states with `over`, thresholds, open span labels, long
  parts that are windowed or cut, and Unicode.

## Exact profile

A request is identical when both responses are byte-identical JSON (Score
legends compared as text, the runtime's `abstain_probability` set aside) and
both sides render the same sequences: the token IDs of every 0.3B sequence and
of every 4B / 9B part and block, and every position the readout reads,
windows included.

| Model | CPU | ROCm |
| --- | --- | --- |
| Vela-2.0-0.3B | 60 / 60 identical | 60 / 60 identical |
| Vela-2.0-4B | 24 / 24 identical | 60 / 60 identical |
| Vela-2.0-9B | 12 / 12 identical | 60 / 60 identical |

- **Rows the engine rejects.** In four of the 0.3B requests one row's schema
  alone is longer than the model's 8,192 tokens. The engine rejects the whole
  request (`too_long`); the runtime answers the other questions and returns
  `max_length_exceeded` for that row's. These requests compare on the
  questions both sides answer, and the runtime's answers to them equal the
  ones in its full response (no other row changes).
- **4B / 9B tensor shapes.** Exactness needs the engine's shapes: prefix rows
  left-padded, blocks right-padded to the longest block, one causal attention
  call per block (`TreeBatch(layout="rows")`, `engines/native/models/forest.py`).
  Packing the same tree into one row computes the same values from fewer
  tokens, with other rounding (next section).
- **Fused kernels.** The forest forward runs the gfx942 fused kernels (norms,
  residual adds, the MLP gate, attention preparation and output gate, the
  gated-delta output norm), which equal the eager operations bit for bit
  (`tests/test_gpu_fast_path.py::test_fused_forest_equals_eager`); the ROCm
  results above include them.

## Approximate profiles (opt-in)

`shared_context` and `batching` run the 4B / 9B trees packed (one row per
prefix, blocks merged by log-sum-exp; `layout="packed"`) and the 0.3B
sequences packed (no padding; local layers in query blocks on long rows).
Same 60 requests on ROCm, `tools/vela2_parity.py --approximate`:

| Model | Decision changes | Max \|Δp\|, answers | Median per request | Max \|Δp\|, span probabilities |
| --- | --- | --- | --- | --- |
| Vela-2.0-0.3B | none | 4.1e-6 | 6.6e-7 | 2.7e-6 |
| Vela-2.0-4B | 1 request: a PII span set | 0.017 | 0.0035 | 0.065 |
| Vela-2.0-9B | 2 requests: PII and entity span sets | 0.017 | 0.0027 | 0.119 |

- "Answers" are every Choice, Score and Noul value, Set label view and
  threshold, including span questions' Noul views; "span probabilities" are
  the mean word probabilities of individual spans.
- The spans that move are short spans in long, windowed documents, where the
  PII thresholds are low (0.05–0.1): a span whose mean probability crosses its
  threshold appears on one side only.
- The same 60 requests took 14.5 s on the packed 4B path against the engine's
  31.7 s, and 17.2 s against 51.2 s on 9B (`vela2-performance.md` has the
  per-request latencies).
