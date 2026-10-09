# Vela 1.0 text models: parity with the legacy router path

The `task_heads` family serves the ten Vela 1.0 text models (Domain, Guard,
Safety, Shield, FactCheck, Feedback, Modality, Hazard, PII, Halu) within the
design's section 17 bars against the router they replace, on CPU and on
ROCm, for every input of the corpus. The legacy side is the router's own
native facade at `main` `61aa7eb2d`, driven as the router drives it; the
runtime side is `Runtime.call`, as its HTTP server calls it. Raw results
(every disagreement, contexts, thresholds) are in `vela1-parity.json`.

- **Date:** 2026-10-05.
- **Runtime commits:**
  - CPU: `efb5ec4d7`, the staging head `b604edeab` with `vela1`'s A/B tool
    commit merged. On it, `exact` lets rows of every length share a batch on
    models that pack rows, so a long input's windows of different lengths run
    in one forward. Every answer is bit-identical to the earlier runs at
    `93a3492c0` (max |Δp| 0 on all 11 jobs).
  - ROCm: `35ff3a8a5`, which adds the fixes for a shared GPU process (device
    work one model at a time, thread-local graph capture) and leaves the CPU
    code as it was. Its answers against the packages' release image are
    below.
  - The `batching` and reduced-copy rows name their own commit.
  - The final head is `35ff3a8a5` with these records on top, so both
    comparisons hold for it.
- **Packages:** the revisions `registry/tables/vela1.py` pins (Domain
  `f6354f54`, Guard `087f9e40`, Safety `6e70e725`, Shield `a981a99e`,
  FactCheck `99ede1ab`, Feedback `47434a7f`, Modality `5384b899`, Hazard
  `5dd25f2c`, PII `6d3300c4`, Halu `ca875312`), the same snapshot files on
  both sides.
- **Legacy side (`tools/legacy_parity.py legacy`):** a Go test compiled into
  `pkg/modelruntime/native` at `61aa7eb2d` that loads every model through the
  facade's own task constructors (`sequence`, windowed sequence, Hazard's
  operating point, token spans, windowed token spans, grounded spans) and
  records each answer and its latency. Two deployments:
  - **CPU:** the router's defaults. That means candle; deployment
    `max_tokens` 512 for sequence models (truncate); Guard and PII windows
    of 512 / 255 over 8,192 / 32,768 tokens; Hazard's packaged operating
    point (2,048 / 1,023 windows over 32,768); Halu 8,192 (truncate). The
    bindings are built in `golang:1.25-bookworm`, because the router image
    at `61aa7eb2d` cannot start on its base: its libraries need glibc 2.39.
  - **AMD:** `config/recipes/vela-amd`, i.e. ONNX Runtime on MIGraphX
    (Guard on the ROCm EP with its 8K graph) through the image's
    `libonnx_semantic_router.so`. The recipe compiles one fixed
    8,192-token session per model, so every request is an 8K forward.
    The recipe leaves Shield out. Shield also ran on the same ORT MIGraphX
    path from its package's `onnx/model.onnx`, but that graph turned out
    not to be the checkpoint (below). Halu ships no ONNX graph, so no
    legacy ORT path exists for it.
- **Runtime side (`tools/legacy_parity.py runtime --inputs`):**
  - the same deployment options, sent as request options;
  - one request at a time, each input twice (the first answer is compared,
    the second timed), then 4 closed-loop callers for 20 s per model;
  - `exact` profile, FP32;
  - CPU: 16 EPYC 9575F cores pinned with `taskset` (node B 64–79; `torch`
    2.10 CPU, oneDNN packed linears). These latencies are not the gate:
    `vela1-performance.md` times both sides interleaved in one cgroup scope;
  - ROCm: one MI325X, in the router's ROCm image stack (Python 3.12.14,
    PyTorch 2.12.0+rocm7.2, HIP 7.2.53211, Triton 3.7.0, fla-core 0.5.2), as
    its user (uid 65532), with encoder graphs and the fused
    rotary kernel on.
- **Inputs:**
  - 547 texts per sequence / token job: the 531 prompts of the e2e
    testdata, six edge cases and ten long documents of 8–384 prompts
    (0.5–48K characters). The edge cases are mixed scripts and emoji,
    contact details, literal `<bos>` / `<eos>` / `[CLS]` strings, one
    character, surrounding whitespace and German PII.
  - 64 grounded triples for Halu: 60 with contexts of 3–40 prompts, 4 with
    400.
  - The runtime run answers the legacy run's recorded inputs (`--inputs`).
    Input digests: `7f23ef84` (CPU deployments), `8d5a8a55` (AMD
    deployments), `f7475ac8` (Shield on ORT, the same 547 texts).
- **Bars (design section 17):**
  - CPU: label agreement 100% outside near ties (top-two margin below
    1e-3), max |Δp| ≤ 1e-3, identical span sets on at least 99.5% of
    inputs.
  - ROCm: label agreement ≥ 99.5%, max |Δp| ≤ 0.02, identical span sets
    ≥ 98%.
  - Span sets compare label and code-point offsets; span probabilities
    compare where the sets agree.

## Jobs

| Job | Model | Head | CPU deployment | AMD deployment |
| --- | --- | --- | --- | --- |
| domain, safety, factcheck, feedback, modality | sequence models | softmax over labels | truncate at 512 | reject over 8,192 |
| shield | Shield | softmax over labels | truncate at 512 | outside the recipe; ORT MIGraphX from its ONNX graph, timing only |
| guard | Guard | softmax over labels | windows 512 / 255 over 8,192, per-label max | reject over 8,192 (ROCm EP) |
| hazard | Hazard | independent sigmoids, packaged operating point | windows 2,048 / 1,023 over 32,768 | same |
| pii | PII | BIO token spans | windows 512 / 255 over 32,768 | reject over 8,192 |
| pii_truncate | PII | BIO token spans | truncate at 512 (the facade's partial result) | — |
| halu | Halu | grounded answer spans | `("User request: …", answer)` pairs, truncate at 8,192 | — (no ONNX graph) |

## CPU

Latencies here are each side's own run: the legacy run hours earlier, the
runtime on node B 64–79 beside the interleaved A/B. `vela1-performance.md`
compares the two sides interleaved on the same cores, and has the
throughput.

| Job | Compared (both rejected) | Agreement | Max abs Δp | Bar | p50 legacy → runtime (ms) | p95 legacy → runtime (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| domain | 547 (0) | 100.00% | 1.5e-04 | pass | 40.38 → 8.21 | 213.65 → 25.19 |
| guard | 545 (2) | 100.00% | 2.9e-05 | pass | 37.28 → 7.98 | 210.52 → 23.61 |
| safety | 547 (0) | 100.00% | 3.5e-05 | pass | 38.05 → 8.07 | 215.50 → 24.64 |
| shield | 547 (0) | 100.00% | 4.6e-05 | pass | 34.35 → 7.56 | 218.32 → 24.60 |
| factcheck | 547 (0) | 100.00% | 2.2e-05 | pass | 35.58 → 7.69 | 232.02 → 23.46 |
| feedback | 547 (0) | 100.00% | 9.7e-05 | pass | 36.27 → 7.44 | 209.35 → 23.05 |
| modality | 547 (0) | 100.00% | 5.8e-06 | pass | 32.57 → 7.71 | 213.46 → 23.68 |
| hazard | 547 (0) | 100.00% | 9.6e-06 | pass | 33.93 → 9.56 | 213.88 → 28.67 |
| pii | 547 (0) | 100.00% | 9.2e-05 | pass | 34.59 → 10.47 | 211.88 → 30.11 |
| pii_truncate | 547 (0) | 100.00% | 7.2e-05 | pass | 82.45 → 10.06 | 244.36 → 27.16 |
| halu | 64 (0) | 100.00% | 4.4e-05 | pass | 1424.37 → 110.38 | 38738.26 → 2808.69 |

## ROCm

### Against the AMD recipe

ONNX Runtime MIGraphX / ROCm EP on the same GPU:

| Job | Compared (both rejected) | Agreement | Max abs Δp | Bar | p50 legacy → runtime (ms) | p95 legacy → runtime (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| domain | 545 (2) | 100.00% | 1.8e-04 | pass | 154.06 → 1.88 | 155.91 → 3.51 |
| guard | 545 (2) | 100.00% | 4.7e-05 | pass | 245.81 → 1.88 | 258.07 → 3.53 |
| safety | 545 (2) | 100.00% | 3.5e-06 | pass | 155.26 → 1.82 | 164.00 → 3.51 |
| factcheck | 545 (2) | 100.00% | 1.1e-04 | pass | 158.24 → 1.82 | 163.86 → 3.48 |
| feedback | 545 (2) | 100.00% | 1.1e-04 | pass | 157.93 → 1.89 | 164.11 → 3.55 |
| modality | 545 (2) | 100.00% | 5.5e-06 | pass | 158.56 → 1.83 | 167.20 → 3.50 |
| hazard | 547 (0) | 100.00% | 2.5e-06 | pass | 13.38 → 1.84 | 14.13 → 3.53 |
| pii | 545 (2) | 98.90% | 2.4e-05 | pass | 134.03 → 1.93 | 136.05 → 3.70 |

### Shield on legacy ORT: its ONNX graph is not its checkpoint

The runtime against Shield's packaged `onnx/model.onnx` on ORT MIGraphX
(547 inputs: 545 compared, 2 rejected by both sides) **misses the bar**:
97.80% label agreement and max |Δp| 0.81. 339 inputs differ by more than
0.02 and 12 labels flip. The cause is the graph, not the runtime:

- Legacy ORT against legacy candle, on the same inputs, flips 14 labels,
  with max |Δp| 0.85 and median |Δp| 0.026.
- The same MIGraphX path agrees with the runtime to ≤ 1.8e-4 on the other
  seven models.
- Candle and the runtime serve `model.safetensors`. They agree on CPU (max
  |Δp| 4.6e-5) and on ROCm (100%, max |Δp| 4.7e-5, below).

So the published ONNX export answers as a different model. That is
presumably why `vela-amd` never deployed Shield on ORT. The row stays in
`vela1-performance.md` as timing only. Classify packages declare no exit
graphs, so the runtime's `onnxruntime` engine cannot serve this graph.

### Against the CPU legacy reference

The router's CPU options, legacy candle on CPU, the runtime on MI325X:

| Job | Compared (both rejected) | Agreement | Max abs Δp | Bar | p50 legacy → runtime (ms) | p95 legacy → runtime (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| domain | 547 (0) | 100.00% | 3.6e-04 | pass | 40.38 → 1.89 | 213.65 → 3.53 |
| guard | 545 (2) | 100.00% | 5.0e-05 | pass | 37.28 → 1.90 | 210.52 → 3.56 |
| safety | 547 (0) | 100.00% | 3.5e-05 | pass | 38.05 → 1.79 | 215.50 → 3.39 |
| shield | 547 (0) | 100.00% | 4.7e-05 | pass | 34.35 → 1.82 | 218.32 → 3.48 |
| factcheck | 547 (0) | 100.00% | 7.1e-05 | pass | 35.58 → 1.82 | 232.02 → 3.49 |
| feedback | 547 (0) | 100.00% | 9.2e-05 | pass | 36.27 → 1.88 | 209.35 → 3.64 |
| modality | 547 (0) | 100.00% | 8.9e-06 | pass | 32.57 → 1.84 | 213.46 → 3.52 |
| hazard | 547 (0) | 100.00% | 1.0e-05 | pass | 33.93 → 1.83 | 213.88 → 3.50 |
| pii | 547 (0) | 100.00% | 1.5e-04 | pass | 34.59 → 1.95 | 211.88 → 3.74 |
| pii_truncate | 547 (0) | 100.00% | 6.3e-05 | pass | 82.45 → 1.92 | 244.36 → 3.66 |
| halu | 64 (0) | 100.00% | 7.2e-05 | pass | 1424.37 → 7.72 | 38738.26 → 56.15 |

### Disagreements

- **CPU:** none. Every label and every span set is identical on all eleven
  jobs; the largest probability difference is 1.5e-04.
- **ROCm against the AMD recipe:** PII spans differ on 6 of 545 inputs
  (`p0103`, `p0246`, `l01`, `l04`, `l06`, `l07`). In each, the runtime
  reports one more span of a single code point that the ONNX Runtime
  MIGraphX path drops: a digit of a phone number, or one character of a
  street address or an organisation, i.e. a token at the edge of its label.
  Against the CPU reference the runtime's ROCm spans are identical on every
  input, so the difference comes from the legacy GPU path's numerics.
- **ROCm against the CPU reference:** none.
- **Rejections:** the two documents longer than 8,192 tokens are rejected
  by both sides wherever the deployment rejects them: the AMD recipe, and
  Guard's 8,192-token window limit on CPU. `pii_truncate` compares the
  facade's partial result (a truncated scan with its spans) with the
  runtime's `input.truncated` answer.

### The router image's stack against the release image

Since `a580be6b9` the router's ROCm image serves with the release image's
PyTorch and ROCm libraries (`rocm-router-image.md`). In the shipped (slim)
image `31d00387c`, `vela1`'s own ROCm runner gives every one of the AMD
recipe's 4,376 answers byte-identically to the release image's (eight jobs of
547; `legacy_parity.py compare --baseline runtime` agreement 1.0, max |Δ| 0.0),
and readiness's golden answers for all 13 `task_heads` built-ins (the ten
text models, Vela Embedding, Vela Reranker and Qwen3-Embedding) equal the
committed file in every value.

Before that, the image installed the official PyTorch wheel. The ROCm
records before it ran in the packages' release image, where readiness's
golden answers were recorded:

- the router image's stack: PyTorch 2.12.0+rocm7.2, HIP 7.2.53211, Triton
  3.7.0, fla-core 0.5.2;
- the release image: PyTorch 2.12.0+git6bbd260, built from source, HIP
  7.2.53211, Triton 3.7.1, fla-core 0.5.2.

On the same code, the router image's stack gives every label and span of
the release image. Most answers are byte-identical, and the rest differ in
their last bits:

| Job | Inputs | Answers | Labels and spans | Byte-identical | Max abs Δp |
| --- | --- | --- | --- | --- | --- |
| domain | AMD recipe | 547 | 547 | 464 (84.8%) | 3.5e-06 |
| guard | AMD recipe | 547 | 547 | 471 (86.1%) | 1.3e-05 |
| safety | AMD recipe | 547 | 547 | 474 (86.7%) | 1.3e-06 |
| factcheck | AMD recipe | 547 | 547 | 481 (87.9%) | 3.1e-06 |
| feedback | AMD recipe | 547 | 547 | 465 (85.0%) | 9.0e-06 |
| modality | AMD recipe | 547 | 547 | 468 (85.6%) | 1.2e-07 |
| hazard | AMD recipe | 547 | 547 | 462 (84.5%) | 1.1e-06 |
| pii | AMD recipe | 547 | 547 | 478 (87.4%) | 4.3e-05 |
| **all** | AMD recipe | 4,376 | 4,376 | 3,763 (86.0%) | 4.3e-05 |
| domain | CPU recipe | 547 | 547 | 462 (84.5%) | 6.1e-06 |
| guard | CPU recipe | 547 | 547 | 469 (85.7%) | 3.8e-05 |
| safety | CPU recipe | 547 | 547 | 469 (85.7%) | 1.3e-06 |
| shield | CPU recipe | 547 | 547 | 471 (86.1%) | 5.7e-07 |
| factcheck | CPU recipe | 547 | 547 | 483 (88.3%) | 3.1e-06 |
| feedback | CPU recipe | 547 | 547 | 463 (84.6%) | 5.6e-06 |
| modality | CPU recipe | 547 | 547 | 465 (85.0%) | 1.2e-07 |
| hazard | CPU recipe | 547 | 547 | 462 (84.5%) | 1.1e-06 |
| pii | CPU recipe | 547 | 547 | 476 (87.0%) | 2.1e-04 |
| pii_truncate | CPU recipe | 547 | 547 | 476 (87.0%) | 3.4e-05 |
| halu | CPU recipe | 64 | 64 | 20 (31.2%) | 3.3e-05 |
| **all** | CPU recipe | 5,534 | 5,534 | 4,716 (85.2%) | 2.1e-04 |

- **Where they differ:** on longer inputs. Answers to inputs of at most
  200 characters are byte-identical on 446–447 of 450 per
  job. The answers that differ have a median input of 379–446
  characters, and Halu's grounded inputs run to thousands. The two builds'
  kernels round some larger shapes differently.
- **Golden answers:** readiness's ROCm references
  (`registry/golden_answers_vela1.json`) are byte-identical on this stack
  for all 13 `task_heads` built-ins: the ten text models,
  Vela Embedding, Vela Reranker and Qwen3-Embedding. Three fresh processes
  gave them, the first with an empty Triton cache. The release image gives
  them byte for byte too, so no golden answer is re-recorded.
- **Across processes:** separate processes on this stack answer byte for
  byte alike:
  - the AMD-recipe inputs' 4,376 answers in seven
    processes: the five A/B rounds of `vela1-performance.md`, and the
    parity runs at `efb5ec4d7` and `35ff3a8a5`;
  - the CPU-recipe inputs' 5,534 answers and Shield's
    547 in three processes, the third with an empty Triton
    cache.
- **The image as built:** the router's ROCm image from `Dockerfile.extproc` at
  `be7366c49` has this stack's pins plus `causal-conv1d` 1.7.0, which these
  models don't run, the HTTP server's `uvloop` and `httptools`, Pillow, and
  Python 3.12.15 for 3.12.14. With this commit's package, it gives every
  golden value (three processes, the first with an empty Triton cache) and
  every answer of the three parity panels byte for byte as this stack does.
- **Quality bar:** the Decision Index scores no Vela 1.0 model, so there
  is no Index delta to measure. Vela 1.0's references are the legacy
  answers, under design section 17's bars, and this stack passes them on
  every job (above).

## Approximate profiles

`batching` coalesces concurrent requests. A request alone runs as on
`exact`, so its answers are the same (`35ff3a8a5`, ROCm, AMD-recipe
inputs: every value identical one request at a time).

The `max_speed` reduced copies (design section 5.4) were recorded against
`exact` on the whole corpus before any consent. Agreement counts the label,
or the identical span set.

BF16 copy on MI325X (heads FP32), against `exact` on the same inputs
(`973842d9e`):

| Job | Inputs | Agreement with exact | Max abs Δp | Floor (99%) |
| --- | --- | --- | --- | --- |
| domain | AMD-recipe | 99.82% | 1.1e-01 | pass |
| guard | AMD-recipe | 100.00% | 6.3e-02 | pass |
| safety | AMD-recipe | 99.82% | 2.0e-02 | pass |
| factcheck | AMD-recipe | 99.82% | 2.3e-01 | pass |
| feedback | AMD-recipe | 100.00% | 3.0e-01 | pass |
| modality | AMD-recipe | 100.00% | 9.5e-02 | pass |
| hazard | AMD-recipe | 100.00% | 1.4e-02 | pass |
| pii | AMD-recipe | 95.23% | 4.9e-02 | fail |
| domain | router CPU | 100.00% | 1.1e-01 | pass |
| guard | router CPU | 100.00% | 3.5e-01 | pass |
| safety | router CPU | 99.82% | 2.0e-02 | pass |
| shield | router CPU | 100.00% | 2.1e-02 | pass |
| factcheck | router CPU | 99.82% | 2.3e-01 | pass |
| feedback | router CPU | 100.00% | 3.0e-01 | pass |
| modality | router CPU | 100.00% | 9.5e-02 | pass |
| hazard | router CPU | 100.00% | 1.4e-02 | pass |
| pii | router CPU | 94.70% | 4.9e-02 | fail |
| pii_truncate | router CPU | 95.06% | 4.9e-02 | fail |
| halu | router CPU | 75.00% | 2.6e-02 | fail |

BF16 copy on CPU (AVX-512 BF16, 16 EPYC 9575F cores, heads FP32), against
`exact` on the router's CPU inputs (`0a2ced483`, both on node B 64–79):

| Job | Agreement with exact | Max abs Δp | Floor (99%) | p50 exact → copy (ms) | p95 exact → copy (ms) | 4 callers, calls/s |
| --- | --- | --- | --- | --- | --- | --- |
| domain | 100.00% | 7.6e-02 | pass | 7.68 → 12.10 | 22.89 → 33.54 | 84.9 → 102.1 |
| guard | 100.00% | 1.3e-01 | pass | 7.35 → 11.67 | 22.24 → 31.21 | 52.3 → 48.3 |
| safety | 100.00% | 1.2e-02 | pass | 7.65 → 12.01 | 23.84 → 31.95 | 84.5 → 108.5 |
| shield | 100.00% | 2.1e-02 | pass | 7.69 → 11.44 | 23.41 → 30.89 | 83.7 → 107.4 |
| factcheck | 100.00% | 2.1e-01 | pass | 7.70 → 11.72 | 23.57 → 31.60 | 83.9 → 108.8 |
| feedback | 100.00% | 3.0e-01 | pass | 7.64 → 11.98 | 23.54 → 32.58 | 84.0 → 106.4 |
| modality | 100.00% | 3.7e-02 | pass | 7.62 → 11.72 | 23.51 → 31.03 | 84.3 → 107.1 |
| hazard | 100.00% | 1.4e-02 | pass | 7.75 → 11.93 | 23.51 → 31.18 | 40.4 → 27.6 |
| pii | 94.52% | 3.5e-02 | fail | 7.91 → 12.10 | 23.52 → 32.52 | 38.2 → 22.8 |
| pii_truncate | 94.88% | 3.5e-02 | fail | 8.97 → 11.69 | 25.81 → 31.36 | 75.9 → 102.8 |
| halu | 70.31% | 6.0e-02 | fail | 118.97 → 109.15 | 2456.87 → 2154.14 | 4.1 → 3.1 |

**The family consents to no reduced copy** (no `BuiltinModel.reduced`
entry for any Vela 1.0 text model); a copy needs ≥ 99% agreement and a
faster path (design section 5.4):

- **GPU:** BF16 fails the floor for the token and grounded heads. It is
  also slower for a single request, because the forward is launch-bound and
  autocast adds a cast per linear.
- **CPU:** BF16 keeps every label of the sequence and scores heads but fails
  the floor for PII and Halu.
  - It is about 1.5× slower per request than `exact`, whose linears already
    run through oneDNN's packed FP32 kernel (Domain p50 7.68 → 12.10 ms).
    Only Halu's long grounded inputs run faster on it (p50 119 → 109 ms),
    and Halu fails the floor.
  - Its higher throughput on short inputs comes from `max_speed` coalescing
    requests for up to 2 ms; `batching` gives that gain without changing a
    value.
- A `float32-packed` copy would duplicate the `exact` path's own kernels,
  and int8 breaks ModernBERT's activations (`decision1-performance.md`).

## Reproduce

```bash
# legacy (bookworm userland with the bindings built at 61aa7eb2d; the AMD
# recipe inside the legacy ROCm image)
python3 tools/legacy_parity.py legacy --recipe cpu|amd --tree <legacy tree> \
  --cache <hf cache> --flat <dir> --repeats 2 --concurrency 4 --out legacy.jsonl
# Shield on the AMD recipe's ORT path, on the CPU run's inputs
python3 tools/legacy_parity.py legacy --recipe amd --jobs shield \
  --inputs legacy-cpu.jobs.json ... --out legacy-amd-shield.jsonl
# runtime, on a legacy run's inputs (its jobs unless --jobs names others)
python3 tools/legacy_parity.py runtime --recipe cpu|amd --inputs legacy.jobs.json \
  --device cpu|rocm:0 --cache <hf cache> --repeats 2 --concurrency 4 --out runtime.jsonl
python3 tools/legacy_parity.py compare --legacy legacy.jsonl --runtime runtime.jsonl \
  --device-class cpu|rocm --context '{"runtime_commit": "..."}' --record record.json
# a copy or a profile against exact: --baseline runtime --legacy exact.jsonl
```
