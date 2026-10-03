# ROCm golden answers and cross-process repeatability (MI325X)

Readiness on ROCm now checks reference answers for all six Decision 2.0
models, and every process on gfx942 answers the Qwen3.5 sizes bit for bit
like the released runtime. A shared autotune cache alone does not guarantee
that, so the runtime pins the released runtime's kernel choices.

- **Date:** 2026-10-03.
- **Device:** AMD Instinct MI325X (gfx942), GPU0 and GPU1 of one node, one
  process per container.
- **Image:** the packages' release image: PyTorch 2.12.0 (ROCm), Triton 3.7.1,
  FLA 0.5.2, causal-conv1d 1.7.0.
- **Packages:** the current Hub revisions, read from the Hugging Face cache:
  Kai `cd49ea38`, Eos `3594047d`, Sol `64235bef`, Nox `25e8f67d`, Lux `78bf3c03`
  and Vega `7aec49ae`.
- **Commits:** the cross-process runs at `cfecb2941` (no pinning) and
  `678ae6fe5` (pinning); golden answers at `678ae6fe5`.

## The current Hub revisions are the built-in models

The six current `main`s are runtime-only revisions: only the packages' bundled
Python changed, which this runtime never imports.

- Each one's model identity equals the built-in table's `model_sha256`; only
  the manifest digest differs.
- The runtime serves them with `--revision <40-hex>`. It verifies them against
  their own manifests, and the golden references and kernel choices apply by
  identity.
- On one GPU with one autotune cache, the pinned revisions and the current
  `main`s give bit-identical golden answers (Kai, Eos, Sol, Nox and Lux
  compared).

The table therefore keeps its pins. Moving the default revision to the current
`main`s is a separate change (docs, examples and tests pin the old revisions).

## Cross-process repeatability

Each run is a separate process. It loads the model as `vllm-sr-runtime serve`
does (verification, the golden check, `--autotune-cache`) and answers the four
scored panels (10,653 prompts, one request each, `exact`). It is then compared
with a reference run prompt by prompt (`tools/cross_process.py`). Identical
means every answer's JSON is byte-identical.

### Eos-0.8B

| Run | Autotune cache (entries before → after) | Against the reference |
| --- | --- | --- |
| Reference: cold, GPU0 | shared, empty → 11 | — |
| Warm, GPU0 | shared, 11 → 11 | 10,653 / 10,653 identical |
| Warm, GPU1 | shared, 11 → 11 | 10,653 / 10,653 identical |
| Cold, GPU1, run beside the reference | its own, empty → 11 | 10,653 / 10,653 identical |
| No `--autotune-cache` | container-local | 10,653 / 10,653 identical |
| **Two cold processes started together, GPU0 and GPU1** | **one shared, empty → 11** | **GPU0: identical. GPU1: 77 / 10,653 identical, 47 decisions changed, max \|Δp\| 0.056, median 0.0002** |
| The release's cache (a copy) | release, 11 → 11 | 10,653 / 10,653 identical |
| Pinned choices, cold | empty → 0 | 10,653 / 10,653 identical to the release's cache |
| **Pinned, two cold processes started together** | **one shared, empty → 0** | **both 10,653 / 10,653 identical to the release's cache** |

### Sol-2B

| Run | Autotune cache (entries before → after) | Against the reference |
| --- | --- | --- |
| Reference: the release's cache (a copy), GPU1 | release, 11 → 11 | — |
| **Two cold processes started together, GPU0 and GPU1** | **one shared, empty → 11** | **GPU0: 75 / 10,653 identical, 35 decisions changed, max \|Δp\| 0.045, median 0.0002. GPU1: 10,652 / 10,653 identical (one css15 prompt, max \|Δp\| 0.0005)** |
| Warm on that cache afterwards, GPU0 | shared, 11 → 11 | 10,652 / 10,653, the same as the GPU1 process (identical to it on 10,653 / 10,653) |
| **Pinned, two cold processes started together** | **one shared, empty → 0** | **both 10,653 / 10,653 identical** |

### What this shows

- **A warm shared cache repeats exactly.** Every later process reads every
  choice from the cache, on either GPU. It repeats the process that wrote the
  cache, not the release: Sol's shared cache differs from the release's on one
  long prompt.
- **A cold cache does not.** FLA times each kernel's configurations at the
  first call of each tuning key. Two processes that tune at the same time can
  choose differently, and one of them then answers almost every prompt
  differently by rounding. The drift exceeds the golden tolerance (0.02), and
  decisions flip. A fleet that starts several replicas at once on one empty
  volume, or without a shared volume, is exposed to it.
- **The release's choices are their own.** The release caches differ from
  fresh processes and from each other, even for the same tuning keys: Eos and
  Sol (16 heads), and Nox and Lux (32 heads), chose differently for
  `chunk_fwd_kernel_o`, `fwd_h`, `kkt_solve` or `recompute_w_u`. Many of these
  differences are neutral, but which ones matter is a property of the kernels'
  layouts, not something a cache can promise.

## Pinned kernel choices

`registry/kernel_choices.json` holds, per Qwen3.5 model and device class
(`rocm:gfx942`), the configuration of each autotuned FLA forward kernel for
each tuning key its released runtime saw:

- `chunk_fwd_kernel_o`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
  `chunk_gated_delta_rule_fwd_kkt_solve_kernel`, `recompute_w_u_fwd_kernel`
  (keyed by head counts and widths: one entry per model);
- `chunk_local_cumsum_scalar_kernel` (keyed by the batch's rows) and
  `l2norm_fwd_kernel` (keyed by the number of 65,536-row blocks).

They come from each model's release autotune cache
(`tools/kernel_choices.py`): the cache its release ran with and that every
later runtime-only revision's parity runs copied forward. The choices are
identical across the release, hotfix and no-eviction revisions.

Before FLA is imported, the runtime writes them as FLA config files and sets
`FLA_CACHE_MODE=full`. A recorded key matches exactly. Any other key, for
example a batch with more rows than the release saw, takes the kernel's entry
of the same structure, or its first entry. No configuration depends on timing,
so a pinned process tunes nothing (0 autotune entries) and loads faster (Eos
22 s instead of 38 s cold). Choices apply only when the installed FLA is the
recorded version (0.5.2); otherwise FLA keeps its per-process tuning. Kai
(dense) has no autotuned kernels.

`--autotune-cache` still persists Triton's compiled kernels and the tuning of
any model without pinned choices.

## Golden answers on ROCm

`registry/golden_answers.json` now carries a `rocm` reference for every
built-in model, recorded with `tools/golden_answers.py` with the pinned kernel
choices, at `678ae6fe5`.

| Model | Current `main` (GPU0) | Pinned revision (GPU1) | Readiness |
| --- | --- | --- | --- |
| Kai-0.6B | `cd49ea38` | `881bee41` | matched 3 / 3 |
| Eos-0.8B | `3594047d` | `ad0aa724` | matched 3 / 3 |
| Sol-2B | `64235bef` | `4b75b521` | matched 3 / 3 |
| Nox-4B | `25e8f67d` | `ce1bdc9d` | matched 3 / 3 |
| Lux-9B | `78bf3c03` | `214ffa43` | matched 3 / 3 |
| Vega-27B | `7aec49ae` | `9b067a95` (with its pinned base) | matched 3 / 3 |

- Each process started with an empty autotune cache. The two revisions answer
  bit-identically on the two GPUs.
- The first recording, with a fresh process's tuning, differed by up to 0.0021
  (Sol, Nox and Vega). That is within the 0.02 tolerance. The references are
  now the pinned, released numerics, so a pinned process matches them bit for
  bit.
- ROCm references differ from the CPU ones by up to about 0.01 (BF16 autocast
  on GPUs against FP32 on CPU), hence one reference per device class.
- Other ROCm architectures have no pinned choices. They tune per process, use
  the fused-kernel-free path, and are checked against the same references at
  the 0.02 tolerance; this record does not cover them.

## Reproduce

```bash
python3 tools/golden_answers.py vllm-sr/Decision-2.0-Eos-0.8B --revision REV --device rocm:0 \
  --autotune-cache DIR
python3 tools/kernel_choices.py vllm-sr/Decision-2.0-Eos-0.8B --autotune-cache RELEASE_CACHE \
  --device-class rocm:gfx942 --fla 0.5.2 --triton 3.7.1
python3 tools/cross_process.py answer vllm-sr/Decision-2.0-Eos-0.8B --revision REV --device rocm:0 \
  --autotune-cache DIR --panel NAME:PROMPTS.jsonl:COUNT ... --answers A.jsonl --receipt A.json
python3 tools/cross_process.py compare REFERENCE.jsonl A.jsonl B.jsonl --output compare.json
```
