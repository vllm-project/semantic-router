# Native engine parity on ROCm (MI325X)

> **Phase 1 record** (Decision 2.0, [#4481](https://github.com/vllm-project/semantic-router/pull/4481) and its follow-ups). The Phases 2–4 records are the `decision1-*`, `vela1-*`, `vela2-*`, `embed-*`, `stores-*`, `router-latency-cpu*`, `router-latency-rocm*`, `rocm-router-image` and `removal-footprint*` files.

The native engine answers exactly as each released package's own runtime on
the four scored panels, on both of its paths, for all six Decision 2.0 sizes.

- **Date:** 2026-10-03.
- **Device:** one AMD Instinct MI325X (gfx942) per run, in the images the
  packages were released with (PyTorch ROCm, Triton, the FLA and
  causal-conv1d kernels).
- **Reference:** the package's own runtime (Transformers remote code,
  `system_one`) on the same image and node.
- **Native side:** `tools/gpu_parity.py`, which loads the package through
  `Decision2Family`, `NativeEngine` and the ROCm accelerator (no package code)
  and answers on the `exact` profile.

## Exact profile, four scored panels

Panels: typed-final (1,600 prompts), css15 (6,547), public231 (231) and
mlx-diag (2,275), 10,653 prompts in all. "Identical" means every answer's
JSON is byte-identical (0 decision changes, max |Δp| 0.0).

| Model | Package revision | Fast path | Eager path |
| --- | --- | --- | --- |
| Kai-0.6B | `881bee41` | 10,653 / 10,653 identical | 10,653 / 10,653 identical |
| Eos-0.8B | `ad0aa724` | 10,653 / 10,653 identical | 10,653 / 10,653 identical |
| Sol-2B | `4b75b521` | 10,653 / 10,653 identical | 10,653 / 10,653 identical |
| Nox-4B | `ce1bdc9d` | 10,653 / 10,653 identical | 10,653 / 10,653 identical |
| Lux-9B | `214ffa43` | 10,653 / 10,653 identical | 10,653 / 10,653 identical |
| Vega-27B | `9b067a95` | 10,653 / 10,653 identical | 10,653 / 10,653 identical |

- **Fast path** (the default on a GPU):
  - every fused kernel of `accel/triton_gfx942.py`, each declared exact
    (`add_rmsnorm`, `residual_add`, `silu_mul`, `attn_prep`, `gated_rmsnorm`,
    `sigmoid_gate`, `gdn_prep`);
  - the lean LoRA (Vega);
  - HIP graphs.
- **Eager path:** `--no-fused --no-graphs`, the backbone modules alone.
- **Hotfix revisions:** Nox `7fc0023a` and Lux `7c6792f7` change only the
  packages' runtime. Their weights are byte-identical, and their default path
  answers these panels byte-identically to the revisions above (hotfix
  records), so this parity carries over to them.

## Shared context

The `shared_context` profile against the released runtime's
`share_context=True` switch, on 207 requests:

- 200 typed-final prompts, each with its own questions plus 15 companion
  questions about the same state;
- the public many-question request (`tools/many_questions.py`) at 2, 4, 8, 16,
  32, 64 and 128 questions.

Every request was shared on both sides (policy `{"min_shared_tokens": 0}`).
The result is 207 / 207 identical for all six sizes.

## Large batches

`tests/test_gpu_fast_path.py::test_attention_prep_above_2gib` runs a
9 × 16,384-token batch at Nox-4B's widths, whose q projection is 2.4 GB. The
fused layer runs it without its eager fallback and equals the eager layer bit
for bit. The released runtime's single-launch kernel fails on such a batch
(Triton's AMD pointer canonicalization); the native kernel launches q and k
separately.

## Commits

The runs span this branch's history:

- eager path at `87879264e`;
- fast path at `0a5a11a4c`;
- shared context at `906055c3e` / `c78c1f350`, and Vega at `df51b0d5f`.

Two later commits keep the same engine:

- **Graph cap** (`df51b0d5f`, on the PR as `f705e2d10`): graphs only up to
  4,096 padded tokens. Kai and Eos were re-run on the fast path, again
  10,653 / 10,653 identical; at `f705e2d10` Kai again.
- **No eviction** (`99208b6d3`): a full graph cache runs new shapes eagerly
  instead of destroying graphs. The six GPU tests pass, including a full
  cache.

## Reproduce

```bash
python3 tools/gpu_parity.py --package PACKAGE_DIR --released RELEASED_ANSWERS.jsonl \
  --panel typed-final:PROMPTS.jsonl:1600 --panel css15:PROMPTS.jsonl:6547 \
  --panel public231:PROMPTS.jsonl:231 --panel mlx-diag:PROMPTS.jsonl:2275 \
  --output parity.json [--no-fused --no-graphs] [--base-path BASE_DIR]
python3 tools/gpu_parity.py ... --profile shared_context --share-policy '{"min_shared_tokens": 0}'
pytest -m gpu tests/test_gpu_fast_path.py
```
