# vLLM plugin prototype — state (keep current; newest first)

Assignment: coordinator notes 2026-09-30 19:40 (serving track), 19:50 / 20:00 (runtime design, `/v1/decisions`),
~23:40 (Tier 1a / 1b split) and the 23:55 continuation. Branch `xunzhuo/decision-2-vllm-plugin`, gist
`09-decision-2-serving.md`. GPU: node A GPU1 (shared lease `owner.serving`), ≤ 3 GPU-h in total. Nothing uploaded.

## Now

- 2026-10-01 ≈01:10 UTC+8 (17:10Z) — **All measurements done; record written; GPU1 released (0.79 GPU-h total).**
  Record: [`vllm-plugin-prototype-2026-10-01.md`](vllm-plugin-prototype-2026-10-01.md).
  - float32: parity with `VLLM_SR_GDN_BF16_INPUTS=1` + `--attention-backend TRITON_ATTN` (the default ROCm
    prefix-prefill kernel needs 128 KiB LDS at float32 / head 256): 42 / 11,053 changes (bf16 63); 30.0 ms p50,
    239 items/s at concurrency 128.
  - Control: Transformers runtime with BF16 parameters changes 64 / 11,053 → the bf16 gap is BF16 residual precision.
  - Changed answers are near-ties (stored margin ≤ .026). Vela: 0 argmax changes (≤ 32,714 tokens). Label-token demo:
    0 argmax changes vs Transformers; 86% prefix-cache hits on a shared state. Co-location: ≤ 7% interference at 1–8
    workers.
  - Next: gist 09, merge into `xunzhuo/decision-2-training`; optional Tier 1b prefix-cache experiment.
- 2026-10-01 ≈00:35 UTC+8 (16:35Z) — **Plugin serves; BF16 parity and runtime measured; second chain running.**
  - `p08-bf16-0930T1611` (mirror `9c0abb957`): every request answered, alias identical. Answer changes vs stored
    predictions: public231 3 / 231, typed FINAL 20 / 2,000, CSS15 33 / 6,547, mlx-diag 7 / 2,275 (max drift .022 /
    .061 / .031 / .124); public231 at concurrency 32 also 3. Bench (400 typed FINAL items, HTTP): single p50 15.2 ms;
    66 / 208 / 449 / 622 items/s at concurrency 1 / 8 / 32 / 128.
  - `p08-fp32-0930T1617`: `--dtype float32` fails at engine start (vLLM's chunked gated-delta kernel asserts on FP32).
    Added an opt-in `VLLM_SR_GDN_BF16_INPUTS=1` custom-op override (`CustomOp.register_oot`; `ecb46f764`).
  - `rt08-0930T1618` (frozen autotune cache): FP32 master p50 22.7 ms, 44.1 items/s, peak 2.91 GiB; BF16-resident
    p50 21.4 ms, 46.8 items/s, peak 2.01 GiB, 186 / 186 Linear converted, bit-identical on all 400 items and 0 changes
    on all four panels (drift ≤ 9e-16).
  - Vela / label / coloc on `9c0abb957` failed at load: HF cache weights are symlinks into the shared
    `/data/dev2/hf-cache/blobs` store, which was not mounted. Relaunched with it.
  - Chain `chain-ecb46f764.sh` (node A GPU1): FP32 + GDN-BF16-inputs plugin parity/bench → Transformers BF16-parameter
    control (`bf16-full`) → vela → labeltoken → coloc.
- 2026-10-01 ≈00:05 UTC+8 — **Continuation (6c2c1341) resumed after the ~20:40 silent stop.**
  - Found: three failed plugin sessions on node A GPU1 (0.093 GPU-h). The last (`p08-bf16-0930T124140`, mirror
    `678ac2060`) loaded the model and failed at the first request: `from __future__ import annotations` turned the
    route's `Request` annotation into a string, so FastAPI treated it as a query parameter (HTTP 400). The
    uncommitted fix (drop the future import) is valid and kept.
  - Code: `/v1/decisions` is now the route (entry point `vllm_sr_decisions`), `/v1/system_one` an alias; a FastAPI
    route test through the service would have caught the bug on CPU; sessions call `/v1/decisions`; panels take an
    optional concurrency; BF16-resident conversion counts non-exact Linear weights instead of failing; new
    `labeltoken` session (Tier 1a demo on the generate runner with stock Qwen3.5-0.8B).
  - Next: mirror, then on node A GPU1 in order: plugin bf16 (4 panels + bench), plugin fp32, runtime (FP32 master +
    BF16-resident, frozen autotune cache), vela, labeltoken, coloc if budget allows; record + gist 09 + merge.

## Inputs (node A)

| Input | Path | Check |
| --- | --- | --- |
| DEV2.0-0.8B (BF16 storage, revision `bede7938`) | `/data/dev2/runs/release/dev2-bf16-0.8B-20260929T224530Z/download/DEV2.0-0.8B` | manifest `a653dde2…` |
| Stored scored predictions | `/data/dev2/runs/release/inputs/dev2-0p8b-t1/derived/*.predictions.jsonl` | = release parity-post digests |
| Gold-free panel prompts | `/data/dev2/private/panels/goldfree/*.prompts.jsonl` | typed-final 1,600; css15 6,547; public231 231; mlx-diag 2,275 |
| Vela-1.0-Encoder-307M-Domain | `/data/dev2/hf-cache/models--llm-semantic-router--Vela-1.0-Encoder-307M-Domain` | router `domain_classifier` |
| Qwen3.5-0.8B (stock) | `/data/dev2/hf-cache/models--Qwen--Qwen3.5-0.8B` | Tier 1a demo |

## GPU-hours (node A GPU1)

| Run | Mirror | GPU-h | Outcome |
| --- | --- | --- | --- |
| smoke-bf16-122544 | `1408d273a` | 0.0164 | engine failed (plugin task rejected by Model Runner V2) |
| smoke-bf16-123241 | `711330eac` | 0.0392 | engine failed (runtime predates Score offsets) |
| p08-bf16-0930T124140 | `678ac2060` | 0.0372 | first request HTTP 400 (FastAPI annotation) |
| p08-bf16-0930T1611 | `9c0abb957` | 0.0986 | BF16 parity + bench done |
| p08-fp32-0930T1617 | `9c0abb957` | 0.0264 | FP32 engine start fails (GDN kernel assert) |
| rt08-0930T1618 | `9c0abb957` | 0.0767 | shipped runtime FP32 master + BF16-resident done |
| vela-0930T1623 | `9c0abb957` | 0.0014 | tokenizer not found (blob store unmounted) |
| label-0930T1624 | `9c0abb957` | 0.0231 | weights not found (blob store unmounted) |
| coloc-0930T1626 | `9c0abb957` | 0.0436 | Vela engine load failed (same) |
| p08-fp32gdn-0930T1629 | `ecb46f764` | 0.0358 | FP32 start fails in ROCm prefix-prefill attention (LDS) |
| rt08-bf16full-0930T1632 | `ecb46f764` | 0.0747 | Transformers BF16-parameter control done |
| vela-0930T1637 | `ecb46f764` | 0.0219 | endpoint plugin aborted the ModernBERT server (fixed `14fe9ac40`) |
| label-0930T1638 | `ecb46f764` | 0.0414 | demo done; shared state over the context (superseded) |
| p08-fp32-tri-0930T1641 | `ecb46f764` | 0.1117 | FP32 parity + bench done |
| vela-0930T1648 | `564d5f8e8` | 0.0364 | Vela float32 + bfloat16 done |
| label-0930T1651 | `564d5f8e8` | 0.0414 | label-token demo + prefix cache done |
| coloc-0930T1654 | `564d5f8e8` | 0.0617 | co-location done |
| **Total** | | **0.788** | |

## Plan

| Step | Status |
| --- | --- |
| Plugin v0: model class, pooler, `/v1/decisions`, pinned build | done |
| DEV2.0-0.8B parity, bfloat16 and float32 | done |
| Vela-1.0 Domain via native ModernBERT classify vs Transformers | done |
| Serving-shape notes (+ co-located measurement) | done |
| Latency / throughput: plugin vs shipped runtime (+ BF16-resident) | done |
| Tier 1a label-token demo (generate runner) | done (0.08 GPU-h) |
| Record | done |
| Gist 09, merge into `xunzhuo/decision-2-training` | pending |
