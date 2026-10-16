# NPU bf16 autocast: Ascend 950DT (Vela 1.0 / Vela 2.0)

> A1 record 草稿（2026-10-16，本地版）。上游落位：`src/model-runtime/docs/records/npu-bf16-ascend950dt.md`，
> 随 record PR 提交（数据采集脚本 `npu-bench/`，原始 JSON 在 36 `sr-build/npu-bench/logs/`）。

## Findings

`--dtype bfloat16` on the NPU accelerator (reference kernels, Ascend 950DT, CANN 9.1):

- **Vela 2.0 0.3B (four-signal decisions, 665 tokens/request): 2.5× throughput.**
  84.8 → 214.1 rps at concurrency 8; 89.7 → 246.6 rps at concurrency 16. Forward p50 drops
  46.1 → 15.9 ms (2.9×). The GDN decoder is matmul-bound, so bf16 autocast converts directly.
- **Vela 1.0 Domain (classify, 195 tokens/request): no change.** 1417 → 1420 rps at c8.
  The encoder already runs at ~5 ms/request wall, far from compute-bound; dispatch and
  per-request overhead dominate.
- **Readiness golden passes on both models under bf16** (bf16 outputs vs fp32-recorded NPU
  references, GPU_TOLERANCE = 0.02).
- **Label flips: zero** across 30 requests × all signals on both models (choice argmax and
  set membership identical between fp32 and bf16).
- **Numeric drift exists above tolerance on raw confidences**: max |Δ| = 0.0376 (V2.0) and
  0.0734 (V1.0); 0.8% (V2.0) / 4.3% (V1.0) of numeric leaves exceed 0.02. Routing decisions
  are unaffected for this corpus, but rules whose thresholds sit on a confidence boundary
  can flip under bf16 — an opt-in accuracy record is required before any default change.

## Setup

- Host: 2-socket Kunpeng 950 7590, 8× Ascend 950DT (driver 25.6.rc1, CANN 9.1.0),
  torch 2.10.0+cpu / torch_npu 2.10.0.post4, runtime main `7313c646c` (accelerator #4827,
  references #4850). Card 3 shared with a co-tenant container (HBM ~90/98 GB, AICore idle).
- Corpus: 30 office-domain prompts (frozen, md5-pinned), four-signal decisions schema
  (473 schema tokens + prompt = 665 input tokens p50); V1.0 via `/v1/classify` (195 tokens).
- Concurrency 8, 120 requests per point (answers pass: 30). fp32 baseline re-collected in
  the same window with the same script.
- Collection: `--dtype bfloat16` on `vllm-srun serve`; readiness golden gate runs at startup.

| Model | dtype | c4 rps | c8 rps | c16 rps | fwd p50 @c8 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Vela 2.0 0.3B | float32 | 82.6 | 84.8 | 89.7 | 46.1 |
| Vela 2.0 0.3B | bfloat16 | 211.0 | 214.1 | 246.6 | 15.9 |
| Vela 1.0 Domain | float32 | — | 1417.4 | 1349.4 | ~5 |
| Vela 1.0 Domain | bfloat16 | — | 1420.2 | 1400.2 | ~5 |

Accuracy diff (fp32 vs bf16, same 30 requests, tolerance 0.02):

| Model | label flips | numeric leaves | max \|Δ\| | over tolerance |
| --- | --- | --- | --- | --- |
| Vela 2.0 0.3B | 0 | 1410 | 0.0376 | 11 (0.8%) — fact_check/jailbreak confidences |
| Vela 1.0 Domain | 0 | 420 | 0.0734 | 18 (4.3%) — class probabilities |

## Implications

1. bf16 autocast is a **working, gated opt-in** on NPU: readiness passes, routing labels are
   unchanged, and the decoder-family speedup is 2.5×. Encoder-family models see no benefit.
2. Keep `--dtype auto` (= published fp32 precision) as the default; bf16 is an explicit
   engine argument. A default switch would need a wider corpus (threshold-boundary rules)
   and a bf16-keyed reference set decision — out of scope here.
3. The existing tolerance gate already accepts bf16 on this device class; no reference
   re-recording was needed.
4. Next lever in the same direction: fused kernels (A2) — the remaining 15.9 ms forward at
   c8 still includes non-matmul overhead that bf16 does not touch.
