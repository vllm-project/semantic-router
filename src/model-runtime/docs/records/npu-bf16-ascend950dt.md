# NPU bf16 precision: Ascend 950DT (Vela 1.0 / Vela 2.0)

`--dtype bfloat16` holds a model's weights in bf16 with the reference kernels
on the NPU accelerator (the `--dtype auto` default keeps each model's
published fp32 precision; this is the held-weights path, not an autocast
context). Measured on Ascend 950DT, CANN 9.1, torch_npu 2.10.0.post4.

## Findings

- **Vela 2.0 0.3B (four-signal decisions, 665 tokens/request): 2.5× throughput.**
  84.8 → 214.1 rps at concurrency 8; 89.7 → 246.6 rps at concurrency 16. Forward p50 drops
  46.1 → 15.9 ms (2.9×). A plausible mechanism is the request size: at 665 tokens the
  bf16 matmuls pay off directly, while the 195-token member stays in the
  overhead-dominated regime — but without a profile this reads as a hypothesis.
- **Vela 1.0 Domain (classify, 195 tokens/request): no change.** 1417 → 1420 rps at c8.
  The member already runs at ~5 ms/request wall; per-request overhead dominates at
  that size.
- **Readiness reports unverified under bf16, by design.** The golden references are
  recorded at `--dtype auto`, so at `--dtype bfloat16` startup checks determinism and
  well-formedness only (`golden references are recorded at --dtype auto, not
  bfloat16`). The accuracy evidence below is an **offline per-request diff** against
  the same host's fp32 answers at the 0.02 tolerance — not a readiness comparison.
- **Label flips: zero on both models over this corpus** (30 requests each; choice
  argmax and set membership identical between fp32 and bf16).
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
- Collection: `--dtype bfloat16` on `vllm-srun serve`; readiness runs at startup and
  reports unverified under bf16 (references are recorded at `--dtype auto`). The diff
  below is the offline per-request comparison.

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

The machine-readable summary lives next to this record
(`npu-bf16-ascend950dt.json`); the raw per-request answers stayed on the
measurement host.

## Implications

1. bf16 precision is a **working, gated opt-in** on NPU: readiness reports unverified
   (as it must off the recorded dtype), routing labels are unchanged on this corpus, and
   the 665-token member's throughput is 2.5×. The 195-token member's runs show no change.
2. Keep `--dtype auto` (= published fp32 precision) as the default; bf16 is an explicit
   engine argument. A default switch would need a wider corpus (threshold-boundary rules)
   and a bf16-keyed reference set decision — out of scope here.
3. Readiness reports unverified under bf16 — the golden references are recorded at
   `--dtype auto` — and accuracy rests on the offline per-request diff.
4. Next lever in the same direction: the follow-up fused-kernel work — the remaining
   15.9 ms forward at c8 still includes non-matmul overhead that bf16 does not touch.
