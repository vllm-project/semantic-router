# 0.6B Milestone 3 amendment (M3b-3): bidirectional Qwen under the (a2) schedule

Frozen 2026-09-28 before any (b2) run.

## Why

- (a2) s2 — the seed whose (a) run collapsed at the end of warmup (SELECT 280 at the
  update-300 stop) — trains under the gentler schedule: SELECT 245 → 449 at update 100
  → 498 at update 200, gradient norm 8–48. The early collapse is a property of the
  backbone schedule (2e-5 peak after 5% warmup at batch 16), not of padding or precision.
- The bidirectional pilots tested the freeze, one-row batches, FP32 and a 5e-6 peak on
  Milestone 2 data; none tested (a2)'s schedule on the Milestone 3 mixture. The
  coordinator's arm (b) asks for the bidirectional model "under the fixed trainer", and
  (a2)'s schedule is now the best-supported fix.

## Arm (b2)

`m3-b2-bidir-s1|s2`: bidirectional official Qwen3-0.6B-Base marker readout with exactly
(a2)'s data, teacher, loss, seeds, schedule (backbone 1e-5, head 2e-4, warmup 10%,
cosine to 1e-6), padded eight-row micro-batches, checkpoints and collapse stop (SELECT
< 350 at update 300). (a2) vs (b2) is a single-factor architecture contrast.

Gates: s2 runs only if s1 clears the stop and reaches BEST SELECT ≥ 450; (d) EuroBERT
runs only if both (b2) seeds do. Queue: GPU1 runs (b2) s1 after (a2) s2 and before
(c) s2. The Milestone 3 cap stays 5.0 GPU-hours; (b2) s2 and (d) are skipped if they
would exceed it.
