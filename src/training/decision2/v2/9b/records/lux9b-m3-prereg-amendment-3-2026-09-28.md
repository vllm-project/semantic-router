# 9B Milestone 3 preregistration, amendment 3: arm DL (arm D's data with arm A's LoRA recipe)

Frozen 2026-09-28 before any arm DL step and before any arm B or D readout (arm A's development
readout was known: proxy tie, typed-DEV Score floor breached, not a finalist).

- **Why:** GPU2–3 are otherwise idle from the end of arm B (≈ 23:10 UTC+8) until arm D finishes
  (≈ 03:00). Arm DL is a new single-factor arm, not a rerun: it trains on arm D's exact TRAIN
  (`d-full`, 205,790 rows / 116.1M tokens, same teacher file and trust region) with arm A's
  validated LoRA recipe (r16 / α32, LoRA lr 5e-5, head lr 2.5e-5, ≤ 16,384 tokens and ≤ 16 rows
  per micro-batch, ≥ 16 rows per update, one epoch). DL − A isolates the added data (A7 + v1 +
  the new A7 human Score arms, minus A's natural24k replay); DL − D contrasts full fine-tuning
  with LoRA on the same data (together with the batch and learning-rate settings each recipe
  carries). It is also the fallback candidate if full fine-tuning erodes Lux, as it did Nox at 4B.
- **Seeds:** 20260926 (GPU3, with preflights, after arm B's pipeline there) and 1 (GPU2, after
  arm B and any arm B formal run there). Two seeds; finalist rule, formal protocol and gate as
  for arms A/B (two-seed mean; primary seed's BEST to the formal runner).
- **Budget:** ≈ 2 × 5.3 GPU-h. Milestone 3 cap raised from 34 to **38 GPU-hours**.
