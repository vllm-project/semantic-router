# Decoder M17 stage 2, amendment 3: wave 4, a third seed (2026-10-02)

Coordinator watchdog, 18:00 UTC+8: node F GPU4–5 are free; start more seeds of the best 4B recipes so far.

This amendment adds one seed (20260928) to two existing arms, on the arms' locked data and with their recipe:

| Chain | Item | Data lock |
| --- | --- | --- |
| node F GPU4 (`M17_STAGE=3`) | `4b-LHS17IB4` seed 3 | `data/4b-s3`, `READY-m17s3.json` |
| node F GPU5 (`M17_STAGE=4`) | `4b-SDMLIB4` seed 3 | `data/4b-s4`, `READY-m17s4.json` |

- Each new seed's measured arm is a new three-seed soup, `4b-LHS17IB4-x3` and `4b-SDMLIB4-x3`: the uniform FP32
  average of seeds 1–3's BEST checkpoints. The two-seed soups stay as built and measured.
- The stop rules are unchanged. The arm cap is 5.0 GPU-h; each arm has used about 2.3–2.5 GPU-h, so one more seed
  fits. The node-F gate is 50 GPU-h; about 19.6 GPU-h is used.
- The measurement and the release rule are unchanged: one Index run on the BF16 release copy, compared with
  `IS-4b-LHA10SDML-bf16`. A candidate is released only if its lower bound is > 0, and only after the integrity
  checks pass.
