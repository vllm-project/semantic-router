# Decoder M6b + Milestone 7 — state (keep current; newest first)

Assignment: coordinator note 2026-09-30 01:50. Budget: 30 GPU-h in total for M6b + M7. GPUs: node A GPU5, node B
GPU3–4. Preregistrations: [M6b](dec-m6b-prereg-2026-09-30.md); M7 (to come, before any M7 GPU job).

## Now

- 2026-09-30 ≈02:40 UTC+8 — M6b preregistered with the item-7 fix (successor script gates `gates public231`,
  reads item 8 from the custodian). Next: mirror, stage and smoke both N6D finalists on node B GPU3 / GPU4.
- Data-track facts that shape M7 (read 02:00):
  - PN1 `@5ad36287` failed its §3 blind label review (12 / 224 gold errors, all false "no" in `pn-near`
    es / fr / ar / ru / ko and Russian `pn-name`). A PN1-r2 rebuild with a fresh certification is under way
    (data prereg amendment `46702e2fc`).
  - HS1 `@171e6f0c`: one confirmed template defect (F2 travel-expense rendered "nights" twice; 150 TRAIN rows;
    fix `05a0d391d`). HS1 is release-safe only after the data track's §4 spot check closes.

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| M6b | 0 |
| M7 | 0 |
| **Total (cap 30)** | **0** |
