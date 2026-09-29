# Decoder M6b + Milestone 7 — state (keep current; newest first)

Assignment: coordinator note 2026-09-30 01:50. Budget: 30 GPU-h in total for M6b + M7. GPUs: node A GPU5, node B
GPU3–4. Preregistrations: [M6b](dec-m6b-prereg-2026-09-30.md) (`833ec0967`), [M7](dec-m7-prereg-2026-09-30.md).

## Now

- 2026-09-30 ≈03:20 UTC+8 — **M6b done: no successor** ([results](dec-m6b-results-2026-09-30.md)). N6D soup
  −2.67 [−4.59, +0.33], ⅔ point −0.89 [−2.88, +1.16] vs DEV2.0-4B; typed FINAL fell (T .647 / .671 vs .688).
  0.284 GPU-h. M7 preregistered with its tooling; next: mirror, `m7-prep.sh` on node B (CPU), data lock, launch.
- Data-track facts that shape M7 (read 02:00):
  - PN1 `@5ad36287` failed its §3 blind label review (12 / 224 gold errors, all false "no" in `pn-near`
    es / fr / ar / ru / ko and Russian `pn-name`). A PN1-r2 rebuild with a fresh certification is under way
    (data prereg amendment `46702e2fc`). M7's P arms follow the preregistered revision rule.
  - HS1 `@171e6f0c`: one confirmed template defect (F2 travel-expense rendered "nights" twice; 150 TRAIN rows;
    fix `05a0d391d`); M7 drops those 75 groups. HS1 is release-safe only after the data track's §4 spot check
    closes.

## Plan

| Chain | GPU | Items |
| --- | --- | --- |
| b3 | node B GPU3 | N7H s1–s3 → N7P s1, s3 |
| b4 | node B GPU4 | N7C s1–s3 → N7P s2 |
| a5 | node A GPU5 | S7H s1–s3 → S7C s1–s3 → S7P s1–s3 |

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| M6b (smokes, CAL fits, two formal collections) | 0.284 |
| M7 | 0 |
| **Total (cap 30)** | **0.284** |

## Incidents

1. M6b first launch stopped before any job: M6's stale `owner.dec-formal` / `owner.m6-formal-smoke` lease entries on
   node B GPU3–4. Only those decoder entries were removed (copies under `m6b/logs/stale-leases/`).
