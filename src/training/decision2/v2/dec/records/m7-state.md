# Decoder M6b + Milestone 7 — state (keep current; newest first)

Assignment: coordinator note 2026-09-30 01:50. Budget: 30 GPU-h in total for M6b + M7. GPUs: node A GPU5, node B
GPU3–4. Preregistrations: [M6b](dec-m6b-prereg-2026-09-30.md) (`833ec0967`), [M7](dec-m7-prereg-2026-09-30.md).

## Now

- 2026-09-30 ≈09:35 UTC+8 (01:35Z) — **2B done: no successor.** `2b-finalists.json` (01:14Z): slot 1
  `2b-S7H-b1` only (L-S7P: H3 .4244–.4251 < .4278 and Score 172–208 < 245; L-S7C likewise). Formal
  (`formal/m7/m7-2b-S7H-b1`, node A GPU5, T = 1 — CAL698 rejected by the 23:15 rule, cache copy `abdfd687…`):
  v3 50.673 (T .555 / H .463) vs DEV2.0-2B 53.437 (T .543 / H .525): **−2.76 [−3.89, +1.90], item 1 FAIL**; 6(b)
  FAIL; items 2, 3, 4 (mlx +.0037 [−.0053, +.0128]), 5 (vs Sol 1.0 16K +4.89 [+2.64, +10.26]), 6(a), 7 (public 231
  173 vs 171, hard 59 vs 57) pass; vs Decider 2B +1.17 [−4.19, +5.40]. Typed FINAL C / N / S 433 / 616 / 158 vs
  445 / 567 / 175: here typed held (+.012) and human transfer fell (−.063, CSS15 mrf −.074, reddit_humor −.054).
  No C1 candidate. `successor/2b-choice.json`: none.
  - 4B: N7P s3 (last seed) ends ≈01:48Z; `4b-finalists.json` ≈02:15Z; formal wrappers armed on node B.
- 2026-09-30 ≈08:00 UTC+8 (00:00Z) — **Continuation worker (the first M7 worker was stopped by the platform at
  ≈23:30Z; no job was lost). H and C arms done on both tiers with three seeds each (no cap block: 4B seeds cost
  1.35 GPU-h, as preregistered).** P arms training; lines, diagnostics and formal wrappers are armed.
  - Lines read (16K, vs `<tier>-I`; gate = type / family floors + CSS-pilot H3 ≥ H3_I):

    | Line | β 1 T / H3 | β ½ T / H3 | β ⅓ T / H3 | Pick |
    | --- | --- | --- | --- | --- |
    | 4b `I` | .704 / .5625 | | | |
    | L-N7H | .749 / .5458 | .741 / .5513 | .731 / .5593 | none (H3 below I at every β) |
    | L-N7C | .648 / .5634 (Choice 440 < 477 floor) | .703 / .5663 | .708 / .5643 | β ½ expected (rules pending) |
    | 2b `I` | .610 / .4278 | | | |
    | L-S7H | .683 / .4421 | .668 / .4388 | .652 / .4378 | β 1 expected (rules pending) |
    | L-S7C | .614 / .4163 (Score 178 < 245) | .618 / .4181 | .621 / .4200 | none |

  - Running (UTC ETA): node B `m7-N7P-s1` (GPU3) / `s2` (GPU4) → ≈00:27, then `s3` on GPU3 → ≈01:50, soup, line,
    diagnostics, `4b-finalists.json` ≈02:20. node A `m7-S7P-s2` (s1 done 23:44) → s3 → ≈01:02, then the line and
    `2b-finalists.json` ≈01:20.
  - Mirror `5ae4cb9cc` (integration merged; M7 formal wrapper, HT-DEV v2 diagnostic and relays; 31 tests) on both
    nodes. Formal wrappers (`ops/m7/m7-formal.sh`) wait for the finalists files: node A GPU5 (2B; smoke → collection →
    report → mlx-diag), node B GPU3 (slots 1, 3) and GPU4 (slot 2). 4B runs are relayed and scored on node A by hand
    (`m6-relay.sh pull / mark`, `m6-score.sh`).
  - **HT-DEV v2 is a diagnostic in M7** (COORDINATION 04:10: the prereg predates it and no amendment adopted it
    before the first development readout, 21:20Z). Collected on the M7 path (same node / image / 16K / T = 1 as the
    line readouts) for `I`, each arm soup and each finalist; gold-free prompts installed in both nodes' decoder panel
    dir; scored on node A (`ops/m7/m7-htdev2.sh`).
  - GPU-h at 23:30Z ≈ 14.4 of 29.7 (arms 11.9, lines 0.96, M6b 0.28, running seeds ≈1.2).
- 2026-09-30 ≈04:30 UTC+8 (20:30Z) — **First seeds done; every arm fits its cap.**
  - `m7-N7H-s1` 1.351 and `m7-N7C-s1` 1.343 GPU-h (the preregistered 1.35 estimate); `m7-S7H-s1` ≈ 0.65 GPU-h.
    The earlier ~1.65 reading was taken while a co-tenant readout shared the GPU. All seeds 2 are past preflight.
  - Line watchers (`ops/m7/m7-watch.sh`, `ed568b34d`) run each arm's line readouts and diagnostics as its soup
    lands, then the tier's finalists: node B GPU4 (4B), node A GPU5 (2B).
  - Expected (UTC): S7H soup ≈21:20, N7H / N7C soups ≈23:10, S7C ≈23:30, S7P ≈01:40, N7P ≈01:55; finalists after
    the last line of each tier. Projected total ≈ 21 GPU-h.
- 2026-09-30 ≈03:25 UTC+8 (19:25Z) — **M7 training live on all three GPUs.**
  - Data lock `ea7540df4` (six files PASS; P arms on PN1-r2 per the revision rule; HS1 rows within the cleared
    revision). The new evaluation-only guard passes on every input and all 512,552 TRAIN rows (`404290ee7`).
  - node B: `m7-N7H-s1` (GPU3) and `m7-N7C-s1` (GPU4) preflight PASS 19:07Z; 989 planned updates each.
  - node A: `m7-S7H-s1` preflight PASS 19:19Z.
  - References: 4b-I reuses M6's node-B 16K readout (same weights); 2b-I read on node A (19:17–19:19Z) from the
    hash-checked S2T copy (`f2ea6ddf…`). hs1-dev / PN1-dev readouts of both references are running as co-tenants.
  - Integration: merged and pushed (`404290ee7`).
  - Watch: 4B seed cost. The first rate reading is ~1.65 GPU-h per seed with a co-tenant readout, above the
    preregistered 1.35; the chain's cap check would then block the third seed of N7H / N7C (a disclosed two-seed
    soup under the stop rules).
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
