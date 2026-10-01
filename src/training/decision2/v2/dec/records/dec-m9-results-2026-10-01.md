# Decoder Milestone 9 — results (HR2 efficacy pilot at 4B; PILOT, NOT RELEASABLE; 2026-10-01)

Preregistration [`dec-m9-prereg-2026-10-01.md`](dec-m9-prereg-2026-10-01.md) (`f22a5797c`, before any GPU job); data
lock [`dec-m9-datalock-2026-10-01.md`](dec-m9-datalock-2026-10-01.md) (`2f20c2223`, PASS); amendments
[1](dec-m9-amendment-1-2026-10-01.md) (`e7f8e9858`: the control becomes M7's N7C) and
[2](dec-m9-amendment-2-2026-10-01.md) (`29057a4de`: H9 arm cap), both before any H9 readout. Development readouts are
never release scores; the formal run is a post-key same-panel comparison labelled **"M9 pilot, not a release
candidate"**. HR2 is `release_safe: false`, so nothing here is handed off, uploaded or sent to the C1 custodian.
Aggregates: [`dec-m9-results-2026-10-01/`](dec-m9-results-2026-10-01/).

## Bottom line

TBD

## What ran

- **H9** (N4XF base 29.40M tokens + the whole HR2 TRAIN block 16.75M tokens, gold only on HR2 rows): three full
  seeds from Nox 1.0 (20260926 / 27 / 28) on node A GPU6–7, image `dbe5f32b`, BEST checkpoints 846 / 847 / TBD
  (SELECT700 family macro .8971 / .8965 / TBD), uniform FP32 soup.
- **Control:** the preregistered C9 (same base + 16.75M recipe-filler tokens) stopped at its seed-1 preflight (one
  gate: zero-step cross-process 698 / 700, drift 0.019; a cold autotune cache filled by both chains at once; no rerun).
  By amendment 1 the control is **M7's N7C** — same base, recipe, seeds and image; filler = the first 14.10M tokens of
  C9's filler; 43.5M tokens (−5.7% vs H9); trained on node B in M7. The early rule E1 did not apply.
- **Amendment 2** raised H9's cap to 5.8 GPU-h from C9's unused budget; the GPU6 chain was replaced in flight and the
  new chain adopted the running H9-s1. In the event the seeds cost ≈ 1.5 GPU-h each, so H9-s3 would also have started
  under the original 4.8 cap: the amendment changed the cap, not the outcome.
- **Paths verified exactly:** DEV2.0-4B's weights read on the M9 node-A readout path reproduce the stored node-B
  readouts (0 answer differences, drift 0.0 on typed DEV, CSS pilot, HT-DEV v2, `hs1-dev`, Score5-typed-DEV; HT-DEV v2
  equals the eval reference), and on the node-A formal path they reproduce the bar run `dev2-4b-t1-derived` (0
  category changes on typed FINAL, CSS15 and public 231; v3 63.151). Every M9 number is therefore comparable with the
  M6–M8 history and with the bar.

## Development (16K, T = 1, node A; paired against `4b-I` = DEV2.0-4B's weights)

TBD table

## HR2 DEV slice (diagnostic; in distribution for H9)

TBD

## hs1-dev (diagnostic)

TBD

## Formal (pilot, not a release candidate)

TBD

## Recommendation (preregistered rule)

TBD

## GPU-hours

TBD

## Tooling and incidents

- New `v2/dec/ops/m9/` (compose, lock, node-A chains with the early rule and an adopt mode, soup, lines with the
  readout-path parity check, HR2 DEV prompts and scoring, rules, formal wrapper, post-soup chain, results summary);
  11 tests in `v2/dec/tests/test_m9.py` (139 decoder + guard tests pass in the image). Small decoder-only changes:
  `drive_arm.sh` (node-A GPU6–7, image override), `launch.sh` (opt-in `HIP_FORCE_DEV_KERNARG=1`), `ops/m6` formal
  library (`M6_4B_NODE=A`). No shared module changed.
- The 4B recipe inputs, the node-B readout autotune cache, the 4B frozen formal caches and M7's N7C soup were moved
  node B → node A over the direct link, each checked by content manifest.
- The first copy of `m9_gpuh.py` also counted the copied M7 parity receipts (+0.226 GPU-h, conservative for the stop
  rules); fixed in `eccf0a232` before the totals below.
- Time stamps: the prereg's "written ≈11:10" and amendment 1's "≈11:35" are about 30 min late; the commit times
  (10:37 and 11:10:50 UTC+8) are authoritative and precede every GPU job and every arm readout respectively.
