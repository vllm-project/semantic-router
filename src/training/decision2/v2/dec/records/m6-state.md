# Decoder Milestone 6 — state (keep current; newest first)

Prereg: [`dec-m6-prereg-2026-09-29.md`](dec-m6-prereg-2026-09-29.md). Budget cap 36 GPU-h.
GPUs: node B GPU3–4 (after research & data's PN1 lend), node A GPU5.

## Now

- 2026-09-29 ≈18:00 UTC+8 — All three training chains are live.
  - The chains are b3 (N6D), b4 (Sol labels, then S6X → N6A → S6D) and a5 (Eos labels, then E6K). Their mirrors
    are `8f5699bdf` on node B and `6fe568271` on node A. Logs: `m6/logs/chain-<ch>.log`.
  - 16K reference readouts `refs` for 4b, 2b and 08b were launched at ≈09:56Z as co-tenants. PIDs are in
    `m6/logs/refs-<tier>.pid`.
  - Expected soups (UTC): S6X ≈11:15, E6K ≈13:40, N6A ≈14:15, N6D ≈15:20, S6D ≈17:15.
  - Next steps:
    - the no-training lines (4b N5BN / Nox, 2b Sol, 08b Eos);
    - the node-B S2T formal reference;
    - each arm's line as its soup lands.
- 2026-09-29 ≈16:55 UTC+8 — Prereg written (`68d31c358`). Tooling is `afb0ea9fe`; data lock parts 1, 2a and 2b
  are `8f5699bdf`, `54b18b6c3` and `4ee4f66b2`.

## Arms

Data lock part 1: [`dec-m6-datalock-2026-09-29.md`](dec-m6-datalock-2026-09-29.md). Chains `ops/m6/m6-chains.sh`.
Markers are under `/data/dev2/runs/dec/m6/status/` on the arm's node:

- `<ARM>-s<i>.DONE` / `.FAILED` / `.STOPPED` for each seed;
- `<ARM>.DONE` / `.FAILED` for each arm; `<ARM>.DONE` is written after the soup;
- the soup's own marker `soup/<ARM>/DONE` holds the path and the per-file SHA-256 list.

| Arm | Tier | Node / GPU | Train / teacher | Status | GPU-h |
| --- | --- | --- | --- | --- | ---: |
| N6D | 4B | B / 3 | `m6-xl-full-59m` `160812e2…` / `lux-all-59m` `a1bafad5…` | s1 preflight PASS 09:24Z, full run | 0.17 |
| N6A | 4B | B / 4 | `m4-xl-full-29m` `c7d51219…` / `lux-all-29m` `dc937c42…` | data locked; chain b4 (after S6X) | 0 |
| S6X | 2B | B / 4 | `m4-xl-full-29m` / own-Sol `sol-29m` `b8dec627…` | s1 preflight PASS 09:34Z, full run | 0.17 labels + s1 |
| S6D | 2B | B / 4 | `m6-xl-full-59m` / own-Sol `sol-59m` `53e4adc8…` | labels locked; chain b4 (after N6A) | 0.17 (labels) |
| E6K | 0.8B | A / 5 | `m6-e8f-r2clean` `f9f3c022…` / own-Eos `eos-e8f-r2clean` `f8202208…` | labels locked (part 2b); s1 next | 0.30 (labels) |

## Lines / finalists / formal

None yet.

## GPU-hours

Per job: `/data/dev2/runs/dec/m6/gpuh-node-{a,b}.json` (`ops/m6/m6-gpuh.py`, launch receipts). The own-Sol label
cost counts against both S6X and S6D for their caps, and once in the total.

| Item | GPU-h |
| --- | ---: |
| data builds, teacher composition, exposure (CPU) | 0 |
| own-Sol labels (node B GPU4) | 0.168 |
| N6D s1 through preflight (node B GPU3, as of 09:29Z) | 0.172 |
| own-Eos labels (node A GPU5; includes the failed 0.0044 GPU-h first attempt) | 0.298 |
| total (as of 09:44Z; training seeds accrue in the per-node files) | 0.64 |

## Incidents / deviations

1. **`m6-xl-full-59m` is 54.74M native tokens, not the prereg's "about 58.8M".** The recipe keeps the A0s rows
   (3.98M) whole and doubles only the pooled share, so the target is 54.84M. The A7 dose, 19.34M, is as stated.
2. **Shared-module fix `6fe568271`.** `v2.dec.teacher_label` now falls back to the single calibrated temperature in
   a 1.0 package's `config.json`: Eos 1.0 ships no `temperature.json`. The first a5 attempt failed after 16 s, before
   any training, and is kept under `m6/attempts/a5-1-eos-temperature/`. Own-Eos labels use the package temperature
   1.0389, and Sol still resolves to 1.30036.
3. **E6K's postrun calibration set is E8F's own CAL file** (`3e34f6cb`, node A default), not CAL698. SELECT700 is
   identical on both nodes. The postrun is used for reporting only.
4. **The lock check's C1 guard now matches on sources only**, as the 9B guard does. The family name
   `pilot_narrative_reading` (100 project-generated A0s rows, also in N4XF) matched the C1 key "narrative". It is
   reported, not failed.
5. **Launch-line slip (no effect).** The first `refs` launch backgrounded a `cd && …` compound, so its PID file
   missed and the 2b launch failed on its redirect before starting. The 4b job ran correctly under `setsid`. The 2b
   and 08b jobs were relaunched with scoped redirects, and liveness was checked by PID and container.
6. **Gist file 04 was briefly deleted (≈09:59–10:01Z).**
   - Cause: my local working copy had disappeared from `/tmp`, so the splice produced an empty file, and a `;`
     let `gh gist edit` run anyway. An empty update deletes the file.
   - Restored byte-for-byte from gist revision `25ac3c64` (45,264 B), with the M6 entry added (47,253 B). No other
     gist file was touched.
   - Rule from now on: gist edits run under `set -euo pipefail` with a size check, from
     `~/.cache/dec-m6/`, never `/tmp`.
