# Decoder Milestone 6 — state (keep current; newest first)

Prereg: [`dec-m6-prereg-2026-09-29.md`](dec-m6-prereg-2026-09-29.md). Budget cap 36 GPU-h.
GPUs: node B GPU3–4 (after research & data's PN1 lend), node A GPU5.

## Now

- 2026-09-29 ≈16:55 UTC+8 — Prereg written. Nothing has launched and 0 GPU-h are used.
  - PN1 generation (research & data) has been running on node B GPU3–4 since 16:22; its lease entries are
    `owner.data`.
  - Next: M6 ops code, then CPU data builds and the data lock, then launches once node B frees.

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
| S6X | 2B | B / 4 | `m4-xl-full-29m` / own-Sol `sol-29m` `b8dec627…` | labels locked (part 2a); s1 next | 0.17 (labels) |
| S6D | 2B | B / 4 | `m6-xl-full-59m` / own-Sol `sol-59m` `53e4adc8…` | labels locked; chain b4 (after N6A) | 0.17 (labels) |
| E6K | 0.8B | A / 5 | `m6-e8f-r2clean` `f9f3c022…` / own-Eos (part 2b) | own-Eos labels running in chain a5 | 0 |

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
| total (as of 09:29Z) | 0.34 |

## Incidents / deviations

None.
