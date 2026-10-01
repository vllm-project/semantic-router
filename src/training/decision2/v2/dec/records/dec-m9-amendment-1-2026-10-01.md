# Decoder Milestone 9 — amendment 1 (the control arm C9 stopped by its seed-1 preflight; the control becomes M7's N7C)

Written 2026-10-01 ≈11:35 UTC+8 (03:35Z), **before any H9 or control readout**: the only development readouts taken so
far are the `4b-I` references (DEV2.0-4B's weights), which involve neither arm. Preregistration
[`dec-m9-prereg-2026-10-01.md`](dec-m9-prereg-2026-10-01.md) (`f22a5797c`); data lock `2f20c2223`.

## What happened

- Chains started 02:58Z on node A GPU6 (H9-s1) and GPU7 (C9-s1), both from the same Nox 1.0 start (identical
  zero-step identity `c5901e2b…`), on a node-A training autotune cache for image `dbe5f32b` that was empty and was
  filled by both processes at the same time.
- **m9-C9-s1 preflight FAIL (03:04Z)** on one gate, `zero_trainer_cross_process`: 698 / 700 SELECT700 argmax agree
  between the zero-step trainer and a separate reload process, maximum probability drift 0.0187 (tolerance: 700 / 700
  and ≤ 1e-4). Every other gate passed, including the one-step cross-process gate (700 / 700, drift 7.7e-8, after the
  cache had filled). m9-H9-s1 passed every gate (zero-step cross-process drift 8.1e-8).
- Reading: an autotune race on a cold shared cache (FLA picks configurations per process unless the cache holds
  them), not the C9 data or recipe. The rule is not reinterpreted: **C9 stops** — no rerun, no replacement seed
  (prereg stop rules; COORDINATION "never rerun failed arms"). C9-s2 / s3 are not started; the early rule E1 does not
  apply (a seed 1 did not finish), so H9 continues to three seeds (GPU6 s1, s3; GPU7 s2).

## Amendment (analysis only; no new training)

The matched-token control becomes **M7's N7C**, the nearest existing arm to C9:

| | C9 (preregistered) | **N7C (M7, used now)** |
| --- | --- | --- |
| Start / recipe | Nox 1.0, N4XF recipe (own-Lux KL 1.0, H7 / H8 gold only, LR 5e-6 / 5e-5, token batching, `even8` + SELECT700) | identical |
| Seeds | 20260926 / 27 / 28 | identical |
| Base | M7's 4B base (58,739 rows) | identical |
| Filler | 34,325 rows, 16.75M tokens of the XL r2 pool, seed `dec-m7-fill-4b-v1` | the first 28,986 of those rows (14.10M tokens; nested in C9's filler, checked in the data lock) |
| Total tokens | 46,159,962 (H9 46,159,507) | 43,507,920 (−5.7% vs H9) |
| Trained on | node A GPU7 (stopped) | node B GPU4 (M7), image `dbe5f32b` |
| Soup | — | `m7/soup/N7C/build/N7C-soup`, per-file list `9bcc0d10…` (BEST 145 / 584 / 1012) |

- The N7C soup is copied from node B to node A over the direct link and checked against its list `9bcc0d10…`. The
  control line is `4b-N7C-a1` (the soup) and `4b-N7C-a1_2` (`[I, N7C]`, which must reproduce M7's finalist
  `4b-N7C-b1_2`, list `d568829a…`). Both are read on the M9 node-A path exactly like H9 (all panels, 16K, T = 1).
- **Primary endpoint, restated:** ΔH_dev2(H9 soup − N7C soup) at α 1 (paired bootstrap 95% CI), reported as a
  **near-matched** contrast (the control has 2.65M fewer recipe-filler tokens). M6b / M7 / M8 found more recipe tokens
  neutral to negative at 4B, so the shortfall is not expected to favour H9.
- Everything else is unchanged: the H9 gates and pick, the formal step (H9 pick only, after the formal-path parity
  run), the recommendation rule with "C9" read as "N7C". If the H9 pick is the α ½ point, M7's formal run of
  `4b-N7C-b1_2` (node B path; v3 61.200, H .571) is the formal control at the same α, read with the parity result.
- **For the coordinator:** a fully matched C9 needs a new preregistered arm (≈ 4.4 GPU-h, ≈ 4.4 h on one GPU). This
  pilot does not rerun it.

## Tooling

`m9-lines.sh` and `m9_rules.py` take the control arm name (`N7C`); `m9-lines.sh control` places and verifies the N7C
soup on node A. Committed before use; the chains are unchanged.
