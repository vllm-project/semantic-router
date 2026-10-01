# 9B Milestone 9, amendment 4: stage 4 = the ½ point K-a12IB of the existing KIB soup (no training)

Written 2026-10-02 ≈00:50 UTC+8 (16:50Z), before any stage-4 job (CPU build or GPU readout). Assigned by the
coordinator after the stage-3 close-out (COORDINATION 2026-10-02 00:35, follow-up B: "a ½-interpolation point with
gates → formal → items 1–7"). It adds stage 4; nothing in the [preregistration](lux9b-m9-prereg-2026-10-01.md)
(`8570a5896`) or amendments [1](lux9b-m9-prereg-amendment-1-2026-10-01.md) (`0b84e0db4`),
[2](lux9b-m9-prereg-amendment-2-2026-10-01.md) (`51c80ddc9`) and [3](lux9b-m9-prereg-amendment-3-2026-10-01.md)
(`787abdc54`) changes. Stage 3 result: [record](lux9b-m9-stage3-result-2026-10-01.md) (`9eb632e03`).

## Why

- K-a13IB (⅓ toward the three-seed KIB soup) passed all seven development gates with *less* near-miss yes-bias than
  C0 (Y1 −.015 [−.028, −.003]; Y3 .134 vs .152) and MLX-DEV-9B level, but tied DEV2.0-9B formally (v3 +0.29
  [−1.57, +1.20]; item 1 fails).
- The largest 9B formal gain so far came at α ½ (K5-a12, +1.62 [−0.19, +2.41]), which failed item 4 through a yes-bias
  that grows with distance from Lux. IB is the first data block that lowered that bias on the K recipe, so the ½ point
  of the KIB soup may keep the yes-bias inside the guard while moving typed accuracy up.
- This is the stage-3 record's proposed lever 1. It needs no training and reuses frozen artifacts.

## Point (one; no α line)

| Point | Construction |
| --- | --- |
| **K-a12IB** | uniform FP32 soup of [KIB soup, Lux 1.0], i.e. α = ½ toward the KIB soup |

- **KIB soup:** the stage-3 three-seed soup on node A, `soup/KIB/build/KIB-soup`, content manifest `c9f945ea…`
  (re-hashed before the build).
- **Lux 1.0 member:** the zero-step checkpoint of m9-KIB-s1 on node A (content manifest `c105bc4f…`; weights
  byte-identical to K-a13's base `m3/pf-D-s1-zero`), the member K-a13IB used.
- `v2.dec.soup` on node A CPU (`m9/launch.sh --cpu`), recorded in `soup/K-a12IB/DONE` and `K-a12IB.built.json`. A
  failed build is not repeated.
- **Data and item 6:** no new data. K-a12IB's TRAIN is the KIB seeds' TRAIN, so item 6 uses stage 3's receipts
  (`exposure/kib-subset.json` `1dfb1843…`: 151,015 of 151,015 rows in x60 ∪ IB1-r3 ∪ IB2; the IB exposure receipt
  lists 0 groups).

## Readouts, gates and finalist

- **Path:** `m9/post-a4.sh` on node A, one GPU of node A GPU6–7 (lease-checked), the stage-1 path:
  - the same eight panels (`dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m9-probes mlxdev`) plus the IB1 / IB2
    DEV diagnostics;
  - `score.sh points C0 K-a12IB`.
- **The same seven development gates vs C0** (`m9_rules.py` unchanged):
  - typed / family floors;
  - the Noul `rule_precedence` floor (≥ C0 − 4 of 400);
  - the Score5-typed-DEV floor;
  - HT-DEV v2 not FLAG;
  - **the yes-bias guard**: Y1 PN1-dev clean gold-no Δ ≤ +.02 with CI lower ≤ 0; Y2 hop ≥ −.03; Y3 `hs1-dev`
    false-yes ≤ +.05;
  - MLX-DEV-9B upper bounds ≥ 0;
  - retention upper bound ≥ 0.
- **Outputs:** typed readout `readout/m9-s4.json` (C0, K-a12IB, K-a13IB); rules `select/9b-finalists-s4.json` (the
  only point is K-a12IB). The gates run **once**; no rerun or re-read of a failed screen.
- **Report only:** IB1 / IB2 DEV per family; the contrast K-a12IB − K-a13IB (HT-DEV v2, probes, PN1, typed).
- **Readings (preregistered):**
  - **H7:** K-a12IB passes the seven gates (the yes-bias and MLX guards are the binding risk).
  - **H8:** K-a12IB's typed DEV T exceeds K-a13IB's (.9344).

## Formal, successor, C1 and release (only if K-a12IB is a finalist)

- **Formal:** the stage-1 path, after the newest COORDINATION notes are re-read, a lock record is pushed and
  `status/formal-s4.GO` is written (`formal-chain.sh` with `M9_STAGE=4`, node A GPU6).
- **Items 1–7** vs DEV2.0-9B (`lux9b/m9/items.py verdict`):
  - item 1: the paired v3 CI against the T = 1 bar `release/dev2-8b-t1-derived` (67.737);
  - items 2–7 as the preregistration.
  - K-a12IB is M9's second formal candidate (the cap is four).
- **Item 8** only if items 1–7 pass:
  - the C1 post-key event, one attempt;
  - the custodian's C1 content recheck r1 cleared IB1-r3 + IB2 (`v2/eval/records/c1-recheck-r1-2026-10-01.md`,
    registry `v2/eval/sealed/c1-recheck-registry.json`), so item 8 may proceed.
- **Release hand-off** only if items 1–8 pass:
  - built on the current 9B `main` (`5de3f9ed` or its card-only successor);
  - under the renamed repository ID `Decision-2.0-Lux-9B` (COORDINATION 2026-10-02 00:05 / 00:25).

## Budget

Readouts ≈ 1.1 GPU-h, formal ≈ 0.25, C1 ≈ 0.1: **cap 4 GPU-h** for stage 4, inside M9's 120 (≈ 41.5 used).
