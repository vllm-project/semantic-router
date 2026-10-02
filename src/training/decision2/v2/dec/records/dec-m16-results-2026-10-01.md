# Decoder Milestone 16 — results (interpolation sweep between each release and its M12 / M13 / M14 arms; 2026-10-01)

Preregistration [`dec-m16-prereg-2026-10-01.md`](dec-m16-prereg-2026-10-01.md) (`9923e8b51`); data lock
[`dec-m16-datalock-2026-10-01.md`](dec-m16-datalock-2026-10-01.md) (`5eb61705f`); state
[`dec-m16-state.md`](dec-m16-state.md). Development readouts are never release scores. Nothing was uploaded; no Index
row was read for selection or tuning.

## Bottom line

- **No M16 model passes successor items 1–7, so M16 has no release candidate.** There is no item-8 spec, no release
  hand-off and no Index request. The releases stay: DEV2.0-0.8B (`4afea305`), DEV2.0-2B (`2973ad4a`) and the released
  LH (`13d42143`).
- **Interpolation clears the development gates that the full arms failed.** 15 of 24 points pass gates 1–7, including the
  MLX-DEV guard. Every 0.8B point except `08b-RAUP-a75` passes. At 2B, five points pass: every a25, plus `2b-RA` a50 and
  a75. At 4B, only `4b-LHA10SD` a50 and a75 pass. Moving towards the release keeps most of the transfer gain. For
  example, the 0.8B a75 points read +.19 to +.20 (M14's full `08b-RAUP` arm: +.197), and 4B `LHA10SD-a75` reads +.045
  (M14's full `4b-LHA10UP`: +.045). It also brings the
  typed floors back.
- **The development gains do not reach the formal v3 bar.** All six finalists tie their bar on item 1 (no CI above 0),
  and all six fail item 6(b), the reduced-panel version of the same comparison:
  - 0.8B `RA-a75`: +3.68 [−2.76, +4.82].
  - 0.8B `RASD-a75`: +1.45 [−4.27, +4.63].
  - 2B `RASD-a25`: +.05 [−.60, +1.67].
  - 2B `RA-a75`: −2.05 [−3.31, +3.14].
  - 4B `LHA10SD-a50`: +.02 [−1.81, +2.65].
  - 4B `LHA10SD-a75`: −.07 [−2.53, +2.38].
- **mlx-diag (item 4) depends on the tier.** Both 0.8B finalists and 2B `RASD-a25` pass. 2B `RA-a75` (−.018) and both 4B
  points (−.015, −.019) fail, with card-eligible CIs below 0 against both bars. The development MLX-DEV guard passed
  all three of those points, so at 2B / 4B the guard does not predict mlx-diag.
- **Every formal path was exact.** Each tier's parity run, the release weights collected on node B's formal path, gave
  0 / 0 / 0 differing answers (typed FINAL / CSS15 / public 231) against a stored run of the same weights:
  - 2B: against bar-t1.
  - 4B: against the stored LH run (v3 67.345 reproduced).
  - 0.8B: against node B's stored M8s reference. The parity run differs from bar-t1 by 9 / 20 / 1, so both 0.8B bars
    bind, as preregistered.

## Lineage and builds

- Each point is \(W(\alpha) = (1 - \alpha)\,\text{release} + \alpha\,\text{arm}\), computed per tensor in FP32 by tensor
  name over full merged weights. The output keeps the arm's layout, tokenizer and config, and
  `decision_config.interpolation` records α, the formula and both paths.
- All 8 release / arm pairs pass lineage (data lock): same architecture, prompt version, head, tokenizer files, and
  backbone and head tensor names and shapes. No pair was skipped. 24 points were built (9 / 9 / 6 per tier; cap 9).
- The arm endpoints (α = 1) are not re-read. Their gates are M12's / M13's / M14's. `4b-LHA10UP-a100` is node B's
  report-only copy of M14's readout.

## Development gates over α (rules ran once per tier on complete inputs)

Gates: M14's (M8 eligibility type / family / Noul floors, Score5-typed-DEV floor, HT-DEV v2 not FLAG, retention macro
CI upper ≥ 0, `hs1-dev` false-yes ≤ reference + .10, 0.8B HT-DEV v2 Δ ≥ 0, transfer macro Δ ≥ 0), plus the MLX-DEV
guard: Noul-ML and Choice-ML 95% upper bounds ≥ 0. The MLX-DEV panel is the decoder MLX-DEV set minus groups that
overlap the tier's arm TRAIN. Δ is point − reference, where the reference is the release read on the same node.

Selection: largest transfer macro Δ (rounded to 4 decimals), then HT-DEV v2 GAIN, then the smaller α. Finalist 2 comes
from a different arm line when one is eligible.

### 0.8B (reference DEV2.0-0.8B, `08b-C0-a`: typed T .613, C / N / S 610 / 212 / 159; rules 18:31Z)

| Point | Typed T | C / N / S | HT-DEV v2 Δ | Retention Δ | Transfer Δ (9 families) | IB DEV Δ | `hs1-dev` false-yes (ref .720) | MLX-DEV Noul-ML / Choice-ML Δ | Failed gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `08b-RAUP-a25` | .644 | 619 / 221 / 191 | +.000 [−.011, +.012] TIE | +.006 [−.006, +.019] | +.108 [+.074, +.141] | +.109 | .747 | +.005 [−.000, +.009] / +.012 [+.000, +.025] | — |
| `08b-RAUP-a50` | .640 | 597 / 221 / 206 | +.018 [+.002, +.034] TIE | +.018 [+.001, +.035] | +.184 [+.149, +.221] | +.189 | .777 | +.003 [−.004, +.009] / +.006 [−.012, +.024] | — |
| `08b-RAUP-a75` | .637 | 572 / 226 / 222 | +.023 [+.004, +.042] GAIN | +.017 [−.003, +.036] | +.192 [+.156, +.228] | +.204 | .798 | −.005 [−.013, +.003] / +.000 [−.022, +.021] | Choice type floor 572 < 586 |
| `08b-RASD-a25` | .646 | 633 / 216 / 184 | +.014 [+.002, +.026] TIE | +.005 [−.007, +.017] | +.102 [+.069, +.135] | +.105 | .726 | +.003 [−.002, +.008] / +.010 [−.004, +.025] | — |
| `08b-RASD-a50` | .650 | 627 / 225 / 188 | +.023 [+.006, +.038] GAIN | +.024 [+.006, +.044] | +.183 [+.148, +.219] | +.189 | .744 | −.001 [−.008, +.006] / +.012 [−.011, +.037] | — |
| **`08b-RASD-a75`** | .636 | 603 / 218 / 197 | +.029 [+.009, +.048] GAIN | +.027 [+.005, +.049] | **+.196** [+.160, +.232] | +.208 | .780 | −.003 [−.011, +.005] / +.004 [−.024, +.032] | — (finalist 1) |
| `08b-RA-a25` | .624 | 598 / 216 / 185 | +.007 [−.004, +.019] TIE | +.011 [−.001, +.025] | +.114 [+.080, +.148] | +.117 | .726 | +.006 [+.001, +.011] / +.007 [−.008, +.023] | — |
| `08b-RA-a50` | .620 | 594 / 219 / 179 | +.026 [+.009, +.041] GAIN | +.029 [+.009, +.051] | +.181 [+.146, +.216] | +.190 | .747 | +.012 [+.005, +.019] / +.006 [−.014, +.026] | — |
| **`08b-RA-a75`** | .627 | 606 / 219 / 179 | +.035 [+.016, +.053] GAIN | +.044 [+.017, +.074] | **+.191** [+.154, +.227] | +.205 | .762 | +.009 [+.000, +.017] / −.002 [−.026, +.021] | — (finalist 2, other line) |

### 2B (reference DEV2.0-2B, `2b-C0-b`: typed T .610, Score 257, floor 245; rules 19:11Z)

| Point | Typed T | C / N / S | HT-DEV v2 Δ | Retention Δ | Transfer Δ (9 families) | IB DEV Δ | `hs1-dev` false-yes (ref .625) | MLX-DEV Noul-ML / Choice-ML Δ | Failed gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2b-RAUP-a25` | .623 | 510 / 235 / 251 | +.007 [−.001, +.015] TIE | −.001 [−.016, +.013] | +.099 [+.074, +.130] | +.089 | .613 | −.000 [−.004, +.004] / +.005 [−.002, +.013] | — |
| `2b-RAUP-a50` | .630 | 529 / 241 / 238 | +.005 [−.007, +.017] TIE | +.005 [−.017, +.026] | +.139 [+.111, +.171] | +.147 | .577 | +.002 [−.004, +.008] / +.010 [−.002, +.022] | Score type floor 238 < 245 |
| `2b-RAUP-a75` | .628 | 532 / 239 / 233 | +.007 [−.008, +.021] TIE | −.001 [−.025, +.024] | +.149 [+.121, +.181] | +.166 | .566 | +.003 [−.004, +.010] / +.008 [−.004, +.021] | Score type floor 233 < 245 |
| **`2b-RASD-a25`** | .616 | 498 / 238 / 249 | +.009 [+.001, +.016] TIE | +.013 [−.002, +.029] | +.099 [+.073, +.130] | +.092 | .604 | +.003 [−.001, +.008] / +.004 [−.002, +.011] | — (finalist 2, other line) |
| `2b-RASD-a50` | .616 | 504 / 237 / 244 | +.011 [−.002, +.024] TIE | +.013 [−.009, +.035] | +.143 [+.115, +.175] | +.150 | .556 | +.003 [−.003, +.009] / +.011 [−.001, +.023] | Score type floor 244 < 245 |
| `2b-RASD-a75` | .597 | 494 / 229 / 232 | +.020 [+.004, +.036] TIE | +.009 [−.015, +.034] | +.150 [+.122, +.182] | +.171 | .527 | +.005 [−.002, +.012] / +.013 [−.001, +.028] | Score type floor 232 < 245 |
| `2b-RA-a25` | .641 | 526 / 236 / 263 | +.002 [−.006, +.009] TIE | −.012 [−.029, +.003] | +.097 [+.072, +.128] | +.093 | .634 | +.002 [−.002, +.006] / +.007 [+.001, +.014] | — |
| `2b-RA-a50` | .659 | 556 / 242 / 257 | +.003 [−.009, +.015] TIE | −.004 [−.027, +.019] | +.141 [+.113, +.173] | +.153 | .580 | +.000 [−.006, +.007] / +.011 [+.001, +.022] | — |
| **`2b-RA-a75`** | .662 | 571 / 243 / 245 | −.002 [−.016, +.013] TIE | −.004 [−.029, +.022] | **+.147** [+.119, +.179] | +.166 | .556 | +.001 [−.007, +.008] / +.011 [+.000, +.022] | — (finalist 1) |

### 4B (reference released LH, `4b-LH-b`: typed T .868, C / N / S 728 / 290 / 371, Score floor 359; rules 19:40Z)

| Point | Typed T | C / N / S | HT-DEV v2 Δ | Retention Δ | Transfer Δ (9 families) | IB DEV Δ | `hs1-dev` false-yes (ref .199) | MLX-DEV Noul-ML / Choice-ML Δ | Failed gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `4b-LHA10UP-a25` | .911 | 775 / 306 / 376 | +.001 [−.010, +.012] TIE | −.015 [−.028, −.004] | +.035 [+.025, +.046] | +.047 | .164 | +.002 [−.002, +.006] / +.004 [−.005, +.012] | retention (CI upper < 0) |
| `4b-LHA10UP-a50` | .915 | 787 / 311 / 366 | +.015 [+.002, +.027] TIE | −.033 [−.054, −.011] | +.043 [+.033, +.055] | +.065 | .164 | +.001 [−.004, +.006] / +.005 [−.007, +.018] | retention (CI upper < 0) |
| `4b-LHA10UP-a75` | .926 | 794 / 320 / 367 | +.020 [+.006, +.034] GAIN | −.034 [−.059, −.008] | +.044 [+.032, +.056] | +.076 | .176 | +.001 [−.005, +.006] / +.001 [−.013, +.015] | retention (CI upper < 0) |
| `4b-LHA10SD-a25` | .902 | 775 / 289 / 380 | −.001 [−.011, +.010] TIE | −.016 [−.031, −.002] | +.033 [+.024, +.044] | +.042 | .170 | +.003 [−.001, +.006] / +.004 [−.006, +.014] | retention (CI upper < 0) |
| **`4b-LHA10SD-a50`** | .893 | 775 / 294 / 360 | +.003 [−.009, +.014] TIE | −.011 [−.029, +.009] | +.042 [+.031, +.054] | +.070 | .170 | +.005 [−.000, +.010] / +.002 [−.010, +.015] | — (finalist 2) |
| **`4b-LHA10SD-a75`** | .899 | 766 / 307 / 366 | −.008 [−.021, +.005] TIE | −.021 [−.047, +.002] | **+.045** [+.033, +.057] | +.079 | .179 | +.006 [−.000, +.012] / −.001 [−.015, +.012] | — (finalist 1) |

- No `4b-LHA10UP` point is eligible, so both 4B finalists come from `4b-LHA10SD`, as the rule allows.
- Retention moves monotonically with α in the `4b-LHA10UP` line. In the `4b-LHA10SD` line, the a25 point fails and
  the a50 / a75 points pass narrowly. At 2B and 0.8B, retention passes everywhere.

## Formal (node B collections, node-A scoring; two-bar successor rule)

The first bar is each tier's release run: bar-t1 (DEV2.0) or bar-lh (stored LH). bar-b is node B's M16 parity run of
the same weights. Items 1, 2, 4, 6(b) and 7 must pass against both bars.

| Finalist | v3 | 1: v3 Δ vs bar 1 [95%] | 2: H | 3: types | 4: mlx-diag card-eligible Δ | 5 | 6(a) | 6(b) | 7: public 231 | Items 1–7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `08b-RASD-a75` | 51.685 | +1.449 [−4.267, +4.628] ✗ | ✓ | ✓ | −.0015 [−.0119, +.0090] ✓ | ✓ | ✓ | ✗ | 159 vs 156 ✓ | **FAIL** |
| `08b-RA-a75` | 53.913 | +3.678 [−2.764, +4.824] ✗ | ✓ | ✓ | −.0037 [−.0139, +.0065] ✓ | ✓ | ✓ | ✗ | 162 vs 156 ✓ | **FAIL** |
| `2b-RA-a75` | 51.386 | −2.051 [−3.311, +3.137] ✗ | ✓ | ✓ | −.0184 [−.0295, −.0075] ✗ | ✓ | ✓ | ✗ | 170 vs 171 ✓ | **FAIL** |
| `2b-RASD-a25` | 53.488 | +.051 [−.599, +1.675] ✗ | ✓ | ✓ | −.0017 [−.0076, +.0040] ✓ | ✓ | ✓ | ✗ | 171 vs 171 ✓ | **FAIL** |
| `4b-LHA10SD-a75` | 67.276 | −.069 [−2.526, +2.379] ✗ | ✓ | ✓ | −.0186 [−.0284, −.0094] ✗ | ✓ | ✓ | ✗ | 174 vs 172 ✓ | **FAIL** |
| `4b-LHA10SD-a50` | 67.368 | +.023 [−1.809, +2.651] ✗ | ✓ | ✓ | −.0153 [−.0230, −.0080] ✗ | ✓ | ✓ | ✗ | 174 vs 172 ✓ | **FAIL** |

- The table shows bar 1. bar-b agrees on every item verdict:
  - 0.8B: `RASD-a75` +1.422 and `RA-a75` +3.650 on item 1.
  - 2B and 4B: bar-b's answers equal bar 1's (the parity runs are exact), so the numbers match.
- Parity v3: 0.8B `m16-08b-C0` 50.263 (bar-t1 50.236); 2B `m16-2b-C0` 53.437; 4B `m16-4b-LH` 67.345.
- Type verdicts are OK / OK / OK for every finalist. Exposure receipts (item 6(a)) use the arm's M12 TRAIN file. The
  overlap-effects runs reproduce every stored pair with 0 problems.
- Item 8 is not reached. Under the prereg, the C1 recheck r1 citation, item-8 spec, `Decision-2.0-{Eos,Sol,Nox}-{size}`
  hand-off and Index request apply only to a model that passes items 1–7, and none does.

## Operations

- GPU-h: **5.95** of the 30 cap. No item started above 13.5 node-GPU-h.
  - Node A, 1.55 (own-node launch receipts): 0.8B readouts, .17 per point, plus the reference MLX-DEV read.
  - Node B readouts, 3.22: 2B .17–.19 per point, 4B .25–.26 per point, plus reference MLX-DEV reads.
  - Node B formal, 1.19 (`GPU-SECONDS.jsonl`, shared co-tenant jobs): 0.8B .36, 2B .35 and 4B .48, including smokes.
  - Interpolation builds, lineage checks, MLX panels and compares were CPU only.
- GPUs used: node A GPU3–5 and node B GPU2–4 only. Formal used node B GPU3 / GPU4, the library's node-B map. Node A
  GPU0–1 were not touched. M15 (node E / F) was not touched; node E served only as the M16 relay transit directory. All
  M16 leases are released (node A 18:30Z, node B 20:27Z).
- No point, chain or collection failed. Preflights, smokes and collections all passed, and no collection was rerun.
- Incidents (no effect on results):
  - The workstation shell started one relay call and one scoring call twice each. Each second copy stopped at its guard
    (the relay's "exists on the target" check, or the `m16-score.sh` lock) and changed nothing. The node-A 4B line
    directories matched node B's filtered content exactly.
  - Scoring fix `ab7be2f15`: m6-score's report-only 2B mlx-diag pair looked for a node-B reference run under M16's root.
    It now uses node A's `m3-S2T-soup-mlx` (bar-t1's run). The first attempt stopped before writing any pair file.
  - Transient SSH connection timeouts made three relay calls fail at a pre-check; one message said "contains weights".
    No files were written, and each call was retried.

## What this suggests next (not preregistered)

- Interpolating towards the release repairs the development profile: typed floors, retention and MLX-DEV all recover.
  It does not create a formal v3 gain. The best formal readings are 0.8B `RA-a75` (+3.7) and `RASD-a75` (+1.4), whose
  CIs are wide and include 0. At 2B and 4B, interpolated points read within ±.1 of the release, except 2B `RA-a75`
  (−2.1). The IB-derived transfer gain on development panels (+.05 to +.20) does not move the formal panel.
- The 0.8B points are the only ones that pass item 4 and every non-v3 item. A second 0.8B collection could tighten item
  1's CI, but that would need its own preregistration: re-collecting a finalist is not in this prereg.
- At 2B and 4B, mlx-diag card-eligible falls by .015–.019 for the α ≥ .5 finalists, even though the development MLX-DEV guard passes. A
  multilingual guard that tracks mlx-diag needs a different development panel; MLX-DEV minus TRAIN overlap does not do
  it.
