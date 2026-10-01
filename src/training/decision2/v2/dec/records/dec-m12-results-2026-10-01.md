# Decoder Milestone 12 — results (additive breadth on each tier's released recipe; 2026-10-01)

Preregistration [`dec-m12-prereg-2026-10-01.md`](dec-m12-prereg-2026-10-01.md) (`1f7fc94be`); data lock
[`dec-m12-datalock-2026-10-01.md`](dec-m12-datalock-2026-10-01.md) (`d506595c2`); state [`m12-state.md`](m12-state.md).
No amendments. Development readouts are never release scores. Nothing was uploaded; C1 was not opened; no Index row was
read for selection or tuning.

## Bottom line

- **No M12 point passes its development gates, so M12 has no finalist.** No formal collection ran, successor items
  1–7 were not reached, the optional transfer-only 4B arm was not trained (its trigger is a passing 4B arm), and there
  are no release hand-offs, C1 item-8 spec or Index requests. The released LH (4B), DEV2.0-2B and DEV2.0-0.8B stay
  the releases at their tiers.
- **Additive breadth works for breadth at every tier, with no HT-DEV v2 cost.** Transfer macro (9 unseen IB DEV
  families) Δ: 4B +.051 / +.045, 2B **+.151**, 0.8B **+.190**, every CI well above 0. HT-DEV v2 ties at 4B and 2B and
  is a **GAIN at 0.8B (+.049)**.
- **Every point still loses one typed or family floor vs its release.** Keeping every released TRAIN row is not enough
  to hold the typed heads: 4B `LHA` loses choice / Noul and retention, 4B `LHA10` loses Score, 2B `RA` loses Score, and
  0.8B `RA` loses one choice family (`attribute_gate`). Each failure sits in a single head, and which head fails varies
  between the two 4B arms.
- **0.8B is the near-miss.** `08b-RA` beats DEV2.0-0.8B on HT-DEV v2 (+.049), retention (+.044, CI above 0), Score
  typed (202 vs 159) and the Score5-typed-DEV check (C0 COLLAPSE, `RA` no collapse), and holds the choice and Noul type
  floors. It fails only the `attribute_gate` family floor (277 / 400 vs 320 / 400 − .10).

## Results (two seeds each, BEST checkpoints souped; references read on the same node)

Gates: the M11 gate (M8 eligibility type / family / Noul floors, Score5-typed-DEV floor, HT-DEV v2 not FLAG, retention
macro CI upper ≥ 0, `hs1-dev` false-yes ≤ reference + .10, 0.8B HT-DEV v2 Δ ≥ 0) plus breadth (transfer macro Δ ≥ 0).
Retention probes use each tier's M12 gold (M10's 3,089 items minus the overlap with the released TRAIN plus the whole
IB pool: 4B / 0.8B 2,203, 2B 2,189; GSM8K is thin, 100–114 items). These numbers are not comparable with M11's
stage-2 4B gold (2,461).

| Point (vs reference) | Typed T (ref) | HT-DEV v2 Δ | Retention macro (Δ) | Transfer macro, 9 families (Δ) | IB DEV macro, 12 families (Δ) | `hs1-dev` false-yes (ref) | Failed gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `4b-LHA` (LH) | .846 (.868) | −.018 [−.034, −.002] TIE | .753 (−.033 [−.063, −.002]) | .973 vs .922 (+.051 [+.038, +.063]) | .951 vs .859 (+.092 [+.078, +.106]) | .226 (.199) | choice type floor 691 < 728 − 24; Noul type floor 274 < 290 − 12; Noul `rule_precedence` floor; retention (CI upper < 0) |
| `4b-LHA10` (LH) | .873 (.868) | +.003 [−.010, +.017] TIE | .763 (−.023 [−.050, +.004]) | .968 vs .922 (+.045 [+.034, +.058]) | .942 vs .859 (+.083 [+.069, +.096]) | .146 (.199) | Score type floor 329 < 371 − 12; `set_reconciliation` family floor 329 / 400 < 371 / 400 − .10 |
| `2b-RA` (DEV2.0-2B) | .641 (.610) | −.004 [−.021, +.012] TIE | .559 (−.010 [−.038, +.017]) | .955 vs .803 (**+.151 [+.123, +.183]**) | .925 vs .750 (+.174 [+.150, +.200]) | .524 (.625) | Score type floor 227 < 257 − 12 |
| `08b-RA` (DEV2.0-0.8B) | .636 (.613) | **+.049 [+.027, +.070] GAIN** | .478 (**+.044 [+.011, +.076]**) | .941 vs .751 (**+.190 [+.152, +.227]**) | .904 vs .697 (+.206 [+.176, +.237]) | .783 (.720) | `attribute_gate` family floor 277 / 400 < 320 / 400 − .10 |

- Typed counts (choice / Noul / Score): 4B LH 728 / 290 / 371; `4b-LHA` 691 / 274 / 388; `4b-LHA10` 743 / 325 / 329.
  2B C0 – / 230 / 257, `2b-RA` 558 / 241 / 227. 0.8B C0 610 / 212 / 159, `08b-RA` 600 / 216 / 202.
- Score5-typed-DEV: no point flags COLLAPSE. The 0.8B reference does (COLLAPSE, NO-GAIN), so the 0.8B floor is the M8s
  amendment-3 rule; `08b-RA` shows only NO-GAIN. The 2B Score head no longer collapses (M11's `2b-LH` did), but it
  still drops 30 typed Score items.
- Retention per probe: 4B `LHA` MMLU .692 / ARC .953 / GSM8K .614, `LHA10` .715 / .950 / .623; 2B `RA` .522 / .864 /
  .290; 0.8B `RA` .412 / .725 / .298.

## Contrasts (report only)

- **Dose, `4b-LHA` − `4b-LHA10`:** HT-DEV v2 −.021 [−.036, −.006] FLAG; typed T −.027 (choice −52, Noul −51, Score
  +59); retention −.009 [−.042, +.024] (MMLU −.022 [−.037, −.006]); transfer +.005. Going from 10% to 25% IB
  buys almost no transfer and costs HT-DEV v2. The typed losses move between heads (LHA: choice / Noul; LHA10: Score)
  rather than shrinking with dose.
- **Additive vs displacing at 25%, `4b-LHA` − M11 `4b-LHB` (same IB rows; M11 readouts copied from node F as
  `4b-LHB-m11`):** HT-DEV v2 −.018 [−.033, −.004] TIE; retention −.024 [−.050, +.002]; typed T +.021 (choice −74,
  Noul +29, Score +79); IB DEV and transfer equal (LHB .952 / .975). Adding the IB rows on top of LH's full mixture
  recovers most of M11's Noul / Score loss, but it moves the loss to choice and costs retention. It does not keep LH's
  typed profile. `4b-LHB` on the M12 gold: HT-DEV v2 vs LH +.000 [−.016, +.017], retention −.008 [−.033, +.017].

## Operations

- GPU-h (launch receipts; finished jobs, no M12 job running at 13:00Z): **10.01** of the 100 cap. Node E 4.44 (0.8B
  `RA` 2.86, 2B `RA` 1.03, readouts .55); node F 5.57 (`4b-LHA` 2.46, `4b-LHA10` 2.18, readouts .88, merges .06).
  Every arm stayed under its cap (4B 5.0, 2B 2.5, 0.8B 5.0).
- Pre-warm first: one job per tier / node (`warm-4b-f`, `warm-2b-e`, `warm-08b-e`) passed preflight before the
  parallel seeds started; all 8 preflights passed. GPUs used: node E GPU0–3 and node F GPU2 / 3 / 6 / 7 only; node E
  GPU4–7 and node F GPU0–1 and GPU4–5 untouched. All M12 lease entries are idle.
- The rules ran once per tier on complete inputs (2B 12:12Z, 0.8B and 4B 12:58Z). The LHB contrast was added afterwards,
  read only and outside the gate.

## What this suggests next (not preregistered)

- Breadth itself is cheap and safe on HT-DEV v2 and retention at 2B / 0.8B. What breaks is one typed head per arm. A
  typed-row teacher (KL to the released model on the typed rows only) or upweighting the released typed rows by the
  breadth share are the obvious next arms. Adding more breadth rows will not fix the typed heads.
- At 0.8B the miss is one choice family. An `attribute_gate`-weighted variant of `08b-RA` is the cheapest test of
  whether a family-level fix suffices.
- At 4B, 10% IB gets ~90% of the transfer gain of 25%. A smaller share (≈ 5%) with a typed teacher is the natural
  4B arm.
