# Decoder Milestone 14 — results (typed-row upweighting on M12's additive arms; 2026-10-01)

Preregistration [`dec-m14-prereg-2026-10-01.md`](dec-m14-prereg-2026-10-01.md) (`e48a66f46`); data lock
[`dec-m14-datalock-2026-10-01.md`](dec-m14-datalock-2026-10-01.md) (`940fb56a2`); amendment
[1](dec-m14-amendment-1-2026-10-01.md) (`efb303494`, formal scoring path; not used); state
[`dec-m14-state.md`](dec-m14-state.md). Development readouts are never release scores. Nothing was uploaded; C1 was
not opened; no Index row was read for selection or tuning.

## Bottom line

- **No M14 point passes its development gates, so M14 has no finalist.** No formal collection ran, successor items 1–7
  were not reached, and there are no release hand-offs, C1 item-8 spec or Index requests. The releases stay: DEV2.0-0.8B
  (`4afea305`), DEV2.0-2B (`2973ad4a`) and the released LH (`13d42143`). No M13 or fast-track candidate was released
  during M14 (the fast track's `08b-RA` failed formal; M13 closed with no model passing items 1–7), so these remained
  the then-current releases.
- **Upweighting the released (typed) rows keeps the breadth but moves the typed loss between heads again.** Transfer
  macro Δ stays large (0.8B +.197, 2B +.156, 4B +.045, every CI above 0), but each arm still fails a typed floor:
  - 2B (Score ×2): Score rises from M12's 227 to 236, still below the floor 245; Choice −20 and Noul −9 vs `2b-RA`.
  - 0.8B (×1.5): Choice falls further (524 vs `08b-RA`'s 600; floor 586), `attribute_gate` 268 (floor 280) and the
    unmet-condition yes-bias rises (.827 vs .720 + .10).
  - 4B (×1.5): typed T .908 vs LH .868 (Choice 799, Noul 309) and **HT-DEV v2 GAIN vs LH (+.028 [+.013, +.043])**, but
    Score 345 (floor 359) and retention −.042 [−.071, −.014].
- **The weights did what was preregistered** (provenance: weights `99b77df5…` / `e1ae6144…` / `3f741da8…`, same windows
  and update counts as M12), and the reference reads on nodes A / B equal M12's node E / F reads exactly, so the M12
  contrasts below are exact: the outcome is the lever, not the path.

## Results (two seeds each, BEST checkpoints souped; references read on the same node)

Gates: M12's / M13's (M8 eligibility type / family / Noul floors, Score5-typed-DEV floor, HT-DEV v2 not FLAG, retention
macro CI upper ≥ 0, `hs1-dev` false-yes ≤ reference + .10, 0.8B HT-DEV v2 Δ ≥ 0, transfer macro Δ ≥ 0). Retention uses
M12's tier golds (4B / 0.8B 2,203 items, 2B 2,189).

| Point (vs reference) | Typed T (ref) | C / N / S (ref) | HT-DEV v2 Δ | Retention macro (Δ) | Transfer macro, 9 families (Δ) | IB DEV, 12 families (Δ) | `hs1-dev` false-yes (ref) | Failed gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `08b-RAUP` (DEV2.0-0.8B) | .592 (.613) | 524 / 219 / 204 (610 / 212 / 159) | **+.028 [+.007, +.049] GAIN** | .445 (+.011 [−.016, +.036]) | .949 vs .751 (+.197 [+.160, +.234]) | .908 (+.211 [+.181, +.241]) | .827 (.720) | Choice type floor 524 < 610 − 24; `attribute_gate` family floor 268 / 400 < 320 / 400 − .10; yes-bias guard |
| `2b-RAUP` (DEV2.0-2B) | .629 (.610) | 538 / 232 / 236 (– / 230 / 257) | +.010 [−.006, +.026] TIE | .564 (−.004 [−.034, +.025]) | .959 vs .803 (+.156 [+.127, +.188]) | .930 (+.180 [+.156, +.206]) | .557 (.625) | Score type floor 236 < 257 − 12 |
| `4b-LHA10UP` (LH) | .908 (.868) | 799 / 309 / 345 (728 / 290 / 371) | **+.028 [+.013, +.043] GAIN** | .743 (−.042 [−.071, −.014]) | .967 vs .922 (+.045 [+.034, +.058]) | .937 (+.078 [+.064, +.092]) | .196 (.199) | Score type floor 345 < 371 − 12; retention (CI upper < 0) |

- Score5-typed-DEV: 2B and 4B no flags; 0.8B COLLAPSE / NO-GAIN as its reference, so the M8s amendment-3 floor applies
  and holds (Score 204 vs 159).
- Retention per probe: 0.8B MMLU .415 / ARC .718 / GSM8K .202; 2B .519 / .864 / .310; 4B .704 / .948 / .579.
- Seeds: BEST at updates 3,086 / 4,109 (0.8B), 1,095 / 1,249 (2B), 858 / 867 (4B). The 4B LoRA merges agree with
  their adapters on 128 / 128 and 127 / 128 SELECT rows (max drift .0066 / .0064) before souping.

## Contrasts vs M12's same-recipe arm (report only; exact path, same TRAIN, same windows)

| M14 − M12 | Typed T Δ (C / N / S) | HT-DEV v2 Δ | Retention Δ |
| --- | --- | --- | --- |
| `08b-RAUP` − `08b-RA` | −.044 (−76 / +3 / +2) | −.021 [−.039, −.002] FLAG | −.033 [−.060, −.005] (GSM8K −.096) |
| `2b-RAUP` − `2b-RA` | −.013 (−20 / −9 / **+9**) | +.014 [+.002, +.025] TIE | +.006 [−.021, +.032] |
| `4b-LHA10UP` − `4b-LHA10` | **+.035** (+56 / −16 / +16) | **+.024 [+.011, +.038] GAIN** | −.019 [−.046, +.006] |

- The only change is the relative loss weight of released vs IB rows within each 64-row update (released share of the
  weight .526 → .625 at 0.8B, .616 → .721 at 2B, .806 → .862 at 4B). It changed the typed profile in different
  directions at each tier rather than restoring the released one.
- Alongside M13 (same references and TRAIN files; report only): M13's self-distillation arms failed the same 2B Score
  floor (`2b-RASD` 237) and the 0.8B Choice floor (`08b-RASD` 574); its `4b-LHA10SD` passed development but tied LH
  formally (item 1) and failed mlx-diag (item 4).

## Operations

- GPU-h (own-node launch receipts; finished jobs, nothing running at 17:12Z): **7.12** of the 80 cap. Node A 3.18
  (`08b-RAUP` 2.90, readouts .28); node B 3.94 (`2b-RAUP` 1.04, `4b-LHA10UP` 2.19, readouts .68, merges .03). Every arm
  stayed under its cap (0.8B 5.0, 2B 2.5, 4B 5.0). No formal GPU job ran.
- All six preflights passed; one pre-warm job per tier and node (`warm-08b-a`, `warm-2b-b`, `warm-4b-b`) passed before
  the other seeds started; the first reference read filled each read cache. GPUs used: node A GPU3–5 and node B GPU2–4
  only; node A GPU0–1 untouched. All M14 leases are released.
- Rules ran once per tier on complete inputs (2B 15:55Z, 0.8B 16:54Z, 4B 17:07Z).
- Incidents (no effect on results): the workstation relay was too slow for readout directories, so readouts moved B →
  node E (M14-only transit directory, removed afterwards) → A; the interrupted first copy was deleted before the hop.
  `m14_gpuh.py` first counted staged M12 and relayed node-B receipts on node A; it now counts own-node receipts only.
  The prereg header's "≈15:20Z" was an estimate; the commit (14:59Z) is authoritative.
- Built but not used (no finalist): `M6_SMALL_NODE=B` in the M6 formal library (`915fcb87a`, test), the node-B formal
  wrapper, node-B master copies (`formal/m14/masters`), and amendment 1's two-bar scoring path.

## What this suggests next (not preregistered)

- Changing the released-vs-IB loss balance does not hold every typed floor; it trades the heads against each other.
  At 4B it traded Score and retention for Choice / Noul and human transfer: it is the first 4B IB arm of M11–M14 with an
  HT-DEV v2 GAIN verdict over LH (M13's `4b-LHA5`, +.019, was a TIE). A 4B arm that keeps this point but protects the
  Score head (Score ×2 as at 2B, or M13's self-distillation on the
  Score rows only) is the natural next test; LH's formal v3 is a high bar (M13's `4b-LHA10SD` tied it).
- At 0.8B, three levers (M13 self-distillation and proxy-family weighting, M14 upweighting) failed the Choice floor and
  the fast track showed the miss is real formally. The 0.8B arm sees the IB pool three times and IB rows are 47% of its
  rows; a single IB copy is the obvious next 0.8B dose.
- 2B: both levers stop about 9 Score items short of the floor (236–237 vs 245).
