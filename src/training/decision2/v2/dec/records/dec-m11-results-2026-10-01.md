# Decoder Milestone 11 — results (LH recipe at 2B / 0.8B; 4B stage 2 with IB1 + IB2; 2026-10-01)

Preregistration [`dec-m11-prereg-2026-10-01.md`](dec-m11-prereg-2026-10-01.md) (`b8c3caf22`); data lock
[`dec-m11-datalock-2026-10-01.md`](dec-m11-datalock-2026-10-01.md) (`7c3aed8c8`); amendments
[1](dec-m11-amendment-1-2026-10-01.md) (`f45c2f481`, node F GPU layout) and [2](dec-m11-amendment-2-2026-10-01.md)
(`3bf17e6c2`, the 2B rules output that lacked probe / `hs1-dev` inputs); stage 2
[`dec-m11-stage2-prereg-2026-10-01.md`](dec-m11-stage2-prereg-2026-10-01.md) (`9c8798576`) and
[`dec-m11-stage2-datalock-2026-10-01.md`](dec-m11-stage2-datalock-2026-10-01.md) (`3773f99a0`); state
[`m11-state.md`](m11-state.md). Development readouts are never release scores. Nothing was uploaded; C1 was not
opened; no Index row was read for selection or tuning.

## Bottom line

- **No M11 point passes its development gates, so M11 has no finalist.** No formal collection ran, successor items
  1–7 were not reached, and there are no release hand-offs or Index requests. DEV2.0-2B and DEV2.0-0.8B stay the
  releases at their tiers. At 4B, `m10-4b-LH` stays the successor candidate; the custodian C1 recheck for IB-trained
  models is not needed this milestone.
- **The LH recipe does not transfer to 2B or 0.8B as-is.** At both tiers the base-start LoRA arm improves or holds
  HT-DEV v2 and retention, but it gives up typed decision floors to the released model (2B: Noul and Score, with a
  Score5-typed-DEV collapse; 0.8B: choice). The label-token arms on the 1.0 lineage do not pass either.
- **Breadth data (IB1 + IB2) adds breadth at 4B but costs typed floors vs LH.** Both stage-2 arms raise the IB DEV
  family macro strongly (all 12 families: +.092 with the in-distribution families, +.030 without; 9 transfer families:
  +.053 / +.051) and tie LH on HT-DEV v2, but `4b-LHB` loses Noul and Score typed accuracy and `4b-LHBx` loses choice
  accuracy and retention.

## Stage 1 — 2B and 0.8B (three seeds each, souped; references read on the same node)

Gates are the preregistered M10 gate (M8 eligibility floors, Score5-typed-DEV floor, HT-DEV v2 not FLAG, retention
macro CI upper ≥ 0), the yes-bias guard (`hs1-dev` false-yes ≤ reference + .10) and, at 0.8B, HT-DEV v2 Δ ≥ 0. Arm LH
is compared with the tier's C0 (the 2.0 release) read on node F, arm NT with C0 read on node E; the two C0 reads are
identical (below).

| Point (vs C0) | HT-DEV v2 Δ | Retention macro (Δ) | `hs1-dev` false-yes (C0) | Failed gates |
| --- | --- | --- | --- | --- |
| `2b-LH` | −.001 [−.020, +.019] TIE | .585 (+.014 [−.001, +.028]) | .444 (.625) | Noul type floor 213 < 230 − 12; Score type floor 175 < 257 − 12; `set_reconciliation` family floor; Noul `rule_precedence` floor; Score5-typed-DEV COLLAPSE |
| `2b-NT` | **+.029 [+.015, +.042] GAIN** | .554 (−.017 [−.030, −.004]) | .580 (.625) | retention (CI upper < 0) |
| `08b-LH` | **+.030 [+.008, +.050] GAIN** | .445 (+.004 [−.009, +.017]) | .682 (.720) | choice type floor 501 < 610 − 24; `attribute_gate` family floor 247 / 400 < 320 / 400 − .10 |
| `08b-NT` | +.018 [−.001, +.035] TIE | .433 (−.008 [−.020, +.003]) | .563 (.720) | choice type floor 571; Score type floor 146 < 159 − 12; Noul `rule_precedence` floor 207 < 212 − 4 |

- Head vs label-token (report only): 2B `2b-LH` − `2b-NT` HT-DEV v2 −.029 [−.048, −.010], typed T −.078 (choice +26,
  Noul −36, Score −115), retention +.031 [+.016, +.045]. 0.8B `08b-LH` − `08b-NT` HT-DEV v2 +.012 [−.009, +.033],
  typed T +.033 (choice −70, Noul +23, Score +100), retention +.012 [−.001, +.025].
- Base retention ceilings (each tier's base vs its C0, the prereg's ceiling report): **2B base .635, +.064 [+.046,
  +.082]**; **0.8B base .495, +.054 [+.034, +.075]**. The bases themselves FLAG on HT-DEV v2 (2B −.057, 0.8B −.095)
  and collapse on Score5-typed-DEV. As at 4B, the released lineage has lost retention relative to its base; LH keeps
  only part of it at 2B (.585 vs .635) and little at 0.8B (.445 vs .495).
- C0 parity: each tier's C0 read on node F, node E and node B's stored M8 `I` readouts agree exactly on five panels (0
  decisions differ, drift 0.0). The M11 path reproduces the released models.
- Soups: three LoRA merges per LH arm (2B merges agree with their adapters on 128 / 128, 128 / 128 and 126 / 128
  SELECT argmax rows; drift ≤ .010), averaged.

## Stage 2 — 4B LH + IB1 + IB2 (two seeds each, souped; reference LH read on the same node)

Arms on LH's recipe (rank-128 LoRA from Qwen3.5-4B-Base with the head, own-Lux KL 1.0 partial) at LH's matched
tokens, with IB rows filling 25% of the tokens: `4b-LHB` (all 15 IB1 + IB2 families) and `4b-LHBx` (without the
in-distribution families `w2c`, `isarc`, `hover`, `gsm2`). Gates are the M10 gate, the yes-bias guard and the breadth gate
(IB DEV macro Δ ≥ 0), all vs LH. The retention probes use the 4B stage-2 gold (M10's 3,089 items minus the 628 GSM8K
items in stage-2 TRAIN: 2,461).

| Point (vs LH) | HT-DEV v2 Δ | Retention macro (Δ) | IB DEV macro, 12 families (Δ) | Transfer macro, 9 families (Δ) | `hs1-dev` false-yes (LH) | Failed gates |
| --- | --- | --- | --- | --- | --- | --- |
| `4b-LHB` | +.000 [−.016, +.017] TIE | .763 (+.001 [−.015, +.017]) | .952 vs .859 (**+.092 [+.078, +.106]**) | .975 vs .922 (+.053 [+.041, +.065]) | .194 (.199) | Noul type floor 245 < 290 − 12; Score type floor 309 < 371 − 12; `rule_precedence` and `set_reconciliation` family floors; Noul `rule_precedence` floor |
| `4b-LHBx` | +.001 [−.015, +.017] TIE | .729 (−.034 [−.049, −.019]) | .889 vs .859 (+.030 [+.019, +.042]) | .973 vs .922 (+.051 [+.039, +.064]) | .161 (.199) | choice type floor 689 < 728 − 24; retention (CI upper < 0) |

- IB DEV per family, `4b-LHB` / `4b-LHBx` / LH: `gsm2` .797 / .589 / .586; `hover` .910 / .660 / .705; `isarc` .939 /
  .659 / .720; `poem` .995 / .993 / .835; `sms` .952 / .935 / .823; `args` .933 / .944 / .861; `copa` .954 / .944 /
  .917; `sentfin` .963 / .968 / .941; `ytspam` .978 / .978 / .946; `argq` .998 / .998 / .978; `snips_rel` and
  `snips_sel` 1.0 for all three. The transfer gain (unseen families) is the same with or without the in-distribution
  families; those families only help on themselves.
- `4b-LHB` − `4b-LHBx` (report only): HT-DEV v2 −.001 [−.015, +.014]; typed T −.044 (choice +76, Noul −65, Score
  −81); retention +.034 [+.018, +.051] (the in-distribution `gsm2` rows hold GSM8K retention).
- LH reference parity: `4b-LH-f` vs M10's stored `4b-LH` readouts exact; `4b-LH-e` vs `4b-LH-f` exact on seven panels
  (incl. retention probes and IB DEV).
- Formal bar note: no stage-2 point reached formal, so the bar rule (LH release if published, else DEV2.0-4B) was not
  applied; at 10:40Z the LH release was not yet recorded on integration.

## Operations

- GPU-h (launch receipts, both nodes; finished jobs, no M11 job running at 10:45Z): **18.17** of the 110 cap. Node E
  8.47 (2B NT 1.28, 0.8B NT 3.57, `4b-LHBx` 2.08, readouts 1.45, pre-warm / merges .09); node F 9.70 (2B LH 1.48, 0.8B
  LH 4.27, `4b-LHB` 2.26, readouts 1.60, merges .09). Stage-2 arms stayed under their 6 GPU-h caps.
- Every preflight passed; one pre-warm job per fresh node / cache before parallel seeds (`warm-2b`, `warm-08b`,
  `warm-4b-e`, `warm-4b-f`). GPUs used: node E GPU0–3, node F GPU2 / 3 / 6 / 7 only; node E GPU4–7 and node F GPU0–1
  and GPU4–5 (amendment 1) untouched. All M11 lease entries are idle.
- Incident (amendment 2): the first 2B rules output lacked the retention / `hs1-dev` inputs (tier-gold probe scoring
  error, two concurrent scoring runs); it was kept aside, the scorer fixed, a scoring lock added, and the rules run
  once on complete inputs. The rerun changed no reason that did not depend on the missing inputs.
- The formal path for 2B / 0.8B on nodes E / F (`M6_SMALL_NODE`, `m11-formal.sh`, frozen masters on both nodes) is
  built and tested but was not used.

## What this suggests next (not preregistered)

- At 2B / 0.8B, a base-start LoRA alone does not keep the released typed decisions. The typed losses look like the
  0.8B / 2B mixtures' Noul / Score / choice rows need more weight (or a typed-row teacher) when starting from a base.
- At 4B, the IB blocks buy breadth without HT-DEV cost, but the 25% token share displaces typed rows. A smaller IB
  share, or IB on top of LH's full token count rather than inside it, is the obvious next arm; the transfer-only result
  says the in-distribution families matter for GSM8K retention, not for transfer.
