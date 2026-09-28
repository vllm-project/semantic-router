# 9B Milestone 3 formal post-key run, arm B: lock

Frozen 2026-09-28 before any formal prediction of arm B. The v3 labels were accessed earlier in
the project, so this is a **post-key same-panel** comparison; public 231 is a public-subset
reproduction, not the official rank.

## Why arm B is a finalist (preregistered rule)

Development readout (typed DEV 1,600 + CSS pilot 1,430, CAL698 temperatures), two-seed mean vs
the same-runtime Lux 1.0 readout: proxy 70.48 vs 70.82 (ΔP −0.34, a tie under |ΔP| < 4);
typed-DEV Choice 797 vs 799, Noul 268 vs 272, Score 325.5 vs 331 (all within 3.0 points); CSS
pilot H .5716 vs .5724 (within 0.015). Arm A (its matched Lux-target control) ties on the proxy
(71.21) but breaches the typed-DEV Score floor (−3.4 points), so it is not a finalist.

## Candidate and comparators

| | Candidate B-s1 |
| --- | --- |
| Weights | Lux 1.0 `bd45a30a…` + arm B LoRA r16 / head, `m3/B-s1/run/checkpoint-0003131` (preregistered primary seed 20260926, SELECT BEST) |
| Training | mx-v2-full-M (pk1 A0s) + A7 natural24k retention replay, TRAIN `55abf2ad…`; KL 0.5 to AutoJev-27B targets (`0d1a71f1…` = pk1 A0s `97a071af…` + AJ-M `ba52dd86…`) |
| Calibration | CAL698 per-type temperatures fitted for this checkpoint (`m3/B-s1-cal/calibration.json`) |
| Adapter / limit | `v2/dec/adapter-spec-infer-dec.json` (`v2.dec.infer_dec`, shared renderer), **16,384 tokens**, over-length inputs invalid, no truncation |
| Runtime | pinned image `f83b1d10…` + FLA, node A GPU2, `TRITON_CACHE_AUTOTUNING=1` with the frozen Milestone 3 cache copy `formal-m3/triton-cache` (already used by the same-renderer Lux1 16K control) |

Comparators (both node A, 16,384 tokens): the native Lux1 run `eval m1/d1-lux1-autotune-cache`
(v3 **65.808**, the stricter and gating comparator) and the same-renderer Lux1 control
`formal-m3/lux1-16k-shared` (65.231, descriptive).

## Panels, reading and gate

- Collection: runner gold-free smoke (`--max-items 8`), typed FINAL + CSS15 + public 231, then
  `mlx-diag`; seal before scoring; report and paired compare with the eval runner
  (`v2/9b/lux9b/m3/formal.sh`, mirror `e9b1231be`).
- **Release gate:** paired post-key v3 95% lower bound > 0 against the native Lux1 16K run.
- Reported: v3, T/H, Choice/Noul/Score, typed families, per-task CSS15 with H with and without
  FLUTE (GoEmotions flagged as a same-source task present in A0s), public 231 E/S/H, typed/CSS
  calibration, invalid counts, mlx-diag. AutoJev provenance caveat disclosed.
- No checkpoint, seed, calibration or limit change after collection; a collection fault is
  recorded and not retried blindly.
