# 9B Milestone 3 formal post-key run, arm DW: lock

Frozen 2026-09-29 (UTC+8) before any formal prediction of arm DW. **Post-key same-panel**
comparison (the v3 labels were accessed earlier in the project); public 231 is a public-subset
reproduction, not the official rank.

## Why arm DW is a finalist (preregistered rule, amendment 4)

Development readout (typed DEV 1,600 + CSS pilot 1,430, CAL698 temperatures) vs the same-runtime
Lux 1.0 readout: proxy **76.06 vs 70.82 (ΔP +5.24 [+3.14, +8.37])**; T .9581 vs .8763, CSS pilot
H .6038 vs .5724 (mean H .5817 vs .5283); typed-DEV Choice 800 vs 799, Noul 367 vs 272 (rule
precedence), Score 366 vs 331; every floor holds. DW is the single α = ½ artifact fixed in
amendment 4 before it was built (chosen after arm D's development readout; disclosed). It is
this milestone's second formal candidate (arm B was the first).

## Candidate and comparators

| | Candidate DW |
| --- | --- |
| Weights | full checkpoint `m3/DW-build/soup`, `model_sha256` `de0dcebd…`: uniform FP32 average of the arm D soup (`9a1d7db8…` = mean of the SELECT-chosen checkpoints of seeds 20260926 / 1 / 2: steps 2,683 / 2,677 / 1,672) and Lux 1.0 in full-checkpoint form (arm D's zero-step checkpoint, `dc9b795c…`) |
| Training behind D | Lux 1.0 full fine-tuning (backbone 1e-5, head 1e-4, one epoch, ≥ 64 rows per update) on 205,790 rows / 116.1M tokens (`34c18d7a…`): mx-v2-full-M (pk1 A0s minus the 752 shortcut-family rows) with own-Lux KL 0.5, A7 v3 core, A7g 45M, v1 A1–A6, A7k / A7s / A7r, A7q 5M. Own-Lux targets only (no third-party teacher); C1 / mlx-diag source guard and isolation passed |
| Calibration | CAL698 per-type temperatures for this checkpoint (`m3/DW-cal/calibration.json`: Choice .859, Noul .722, Score .247) |
| Adapter / limit | `v2/dec/adapter-spec-infer-dec.json` (`v2.dec.infer_dec`, shared renderer), **16,384 tokens**, over-length inputs invalid, no truncation; packaging profile `qwen-full` (7,940,895,744 loaded parameters, the Lux 1.0 architecture) |
| Runtime | pinned image `f83b1d10…` + FLA, node A GPU4, `TRITON_CACHE_AUTOTUNING=1` with the frozen Milestone 3 cache copy `formal-m3/triton-cache` |

Comparators (node A, 16,384 tokens): native Lux1 `eval m1/d1-lux1-autotune-cache` (v3 **65.808**,
the stricter, gating comparator) and the same-renderer control `formal-m3/lux1-16k-shared`
(65.231, descriptive).

## Panels, reading and gate

- Runner gold-free smoke (`--max-items 8`), typed FINAL + CSS15 + public 231, then `mlx-diag`;
  seal before scoring; report and paired compare (`v2/9b/lux9b/m3/formal.sh`, mirror `e78a79a9c`).
- **Release gate:** paired post-key v3 95% lower bound > 0 against the native Lux1 16K run.
- Reported: v3, T/H, Choice/Noul/Score, typed families, per-task CSS15 with H with and without
  FLUTE (GoEmotions flagged as a same-source task in A0s), public 231 E/S/H, typed/CSS calibration,
  invalid counts, mlx-diag. Disclosed: typed families share mechanisms (not text) with A7 Stage4
  generators, as the A7 track recorded.
- If the gate is met: the checkpoint and calibration go to a private HF staging model repo and
  are reported immediately with the scored run directory; no public artifact.
- No checkpoint, calibration or limit change after collection; a collection fault is recorded
  and not retried blindly.
