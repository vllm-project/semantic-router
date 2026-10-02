# Decoder M17b wave 6, amendment 2: the half-LR arm `4b-LHS17IB4-lrh` becomes a release candidate (2026-10-03 ≈21:15Z)

**Disclosed:** this was written after the result was read. The prereg listed `4b-LHS17IB4-lrh` (the arm factory's
two-seed soup of `4b-LHS17IB4` at half the LoRA / head learning rate, 5e-5) as an *information point*, assuming single
arms sit below the release, as every earlier single arm did. Its one Index read (`AF-4b-LHS17IB4-lrh-bf16`, node A,
panel-8, 120,224 ok + 2 unsupported, the release run's panel) is far above the release's point.

The release rule is unchanged and decides on its own: under the Index-first gate any measured candidate qualifies if
the full-panel paired bootstrap vs `d55528d1`'s run has a 95% lower bound above 0. So this amendment only adds the arm
to the candidates; it changes no candidate, weight or measurement.

## The candidate

- `4b-LHS17IB4-lrh`: uniform FP32 soup `0b40be94…` of seeds s1 (20260926, BEST `checkpoint-0001249`) and s2 (20260927,
  `checkpoint-0001254`). LoRA r128 on Qwen3.5-4B-Base `@1001bb4d`, scoring head, typed-row self-distillation, trained on
  M17's locked `4b-LHS17IB4` TRAIN `dfed3944…` (audited in audit6). BF16 release copy `1cef169d…`.
- Integrity so far: the IX1 86-request parity gate PASS (86 / 86, max |Δp| 0); IF3 covered by audit6.
- Next: formal R3 (node F, T = 1, scored on node A), then the release path if the bootstrap's lower bound is above 0.

## What it says for the next wave (no values)

Its gains are exactly the 4B deficits (RAGTruth, Home appliances, API-Bank, FinEntity, PhishNChips, GSM8K,
iSarcasmEval), and it gives back some of the strongholds (HoVer, WinoGrande, BANKING77, CLINC150). A lower step size
keeps the arm nearer the base, which is also what cross-arm averaging did, only more weakly. Wave 7 therefore adds
lower-LR arms (preregistered separately).
