# Own-Lux 9B short-rule screen: development HOLD

This receipt closes the bounded experiment frozen in
`lux9b-short-rule-replay-screen-prereg-2026-09-27.md`. It is a development
result, not a JevArena v3 or JevBench result for a new release candidate.

## Fixed experiment and observed result

- Direct weight start: our Decision-1.0-Lux-9B, pinned at
  `bd45a30aee8c84032791c245c70f86dee5389cc8`.
- TRAIN: 6,094 rows, SHA-256
  `a9264ff9f97cba440d71317b514b6f1f9828146d8047d2ac229812013260e9eb`;
  5,824 human rows plus 270 complete short-rule groups. Added inputs were
  6.8185% of input tokens. SELECT: 600 fixed rows, SHA-256
  `d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38`.
- The 32-prompt zero-step source check had zero category changes and zero
  maximum probability drift. The separate one-step arm had finite loss and
  gradient, with 32.85 GiB peak GPU allocation. Neither supplied weights to
  the 128-step arm.
- The 128-step arm completed on one GPU. SELECT counts at steps 32, 64, 96,
  and 128 were respectively **333, 338, 334, 338 / 600**. The fixed
  accuracy/Brier/earliest tie rule selected checkpoint 64. Final step 128
  remained 338/600 = **0.563333**, with family-macro Brier **0.302419** and
  finite logged loss/gradient. The existing human-only control BEST was
  358/600 = **0.596667**.
- The preregistered screen threshold was **0.576667**. BEST 0.563333 missed
  it by 0.013334 (8 SELECT items), so the arm is **HOLD**. The run stopped
  before CAL, full typed DEV, CSS pilot, public JevBench, or protected v3
  inference. No formal score can be inherited from another 9B checkpoint.

The result argues against adding this fixed, small short-rule replay to the
5,824-row human mix with this optimizer budget. It does not isolate data
composition from learning schedule or establish that short rules never help.
The next 9B experiment should test a separately frozen mechanism that better
protects human transfer, using the same source and an unchanged SELECT gate.
The failed screen will not be extended by choosing a favorable checkpoint or
loosening its threshold.
