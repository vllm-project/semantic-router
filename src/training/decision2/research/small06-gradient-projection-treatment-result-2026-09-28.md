# 0.6B fixed gradient-projection treatment: SELECT HOLD

**Decision:** one and only one full projected-gradient treatment completed its
frozen 466 updates. The fixed-rule BEST checkpoint is step 466. It failed the
predeclared SELECT advancement gate; do not inspect typed DEV, CSS pilot,
JevArena FINAL, JevBench or other revealed formal/public labels for this arm.
Do not package or upload this candidate. Preserve the ordinary control and the
failed treatment as separate research artifacts.

## Matched comparison

Both arms start from the same official Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd` with the same full
7,455-row rights-clean v2 TRAIN, 700-row SELECT, 700-row untouched CAL, seed,
epoch-zero order, 4,094,489 native TRAIN tokens, CE + 0.5 Brier objective,
FP32 parameters and BF16 backbone compute, shared head, AdamW schedule,
global clip, eight fixed checkpoints, and 466-update horizon. The only
treatment difference is per-type projection of shared-backbone gradients
against original other-type gradients in Choice→Noul→Score order, with the
combined backbone norm matched to the ordinary sum. The head gets the
ordinary gradient. No third-party Decision weights initialize either arm.

| Frozen SELECT measure | Ordinary control BEST466 | Projected BEST466 | Prospective advancement |
| --- | ---: | ---: | ---: |
| Correct / 700 | 562 | 547 | ≥569 |
| Six-family macro accuracy | .772593 | .713426 | ≥.78259 |
| Quantized Score / 90 | 32 | 32 | ≥37 |
| Human GoEmotions Choice / 200 | 161 | 159 | ≥156 |
| Six-family macro Brier, lower better | .142634 | .168494 | Report |

The frozen selector ranks checkpoint family-macro accuracy, then normalized
Brier, then earliest step. It selected step 466 over step 384: both had macro
accuracy .713426, while the former had lower Brier. An earlier step 320 did
reach Score 38/90, but its SELECT total was only 529/700 and macro accuracy
.694537. It was not the selected checkpoint and must not be substituted.
The selected arm is 15 answers and .059167 macro accuracy below the existing
same-start ordinary control, and misses three of four prospective capability
thresholds. This is a negative development result, not a formal 2.0 score.

## Integrity, cost and limits

- Implementation: signed code commit `b44d61d9d9f722176ee89c8f4fc3867bf2fee5f7`.
  The archived ordinary `train.py` SHA-256 remains
  `0111bbfd0e6a372661a88c44e0c716c18a94b78f651c0662fa2b18bbe59ad96d`.
  Treatment source SHA-256 is
  `c5be12159017cd332f45e9747ce008df2af0a09c617524fbef10fa48368d8e6a`;
  projection algorithm SHA-256 is
  `5cc2e5cf0c907f44597a08e383f23650ffb43803e23235151136da77dd5434e7`.
- TRAIN SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`;
  SELECT SHA-256 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`;
  CAL SHA-256 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  The new CPU plan matched the archived ordinary plan apart from creation time.
  Archived ordinary plan SHA-256 `20174e684605aadbc19b3b16b32cbbcc091a8f0f6bdeaaa28a7898157b1f5201`;
  projected CPU plan SHA-256 `69a2fa229bf6bdb52b58f5521ce1065a9d01f8d894f3cc9e55bd08b9f6e6c009`.
- A separate one-update save/reload preflight passed zero-step source parity and
  post-update checkpoint parity on 32 fixed SELECT prompts: zero changed
  categories and maximum option-probability drift 0.0 in both comparisons.
  Its receipt SHA-256 is
  `ce0e7789e880aeca56a2b3f8f2eec075016acd731330cd8ea012548eedab537b`.
  Its first unchanged window had Choice 11, Noul 5, Score 0 and zero projected
  pairs; it cannot by itself validate Score behavior. CPU tests did cover a
  three-type conflict and projection-disabled ordinary-gradient equivalence.
- The full treatment logged 321 Score-containing windows, 353 windows with an
  actual projected pair, 1,138 projected pairs in all, and finite task and
  global gradient norms at every step. Its median optimizer window was 2.183
  seconds. The conservative whole-run GPU-visible wall time was 1,119.95
  seconds = 0.31110 GPU-hour, below the frozen 2 GPU-hour cap. The one-update
  preflight added 0.00546 GPU-hour. No numerical stop or reselected window
  occurred.
- Private receipt hashes for independent audit: provenance
  `d969c1866f3d5917b43de777dbd6e8b227708f0ddc648378a239cc3240e1e49f`,
  full metrics
  `52d7b99448b7bc854c7fad997ab74f0b7622f1226a40e4284dd373d6d500f752`,
  BEST selector
  `e96277b9e0b07b7ef8c8df4b2fcbb71b7bdb9e8b4b8ed58816d8381dc221e3b6`,
  completion
  `f35dd46e7f4a09d94532eee6748d0daddb04adfe1bf4d94c031cd029a98c9d49`,
  selected SELECT predictions
  `5b611e47303a5723fcbc4bc290a4ae5830671a45884d7f7c5027bd8d24fcdb5e`,
  selected checkpoint metadata
  `b72a641bee8ecd75a3cca79433ae1ebc235e99232769791dfef4948e52a318db`.

The read-only conflict preflight previously found a negative pair involving
Choice or Score in six of eight fixed mixed-type windows, but a direct
Choice–Score negative pair in only one. This supports broad cross-task
interference, not a specific causal explanation for the original Choice/Score
regression. The completed projection treatment did not improve the selected
SELECT result. Its failure does not rule out other, separately preregistered
data or architecture approaches.

The first launch attempt was rejected before any optimizer update because a
wall-timing receipt had been placed in the required-empty output directory.
Moving that receipt outside the output directory allowed the single fresh arm
to run; the failed-at-start receipt remains private. No completed result or
control was reset.
