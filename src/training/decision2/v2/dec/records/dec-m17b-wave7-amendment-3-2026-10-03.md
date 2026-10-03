# Decoder M17b wave 7, amendment 3: a two-arm low-LR soup, a top-3 rule and the two missing half-LR arms (2026-10-03 ≈23:50Z)

**Disclosed:** written after two wave-7 reads, before any other. Earlier records:
[prereg](dec-m17b-wave7-prereg-2026-10-03.md), [amendment 1](dec-m17b-wave7-amendment-1-2026-10-03.md),
[amendment 2](dec-m17b-wave7-amendment-2-2026-10-03.md).

## The two reads so far (no values; paired bootstrap vs the release's run)

- `4b-LRHxXALL-m50` is not a successor: its point is slightly above the release's and its lower bound is below 0.
  It trades the release's gains (RAGTruth, Home appliances, PhishNChips, VAST, FinEntity, API-Bank) for the full-LR
  soup's strongholds (HoVer, WinoGrande, cfcolor, BANKING77). Amendment 2's rule therefore builds
  `4b-LRHxXALL-m75`, which is on the Index.
- `4b-SDMLIB4-lrh` has its point above the release's; the bootstrap is running. It is the second half-LR arm at
  that level, and its profile complements the release's arm. It gains on cfcolor, ContractNLI, CLadder, BANKING77
  and WinoGrande, and loses on When2Call, HoVer, BPoMP, New Yorker and RAGTruth.

Still unread: `4b-LRHxALL`, `4b-LRHxALL-L2`, `4b-LHS17ML-lrh`, `4b-LHS17IB4X-lrh`, `4b-SDML-lrh`,
`4b-LRHxXALL-m75` and `4b-LHS17IB4-lrq`.

## Added candidates (gate and measurement unchanged)

| Candidate | Members (FP32) | Built when |
| --- | --- | --- |
| `4b-LRH2` | ½ `4b-LHS17IB4-lrh` + ½ `4b-SDMLIB4-lrh` | now (CPU, node C) |
| `4b-LRHxTOP3` | uniform over the three half-LR arms with the highest single-arm points | all five half-LR arms read; skipped if its set equals an earlier candidate's |
| `4b-LRHxALL7` | uniform over seven half-LR arm soups: the five, plus `4b-LHS17SD-lrh` and `4b-LHS17UP-lrh` | both new pairs DONE |

- `4b-LRH2` keeps the two half-LR arms that are each at or above the release, and nothing else. Members that join
  dilute the release's arm.
- `4b-LRHxALL7` is the low-LR analogue of the earlier release `4b-SDMLxALL`, which used the same seven recipes at
  full LR. The arm factory is asked for `4b-LHS17SD-lrh` and `4b-LHS17UP-lrh`: two seeds each, the released arms'
  locked TRAIN and weights files, LoRA / head LR 5e-5.
  - IF3: `4b-LHS17UP` trains on the `4b-LHS17SD` file, which audit6 covers. No new TRAIN file.
- Order: `4b-LRH2` on the first free lane, then the rest of amendment 2. `4b-LRHxTOP3` and `4b-LRHxALL7` follow
  when they are built.
