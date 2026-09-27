# JPT-9B native same-panel development result

This closes the bounded [prospective comparison](jpt9b-native-samepanel-dev-prereg-2026-09-27.md).
The full-panel predictions were sealed before development gold was read. These
are **development and public-subset diagnostics**, not JevArena v3 FINAL,
CSS15, official JevBench, or Decision Index results. No candidate was trained
or selected in this experiment.

## Model and execution identity

`kirp/jpt-9b` revision `7114b0c3d9bea6b82dfa2d0691e8d5562cd26d4e`
was loaded through its pinned `llm2jev` source revision
`2b252d504972764211ef172c1155ac0fedc9c3de`, native `HF` backend and
temperature 1.087. Safetensors headers give **9,409,813,744 parameters**
across 760 tensors. The exact committed adapter, loader and scorer hashes are
in the preregistration. The 32-item Choice/Noul/Score smoke reproduced the
earlier native answer maps exactly, with zero invalid answers.

The three full prediction files had the exact expected 1,600/1,430/231 IDs,
source-input hashes, type maps, finite numbers, and zero missing answers.
Their seal SHA-256 is
`528da0323b26c326e2654a48395e4c8788573fac7125d9f7efc2996e85f27d4f`.
No protected FINAL or CSS15 prompt or label was read. Four GPU containers,
including the smoke, consumed **285.906 GPU-seconds = 0.079418 GPU-hours**
(30.649, 106.482, 103.357 and 45.418 seconds). All exited successfully.
The two full panels were concurrent on separately verified idle devices; the
total was far below the frozen two GPU-hour ceiling.

| Frozen artifact | Typed DEV1600 | CSS pilot1430 | Public JevBench231 |
| --- | --- | --- | --- |
| Native prediction SHA-256 | `32ef352d64d2b8d4d44da4e9b2adbd8177a1f1b12b386024a91a945aac3f5f3f` | `e4cc4f74238fd4e0994681c91fd377830d9b418221ab5eb503fa37e2ba55165f` | `a8e7e2c75d3449fb1132821827fbf22a82b4852939d5585593b30aec0338e357` |
| Native manifest SHA-256 | `fca2831ee09f315d4fd4f7c600fa6a4cd2c61e61b0da01757a16ed9277a1b33d` | `5827ace1b5a2498d45a2986724d2160b6907eebd6820e0bdca41140ab98d6acc` | `4048d11ba983df93e395e5d6e82cfdb82608747b356f24a0b3631ead76b0b5c4` |
| Score report SHA-256 | `02ec4b56debcea9802d7e7c45784cbe1f8df340a1066e042062f37d27065332d` | `584e4980af3c0e3475d23b6c69691ec30d6093912df74f78bcb71e63afbc88a6` | `5f34a80a850ef8a465e7a46e629534c964e5f7dcc2501a8387e75d5ec3dc622a` |

The public scorer's optional native-manifest schema expects `model_sha256`,
`adapter_sha256`, `predictions_sha256` and row `input_sha256`; JPT's collector
records a complete, separately sealed file-hash map and row
`source_input_sha256` instead. The first public scoring attempt stopped before
writing a report. The unchanged scorer was then run through its documented
inline model-ID/revision/input-hash path; its prompt/target manifest and all
231 rows passed. The independent JPT model-file manifest remains attached to
the sealed prediction set. No prediction, temperature or target changed.

## Same-input development outcomes

| Model | Actual loaded parameters | Typed DEV correct | CSS pilot median task macro-F1 | Public 231 correct |
| --- | ---: | ---: | ---: | ---: |
| JPT-9B, pinned native | 9,409,813,744 | **1,436/1,600 (89.75%)** | 0.52544 | **197/231 (85.28%)** |
| Own Lux 1.0, frozen prior receipt | not remeasured in this run | 1,388/1,600 (86.75%) | **0.57011** | 183/231 (79.22%) |
| Own Lux-start BEST224 research candidate | 7,984,174,080 unmerged | 1,398/1,600 (87.375%) | **0.57398** | 183/231 (79.22%) |

The Lux 1.0 typed/CSS observations are the previously frozen same-panel v2
baseline described in `jevk5-native-panel-2026-09-26.md`; raw paired DEV/CSS
receipts were unavailable to this run. Its public 231 score was independently
checked against the same public prompt/target digests and scorer version.
BEST224's own source-native report hashes are `fd321a43...0af333`
(typed), `fe12a134...7f36f33` (CSS), and `81330996...b6264` (public).
Those exact reports, not a newly chosen checkpoint, supply its row.

JPT's typed Choice/Noul/Score correct counts are **795/800, 298/400,
343/400**. Its typed Brier is 0.07823 and 10-bin ECE 0.04784; BEST224 has
799/800, 276/400, 323/400, Brier 0.09476 and ECE 0.05718. JPT's Score
point accuracy improves, while its Score expected-value MAE is worse
(0.19547 versus 0.18134). Thus the ordinal improvement does not establish a
uniformly better Score probability distribution.

JPT's CSS pilot task macro-F1 is 0.52544 discourse, 0.42263 implicit hate,
and 0.71154 stance; BEST224 is 0.57398, 0.40844 and 0.68026. JPT's median
loss comes from discourse despite a stance gain. CSS micro accuracy is almost
unchanged (55.38% JPT versus 55.52% BEST224). CSS median full-sum Brier is
worse for JPT (0.61719 versus 0.57981). Public easy/standard/hard correct
counts are **48/48, 68/72, 81/111** for JPT and **48/48, 67/72, 68/111**
for both own rows; nearly all the public gain is on hard questions.

Secondary paired uncertainty analyses, computed after development labels
were opened, are descriptive rather than a model-selection gate:

- JPT minus BEST224 typed family macro accuracy: **+2.375 percentage
  points**, 95% group bootstrap CI **[+1.00, +3.81]**, 400 four-row groups;
  report SHA-256 `58f5bca6e8eddc889488191fa6d0e966830ab57119e8c513a8192c8b43d909c8`.
- JPT minus BEST224 CSS pilot median task macro-F1: **−0.04853**, 95% paired
  within-task item bootstrap CI **[−0.08831, −0.00800]**. Micro-accuracy
  interval includes zero; report SHA-256
  `c103676989c8208ea5d40694fe3279463ac00244403138b8533338c2adaf2030`.
- JPT minus each own row on the public 231: **+14/231 = +6.06 percentage
  points**. Tier-stratified paired 95% interval is **[+2.16, +10.39]**
  against BEST224 and **[+2.60, +9.96]** against Lux 1.0; report SHA-256
  `28b7c3fa052f595cca28a84e0b38687bf8cb02b0b209ec3265e9830021bc7cf6`.

The external model is a strong typed/public comparator but is not a clear
human-transfer winner. It also has about 18% more parameters than the own
unmerged BEST224, so the public score delta is not a same-parameter Pareto
proof. The next 9B design experiment should test cross-source human-transfer
improvements without sacrificing Noul/Score accuracy; any subsequent claim
still needs an independent frozen panel. Historical Decision Index numbers
were not mixed into this comparison.
