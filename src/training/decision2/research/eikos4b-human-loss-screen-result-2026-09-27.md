# Eikos 4B human-source loss screen: result and stop decision

This is the result of the prospective, single-treatment protocol in
[`eikos4b-human-loss-screen-prereg-2026-09-27.md`](eikos4b-human-loss-screen-prereg-2026-09-27.md).
It is a development ablation, not a model release or sealed JevArena result.

## Frozen comparison and execution audit

The preserved clean-v2 control and this treatment share the pinned Eikos-4B
source revision `582ffb13f19a4da3f455e3db198584190bd7755b`, unchanged
TRAIN/SELECT/CAL bytes, ordered admitted TRAIN rows, 20260926 seed, optimizer,
one-epoch batch schedule, 232 updates, native serving adapter, and the pinned
ROCm image digest `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
The only treatment is 1.5× loss weight for the 3,974 admitted human-source
TRAIN rows; the other 3,444 rows, including all 516 Score rows, retain 1.0×.
The original control was not rerun or modified.

The new isolated run completed all **232/232** optimizer updates with eight
durable checkpoints (steps 32 through 224 every 32, plus 232) and exit code 0.
Its per-step `(step, examples, tokens)` trace exactly matches the 232-step
control trace: **7,418 admitted examples** and **4,402,743 input tokens**.
Before any optimizer update, all **700** source SELECT predictions matched
the preserved control *byte for byte* (SHA-256
`7fef6e2fcaa42e17ed1187533230b7210d566129ee60721e051f45f1e17ec7ff`):
zero categorical changes and maximum probability drift **0.0**. The parity
comparison ignored gold labels. The old initial LoRA tensor was not archived,
so exact source output parity plus the same seed/code/source is the available
initialization evidence, not a tensor-by-tensor proof.

Single-GPU training occupied the assigned GPU from
`2026-09-27T01:45:53.140382793Z` to `2026-09-27T02:09:41.681871583Z`.
This is **23 min 48.54 s**, or **0.397 GPU-hours** of allocated training
wall time, including internal SELECT and checkpoint saves; it is not a GPU
utilization measurement. The run and control artifacts remain private.

The frozen implementation is signed commit
`0b335e341cc01de0dd3847217f31ea786e9e2e4c`. Private receipt SHA-256s
permit audit without exposing machine locations or restricted rows:

| Receipt | SHA-256 |
| --- | --- |
| `provenance.json` | `eefed403fcdc186fd8ac6daa08a885e77bd0fd1a98010e1fbc62ee7642cfd6d5` |
| `events.jsonl` | `5bfe5e203159166debdadc4d6985c95d4680a93a826f8411263b799d1040bd76` |
| `COMPLETE.json` | `07d9772eeb82b0315a7690680c3d1be78670250477db4600e01cc7ad587cde7f` |
| `baseline-parity.json` | `c8d14a8bfe701b30896410f24765f10b39bb0bda0b09707c3f742a83e00c2cca` |
| `NATIVE_BEST.json` | `f771b86c0c9a69e100dec4747e1f4feb34be24771eabba179ca368949a4ddbf6` |
| Selected native predictions | `ffe6d3c685d50477c94d52c412640e02552ae7bb6ab27bf876d2de13eff17ae5` |

## Frozen released-native SELECT gate

The completed native SELECT run compared the original source and all eight
checkpoints on the unchanged SELECT700 using the released
`serve.Decider` path. The predeclared winner is highest family-macro accuracy,
then lowest Brier, then source or earliest step. The advancement gate is
conjunctive: human GoEmotions at least **337/400**, family macro at least
**.822222**, Score targeted median at least **74/90**, and no invalid or new
source-length truncation. Both runs selected checkpoint **0232** natively;
the treatment's fast trainer preferred checkpoint 0224, so the native
selection was necessary. No alternate treatment checkpoint exceeded 334/400
human correct.

| Frozen SELECT700 measurement | Control | Treatment | Delta |
| --- | ---: | ---: | ---: |
| Correct | 595 | 602 | +7 |
| Family-macro accuracy | .827222 | .839444 | +.012222 |
| Family-macro Brier, lower better | .102390 | .103655 | +.001265 |
| Human Choice / 200 | 159 | 162 | +3 |
| Human Noul / 200 | 172 | 172 | 0 |
| **Human subtotal / 400** | **331** | **334** | **+3** |
| Score targeted median / 90 | 75 | 78 | +3 |
| String composition / 40 | 19 | 20 | +1 |
| Narrative reading / 130; abstention / 40 | 130; 40 | 130; 40 | 0; 0 |

Both selected prediction files contain the same 700 IDs and input hashes.
All treatment answers are valid finite distributions with keys in domain.
Native input-token counts match the control row by row; the maximum is 236,
with no new truncation. The human paired transitions are five control-wrong
to treatment-correct and two in the reverse direction (Choice 3/0, Noul
2/2). Score transitions are 4/1. These small development changes are not a
transfer claim or a blind confidence interval.

The native selector exited 0 after **5 min 5.89 s**, another **0.085 GPU-hours**.
Training plus native SELECT consumed **0.482 allocated GPU-hours** on one GPU.
These times exclude any historical control compute;
GPU allocation time does not measure actual device utilization.

## Subsequent diagnostics and conclusion

**STOP: the frozen human gate failed by three items** (334/400 versus the
required 337/400), even though family-macro and Score floors passed. The
protocol therefore forbids CSS pilot, XNLI/PAWS-X validation, typed DEV, and
exposed public diagnostics for this treatment. None was run. CAL labels were
not used for calibration or model selection. Neither protected FINAL nor
release labels, Jev outputs, or a Hugging Face publication were used.

This one-variable comparison shows that increasing the existing human-source
loss mass moves the matched SELECT checkpoint's Score and overall accuracy
up, but fails the predeclared human gain. It cannot establish source-disjoint
transfer, because that panel was intentionally gated off. Do not rescue this
run by post hoc checkpoint choice or threshold changes. A future prospective
arm needs independently sourced human and ordinal Score validation before
training, with its source IDs and contamination audit frozen, while retaining
the same native runtime and an untouched control. This failed arm remains an
archived research result, not `dev-2.0-4b`.
