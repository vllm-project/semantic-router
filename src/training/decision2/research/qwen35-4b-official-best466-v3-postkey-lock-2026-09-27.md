# Official-source 4B: prospective JevArena v3 post-key lock

**Status: locked for a separately authorized formal readout; no candidate
FINAL, CSS15, or public-231 predictions or scores have been run.** This record
continues the [signed development result](qwen35-4b-official-base-full-development-result-2026-09-27.md).
The exact protocol is in the adjacent
[`postkey-roster` JSON](qwen35-4b-official-best466-v3-postkey-roster-2026-09-27.json),
validated by [`lock_postkey_qwen4b.py`](../jev_arena/lock_postkey_qwen4b.py).
The final private, immutable lock SHA-256 is
`ab2602d5a1aa7b3284a61f207fec45c7758b2ba879c12680e5fd8308b4fbd494`.
It binds the actual model files, scorer files, three gold-free prompt files,
reused controls, and repeatability and overlap receipts.

## Fixed candidate and calibration

- Direct weight origin: official `Qwen/Qwen3.5-4B-Base` revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`, plus the new Decision
  head and LoRA; exact SELECT-chosen BEST update 466. Its native model plus
  source fingerprint is
  `dc2a8267ec7315a48aa1b2c31e7157e79e50582c4061177bed32ca0ed2735372`.
  Actual native loaded parameters, counted after loading the LoRA source and
  head, are **4,240,848,384**; trainable parameters are 2,632,192.
- The pinned native adapter is
  `decision2-typed-benchmark-adapter-v2-calibrated`, SHA-256
  `33dae46ec1aa812f0a75365e2757a60f17b8d9474fdd15d9a45524437c1b7d56`,
  with max input length 8,192 tokens and no truncation.
- The frozen CAL700 native NLL fit returned Choice temperature
  `1.217267172620981`, Noul `1.044386023961823`, and Score `0.05`.
  The CAL report SHA-256 is
  `a63b5d34d7f6f5d2c21c3573db6e64341b0cd13798f9aa004305572ef4d50652`.
  Score reached the fixed temperature search's lower bound, so potential
  overconfidence on transfer is a material risk. The fit is kept as selected;
  no DEV or formal label will retune it.
- Two separate CAL-derived, gold-free 32-row native inference processes,
  each with the frozen calibration, returned 32/32 valid answers. There were
  zero category changes and zero p99/max probability drift. Their comparison
  receipt SHA-256 is
  `c3223972f3919f25090e9c60b135b6e914956076836d2434b0ab70ef7ace2ee1`.

## Panel and controls

The locked JevArena v3 main panel is typed FINAL 1,600 items / 2,000 answer
slots and CSS15 6,547 items; public JevBench is a separate 231-item panel.
Their gold-free prompt SHA-256 values are pinned in the roster. The
rights-clean v2 TRAIN 7,455 versus all three panels has **zero** repeated row
IDs, raw or normalized state matches, and near matches under the frozen
state-level SimHash/sequence rule. This cannot rule out unseen semantic
paraphrases or base-model pretraining exposure. The private overlap audit
SHA-256 is
`0b7f5324b40e17a1add57d6b7ef345960c971cb5838daa0d1bb7e0e2e65c7351`.

The archived Decision 1.0 Nox 4B and Kev 4B predictions were reused only
after the lock tool checked every row's prompt digest, answer-key set, model
ID, exact weight revision, adapter version, byte hash, and the previous
pre-key prediction audit. All six files passed for the same typed/CSS/public
prompts. Their native adapters remain model-specific; all models will be
scored by the pinned common scorer. Neither the older third-party-start 4B
candidate nor its reported performance is a candidate or comparator in this
lock.

The planned main score is fixed at `100 × sqrt(T × H)`: typed FINAL
four-family macro accuracy `T` and CSS15 task-median macro-F1 `H`. Missing,
invalid, and over-budget answers fail at the full denominator. Paired
uncertainty uses 5,000 bootstrap replicates with seed 20260927. The run is
eligible for release only if its v3 aggregate is at least **3.0 points** above
the same-panel own Nox 1.0 score and it passes package/lineage checks; all
individual regressions must be disclosed. The run is
explicitly **post-key same-panel**: project formal labels were accessed in
earlier experiments. The new candidate was selected on SELECT and a separate
DEV/pilot screen, then locked before its own formal predictions. Formal
results must not be called a never-unsealed blind test.

## Resource and failure record

CAL fitting took 48.880 seconds, or `.01358` one-GPU hours. The two
calibrated repeat processes took 30.200 and 30.179 seconds, or `.01677`
one-GPU hours combined. This phase therefore consumed `.03035 GPU-hours`;
CPU-only overlap and lock checks are excluded. Each GPU container exited 0.

The first **lock command**, before any formal inference, failed closed on a
clerically shortened public-prompt hash in the new roster. It was corrected
to the already frozen 64-character hash from the preexisting overlap audit;
no model, data, prompt, calibration, scoring code, or comparator prediction
changed. The corrected full closure check passed; after the +3.0 release
rule was added to the prospective roster, a second immutable lock was written.
The formatting gate then changed only the lock verifier's source bytes, so a
third immutable lock was written and is the final one named above. The prior
lock receipts are retained; no
selected checkpoint was changed after the development result.

**Next action:** after independent lock review, run the candidate on the
three locked gold-free panels, seal all predictions and manifests before a
scorer opens labels, then compute the fixed v3 and public-231 comparisons.
Actual release still requires formal performance, package materialization,
download/readback and native-output parity. The existing 0.6B full-bundle
packager is size-specific and must not be reused unmodified for this LoRA 4B.
