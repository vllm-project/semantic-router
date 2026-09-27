# Official Qwen3.5 4B Posttrained: completed matched arm and development HOLD

**Decision: HOLD.** The authorized official-source arm completed its fixed
466-update budget and its SELECT-selected checkpoint passed independent native
reload. Its one frozen typed DEV/CSS-pilot screen then failed the prospective
promotion gate. This candidate is **not** eligible for CAL model use, JevArena
v3, public JevBench, Hugging Face upload, or a release claim. No alternate
checkpoint was chosen after seeing development labels.

## Identity, execution, and freeze

The sole initialization change against the completed Base control was
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` instead of
`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`.
The [prospective full-arm lock](qwen35-4b-posttrained-full-clean-v2-lock-2026-09-27.md)
and CPU audit bound identical TRAIN/SELECT/CAL native token IDs, the same
rights-clean v2 TRAIN 7,455 rows / 4,194,465 unpadded tokens, one epoch,
466 optimizer updates, LoRA/head architecture, loss, optimizer, seed and
runtime. CAL was parsed and hashed for split isolation only; it was not
tokenized, passed through the model, scored or used for calibration.

The sealed private lock SHA-256 was
`7a3fe34eef8a4845edb71bead0aa12660993d0b81ad3a99184714f69e37809cd`.
The pinned image digest begins `f83b1d10f14d`; the exact 57-file code mirror
digest was `8d7809920cd48f38b9f2980f2e888006ae3f07753714ec666a8cec6fff009118`.
The single authorized container exited zero at 466/466 updates with eight
scheduled checkpoints and 466 finite train loss/gradient events. Its
training receipt SHA-256 is
`fbcd895ba99d8ffe21e667580659f4a4e9531e78b94d250e9e12b2a8a1a81e38`;
`COMPLETE.json` SHA-256 is
`cc0e705d461579dde134629a82f996715240d787ca1c5f959b610aa3f0392d2f`.
Training consumed **1.272628 one-GPU hours**, below the 3.0-hour cap. No
training retry, OOM, nonfinite event, early stop or parameter change occurred.

The fixed SELECT family-macro-accuracy/Brier/earliest selector chose
**BEST448**: 632/700, macro `.895926`, normalized Brier `.074020`.
Update 466 was 630/700, macro `.894259`; the completed Base selected
BEST466 at 636/700, macro `.910278`. All were same-panel valid answers.
The Posttrained `BEST.json` SHA-256 is
`0d1c5055fb324d60b237652e90c6187136170ed4d9a6673ee89666826fa89420`.
A fresh native reload of BEST448 reproduced all 32 fixed SELECT predictions
(Choice 13, Noul 14, Score 5) with zero categorical changes and zero p99/max
probability drift. Its private receipt SHA-256 is
`a7195ec62320b77293f629a0776d7cb487a1d2db9acd64c933a7ce02106bcf3b`.

## One sealed development readout

BEST448 was collected once, uncalibrated, with the same native typed adapter
and 8,192-token cap as the completed Base control. The gold-free typed DEV
1,600 and CSS pilot 1,430 input SHA-256 values were
`a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`
and `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`.
The candidate's two manifests bind the same model fingerprint
`3d8036263a9153a96eb53f034874f9d6cd8d63a930392804d25ffd427d0eb713`,
adapter fingerprint
`8e8115b09a7e2442974185c75a343f08a1d043deb749f65f28db9c446fc30803`,
temperature 1.0 and 3,030/3,030 valid, nontruncated answers. The private
prediction seal SHA-256 is
`0742a4417bd631be67dc42761ce3fff3bfbc494b89cbcd9946aea6e38773883c`.
The DEV and CSS prediction SHA-256 values are
`7874592ea2dcbf4c868a29a98f3288469a96b32602773dd36dd0df094948754b`
and `2c72bcc40079ce0e6d8450e74479dcd6706b5c8aa16b29aaf85d500a1e5998af`.
The unchanged typed/CSS scorer SHA-256 values match the Base screen:
`d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`
and `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`.
The private score-report SHA-256 values are
`491480c87299bde3b4e2bc40a2047dcb68866921c9fd71ff3a657d47768fd5bd`
and `a3888a42a91a8e8367da073b4c2875832f0b3d0985917d8d09beb554beab45bc`.

| Development metric | Official Base BEST466 | Posttrained BEST448 | Change |
| --- | ---: | ---: | ---: |
| Typed four-family macro `T` | .820625 | .717500 | **−.103125** |
| Typed correct / 1,600 | 1,313 | 1,148 | −165 |
| Choice / 800 | 720 | 745 | +25 |
| Noul / 400 | 230 | 228 | −2 |
| Score / 400 | 363 | 175 | **−188** |
| Typed normalized Brier ↓ | .119841 | .198853 | +.079012 |
| Typed ECE10 ↓ | .023122 | .145085 | +.121963 |
| CSS pilot task-median macro-F1 `H` | .520209 | .512524 | −.007685 |
| CSS pilot micro correct / 1,430 | 734 | 730 | −4 |
| Frozen proxy `100 × sqrt(T × H)` | 65.33730 | 60.64123 | **−4.69606** |

CSS pilot macro-F1 by task was discourse `.520209 → .512524`, implicit
hate `.364611 → .362970`, and SemEval stance `.638807 → .665938`.
The promotion rule required the Base proxy +2.0, Noul at least 230/400,
Score at least 355/400, and CSS H at least .520209, plus other floors. It
fails multiple independent conditions. Typed DEV and CSS pilot are
**development evidence**, not JevArena v3 or a release comparison.

### The Score failure and selector blind spot

The candidate predicted ordinal Score levels 0/1/2 in **304/0/96** of 400
DEV questions; the Base predicted **101/95/204**. The candidate assigned
gold-level-1 questions mean probability `.0342` to level 1 versus Base
`.5664`, and mapped **118** gold-level-2 cases to level 0 versus Base
**11**. Score Brier rose `.091505 → .439844`, ECE10
`.154857 → .462284`, and expected-value MAE `.262087 → .691981`.
This is a severe middle-level and high-level failure, despite the
candidate's Choice improvement.

An aggregate audit of frozen rights-clean v2 found only **516 Score TRAIN**
rows, of which **102 have exactly three levels** (label 0/1/2 counts
33/33/36). Every one of the **90 SELECT Score** rows has **five levels**;
the selector contains zero three-level cases. SELECT 86/90 at BEST448
therefore did not test the same cardinality as the three-level DEV Score
family. This coverage mismatch is directly observed. The contrast does
not by itself prove that official Posttrained initialization caused the
collapse, nor that a new Score corpus would cure cross-source transfer.

The first DEV collector exited zero. A transient GPU0-busy telemetry check
then stopped the controller **before any CSS container existed**. After a
fresh idle check, the first CSS collector exited zero; DEV was not rerun.
The two readouts used `.040345 + .034355 = .074700` one-GPU hours; reload
used about `.006830`. A host-side score command initially named the
container-only `/usr/bin/python` and failed before either scorer executed
or wrote a score. The corrected host `python3` command scored the sealed
predictions once. Both operator incidents and their unchanged inputs are
retained privately. Total candidate training, reload and development
readout used **1.354158 one-GPU hours**. A machine-readable private HOLD
receipt binds the source, model, prediction seal, both scorers, failed
gates, incidents and resource use; its SHA-256 is
`dbfc4cf0120a23d0e006ebd06eac726a6e4132c887c9c081ce61f86ab08bf092`.

## Next experiment: CPU feasibility first, no GPU authorization

The fastest relevant hypothesis is **three-level Score coverage with
source-independent transfer**, starting from the same official **Base**
revision as the completed Base control, not from this held Posttrained
checkpoint. Base retained far more three-level competence and was closer
to the release threshold. A candidate intervention would preserve the
same model/head, native adapter, seed, objective, update schedule and
approximately 4.194M native TRAIN-token budget. It would retain every
human-labeled Choice source and the existing Noul/Score rows. Of 3,908
Choice TRAIN rows, 2,240 are in the four identified GoEmotions, COSMOS,
SNLI and FLUTE human-label families; the remaining 1,668 are **only an
upper bound** on replacement candidates until their original sources are
audited. Replace only an eligible, prospectively identified synthetic subset
with new complete, program-oracle-verified three-level Score groups.
An initial aggregate CPU inventory found 976 unique-group Choice rows in
six non-replay `stage4_arithmetic/automaton/dense_table/registers/relations/scope`
families within that upper bound. This establishes a potential row-count
pool, **not** source-rights, difficulty, token-budget or replacement
eligibility. The exact row IDs remain unselected.
The exact replacement IDs and raw/padded token exposure must be frozen
before training; if a matched 7,455-row, 466-update budget cannot be
constructed, the arm remains HOLD. This is a **data-only** causal contrast
against the already completed Base control; do not rerun that control.

Build a separate gold-free three-level **SELECT3 gate** from independent
source groups and rendering families. Use it once on the SELECT-selected
BEST, **not** to reselect among old or new checkpoints; the original SELECT
continues to fix BEST so the control is comparable. Before seeing any
model prediction, fix the new gate's number of independent groups, balanced
0/1/2 variants, semantic mechanisms, answer and shortcut audits, blind
review, pass margin and paired-group uncertainty rule. Keep full triplets
together across all splits. Score both the completed Base control and the
single new arm once on this same gate. The already-consumed Score SELECT r2,
held Score TRAIN v6, typed DEV labels, CSS pilot labels, formal labels and
public items cannot be relabeled as a fresh gate or reused as newly
independent training material.

CPU feasibility must first enumerate source IDs/groups, source rights and
licensing, per-row native token lengths, matched synthetic-Choice replacement
pool, exact/near/semantic overlap against TRAIN/SELECT/CAL, typed DEV,
CSS pilot, available formal prompt rosters and public panels, then verify
complete 0/1/2 oracle triplets and shallow-feature resistance. Preserve
quarantined rows and reasons. The prior Score v1–v5 shortcut failures, v6
limited English pilot and consumed r2 selector are evidence that a large
synthetic count alone is insufficient. No new corpus, selector or training
lock is admitted by this note. Even if such an arm passes SELECT3 and one
development readout, the project has already accessed v3 keys; a subsequent
release requires prospectively frozen same-panel reporting and genuinely
independent corroboration. It also needs separate Noul/exception and CSS
transfer improvements; fixing Score alone is not a family release claim.
