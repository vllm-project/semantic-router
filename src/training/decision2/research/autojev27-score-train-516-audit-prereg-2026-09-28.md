# AutoJev 27B teacher: complete TRAIN Score coverage audit

**Status:** prospective protocol, frozen before the 516-row GPU pass. This is
an internal teacher-source experiment, not a student, JevArena, JevBench, or
model release score. The earlier 96-group pilot used only 32 of these Score
rows; this audit examines every Score row in the original TRAIN bytes without
sampling or answer-based selection.

## Fixed inputs and CPU audit

| Component | Freeze |
| --- | --- |
| Teacher | `denis-pplx/autojev-27b@6f5b557e037f5edb25c7dc92dbc6553e5a19c015`; clean native source commit `ee63c1515980491a742f0bd0685c8dc5ca1f00c3`; package SHA-256 `d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2` |
| TRAIN | Rights-clean v2 TRAIN 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`; data manifest SHA-256 `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8` |
| Score roster | All 516 Score rows in file order, 516 distinct IDs/input identities and **441 independent groups**; gold-free identity/group/source/family/level-count SHA-256 `a2d759bc47b0a8cc116c47d99762f73a2fde175f58bc551c4a0c58b397828d3b` |
| Source families | stage4 ordinal 279, targeted quantized median 150, stage4 dense table 55, stage3 logic replay 32; source buckets stage4 composition 334, targeted programmatic 150, stage3 replay 32 |
| Languages | English 293, Chinese 223 |
| Native level counts | 3: 102, 4: 59, 5: 211, 6: 58, 7: 47, 8: 39 |
| Gold classes | 0: 109, 1: 106, 2: 111, 3: 87, 4: 62, 5: 27, 6: 9, 7: 5; classes at high levels have small denominators |
| Native context | The pinned AutoJev processor and unmodified prompt renderer yielded 192 minimum, 276 median, 3,293 p95, 5,368 maximum tokens; 0/516 exceed its **8,192-token** input limit. Token-count roster SHA-256 `79ad91e24fb929696aa7364c4ed5e2cadecc50411507e436af3feb377c883b19` |
| Runtime | One freshly confirmed idle GPU on an authorized node; image `decision20-train-fast:host2` SHA-256 `f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`; **15-minute wall cap** including package verification/model load, exactly one run and no changed prompt, calibration, sampler or checkpoint retry |
| Code | [`autojev_score_train_audit.py`](autojev_score_train_audit.py), signed local commit `8e666c87a9242270014b7e0e8ff1b67183097698`, SHA-256 `8bdd2e0e7b4465b9066595642584c89c90841163590b1b65c7dd306b5dce3537` |

The row-level source audit finds all 334 stage4 rows marked as objective
generator, all 32 replay rows tracing to an earlier objective generator, and
all 150 targeted rows marked `targeted-oracle-v1`. This is stronger than
assigning the mixed full-TRAIN manifest's rights to every Score row. The
published teacher card declares Apache-2.0. Retained teacher vectors remain
private and tied to internally generated TRAIN rows; no source text or vectors
are uploaded to a public model repository. Exact source terms still require
rechecking before any dataset redistribution. The existing rights-clean
manifest reports zero exact/near context overlap with SELECT, CAL, CSS and
its named synthetic typed panels, but approximate similarity is not proof of
semantic independence; overlap with the third-party teacher's own training
corpus is unknown. The 516 rows contain only 441 independent groups and four
related generator families, so they cannot substantiate cross-source transfer.

## Locked native inference and outputs

The input to each teacher call is exactly one typed Score question with the
TRAIN state, instructions and ordered live level descriptions. The gold label
never enters the request. The native released `DecisionModel`/`answer`
returns a probability for every level; no chat generation, forced class,
truncation, temperature fit or prompt search is allowed. The code checks the
release tree/source revision, 26,086,635,760 loaded parameters, TRAIN bytes,
roster identity, contiguous 0-based levels, internal source metadata, finite
normalized probabilities and native return type.

The private aggregate schema `decision2-autojev-score-train-audit/1` records
the 516-row denominator, valid/invalid/context-overflow/tie and unique-max
agreement, summed gold probability and half Brier. It reports each metric by
native level count, source family and gold class, plus aggregate input/code/
model/source hashes. Invalid or missing answers stay in the denominator as
failures. The separate private mode-0600 artifact
`decision2-autojev-score-teacher-distributions/1` may contain only record and
input identity, group, source/family, level count and native probability map;
it contains no state, option text or gold label. It is written **only if all
516 native outputs are structurally valid and every source remains internally
generated**. An unsupported native context/candidate limit counts invalid;
an unexpected error aborts the run. If validity, rights or the 15-minute cap
fails, retain the private failure/aggregate, withhold the vector artifact and
mark teacher coverage HOLD. Do not retry with a changed rule after seeing
results.

GPU evidence will be descriptive TRAIN source coverage only. Any future KL
student ablation needs its own prospective registration: eligible official
Qwen or own Decision 1.0 student origin; fixed teacher/softening/KL weight;
same data groups, tokens and update budget as a no-teacher control; independent
SELECT and source-disjoint transfer gates; and calibration/retention checks.
This audit does not authorize training or promotion of a student.

**No 516-row teacher result exists at preregistration.**

## CPU source lock before GPU

The exact signed source mirror reproduced the code SHA in the table and the
CPU dry run reproduced the 516-row roster, all histograms and exact native
token lengths above. The independent full package verifier again passed with
model config SHA-256
`bacbcbb281a53af5ef5cc6c9028601097d155bf981129f18a727219517921dcd`
and runtime source tree SHA-256
`550ccd857350c6771a1de03e4bcba9fb4412247b58bcdc9e8ac03de2ab9641a5`.
No GPU teacher question has been evaluated in this audit. The physical GPU
must be checked again against live processes and containers immediately
before the single run.

## One frozen native run: result and limitations

The sole run completed in 126 wall seconds, **0.0350 GPU-hours**, within the
900-second cap. All 516 native distributions passed type, level-key, finite
mass and normalization checks; there were **zero invalid/overflow**, 20
maximum-probability ties, and **324/516 = 62.79%** unique-max agreement with the
TRAIN oracle. The overall mean gold probability was **0.4553** and mean half
Brier **0.2459**. Ties count as wrong. The aggregate SHA-256 is
`ff703bd206ae9958bd4afc6afbfc90c583cb85ef4a8263ddc6ba7ea7156f59ec`.
The separate 516-row distribution artifact was admitted under the frozen
internal-source/structural rules, remains private mode 0600, and has SHA-256
`072cd519657caaa883eea1f5077789e5bacbf85f8ee20ab44cd562acc317701b`.
It contains no raw text, option descriptions or gold labels. Source, code,
TRAIN and roster hashes match the CPU lock. The task-owned container exited
and the reserved GPU returned to idle memory.

| Native levels | Rows | Correct | Ties | Mean gold probability | Half Brier / row |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 102 | 75 | 2 | 0.6129 | 0.1738 |
| 4 | 59 | 34 | 3 | 0.5225 | 0.2230 |
| 5 | 211 | 153 | 6 | 0.4399 | 0.2357 |
| 6 | 58 | 28 | 4 | 0.3751 | 0.2990 |
| 7 | 47 | 18 | 2 | 0.2999 | 0.3468 |
| 8 | 39 | 16 | 3 | 0.3320 | 0.3243 |

| TRAIN family | Rows | Correct | Ties | Mean gold probability | Half Brier / row |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage4 ordinal | 279 | 157 | 11 | 0.4699 | 0.2509 |
| targeted quantized median | 150 | 120 | 3 | 0.4472 | 0.2195 |
| stage4 dense table | 55 | 19 | 6 | 0.2827 | 0.3554 |
| stage3 logic replay | 32 | 28 | 0 | 0.6636 | 0.1382 |

| Gold class | Rows | Correct | Ties | Mean gold probability | Half Brier / row |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 109 | 96 | 3 | 0.5634 | 0.1623 |
| 1 | 106 | 76 | 7 | 0.5190 | 0.2040 |
| 2 | 111 | 72 | 3 | 0.4968 | 0.2249 |
| 3 | 87 | 39 | 4 | 0.3639 | 0.3057 |
| 4 | 62 | 28 | 1 | 0.3179 | 0.3427 |
| 5 | 27 | 9 | 1 | 0.2969 | 0.3560 |
| 6 | 9 | 2 | 1 | 0.2546 | 0.3988 |
| 7 | 5 | 2 | 0 | 0.3422 | 0.3149 |

The larger audit substantially tempers the earlier 25/32 Score pilot: the
pilot was not representative of the 516 rows. Apparent difficulty increases
with option count and gold level, and dense-table items are a clear weak
family. These factors are confounded by construction, and classes 6–7 have
only 9 and 5 rows. The data are synthetic related families, not independent
real-world transfer. No current 2.0 student has improved from these teacher
vectors. An unconditional KL target across all 516 would also expose the
student to 192 wrong or tied teacher maxima; it is not justified by this
audit alone.

**Next discriminating experiment, not yet executed:** pre-register a
matched-budget official-Qwen or own-1.0 Score student control using the same
TRAIN rows, token count, updates, initializer and native inference. Compare
hard-label CE against a fixed hard-CE plus low-weight teacher-KL arm; keep
hard labels active on every row and pre-specify how wrong/tied teacher vectors
are masked or downweighted using TRAIN labels. Give the control the identical
row schedule and masking weights, replacing KL with CE where needed. Evaluate
SELECT, Score-class calibration, long-input and source-disjoint transfer,
including dense-table and high-level slices, before considering any
JevArena/JevBench or release result. No student arm or promotion follows
automatically from this source audit.

### Post hoc CPU slices from the unchanged private artifact

After the sealed result, the fixed-result
[`autojev_score_train_posthoc.py`](autojev_score_train_posthoc.py) (signed code
commit `30148cbda76fd475cfa10fbebd0ebb744edd0aa8`) joined only the pinned
TRAIN bytes and the two exact private result SHA values. It made **no model
call and used no GPU**. Its private mode-0600 slice receipt SHA-256 is
`1be46a22c97448a172b8de12600f7f5c8110d6bba335555c053f9747076bc642`.
These comparisons were *not* preregistered primary metrics; they are
descriptive follow-up for designing a future control.

| TRAIN slice | Rows | Correct | Ties | Mean gold probability | Half Brier / row |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 levels, all languages | 102 | 75 | 2 | 0.6129 | 0.1738 |
| 4–8 levels, all languages | 414 | 249 | 18 | 0.4165 | 0.2637 |
| English, all levels | 293 | 185 | 9 | 0.4430 | 0.2498 |
| Chinese, all levels | 223 | 139 | 11 | 0.4716 | 0.2409 |
| English, 3 levels | 50 | 35 | 1 | 0.6113 | 0.1753 |
| English, 4–8 levels | 243 | 150 | 8 | 0.4084 | 0.2651 |
| Chinese, 3 levels | 52 | 40 | 1 | 0.6144 | 0.1723 |
| Chinese, 4–8 levels | 171 | 99 | 10 | 0.4281 | 0.2617 |

The 3-level to 4–8-level gap is **73.5% versus 60.1%** agreement. Aggregate
English/Chinese agreement is **63.1%/62.3%**, but their level/source mixes
differ. Both language slices are synthetic TRAIN questions, so this is not a
real-world multilingual transfer finding. The next student ablation should
report language crossed with level count and source family before attributing
any gain to multilingual generalization.
