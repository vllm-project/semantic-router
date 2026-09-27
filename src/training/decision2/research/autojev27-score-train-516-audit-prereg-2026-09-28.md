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
