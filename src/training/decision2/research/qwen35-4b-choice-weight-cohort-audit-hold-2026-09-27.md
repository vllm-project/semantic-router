# Official Qwen 4B: Choice-only reweighting review

**Decision: HOLD; do not allocate a GPU to the proposed 2,120-row Choice-weight arm.**
This is a prospective review of the follow-up suggested in the
[fixed 4B result](qwen35-4b-official-best466-v3-formal-hold-2026-09-27.md),
not a new model score or a revision of that result. No optimizer, protected
answer key, calibration fit, or checkpoint search was used here.

## Frozen cohort audit

The rights-clean v2 TRAIN bytes match SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
The proposed exact-source and `task_type=choice` predicate selects **2,120
unique rows in 2,042 source groups**. The SHA-256 of the sorted JSON array of
their row IDs, serialized with UTF-8 and compact separators, is
`97085c734f170dac4fc7cbc5e2c28d03167e19ede92e18705f8bb8364d00d368`.
The source manifest is SHA-256
`61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
This audit read the fixed private TRAIN, SELECT, CAL, and **gold-free** CSS15
prompts; it did not read CSS15 labels.

| TRAIN source ID | Choice rows | Groups | Original source / split | Per-row original IDs |
| --- | ---: | ---: | --- | ---: |
| `google_goemotions_official_train` | 1,400 | 1,400 | Google GoEmotions, official TRAIN, pinned revision `2adf640a14f11025ae5a9d0ec493b78530d276d3` | 1,400 |
| `legacy:cosmos_qa` | 448 | 389 | COSMOS QA TRAIN | 448 |
| `legacy:snli` | 272 | 253 | SNLI TRAIN | 272 |

Every proposed row has an original record ID in its audit metadata. The
legacy COSMOS and SNLI row metadata do not themselves contain upstream
revision fields; retain their parent source-file hashes and manifest as the
version evidence. The 2,120 IDs, groups and canonical input hashes have zero
detected intersections with SELECT 700 and CAL 700. The fixed rights-clean
builder also recorded exact and approximate context exclusion against the
gold-free CSS panel; that rule does not prove semantic independence or rule
out foundation-model pretraining exposure.

The 120 `css_flute_official_train` Choice rows are **excluded** from the
proposed cohort. FLUTE is a CSS15 evaluation task: its 120 official TRAIN
record IDs and 500 CSS15 test record IDs are distinct, but they still belong
to the same supervised task. Conversely, GoEmotions is a different dataset
from the CSS15 task named `emotion`, yet they share a broad emotion
classification mechanism. “Source-disjoint” here means original dataset and
record isolation, not a claim of cross-mechanism generalization. COSMOS and
SNLI also provide Choice/NLI supervision that may be related to some human
transfer tasks despite distinct datasets.

## Why this arm is a poor next experiment

The completed official-Base BEST466 was **53.218** on the fixed JevArena v3
panel versus own Nox 1.0 **56.470**. Its Choice answer accuracy was already
slightly higher (`.69375` versus `.69000`), while Noul was **9 percentage
points lower**, Score **4.25 points lower**, and the typed exception family
**16 points lower**. CSS15 task-median macro-F1 was lower (`.496323` versus
`.519046`) despite gains on emotion and FLUTE; IBC and wiki politeness had
large losses. Increasing the gradient share of Choice-only examples has no
direct mechanism to repair the main typed deficits and may reduce the
relative Noul/Score share. GoEmotions also supplies 1,400 paired Noul rows
from the same groups, which this treatment would leave at weight 1.0.

SELECT 700 contains a narrow GoEmotions Choice slice and synthetic anchors.
The existing human CSS pilot covers three tasks, not the 15-task distribution
that exposed the failure. A SELECT gain or another three-task pilot gain
would therefore be weak evidence for the release objective. Past own-Nox
Choice-weight screens stopped at zero-step identity; they provide no positive
training result to rescue the hypothesis. The official-Base source passed its
own zero-step admission, but that only makes the treatment executable, not
well targeted.

The formal v3 labels were accessed earlier in the project. Reusing that panel
to choose a weight or checkpoint would be post-key selection, even if the
candidate predictions were sealed before each later score. Any future
release claim needs an independently untouched corroboration source. For
these reasons the proposed `1.0 → 1.5` Choice weight is **not authorized** by
this review; its row audit is retained so the idea is traceable, not treated
as a completed ablation.

## More discriminative next step

Before another 4B optimizer arm, freeze a **source-disjoint development
screen** that samples the observed failure mechanisms: Noul with changed
evidence, nested exceptions, ordinal Score across all three levels, and at
least one human-labelled transfer task outside the current three pilots.
Set source/record/group separation, annotation rules, minimum per-cell counts,
adapter, prompt, score, and stop thresholds before any candidate inference;
seal gold-free predictions before scoring. A new Score data pilot may supply
this only after its independent quality gates pass; quarantined pilot rows
cannot be counted as validation. Do not read v3 again to make this screen.

Then test **initialization** as a single, higher-information 4B axis: the
same rights-clean v2 7,455 rows, native head/LoRA code, 466 updates, token
budget, SELECT-only checkpoint rule, and CAL-only temperature method, but an
official Qwen3.5-4B general post-trained/Instruct starting revision instead
of the already completed official Base control. Pin the exact official model
ID, revision, loaded parameter count, license, source hashes, and zero-step
native behavior in a separate signed preregistration before any update. Each
initialization needs its own repeated zero-step identity check; different
source models are **not** required to return identical predictions. The
candidate must first pass a fixed new development screen with non-collapsing
Noul/Score and higher combined score, then a separately locked post-key v3
same-panel comparison and untouched corroboration. If an official compatible
post-trained 4B source cannot be verified, this arm remains unstarted.

This alternative tests whether broader general instruction priors, rather
than greater Choice loss mass, recover the observed generalization gap. It
reuses the completed Base BEST466 control without retraining it; it does not
claim a causal estimate for a data-weight intervention. No new training is
authorized until the new screen and source preflight are frozen.
