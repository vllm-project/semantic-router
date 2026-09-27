# VitaminC contrastive evidence as a prospective native Score source

**Decision: HOLD for student training.** This is a CPU-only, TRAIN-only source
screen, not a new model arm, benchmark result, or claim of transfer. It used
**0 GPU-hours** and opened no publisher development/test rows, protected answer
keys, model predictions, or JevArena/JevBench labels. No source row, claim,
document, or private path was committed or placed in the research gist.

## Why this source is different

The [publisher paper](https://aclanthology.org/2021.naacl-main.52/) and
[repository](https://github.com/TalSchuster/VitaminC) describe contrastive
Wikipedia revisions: the same claim is paired with evidence before and after
a factual edit. In the original three-way fact-verification task the label is
`REFUTES`, `NOT ENOUGH INFO`, or `SUPPORTS`. A native System One Score question
can give the ordered **evidence relation** 0/1/2 those precise meanings. This
is not generic ordinal severity or an answer to rule-precedence Score items.
The original paper reports that a claim-only classifier was at 50% on its
real contrastive subset, while noting residual word-overlap artifacts.

The pinned publisher `vitaminc.zip` has SHA-256
`49d82dc1690cbee420d18e2c26f687a7937710bb211845d2571430dfd4dc0337`.
The [DATA_LICENSE](https://github.com/TalSchuster/VitaminC/blob/eb532922b88b199df68ed26afeb58dca5501b52f/DATA_LICENSE)
ties Wikipedia-derived material to the applicable article terms, with CC
BY-SA 3.0 fallback. Synthetic rows also derive from FEVER, so this screen
keeps them **out** of its proposed pilot. The original archive has no
`big_bench_canary` field; the code uses the original publisher archive rather
than an HF mirror. This is English Wikipedia content and provides no
non-English Score coverage. Source text and a derivative TRAIN set are not
being redistributed.

## CPU findings and limits

The aggregate auditor reads only `vitaminc/train.jsonl` inside the pinned
archive and fails closed on the real/synthetic schemas. It found 370,653 TRAIN
rows, 112,426 publisher `case_id` groups, and 22,198 Wikipedia `page` groups.
Class counts are 185,714 supports, 131,958 refutes and 52,981 not-enough-info;
248,953 rows are marked real and 121,700 synthetic. Across all TRAIN rows,
3,155 normalized claim–evidence pairs repeat, so full-corpus admission would
require page- and case-aware deduplication.

A deterministic **screening subset**, not an admitted TRAIN partition, used
only real revisions with paired opposite labels and two distinct evidence
texts for each claim. It selected one case per page from two strata: 128 cases
where the contrast is support/refute and 128 where it is
support/not-enough-info. There are 256 independent pages, 256 cases and 820
rows, with no duplicate normalized claim–evidence pair in the selection.
Selected label counts are 410/255/155 for support/refute/not-enough-info.
Every selected claim has both a supporting and an opposing evidence sentence;
even a claim-memorizing classifier therefore cannot exceed **50%** on these
820 paired rows. This is only a necessary anti-shortcut property. It does not
exclude evidence-only, lexical, article-topic or edit-position shortcuts, and
the selected labels are not balanced across all three levels.

The exact native segmented-option prompt was measured with the already pinned
Decision 1.0 Eos 0.8B tokenizer (tokenizer JSON SHA-256
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`).
All 820 requests fit an 8,192-token cap; token lengths are median 180, p90
212, p99 295 and maximum 576. Thus this source is **short**. It does not
address long-document Score. The 0.8B archived hard-replay substitution had
a fixed 192-row Score block of 201,359 tokens, needing at least 155,681
replacement tokens under its ±1% total-exposure gate. This sampled VitaminC
pilot is plainly not a drop-in replacement for that arm; the result does not
prove an upper bound over all real VitaminC rows or prohibit a separately
preregistered training-budget design.

The [aggregate-only script](../training/data/audit_vitaminc_score.py) and
[five focused tests](../training/data/tests/test_audit_vitaminc_score.py)
are the reproducible CPU method. Script SHA-256 is
`20223687f5514830d145bcf3b58cfa58a1c3efc8a5cfd88078b1fdfd68963481`;
the private native-length aggregate receipt SHA-256 is
`53b0539059d45864de74e4583c519777e160f6c6597113d5471b1034eb862c98`.
The exact source-code mirror matched before execution. Raw receipts and the
publisher archive remain in the private experiment environment.

## Admission and discriminating test

The previously pinned 35-role protected prompt inventory (SHA-256
`c6d3f497b4385ff48817e9a6f98f63baf529b99b27a132a190164d5d729c18c2`)
was unavailable to this run. The script requires its exact manifest and role
file hashes; absent or changed inventory returns HOLD. **No training/eval
overlap clearance has been claimed.** The earlier near-overlap scanner omitted
9,444 long protected leaves, so even a zero on its bounded lexical screen
would not prove semantic separation.

Before any optimizer run: (1) recover the pinned gold-free protected inventory
and quarantine whole source pages on exact/near matches; (2) independently
review blinded source labels against the specific 0/1/2 rubric, especially
whether lack of evidence means the same thing across examples; (3) run
evidence-only and edit-position negative controls; (4) reserve a different
publisher/domain for source-disjoint transfer and a separate untouched
rule/state Score diagnostic; (5) freeze one size-specific equal-token
treatment/control with the same allowed model initialization and native
inference path. Publisher held-out VitaminC revisions would test within-source
generalization, **not** cross-source transfer. [AVeriTeC](https://github.com/MichSchli/AVeriTeC)
is one *unopened* future cross-source candidate, but its original task has a
fourth conflicting-evidence label and a different web-evidence protocol; it
cannot be silently collapsed into this three-level Score rubric.

For this turn the only justified conclusion is that publisher VitaminC is a
promising, genuinely new **contrastive evidence-relation** source with a
measurable anti-claim-only pairing, yet it is not ready for Decision 2.0
training or any SOTA/release claim.
