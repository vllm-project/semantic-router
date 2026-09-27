# Score v8.4: CPU pilot quality handoff

**Status: PENDING_INDEPENDENT_BLIND_REVIEW.** The prospective
[v8.4 design](score-v8p4-multisource-pilot-prereg-2026-09-27.md) was signed
before any v8.4 candidate. This is a data-quality pilot and supplies no
training-effect or model-release evidence. No GPU, optimizer, teacher output,
model selection, CAL fitting, or protected benchmark answer was used.

## Frozen candidate and reproducibility

The generator and its five unit tests are in signed commit `e162f3589`.
The audit's ability to read gold-free structured prompt states was fixed in
signed commit `244b05fa5`; the candidate rows and private seed did not change.
The generator SHA-256 is
`e03850b5181e9a34ee41f70d20a8f6fda9b30659e882f3ff37d338f76341fcca`
and the final audit code SHA-256 is
`d4aa1c607cf078f5921c008cff629737c436e4668c59ef95ad7578ce29620262`.
The one private 32-byte seed has digest
`85783a360d6ff74f36fc0fda27816c81875414d35cfe175efc783483f895a036`.
The candidate manifest SHA-256 is
`827c56ad419eccdf4fcfa93cb7a1e5915eeef87e179ff2ad52404dc5e2567dbb`.

The pilot has exactly 30 independent case groups: five per mechanism and
three counterfactual Score levels per group. TRAIN has 18 groups/54 rows;
SELECT has 12 groups/36 rows. The 90 rows include 54 English and 36
Chinese-language rows. Each mechanism contributes 15 rows; levels 0, 1 and 2
are balanced within each role and mechanism. The source documents and
counterfactuals remain private.

The fixed Qwen/Qwen3.8-27B tokenizer revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Counting state,
instruction and options, 72/90 rows exceed 700 tokens and 6/90 exceed
1,500 tokens; the range is 264–1,628 tokens. The counts measure input
length, not evidence that a long input is useful or realistic.

## Automated checks and failure history

The source-level and rendered-text oracles agree for all 90 rows. The audit
checked the frozen parent TRAIN/SELECT/CAL, earlier Score pilot TRAIN and
gold-free SELECT packets, and 28 gold-free protected prompt inventories.
Across 83 candidate-to-source comparisons there were zero flagged exact or
bounded near matches. The within-candidate scan found zero cross-group near
pairs and zero one-field decoders of all three levels. All 30 groups remain
after these mechanical checks; each of the six mechanisms has five groups.
The final audit receipt SHA-256 is
`0c4322f6a67d5b9bbbbe8d78190c49bbe0369d6acaeb2d672893a64646c6408e`.

The first full audit attempt stopped before producing a receipt because some
gold-free protected prompts use structured JSON states. The audit parser was
corrected to normalize those states without opening labels, and the exact
signed code was mirrored before the successful run. Earlier generator
preflight also caught punctuation parsing and stock-arithmetic defects before
the candidate was sealed. The seed was not rerolled, and no case was replaced.
Local `make check` including `make test-training-contracts` passed; the
focused generator tests passed five of five.

## Independent review boundary

The TRAIN gold-free packet SHA-256 is
`18a36d7bbe0d2f4e1dbfde2e3a82c0721b88004e82d33ab13405f0d9bb3dc440`;
the SELECT packet SHA-256 is
`ed5134081ebfeaf6ef4f0554c1a323f3d0eccb5a06820faf70a03b734fccd5d6`.
An isolated private handoff contains only those packets, an aggregate audit
summary, reviewer instructions and a hash manifest. The sealed keys and
labeled source rows are elsewhere. An independent reviewer must solve all
90 items, inspect all 30 counterfactual triplets, review Chinese language
quality, document realism, evidence necessity, ambiguity and shortcut cues,
then seal labels and findings before any oracle comparison.

Automated agreement and overlap checks do not establish semantic
independence or editorial quality. Especially scrutinize whether the long
archived context reads as realistic material rather than repeated padding.
An unresolved contradiction, unsupported decisive fact, systematic shortcut
or insufficient retained-group count makes the pilot **HOLD**. No training arm
may begin from this packet until independent quality review and the separate
matched training-arm preregistration are complete.
