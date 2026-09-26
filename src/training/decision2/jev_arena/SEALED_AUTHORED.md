# Sealed authored typed-decision candidate

This is a proposed sixth JevArena axis. Its release panel is generated from
original, deterministic English-language case specifications rather than
copied benchmark questions. It is **not** eligible for the six-axis rank until the panel's
programmatic checks, overlap audit, and blind editorial review all pass.
Opening the already frozen synthetic FINAL or CSS15 gold is never part of
building or reviewing this panel.

## Preregistered design

- Release: 1,296 distinct record instantiations, one scored prompt and one
  question per group. DEV: 144 separately seeded record instantiations. No main group
  or variant can contribute more than once to the item denominator.
- This version is **English-only**. It does not make a multilingual claim.
  Multilingual transfer diagnostics belong to a separately frozen protocol;
  no machine-translated variants enter this score or silently alter JevArena v2.
- Balanced grid: three types (`choice`, `noul`, ordinal `score`) × four
  challenges (`near_distractor`, `long_context`, `insufficient_evidence`,
  `rule_precedence`) × six document domains × 18 release or 2 DEV groups.
  Thus each release type has 432 items, each challenge 324, each domain 216.
- Every question asks about one target record in a multi-record packet. Only
  a uniquely identified FINAL record counts. Drafts, neighboring IDs, older
  notices and unrelated logs are explicit distractors. The instructions
  state how missing evidence is treated, so the key is mechanically decidable.
  Choice applies an ordered four-action policy, with plausible escalation
  actions for incomplete evidence; Noul asks about sufficiency
  of four named checks; Score counts the four verified checks on a 0–4 scale.
- A long-context item contains at least 2,800 whitespace-delimited words;
  token budget and validity are also measured by each native model adapter.
  Near-distractor items use adjacent identifiers. Rule-precedence items
  contain a conflicting draft record with the target identifier.
- The source case spec, visible packet, prompt, and target are separate
  artifacts. A direct spec oracle and an independent parser of the rendered
  packet must agree for every item. The parser also checks unique target
  FINAL rows, option keys, answer range, and source-input digest.
- Release and DEV use different private 256-bit seed files. The manifest
  commits to seed SHA-256 and builder code SHA-256. Rebuilding with the same
  seed bytes and code must reproduce every artifact hash.
- Protected overlap checks use **only** TRAIN/SELECT/CAL and benchmark prompt
  files, never final labels. They check exact canonical inputs, normalized
  state text, and high-similarity token shingles against existing Decision
  data, synthetic DEV, CSS pilot, gold-free CSS15, RQ panels, public JevBench,
  and public Decision Bench v4. Suspect pairs require review or rejection.
- The automated gate also checks all four Choice actions appear, both Noul
  answers appear, all five Score levels appear, and long-context word floor.
  These are necessary checks, not evidence that the prompts read naturally.
- A stratified blind review packet samples two groups from every
  type × challenge × domain cell (144 release examples). An actual reviewer must sign off on clear
  wording, unique answer, realistic distractor role, and correct domain
  framing. Agent-generated or automatic agreement is not represented as
  human approval. Any unchecked cell blocks the release axis.

The authored score report has `score_version: jevarena-authored-score/1`,
`phase`, `items`, `independent_groups`, prompt/target SHA-256, model identity,
`macro_family_accuracy`, and a `quality_gate.status` of `passed` or
`blocked`. It breaks down validity and accuracy by type, challenge, domain,
and the 12 type/challenge families. Missing and invalid predictions are
wrong. The family macro is the unweighted mean of the 12 family accuracies.
The source inventory contains just three semantic operations (ordered
Choice policy, Boolean evidence sufficiency, and 0–4 evidence count),
12 type/challenge mechanisms, and 16 logical-question templates after the
Noul polarity is included. The six domains mainly substitute vocabulary;
four heading styles change wording rather than reasoning. Thus the manifest
field `independent_groups: 1296` means one nonduplicate item per record
group, **not 1,296 independent scenario designs**. The separate
`template_diversity.py` audit verifies these counts, challenge construction,
and a reproducible private 20-item ambiguity-review sample. Even if an
editor approves wording, this remains a narrow synthetic operation axis.

Confidence intervals resample record groups and show a separate
template-cell sensitivity view. The scorer records deterministic 2,000-replicate
scenario and domain-cell bootstrap intervals. The latter resamples six
domain cells within each of the 12 families; neither interval removes
synthetic-template bias.

## Release seal

`build --phase release` writes gold-free `prompts.jsonl` separately from
mode-0600 `targets.jsonl`, source specs, and blind review packet. The manifest
records each SHA-256, the private seed commitment, builder code digest,
language scope, original source origin, counts, and automated audit. The
builder refuses to overwrite an existing panel. A reviewer approval must bind
the review-packet SHA-256, name a real reviewer, and mark all 144 sampled
examples across the 72 cells as clear, uniquely answered, validly distracted,
and appropriately framed. The authoring agent cannot create this attestation
on behalf of a human.

Release scoring refuses to inspect answers while the quality gate is blocked.
It then requires the native prediction receipt and a frozen selection roster
(`jevarena-authored-selection-lock/1`) bound to the panel manifest and each
model's ID, revision, model SHA-256, and adapter SHA-256. The roster's
`selection_basis` must be `development_only`. DEV scoring may run with the
blocked gate for diagnostics, and such scores are never rank eligible.

DEV may inform feasibility and adapter validation. Release targets are held
separately and cannot be used for model training, checkpoint choice,
calibration, or prompt refinement after freezing. If the 1,296 release
groups or review gate cannot be completed at the specified quality, the
panel remains a blocked candidate and is excluded from JevArena v2 rank.
