# R116 MASSIVE v4 localized-option pilot preregistration

The independent v3 review found that nearest-key agreement can hide option
scope gaps, parallel meaning drift, and unnatural requests. This v4 pilot tests
a stricter **generation and review gate** on new rights-clean source groups. It
does not revise the frozen v2/v3 candidates, authorize training, or estimate
full-corpus quality.

## Frozen inputs and selection

- Start from the unapproved, English-screened MASSIVE v2 TRAIN: manifest SHA-256
  `193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10`,
  TRAIN SHA-256
  `55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80`.
  Its source license is CC BY 4.0, with Amazon MASSIVE and upstream SLURP
  attribution and the original LICENSE/NOTICE retained.
- Reuse the fixed 41-description, six-language option table, SHA-256
  `8b71e471bef575b15c3e0745ac8c81a031b88e048f5694a9d5fbdac8b09d4a43`.
  This is a coverage constraint, not a claim that every description is
  semantically adequate.
- Exclude all ten v3 source IDs using the sealed v3 key, SHA-256
  `db9e53762879a6ebc604927125836a4e8b19982984b19a600675a63642710e25`.
  No v3 group may reappear and no failed selection may be refilled.
- A feasibility count found 93/481 v2 groups whose six options are all in the
  fixed translation table; 83 remain after v3 exclusion. Select **one** whole
  seven-locale source group for each of these twelve fixed intents:
  `alarm_query`, `alarm_remove`, `audio_volume_down`, `audio_volume_mute`,
  `audio_volume_up`, `email_addcontact`, `email_query`, `email_querycontact`,
  `general_joke`, `iot_hue_lightoff`, `qa_currency`, `qa_maths`.
- Among eligible groups for each intent, choose the lexicographic minimum of
  SHA-256 over the NUL-separated UTF-8 values
  (`decision2-massive-v4-locale-pilot-v1`, intent, source ID), breaking a hash
  tie by source ID. This is fixed before inspecting selected utterances.
  Require seven distinct locales, aligned A–F options and source labels, and
  distinct translated choices. Abort on any lineage, rights, quota, or hash
  mismatch.

## Blind packet and review gate

Freeze 12 groups × six non-English locales = **72 pairs**. Each blind row
contains the English anchor, localized utterance and instruction, English and
localized six-option sets, opaque row/group IDs, and locale. A separate key
contains source IDs, intent and option letter. Keep both private with restricted
permissions and hash receipts; expose the reviewer to the packet alone.

An independent reviewer who did not build the packet must verify its SHA-256
before reading, decide every row's inferred answer and **exact option fit**,
check each translated option for action/scope entailment and distinctness,
compare entities, time, polarity and requested action across all six
utterances, and judge request naturalness and ambiguity. The reviewer records
row and whole-group failure categories and seals a private verdict SHA-256
before receiving the separate key. A nearest option or matching key cannot
override an option-scope, semantic, or wording failure.

Any failure quarantines the **complete** source group; there is no relabeling
or refill within this pilot. The pilot advances to a larger audit only if all
12 groups pass all six locales with exact option fit, preserved parallel
meaning, natural unambiguous wording, distinct translated options, and
inferred/key agreement. Even a clean pilot does not approve the remaining
source groups or model training: `training_approved=false` until a separate
broader gate. No GPU optimizer step, held-out TEST, or FINAL evaluation is in
scope.
