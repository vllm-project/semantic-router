# MASSIVE v5 TRAIN-only intent-ID pilot: prospective method

Status: **design preregistered; no rows generated or approved at this point**.
This method must be signed before the first v5 source selection. The pilot is a
small test of multilingual intent supervision, not an arena evaluation or a
claim about numerical Score, Noul, compositional reasoning, or transfer.

## Why the previous conversions are blocked

The v1 English audit found 483/600 unambiguous option matches, 52 ambiguous
requests, 57 requests without a valid option, and eight source-label
conflicts. The v2 six-locale blind pilot had 78/108 strict parallel-semantic
passes and only 10/18 complete source groups passing. Its 108/108 agreement
with the nearest option was therefore not a quality result. Translating the
long option descriptions in v3 could not repair source-to-option mismatches or
changes in the human-localized utterance. In v4, blind key agreement was
72/72, but only 32/72 rows simultaneously passed exact option fit, full
parallel meaning, and natural wording; 0/12 six-locale source groups passed.
The v2/v3/v4 packets and verdicts remain frozen and blocked from training.

## Task and fixed selection

Use the official MASSIVE 1.1 archive with SHA-256
`4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`.
Retain only official `partition=train` utterances. A source ID is one lineage
group spanning `en-US`, `ar-SA`, `de-DE`, `es-ES`, `fr-FR`, `ja-JP`, and
`zh-CN`. Keep a group only when the six non-English rows each have at least
two of three source judgments jointly supporting intent (score 1 or 2),
grammar (at least 3/4), and target-language presence. This is an initial
filter, not semantic certification. Exclude every source ID selected into the
old 600-group MASSIVE v1 candidate; that also excludes its v2/v3/v4 subsets.
Do not read official TEST labels or select from official DEV. Check near
overlap against DEV utterances without using their intent labels.

The **only** four classes in this pilot are `alarm_set`, `weather_query`,
`play_music`, and `general_joke`. Their fixed model-facing English codebook is:

| Official intent ID | Short codebook meaning |
| --- | --- |
| `alarm_set` | Set an alarm |
| `weather_query` | Ask about weather |
| `play_music` | Play music |
| `general_joke` | Ask for a joke |

The task is four-way **Choice**: one original human-localized utterance in
the state, one fixed English instruction to choose the best matching official
intent ID, and these four ID/gloss options. The option descriptions are fixed
short codebook entries, never machine translations of long English option
descriptions. This is a deliberately bilingual prompt for non-English rows;
it measures a local utterance mapped to a supplied English taxonomy, not the
naturalness of a translated instruction. All four options appear in every
row; deterministic option order is fixed across locales within a source
group. The source intent gives a provisional key, never an automatic semantic
pass.

The pilot has exactly **three source IDs per intent, 12 independent groups and
84 rows** (12 per locale). Within each intent, rank eligible source IDs by
SHA-256 of NUL-separated UTF-8 components
`decision2-massive-v5-stable-intent-choice-v1`, `select`, intent, source ID;
break ties by source ID. Apply the source-ID exclusions and protected-set
overlap gate before taking the first three. Abort if any stratum has fewer
than three clean groups. Once frozen, any editorial failure quarantines its
whole seven-locale group, with **no refill or relabeling** in v5.

MASSIVE's intent ID alone is not ground truth for whether an utterance uniquely
expresses one of these four actions. An independently qualified reviewer must
confirm source-key fit and the exclusion of the other three choices. The pilot
is unsuitable for Noul: a one-vs-rest false target, a true/false existence
claim, or an explicit none-of-the-above answer cannot be inferred from the
source intent label alone. Creating any Noul rows requires a separate
prospective candidate-list and negative-adjudication protocol. No Score rows
are derivable from this source.

## Isolation, rights, and packet protocol

The source is CC BY 4.0. Keep the original MASSIVE LICENSE and NOTICE, with
the upstream SLURP attribution, beside any private candidate. Preserve the
archive, locale-file, rights-file, builder, input-inventory and output hashes
in a private manifest. Do not commit source utterances or unrestricted data.
The v5 pilot is TRAIN-only and `training_approved=false` throughout this
build. Do not use official DEV/TEST, JevArena FINAL, or any protected answer
key for sample selection, quality editing, model training, or evaluation.

Before selection, compare **utterance-only** normalized exact hashes and
near text (threshold 0.94, with retrieval limitations disclosed) against
the prior MASSIVE candidates, the current TRAIN/SELECT/CAL partitions and
all available gold-free benchmark and authored prompt rosters. Also compare
the complete model-facing prompt after selection. An exact or near hit in any
locale removes the whole source group. Compare candidate locales with one
another for inadvertent duplicate source content and check every ID and
lineage group is split-isolated. Record each protected input path only in a
private manifest; a public note may list roles, counts and SHA-256, never
private infrastructure or raw utterances. If any protected roster required
by the builder is absent or its hash changes, fail closed.

Use fresh random private salts to assign opaque review row/group IDs; do not
encode source IDs, intents or labels in the visible IDs. Freeze (1) private
candidate/key, (2) a gold-free local Choice packet with all 84 prompts, and
(3) a gold-free seven-locale comparison packet, all with SHA-256 and restrictive
permissions. The local packet exposes only independently shuffled row IDs,
not the source-group join token, so an English row cannot be located as the
anchor for a particular localized row. A reviewer must first infer each local
request's answer and option exactness **without** the English parallel anchor,
and seal those judgments. Only then may the reviewer inspect the comparison
packet and seal
parallel meaning, entity/action/time preservation, naturalness, ambiguity
and group verdict. The source-intent key remains sealed until both blind
reviews and their hashes/timestamps are independently verified. Any shortcut,
stereotyped codebook cue, opaque-ID leakage or dangling instruction blocks the
candidate. The reviewer must have qualified understanding of each locale;
unsupported locales remain HOLD, not machine-certified.

Advance only if all 84 rows have one exact option, natural intelligible
wording, independent/key agreement and correct intent semantics, and all 12
groups preserve the requested action and material entities across seven
locales. A single failed group makes the **pilot method HOLD**; failures are
reported by category with source groups private. Even a 12/12 pass permits
only a larger independently reviewed TRAIN audit, not immediate training or
a multilingual performance claim. No model inference, GPU optimization,
HF upload or held-out label opening is part of v5.
