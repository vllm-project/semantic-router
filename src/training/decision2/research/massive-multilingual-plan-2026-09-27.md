# Human-localized multilingual data feasibility and preregistration

The current rights-clean v2 TRAIN7,455 has declared EN6,085/ZH1,370 and no
declared Arabic, German, Spanish, French or Japanese rows. The established
JevArena main panels are overwhelmingly English, so they cannot support a
multilingual release claim. The previous M2M100 TRAIN-only translation smoke
failed quality, especially ordinal Score, and remains `training_approved=false`.

I audited the [official MASSIVE 1.1 release](https://github.com/alexa/massive),
the [dataset card](https://huggingface.co/datasets/AmazonScience/massive) and
the [ACL paper](https://aclanthology.org/2023.acl-long.235/). It consists of
parallel human-localized assistant utterances with intent labels; the dataset
license is CC BY 4.0, with attribution to Amazon and the underlying SLURP
source in its NOTICE. This is useful **Choice** supervision and does not solve
Score or long-context reasoning. The official source archive SHA-256 is
`4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`.

The audited locales are en-US, ar-SA, de-DE, es-ES, fr-FR, ja-JP, zh-CN.
Each contains 16,521 source IDs, with 11,514 TRAIN, 2,033 DEV and 2,974
TEST; all six non-English files agree with English on ID, split and intent
for all 16,521 rows. Their extracted file SHA-256 values in this order are
`c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e`,
`b604b44d3bb94e4f71f64d041229e16ceb42d7b89385df7c82873d2514737186`,
`5e09cc550b38d37faf002e0ff42103acd330906e23032689e8f071e1cf3ff621`,
`310462a79fa181ff83c643a8d356c7b8155fd37a25e80a77ba3ca9b29305c4a5`,
`f9bf3db170ad415b389e4c9594dd0f8f80c38188143e05cc4459a6fa7df7cf49`,
`c22df382db6aa4a23dd1e7f62a2ac8f01c6158865771ad25201696be7201ab79`,
`992bf0bef3d678f08c27e514739bc851163e8f40f530bfb4d5970a2c24408ace`.

Precommitted filter: require at least two of three localization judgments to
mark intent as matching or reasonable (`intent_score` 1/2), grammar at least
3/4, and target language present; retain a source ID only when **all six**
non-English locales pass. This leaves 9,191 TRAIN source IDs, covering all
60 intents; 1,626 official DEV IDs and 2,385 official TEST IDs also meet the
same filter, though TEST labels will not be used for development. Arabic is
the limiting locale at 9,789/11,514 quality-passing TRAIN rows. These
counts precede cross-benchmark overlap quarantine.

Next controlled data arm: select at most 600 source TRAIN IDs, stratified to
cover 60 intents, keeping all seven parallel locale rows in one lineage
group. Produce dynamic six-option Choice examples with two or more same-
scenario hard negatives where possible, deterministic option permutation,
and localized question instructions; English option descriptions remain an
explicit cross-lingual limitation. Reserve about 200 separate official DEV
source IDs for model development. Exclude exact and approximate context
overlap with all protected TRAIN/SELECT/CAL and gold-free evaluation prompts
before training; preserve the original official TEST untouched. Run a blind
semantic review of option meaning and localized instructions before any GPU
arm. If labels or language judgments are ambiguous, drop the group instead
of treating it as clean supervision.

This arm may be mixed with rights-clean v2 only after its private manifest,
rights notice, group isolation, token counts and overlap receipts are fixed.
First compare source and clean-v2 candidates on a group-disjoint multilingual
DEV, with paired English regression checks; do not tune a JevArena release
score on the public MASSIVE TEST. No multilingual gain is claimed yet.

## R1 private feasibility candidate, 2026-09-27

The first production build failed closed at its final TRAIN/DEV near-context gate
(133 near-matching selected DEV rows). The source-supply-aware builder then
quarantined every DEV source ID with any locale near a selected TRAIN context
before DEV sampling, without lowering the 0.94 threshold. This yielded an
immutable private review candidate under `data/massive_multilingual_feasibility_v1/`.
No GPU training used these rows. `training_approved=false` pending an
independent gold-blind semantic review and source-ID audit.

- Source code: signed commits `40785db`, `7310120`; builder SHA-256
  `b91a1651665e002937cddf72ffe52cfc6f861389371b2b1913cfe26c86ee18ca`;
  ten focused CPU contract tests pass.
- Official MASSIVE 1.1 archive SHA-256
  `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`;
  seven locale source files, LICENSE, NOTICE and all protected input hashes are
  pinned in the builder and candidate manifest. Source/SLURP attribution and
  CC BY 4.0 notice are mandatory for any later private dataset package.
- Candidate manifest SHA-256
  `861956422e90a6dd6e064b0ce9fc60df0344d184aeb0a8745931e71f7faeba62`.
  TRAIN `train.private.jsonl`: 600 source IDs x 7 locales = 4,200 rows, SHA
  `4c57dac9d5dd39cf3e0920cda7c1e975f5e2dacf46379e7232797efb3ae2d9ba`.
  Official DEV `dev.private.jsonl`: 200 source IDs x 7 locales = 1,400 rows,
  SHA `f86b47dc200cf123c06902dbb8583ad21239bf543c802c651d59be8df4c35fb9`.
- Blind review packet `semantic-review.private.jsonl`: 60 TRAIN source groups,
  all seven locales, 420 rows; SHA
  `28c18f70ab332dc53983804cf066f7b242f5ec5f6f8e3c2e43f866fb3f833c95`.
  Separate sealed answer key SHA
  `cd4ff895b48b6aafb496d5d33e5d3e01541350207ec76f2f994b469ec08a178c`.
  The independent reviewer received both paths and will record blind choices
  before opening the key.
- Protected-reference screening: 1,181 shortlisted TRAIN and 575 shortlisted
  DEV groups were audited against frozen train/selection/calibration, gold-free
  development/transfer/pressure/public/multilingual prompts; 289 groups
  quarantined. Relative to selected TRAIN, 218 of 480 remaining DEV candidate
  groups were further quarantined (16 exact and 327 near-triggered locale rows).
  Final TRAIN/DEV source-group overlap, exact context and indexed near context
  were zero. SimHash candidate retrieval is approximate, so this is not proof
  against every paraphrase.
- TRAIN covers 60 intents; DEV covers only **52** after strict isolation. The
  missing intents are `audio_volume_other`, `audio_volume_up`, `cooking_query`,
  `general_greet`, `iot_hue_lightoff`, `iot_hue_lightup`, `music_settings`, and
  `transport_taxi`. The filtered official DEV pool lacked
  `audio_volume_other` even before quarantine. Do not report a 59- or 60-intent
  DEV result for this candidate. TRAIN source-intent minimum is two groups;
  some DEV intents have one group.
- Full private output parser revalidated 4,200/1,400 rows, 600/200 isolated
  seven-locale groups, consistent answer/options per group, localization
  votes >=2, output directory mode 0700, and unapproved status. Exact option
  position histograms are in the private audit log. No original MASSIVE TEST
  labels or JevArena FINAL gold were read.

## R2 independent semantic verdict, 2026-09-27

An independent reviewer fixed all blind judgments before opening the separate
answer key. The private blind receipt SHA-256 is
`62f2aad82fa61c0de432de09d09ae56e48d4d10ec860cdaf9824616a849b30b8`;
the subsequent private key-adjudication receipt SHA-256 is
`78f5d2529587b4ae6a39e692bb8fafc81f21f7356f4213c22a008efcd4b55056`.
These receipts contain the source-ID-level findings and are not public data.

**Decision: block the v1 candidate from training.** The reviewer inspected the
gold-blind English utterance and dynamic six-option set for one source group in
each of the 60 intents, plus all seven locales for 15 risk-focused groups
(105 localized rows). Only 50 of 60 blind English choices matched the source
intent key. Some disagreements are clear defects: an utterance asks for recipe
steps but carries the `cooking_query` label; a time conversion request carries
`datetime_query`; some questions have no matching offered action. Other rows
with matching keys remain ambiguous because they lack a requested action,
request two actions, or have competing plausible options. The reviewer also
found cross-locale meaning drift despite passing localization votes. All 420
key labels matched the serialized TRAIN answers, isolating the issue to
semantic quality rather than packaging.

The sampled review cannot certify the remaining 540 TRAIN source groups or
native-speaker quality in every locale. Keep `training_approved=false`,
quarantine affected complete seven-locale groups, audit all 600 English
utterances against their actual option sets, and expand native-language review
before a revised frozen candidate or any GPU training. The 52-intent DEV
coverage limitation also remains. No multilingual model gain is claimed.

## R3 complete English blind audit, 2026-09-27

A separate gold-blind packet presented all 600 selected English TRAIN
utterances with their actual dynamic six-option sets, without source IDs or
labels. Packet SHA-256:
`a490ff6dd453ba572742ef82b4174f188c5f61a119e655de66a69c9156fd8c55`.
The reviewer inspected and recorded all 600 choices before opening the
separate key; sealed blind decisions SHA-256:
`57ad5324fdac94a5427fd48d74f42a87d6630557ffb3394395838ecdab6c5535`.

Post-key, the English-only verdict is 483 pass, 52 ambiguous, 57 absent option,
and eight clear source-label conflicts. The exact 600-row private verdict
receipt SHA-256 is
`ce4ca7f106cc323b53d33c383cdb2544e13835baf61ca123f85de0af31cc11ed`.
Unioning every non-pass source group with the earlier independent risk flags
quarantines 119 complete seven-locale groups; the private exclusion receipt
SHA-256 is
`99af615dff6bfd2f0d6c10b692013408883d5b67c7b016a3d0f36dc860ad1e21`.
This leaves 481 English-passing groups over 59 of the 60 source intents: the
scarce cooking-intent class has no clean selected group after review. These
figures are not multilingual quality rates. The prior seven-locale risk sample
found translation drift, and the other localized rows have not received
independent native-language approval. `training_approved=false`; no training
used these rows. A new candidate must preserve the exclusions, restore any
desired intent coverage from separately reviewed source groups, and pass a
separate localization gate.

## R2 independent semantic block and expanded review, 2026-09-27

Independent reviewer sealed gold-blind choices before opening the key for the
v1 one-per-intent packet. Blind receipt SHA-256
`62f2aad82fa61c0de432de09d09ae56e48d4d10ec860cdaf9824616a849b30b8`;
post-key verdict SHA-256
`78f5d2529587b4ae6a39e692bb8fafc81f21f7356f4213c22a008efcd4b55056`.
Verdict: **BLOCK_FOR_TRAINING**. Fifty of 60 English blind choices matched
source labels; ten mismatched, some gold-matching examples still had ambiguous
or absent action semantics, and four translations raised meaning/target
concerns. Seventeen unique reviewed source IDs were flagged. Mechanical
seven-locale labels agreed with the source record for all 420 reviewed rows;
this does not establish semantic validity. Full reports stay private under
`reviews/massive-independent/`. The frozen v1 candidate remains untouched and
`training_approved=false`.

A separate private gold-blind packet for **all 600 English TRAIN groups** is
now frozen under `reviews/massive-expanded-english-all600-v1/`, bound to v1
manifest SHA `86195642...` and TRAIN SHA `4c57dac9...`. Blind packet SHA
`a490ff6dd453ba572742ef82b4174f188c5f61a119e655de66a69c9156fd8c55`;
separate answer key SHA
`05e9affa7a49548a357c2b112ea183b658fa0b4da0fc4d1f8b1deed8561a5367`;
receipt SHA
`ce4f6de16fe0c54243136c02c5f1451b404687026030b4c1c2178d888ceeeb0b`.
Directory/file permissions are 0700/0600. The independent reviewer will
adjudicate English before opening the key; high-risk locale checks and any
source-group quarantine follow. Expanded review helper/test are in signed
commit `005bd5b` (11 focused CPU tests pass). No model training is approved.

## R4 English-filtered v2 and implementation port, 2026-09-27

The complete independent all-600 English verdict, 119-source-group exclusion
roster, and post-key review receipt were verified against SHA-256
`ce4ca7f106cc323b53d33c383cdb2544e13835baf61ca123f85de0af31cc11ed`,
`99af615dff6bfd2f0d6c10b692013408883d5b67c7b016a3d0f36dc860ad1e21`,
and `105c17915da980e4a7a2b2f747baa05c5fe9c9b4eb1bb3ab3a86ae34bafd4283`.
The fail-closed filter matched every one of 600 review IDs, checked the
independent exclusion set and count, then excluded all non-pass and prior
flagged groups without refilling from unreviewed rows. Source research filter
signed commit `24a3ed2`, SHA-256
`f54414ed1f5febb26693e3ea470d03fd7f3c4e1d4e15df0675e9ce492fc09c82`.

The immutable private v2 candidate lives at
`data/massive_multilingual_review_filtered_v2/`. Manifest SHA-256
`193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10`;
TRAIN SHA-256
`55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80`,
481 full seven-locale source groups = 3,367 rows and 59 intents. Official DEV
unchanged at 200 groups = 1,400 rows and 52 intents. Original source
LICENSE/NOTICE are copied with exact SHA and private 0700/0600 permissions.
`cooking_query` has no retained TRAIN group: both selected examples failed
blind semantic review; the only other quality-qualified source group triggered
a near-context overlap with a protected reference. Sparse retained intents
include `audio_volume_other`, `general_quirky`, and `transport_query` at two
source groups each. This is a feasibility candidate, not a 60-intent model.
It remains `training_approved=false`: six-language semantic preservation and
independent v2 review are pending; **no GPU training was started**.

Audited builders, review filter, three focused test modules and non-raw-text
audit docs were ported to the isolated vLLM Semantic Router Decision 2.0
branch in signed commit `a68f801bd5defd26efb403d754699838ed2ebe0d`.
Repository `make impact ENV=cpu` selected the training domain;
`make check` including `make test-training-contracts`, 12 new focused tests,
Ruff, Black, pre-commit security/structure checks, and staged-file codespell
passed. The formatted port has different source-file bytes from the original
research commits; frozen v1/v2 manifests retain their original builder
receipts and must not be silently regenerated under the port.

## R5 preregistered six-language pilot before packet construction, 2026-09-27

Signed vLLM Semantic Router commit `4cb095bb0b3bab52ffbf6d6ec0c9c92e15383e4f`
freezes the builder, contract tests, and public-safe sampling rules **before**
any new localized semantic verdict or packet was inspected. Builder SHA-256
`3f990ace42931719646151e2af2afebbcc0935fc92de15c8aa8953d110fa00fd`;
`make impact ENV=cpu` selected training, `make check` including the full
training contracts and focused pilot tests passed.

Input is the still-unapproved private v2 candidate manifest SHA
`193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10`,
TRAIN SHA `55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80`.
Precommitted seed `decision2-massive-v2-locale-pilot-18-groups-v1`. Select one
source ID by minimum SHA256(seed,intent,source_id) in each of 18 fixed intents;
all six non-English locales per group give 108 paired English/translated
items, 18 per locale. Sparse intents:
`audio_volume_other/general_quirky/transport_query/music_dislikeness/recommendation_movies/iot_hue_lighton`;
prior risk:
`datetime_convert/qa_definition/play_podcasts/social_post/transport_ticket/recommendation_events`;
broad scenarios:
`alarm_set/email_sendemail/weather_query/lists_query/takeaway_order/iot_hue_lightoff`.
All selected groups are independent. The packet hides source ID, intent,
selection stratum and gold; the private key is separate. `cooking_query` is
absent in v2 and cannot be sampled. No outcome-dependent substitution,
training, or FINAL use is permitted. This 18-group pilot cannot certify all
481 retained groups even if every reviewed translation passes.

## R6 immutable six-language pilot packet, 2026-09-27

Built *after* the R5 signed preregistration from the exact committed builder
SHA `3f990ace42931719646151e2af2afebbcc0935fc92de15c8aa8953d110fa00fd`.
Private packet directory `reviews/massive-v2-locale-pilot-v1/` is immutable
and permissioned 0700; files are 0600. Blind reviewer packet SHA-256
`5ed2c30dde9ac2def316c070b021b19fed1b0858cbb46abc8a3607638830abb5`;
separate unopened key SHA-256
`9d67a07587bf28e5bc17ca097d5bce64926523ca15d0d71a97142f303af7379f`;
receipt SHA-256
`a996345ec056491ce2c33c9c2f20a3b63e2e24fad1d0705904efe2c3a7c4ef25`.
The packet has 108 opaque review IDs, 18 independent seven-locale source
groups, 18 translated pairs in each of ar-SA/de-DE/es-ES/fr-FR/ja-JP/zh-CN.
Its top-level fields are only review ID, opaque group token, locale, paired
English/localized utterances and instructions, and the exact six shared option
descriptions. Source IDs, intents, labels, selection strata and answer keys
were absent on inspection. Original MASSIVE LICENSE/NOTICE SHA-256 remain
`c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435`/
`b90534ccd20c6f0e1e5239567af0d150496339542b75a15bfbc3e1e737593ddb`.
The source v2 data is unchanged, `training_approved=false`, no independent
language verdict has been inspected, no model training and no FINAL use.

## R7 independent 108-row six-language blind verdict, 2026-09-27

An independent reviewer verified the frozen prompt-only packet SHA-256
`5ed2c30dde9ac2def316c070b021b19fed1b0858cbb46abc8a3607638830abb5`
and manually compared all 18 source groups × six translated utterances against
paired English utterances and the six shared options. The row-level verdict was
**sealed before answer-key exposure**: private verdict SHA-256
`a11e8add7ff18f9960c9735e9eab25bcd4c8436dde3b8fd65fc0a021b2658f1c`,
manifest SHA-256
`a9be0cec6295f75e3315707561927d450cab55e6bfab54520dd4174b8d8900ba`.
No source ID, intent, target label, selection stratum, or gold was present in
reviewer inputs. The reviewer inferred a nearest option independently and
marked it invalid when a unique exact option was absent.

| Locale | Strict parallel pass / 18 | Unique valid option / 18 | Utterance localized / 18 |
|---|---:|---:|---:|
| Arabic | 11 | 15 | 17 |
| German | 13 | 16 | 18 |
| Spanish | 12 | 16 | 18 |
| French | 15 | 15 | 18 |
| Japanese | 14 | 16 | 17 |
| Chinese | 13 | 14 | 18 |

The strict pass requires exact or minor semantic equivalence *and* a unique
valid label. Named entities or routes were changed in several parallel
translations without a declared adaptation rule; some pairs changed the
requested action or medium. Two entire sampled source groups have no exact
matching option for their request, even before looking at any hidden key.
All 108 localized instructions are translated, but the six option descriptions
remain English in every row. This can be a deliberate bilingual label format;
it cannot support a claim that the entire prompt is localized. Minor code
switching appears in two non-English utterances. The 18-group pilot is a
stress sample, not a representative accuracy estimate for all 481 retained
source groups.

**Gate: BLOCK full multilingual training.** Correct option coverage, define
entity-adaptation and prompt-language policies, quarantine or rewrite failing
seven-locale groups, and repeat independent review on a separately frozen
packet. A limited pilot is also blocked on the current 108-row evidence; a
repaired small pilot may be considered after a fresh blind review. The
underlying v2 candidate remains `training_approved=false`, unchanged, and
no multilingual optimizer step was run. Post-key label agreement is pending;
this blind verdict must remain immutable.

## R8 sealed post-key join of the six-language pilot, 2026-09-27

Only after the R7 blind verdict was sealed did the reviewer open the separate
private key, verifying its frozen SHA-256
`9d67a07587bf28e5bc17ca097d5bce64926523ca15d0d71a97142f303af7379f`
and re-verifying the unchanged packet, verdict, and blind-manifest hashes.
The review-ID and opaque-group join was one to one: 108 rows in 18 independent
source groups, six locales each. Private post-key row receipt SHA-256
`f2888be0091041db9fa7029f3f5b768507c2bc309c555f7da6d2d98045d889a8`;
post-key manifest SHA-256
`915e800bdf804d92bb0a046264fc9262a8fe8b5de91ef8929d3e1cd13afbeff1`.
No blinded judgment was changed after seeing the key.

The independently inferred *nearest* option matched the source key for all
108 rows, 18/18 per locale. This key agreement coexists with only **92/108**
rows having an unambiguous exact option under the blind rubric and only
**78/108** passing the strict parallel criterion. Semantic tiers were
69 exact, 16 minor, 10 entity-adapted without a stated policy, 9 material
meaning changes, and 4 unusable translations. Only **10/18** source groups
passed across all six translated locales, so row-wise salvage cannot preserve
all parallel groups. All six source keys for each of two sampled tasks matched
the reviewer's nearest category even though the exact request was broader than
or different from the available action label. Blind nearest-key agreement must
therefore not be presented as translation fidelity.

**Decision remains BLOCK_FOR_TRAINING for v2.** A corrected, versioned source
requires a stated entity-localization policy; repaired or quarantined groups
for missing-option, meaning-change and nonsensical cases; and clarity on whether
English option descriptions are a deliberate bilingual input format. Freeze a
new packet and get an independent gold-blind review before any optimizer step.
The current source v2 files and `training_approved=false` manifest remain
untouched. The risk-enriched 18-group pilot does not estimate full-corpus
quality; it establishes concrete failure modes and a failed gate.
