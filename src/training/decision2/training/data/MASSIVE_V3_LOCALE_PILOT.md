# MASSIVE v3 localized-option pilot

This is a **small, preregistered semantic review**, not a TRAIN dataset or an
approval for model optimization. The source is the unapproved English-filtered
MASSIVE 1.1 v2 candidate (manifest SHA-256
`193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10`,
TRAIN SHA-256
`55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80`).
Its complete seven-locale groups were already isolated from protected panels;
the official DEV covers only 52 intents. No official DEV, TEST, public
benchmark, or JevArena FINAL answer is used to select or relabel this pilot.

## What the independent v2 review found

The previous pilot sampled 18 independent source groups, six translations per
group. Its blind verdict was sealed before the key was opened. Of 108 rows,
78 met the reviewer's strict parallel-semantic rule, 92 had a valid exact
option, and only **10 of 18 complete groups** passed all six locales. Every
option description remained English. The 30 strict failures divide into 16
rows without an exact valid option and 14 rows with a usable label but a
material semantic shift, adapted entity, or other failed parallel criterion.
Twelve of the 16 invalid options came from two entire groups: asking to watch
a drama did not uniquely mean a movie, and asking to set a time zone did not
mean converting a stated time between zones. Other failures involved missing
actions, nonsensical or abstract routes, a blog/podcast mismatch, social
posting omission, and entity or purpose drift. These are defects of the
source-to-option mapping or localized utterance; translating option text alone
does not repair them.

The v2 private evidence is frozen at blind packet SHA-256
`5ed2c30dde9ac2def316c070b021b19fed1b0858cbb46abc8a3607638830abb5`,
sealed verdict SHA-256
`a11e8add7ff18f9960c9735e9eab25bcd4c8436dde3b8fd65fc0a021b2658f1c`,
post-key receipt SHA-256
`915e800bdf804d92bb0a046264fc9262a8fe8b5de91ef8929d3e1cd13afbeff1`.
V1 and v2 source files remain immutable.

## Frozen selection and intervention

The v3 builder joins the **exact sealed v2 packet, key, blind verdict and
post-key receipt** by opaque review ID and validates their hashes. A source
group enters only if **every** one of its six localized rows has
`strict_parallel_pass=true`, `label_validity=true`, preserved source intent,
`semantic_tier` of `exact` or `minor`, and a target-language utterance. The
other eight groups are quarantined intact, including their English row. There
are no refills from the other 463 unreviewed source groups. The resulting
pilot has exactly ten groups and **60 translated pairs, ten per locale**.

The only intervention is a manually drafted locale-specific natural-language
description for each of the six options. Each group retains its original
utterances, instructions, option keys and gold option key. All 41 distinct
English option descriptions present in the selected ten groups have separate
Arabic, German, Spanish, French, Japanese and Chinese drafts. The builder
rejects missing translations, changed option keys, duplicate localized
choices, unexpected source hashes, partial groups and any change to the
sealed independent verdict. Draft translations are **not yet validated**;
the independent review must assess fluency and meaning.

The blind packet includes paired English and localized utterances and
instructions, plus English and localized option descriptions with keys. It
does **not** contain a source ID, source intent, label or answer. Those live
in a separate private key. Reviewers should first record the option their
own reading supports, then whether every localized option preserves its
English meaning, whether exactly one option matches the localized request,
and whether the local utterance preserves the English task and entity. A
group with **any** failed locale remains quarantined; neither a nearest-option
guess nor agreement with the hidden key overrides semantic invalidity. A
new blind verdict must be sealed before opening the new key.

```bash
python -m training.data.build_massive_v3_locale_pilot \
  --candidate /private/decision20/data/massive_multilingual_review_filtered_v2 \
  --prior-review /private/decision20/reviews/massive-v2-locale-pilot-v1 \
  --output /private/decision20/reviews/massive-v3-locale-pilot-v1
python -m unittest training.data.tests.test_massive_v3_locale_pilot
```

The official MASSIVE 1.1 source archive SHA-256 is
`4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`.
MASSIVE's CC BY 4.0 LICENSE and NOTICE are copied beside the private packet;
the NOTICE carries upstream SLURP attribution. Cite
[MASSIVE](https://aclanthology.org/2023.acl-long.235/),
[SLURP](https://aclanthology.org/2020.emnlp-main.588/) and the
[official repository](https://github.com/alexa/massive). No raw utterance or
private key belongs in Git, a public gist, or an unrestricted HF dataset.
The small, deliberately preselected pilot cannot certify the other 471 v2
source groups or full multilingual training. `training_approved` stays false.
