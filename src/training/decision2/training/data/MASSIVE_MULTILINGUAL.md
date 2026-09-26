# MASSIVE 1.1 multilingual Choice feasibility data

`build_massive_multilingual.py` creates a private, reproducible research candidate
from the official MASSIVE 1.1 archive. It is intentionally **not approved for
training** until the localized semantic review has been signed off. The
builder does not train a model, read MASSIVE TEST labels, or access JevArena
FINAL gold.

The builder verifies the official archive, all seven extracted locale files,
LICENSE, NOTICE, and every frozen protected reference by SHA-256. The
references include Decision 2.0 TRAIN/SELECT/CAL, gold-free DEV/CSS/RQ,
public benchmark prompts, and multilingual DEV prompts. All expected hashes
are source constants. A changed input fails rather than silently changing the
experiment.

```bash
python -m training.data.build_massive_multilingual \
  --archive /private/massive/amazon-massive-dataset-1.1.tar.gz \
  --source-directory /private/massive/1.1 \
  --workspace-root /private/decision20 \
  --output /private/decision20/data/massive_multilingual_feasibility_v1
```

The text-bearing outputs stay private: `train.private.jsonl`,
`dev.private.jsonl`, and `semantic-review.private.jsonl`; the separate
`semantic-review-key.private.jsonl` contains its withheld answers. The manifest contains
hashes/counts and attribution without raw text. Do not commit the outputs or
publish the review packet. Keep MASSIVE's CC BY 4.0 notice and cite both
[MASSIVE](https://aclanthology.org/2023.acl-long.235/) and
[SLURP](https://aclanthology.org/2020.emnlp-main.588/), as requested by the
[official repository](https://github.com/alexa/massive).

The quality rule requires all six non-English localizations to have three
recorded judgments and at least two **jointly** pass intent (1 or 2), grammar
(3 or 4), and target-language presence. The official archive has a few rows
with fewer than three judgments; they fail this rule and are counted in the
manifest. The preregistered quality pool is 9,191 TRAIN and 1,626 DEV source
IDs before cross-benchmark quarantine. The pool is highly imbalanced:
`cooking_query` has only three qualifying TRAIN and one DEV source ID, so a
strict ten-per-intent split is impossible. The builder uses a deterministic
water-fill shortlist and selects 600 TRAIN source IDs (4,200 locale rows) and
200 official DEV IDs (1,400 rows), respecting source supply and exact-context
uniqueness. It quarantines every DEV group with any locale context near a
selected TRAIN context, then selects the 200 DEV groups and rechecks the final
two partitions. TRAIN covers all 60 intents. The quality-filtered official DEV
pool has **no** `audio_volume_other` ID and can cover at most 59; the frozen v1
candidate covers **52** after protected-reference and TRAIN-neighbor
quarantine. Its missing intents are `audio_volume_other`, `audio_volume_up`,
`cooking_query`, `general_greet`, `iot_hue_lightoff`, `iot_hue_lightup`,
`music_settings`, and `transport_taxi`. This candidate cannot measure transfer
across all 60 intents. The manifest records actual per-intent counts. All seven
locale rows share one group ID and one
deterministic six-option intent set and permutation. Available same-scenario
intents supply at least two hard negatives; single-intent domains cannot.
Questions are localized, while option descriptions are intentionally English.

The gold-blind review packet presents one seven-locale source group per intent
with its English source, localized utterance, and option meanings. Reviewers
should record their option choice before the separate key is opened. They should check localized instruction fluency, option meaning, and
known close-intent pairs, especially cooking query/recipe, QA
definition/factoid, light changes, and volume changes. A human review of this
packet is a separate gate; the source localization votes do not establish
that our new dynamic distractors are semantically unambiguous.

The frozen v1 packet failed independent blind semantic review: 10 of 60
English choices disagreed with their source labels, several gold-matching
choices were ambiguous, and four localized rows had translation concerns.
Those 17 unique reviewed source groups must be removed as whole seven-locale
groups from any successor. v1 remains `training_approved=false`. A separate
`build_massive_expanded_review.py` helper makes a gold-blind English packet for
**all 600** selected TRAIN source groups; its key remains separate. Expanded
English adjudication and a bounded native-language review of risk intents are
required before a new immutable candidate can be considered for training.
The `build_massive_review_filtered.py` helper takes the frozen candidate,
expanded-review receipt and a complete independent English verdict JSONL
(`review_id`, `verdict`, `reason` for every source ID). It retains only `pass`
groups, forcibly excludes all 17 initial-review flags, and never refills from
unreviewed rows. It copies the original `LICENSE` and `NOTICE.md` into the new
private output, preserves the official DEV unchanged, and still writes
`training_approved=false`. A separate cross-locale review and independent v2
verdict remain required.

The frozen English-filtered v2 candidate retained 481 of 600 TRAIN source
groups (3,367 seven-locale rows), covering 59 intents. `cooking_query` lost
both candidate groups under the semantic audit; the third quality-qualified
official TRAIN group collided with a protected reference in the near-context
screen, so it was not substituted. Its unchanged official DEV has 200 source
groups, 1,400 rows, and 52 intents. V2 manifest SHA-256 is
`193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10`;
TRAIN SHA-256 is
`55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80`.
The sealed 600-row independent verdict, whole-group exclusion roster and
adjudication receipt have SHA-256
`ce4ca7f106cc323b53d33c383cdb2544e13835baf61ca123f85de0af31cc11ed`,
`99af615dff6bfd2f0d6c10b692013408883d5b67c7b016a3d0f36dc860ad1e21`,
and `105c17915da980e4a7a2b2f747baa05c5fe9c9b4eb1bb3ab3a86ae34bafd4283`.
The v2 manifest is **not** training approval: semantic preservation across
the other six languages and independent v2 review are still pending.

The sealed v2 six-language review has since found only ten of 18 sampled
source groups strictly valid in all six translations. The separate
[`MASSIVE_V3_LOCALE_PILOT.md`](MASSIVE_V3_LOCALE_PILOT.md) preregisters a
small, still-unapproved localized-option follow-up and records the defect
taxonomy. Do not treat that pilot as approval of the remaining v2 TRAIN.

These immutable v1/v2 manifests bind the research worktree's original source
bytes (builder SHA-256
`b91a1651665e002937cddf72ffe52cfc6f861389371b2b1913cfe26c86ee18ca`,
signed research commits `40785db`/`7310120`; v2 filter signed research commit
`24a3ed2`). This repository copy was formatted to the vLLM Semantic Router
Python style; a fresh run produces a **new** manifest hash and requires a new
review. Do not rewrite the frozen candidate or relabel it approved.

For any later private dataset package, carry exact copies of the original
MASSIVE `LICENSE` and `NOTICE.md` beside the rows. The NOTICE retains SLURP
attribution; the manifest's prose alone is not a substitute.

Run the contract tests with:

```bash
python -m unittest training.data.tests.test_massive_multilingual
python -m unittest training.data.tests.test_massive_expanded_review
python -m unittest training.data.tests.test_massive_review_filtered
```

The near-overlap audit uses the existing eight-band SimHash candidate search
and SequenceMatcher threshold 0.94. It detects likely near duplicates but
cannot prove that no paraphrase overlap remains. The exact context and source
group checks are deterministic. Any semantic sign-off or subsequent training
must be recorded separately with the immutable output manifest SHA.
