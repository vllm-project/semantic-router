# Decision 2.0 pilot data

`build_pilot.py` makes controlled, hard-label training arms from an existing
TRAIN JSONL plus three new programmatically checked text families. It does not
import or read benchmark generators or benchmark dev/final samples as training
data. SELECT and CAL can be separate flattened audit-only inputs or reserved
by source group from the caller-supplied legacy pool. They never appear in an arm.

## New families

| Family | Oracle and skill | Prompt form |
| --- | --- | --- |
| `pilot_string_composition` | Sequential symbol transformations | Badge printer prose with two or three ordered actions |
| `pilot_narrative_reading` | Actor, item and pronoun referent from a short note | Archive handoff prose; Choice and Noul |
| `pilot_open_world_abstention` | Direct card-field lookup with half of target attributes absent | Partial museum cards; `not stated` is a genuine option |

These mechanisms and wording differ from the benchmark's JSON record gates,
priority rules, set comparisons, transition tables, constraint search,
exception rules, evidence joins and resource ledger. They are still synthetic
and do not replace natural or external evaluation data.

## Build

Run on the training machine using the exact local tokenizer revision used by
the trainer. The first example assumes separate held-out SELECT and CAL files.
Choose row counts after auditing the old pool and row lengths.

```bash
python3 -m training.data.build_pilot \
  --legacy-train /path/to/legacy-24k.train.jsonl \
  --select /path/to/select.jsonl \
  --cal /path/to/cal.jsonl \
  --output-dir /path/to/private-run-data \
  --seed decision2-pilot-v1 \
  --arm legacy_6k:6000:0:0:0 \
  --arm replace_5pct:5700:100:100:100 \
  --tokenizer /path/to/exact/tokenizer \
  --tokenizer-revision EXACT_REVISION \
  --max-row-tokens 8192 \
  --stratified-replacements \
  --source-license-evidence /path/to/private-license-evidence.json
```

When no separate SELECT/CAL files exist, reserve complete source groups from
the legacy pool **before** sampling any training arm. Other legacy groups with
an exact or detected near input match to either holdout are quarantined as
whole groups; the holdout manifest records the excluded count and family mix.
With `--max-row-tokens`, whole source groups containing any overlength row
are excluded before partitioning, and every excluded group and reason is
recorded in the private holdout manifest.
The following requests two 6,000-row arms from a 24,000-row pool, with 600
SELECT and 300 CAL rows:

```bash
python3 -m training.data.build_pilot \
  --legacy-train /path/to/legacy-24k.train.jsonl \
  --derive-holdouts --select-count 600 --cal-count 300 \
  --output-dir /path/to/private-run-data \
  --seed decision2-pilot-v1 \
  --arm legacy_6k:6000:0:0:0 \
  --arm replace_5pct:5700:100:100:100 \
  --tokenizer /path/to/exact/tokenizer \
  --tokenizer-revision EXACT_REVISION \
  --max-row-tokens 8192 \
  --stratified-replacements \
  --source-license-evidence /path/to/private-license-evidence.json
```

Derived files are `select.jsonl` (`split=select`, `evaluation_role=select`),
`cal.jsonl` (`split=cal`, `evaluation_role=calibrate`) and
`holdouts.manifest.json`; their exact token totals are recorded when a
tokenizer is supplied. The full legacy pool is the source for all three
partitions, so these are **pilot monitoring/calibration splits**. If an earlier
Decision model trained on that pool, these splits are not independent for that
model. Do not present them as final or external evaluation. Use the frozen
benchmark and independently sourced human-labeled test data for release claims.

The format for each `--arm` is
`name:legacy_count:composition_count:reading_count:abstention_count`. Groups
from the legacy file are sampled intact, deterministically. Each family is
generated once per seed, so arms with different counts use nested subsets.
For a token-matched data intervention, put a legacy-only baseline first and
add `--token-match-replacements`; every later arm must replace exactly one
legacy row per added new row. The tool chooses complete singleton legacy
groups whose exact token cost closely matches the new rows, then records the
removed groups' family/type/language mix and input-ID commitment. This keeps
the row and token budgets close but biases which legacy rows are removed.
For a complementary composition comparison, use `--stratified-replacements`
instead. It removes complete groups in proportion to the baseline legacy
`family × task_type × language` strata and records exact quotas, selected
counts, and token imbalance. The two strategies answer different questions;
neither isolates a single causal factor because new rows also differ in
length, wording, and labels.
The script rejects an exact ID, group, or input overlap across TRAIN, SELECT
and CAL. Near duplicates use an approximate SimHash candidate search followed
by text similarity; the default policy also rejects them. The explicit option
`--near-duplicate-policy report` retains a suspected near duplicate, and the
manifest records it. Identical TRAIN inputs with conflicting labels are
rejected; repeated inputs with the same label are counted. The script never
silently drops suspect rows.

Exact token counts require `--tokenizer`; `--max-row-tokens`,
`--max-arm-tokens`, and `--match-arm-tokens-percent` are unavailable otherwise.
Counting uses the decoder v2 segmented prompt boundaries, matching the
current training reference's `encode`. The tokenizer path must be local, and
the exact revision is recorded. The matching tolerance is checked against the
first listed arm. This is a constraint check, not an automatic rebalancer.
`--tokenizer-revision` is required whenever a tokenizer is supplied.

Arm outputs are `<arm>.train.jsonl` and `<arm>.manifest.json`. Within this source
repository the output must be under ignored `runs/` or `.private/`; an
external private directory is preferable. Files are never overwritten without
`--overwrite`.

## Row contract

Every training-arm row has `id`, `state`, `instructions`,
`options[{key,description}]`, zero-based `label`, `task_type`, `family`,
`group_id`, `language`, `split=train`, `source`, `evaluation_role=train`,
`render_template`, `audit_metadata`, and `input_sha256`. All pilot arms have
hard gold labels. `target_probs` and `teacher_probs` are rejected; replay
distillation requires a separate protocol and file.

`input_sha256` is SHA-256 of UTF-8 canonical JSON over exactly
`{state,instructions,options,task_type}` with sorted keys, compact separators
and `ensure_ascii=False`. For old rows with a different existing hash, the
original is preserved as `audit_metadata.original_input_sha256`. Legacy source
objects become stable `legacy:<dataset|generator|type>` IDs, with the original
object retained in private audit metadata. Candidate descriptions preserve
their original finite JSON type, so the decoder's canonical rendering of old
descriptions is unchanged. Missing
render templates receive an explicit `legacy_unspecified` value. The manifest
contains input and output file SHA-256, family/type/language/source counts,
token totals when measured, generator code SHA, source license/attribution
placeholders, and TRAIN/SELECT/CAL overlap reports. Resolve every legacy
source license and attribution placeholder before publishing training data.
For a private run with documented source records, pass
`--source-license-evidence /path/to/evidence.json`. This JSON file has a
`sources` map keyed by normalized source ID, with a nonempty `license`,
`attribution`, and `evidence` string for each entry. Every source used by an
arm or its holdouts must be covered. The manifest embeds the records and the
evidence file SHA-256. Mixed-source records can still require release review;
source documentation alone does not authorize redistribution of training
rows.

## Verification and limits

```bash
python3 -m unittest discover -s training/data/tests -v
python3 -m py_compile training/data/build_pilot.py
```

The near-duplicate search is approximate. Shared semantics with a different
surface form may escape detection. Programmatic labels validate the small
mechanisms above; they do not establish natural judgment, calibration under
disagreement, or a credible Jev comparison. Keep evaluation on the unified,
frozen benchmark and independently sourced natural sets.

## Independent pilot selection and FLUTE training candidate

The older legacy-derived SELECT/CAL files share lineage with published
Decision 1.0 weights. They are only loader and lineage smoke data. For model
selection and per-type calibration, make fresh holdouts from the three CSS
**pilot** tasks plus original Noul and Score examples. This builder reads only
`css-pilot.prompts.jsonl` and `css-pilot.gold.jsonl`; it never reads labels
from the 15 CSS evaluation tasks. It checks every pilot context against the
full legacy TRAIN pool by raw and normalized text SHA-256, removes complete
repeated-context groups, and keeps SELECT/CAL context-distinct.

```bash
python3 -m training.data.build_css_pilot_holdouts \
  --panel-dir /path/to/private-css-panel \
  --legacy-train /path/to/legacy-24k.train.jsonl \
  --output-dir /path/to/fresh-css-pilot-holdouts \
  --tokenizer /path/to/exact/tokenizer \
  --tokenizer-revision EXACT_REVISION \
  --max-row-tokens 8192 \
  --select-per-task 200 --cal-per-task 100 \
  --synthetic-select-per-type 0 --synthetic-cal-per-type 100 \
  --seed decision2-css-independent-selectcal-v1
```

The audited reference build has SELECT 600 Choice rows (SHA-256
`d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38`)
and CAL 300 Choice, 100 Noul, 100 Score rows (SHA-256
`b5235c72269d8f13a28fa18019e29172f7c75ccc7204a63fa6fba9f20f096ebb`).
Raw and normalized CSS pilot context overlap with all 24,000 legacy states
was zero. These are selection and calibration data, not final release tests.
The synthetic CAL tasks are narrow programmatic checks; temperature estimates
need uncertainty reporting and final independent validation.

The CSS replication package is MIT, but its task data retain their original
creators' rights. The official [ColumbiaNLP FLUTE dataset card](https://huggingface.co/datasets/ColumbiaNLP/FLUTE)
lists AFL-3.0 and an official train partition. The pinned SALT preparation
uses that train file for its FLUTE classification task. The private FLUTE
builder excludes all CSS pilot/evaluation source IDs, raw context hashes,
normalized context hashes, and input hashes before it accesses candidate
labels. It then removes repeated normalized candidate contexts. This leaves
5,978 rows from 6,484 source rows (500 panel IDs, two extra exact-text
overlaps, four repeated contexts). The reference pool SHA-256 is
`4ffc5305fc2e094b3f80f32da0b34fc3203e23bbbd55f12c93a44bc6fbc8fd22`.

```bash
python3 -m training.data.build_css_flute \
  --data-root /path/to/pinned-salt-snapshot \
  --replication-root /path/to/pinned-replication-snapshot \
  --panel-dir /path/to/private-css-panel \
  --control-dir /path/to/stratified-legacy-6k \
  --select-file /path/to/fresh-css-pilot-holdouts/select.jsonl \
  --cal-file /path/to/fresh-css-pilot-holdouts/cal.jsonl \
  --output-dir /path/to/fresh-flute-candidate \
  --tokenizer /path/to/exact/tokenizer \
  --tokenizer-revision EXACT_REVISION \
  --max-row-tokens 8192 --sample-count 1000 \
  --seed decision2-css-flute-1k-v1
```

The FLUTE-only 6k ablation contains 5,000 group-stratified legacy and 1,000
FLUTE train rows; reference SHA-256
`8e653e59f10a0c7448a785507576d5b6e562c361fad8d664966a57fac98ad9aa`.
The primary pilot candidate adds the same 300 programmatic rows used in the
earlier stratified arm and uses 700 FLUTE rows. It shares the exact 5,000
legacy IDs with the FLUTE-only ablation. The reference SHA-256 is
`4e82651181fd4f9b11e82718275a7370b82cbe810f20bbe87db9786cfdf888ad`
at 4,506,153 exact Qwen3.5-4B input tokens. Both are shorter than the pure
legacy 6k control (5,212,288 tokens); comparisons are not single-factor
causal tests. A model trained on FLUTE must report its CSS FLUTE test cell as
**same-task supervised**, outside a zero-shot transfer aggregate.

```bash
python3 -m training.data.build_combined_pilot \
  --legacy-control /path/to/stratified-legacy-6k \
  --program-arm /path/to/stratified-program-6k \
  --flute-arm /path/to/fresh-flute-candidate \
  --select-file /path/to/fresh-css-pilot-holdouts/select.jsonl \
  --cal-file /path/to/fresh-css-pilot-holdouts/cal.jsonl \
  --panel-dir /path/to/private-css-panel \
  --output-dir /path/to/fresh-combined-candidate \
  --tokenizer /path/to/exact/tokenizer \
  --tokenizer-revision EXACT_REVISION \
  --max-row-tokens 8192 \
  --legacy-selection-seed decision2-css-flute-1k-v1 \
  --seed decision2-combined-6k-v1
```

## Nested training budgets from the frozen combined arm

The combined 6k pilot can be trained first at 1,024 and 2,048 rows. The
budget builder accepts only its frozen TRAIN SHA-256, allocates the same
5000:300:700 legacy/programmatic/FLUTE source ratio by largest remainder,
interleaves family, task type, language, and source strata, and preserves
complete legacy source groups. The 1,024-row IDs are a strict subset of the
2,048-row IDs. SELECT/CAL are copied byte-for-byte from the independent
pilot holdouts. The builder validates both budgets with the trainer loader,
checks exact and near-duplicate holdout overlap, and reads only ID/hash
fields from the CSS pilot/evaluation gold files for contamination checks.

```bash
python3 -m training.data.build_budget_subsets \
  --source /path/to/frozen-combined-6k/combined_6k.train.jsonl \
  --select-file /path/to/fresh-css-pilot-holdouts/select.jsonl \
  --cal-file /path/to/fresh-css-pilot-holdouts/cal.jsonl \
  --panel-dir /path/to/private-css-panel \
  --tokenizer /path/to/exact/Qwen3.5-4B-Base-tokenizer \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-row-tokens 8192 --seed decision2-combined-budget-v1 \
  --output-dir /path/to/fresh-combined-budget-v1
```

| Budget | Legacy | Programmatic | FLUTE | Exact input tokens | Max row tokens | TRAIN SHA-256 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1,024 | 853 | 51 | 120 | 754,164 | 5,938 | `7ab3df2a4e2c5ac74e12923b6ba94ef192ed90b94a8a612138ff9fe0cc8b1ddd` |
| 2,048 | 1,707 | 102 | 239 | 1,540,092 | 7,751 | `fb8d1abf9a21bf0452b8c65ca219d82d3a6717e32184127bfff91cf36994bfea` |

The reference manifest SHA-256 is
`b236cdfe149689306a620143f61575c8b55c736f3609055d2bc1ff4d25308b04`.
These are training budgets from the same data mixture, not independent
experiments about the value of one source. No CSS evaluation labels enter the
builder. A model trained on either budget must identify CSS FLUTE evaluation
as supervised same-task.

### Pure legacy 1,024-row control

For an early matched comparison, use the untouched legacy 6k source as a
separate control. The builder copies all 853 legacy rows in `combined_1024`,
then selects 171 whole singleton legacy groups to replace the 51 original
programmatic and 120 FLUTE rows. Choice/Noul/Score and the joint task-type ×
language counts match exactly. It counts tokens with the same Qwen tokenizer,
selects replacements close to the removed rows' lengths, and makes one
within-stratum swap to close the remaining aggregate token gap.

```bash
python3 -m training.data.build_pure_legacy_control \
  --legacy-source /path/to/stratified-legacy-6k/legacy_6k.train.jsonl \
  --combined-source /path/to/fresh-combined-budget-v1/combined_1024.train.jsonl \
  --select-file /path/to/fresh-css-pilot-holdouts/select.jsonl \
  --cal-file /path/to/fresh-css-pilot-holdouts/cal.jsonl \
  --panel-dir /path/to/private-css-panel \
  --tokenizer /path/to/exact/Qwen3.5-4B-Base-tokenizer \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-row-tokens 8192 --seed decision2-pure-legacy-1024-v1 \
  --output-dir /path/to/fresh-pure-legacy-control
```

The reference TRAIN SHA-256 is
`437055a3d8d222e7e181ff7484f11988d830596e4301708feb5fccf84edfb5f6`;
the manifest SHA-256 is
`a6dfa42718138a9b76433d92a33d33bc6384df3283e1397b292a7803828a1a40`.
Both this control and `combined_1024` contain exactly 754,164 input tokens,
with maximum row length 5,938. All 171 replacement task-type × language
strata are feasible and their row-count differences are zero. Within-stratum
token differences are Choice/en −3, Choice/zh +3, Noul/en 0, Noul/zh 0;
the aggregate difference is zero. SELECT/CAL are byte-identical to the
independent holdouts and exact, near-duplicate, and CSS panel hash checks are
zero. The manifest records full family/source counts and the shared-core ID hash;
the training JSONL contains every selected ID.
Length matching favors shorter legacy examples, so this comparison does not
isolate data novelty from source, family, and label distribution changes.

## Further CSS training-source audit

The CSS authors' [data loader](https://github.com/SALT-NLP/LLMs_for_CSS/blob/main/data_loader.py)
formats ConvoKit utterances as `speaker: text` and concatenates utterances
within a conversation. The official [Wikipedia politeness corpus](https://convokit.cornell.edu/documentation/wiki_politeness.html)
has 4,353 three-class examples under CC BY 4.0 and stable utterance IDs.
Its downloaded ZIP SHA-256 is
`90ec6c2c6e05d064a805d2e4be4a8d442f370b31f0025798e8eaf62d0014ba48`.
All 498 CSS politeness test contexts map by raw SHA-256 to this archive,
covering 501 source records because of repeated contexts. Excluding all CSS
pilot and evaluation context hashes leaves 3,852 rows; normalized-context
deduplication and dropping three conflicting-label groups leaves 3,837
candidate contexts. An additional approximate near-duplicate check against
the source records mapped to CSS test contexts excludes eight more, leaving
3,829 private candidate rows. This is a **derived train complement**, not an
official train split. Its labels must stay out of the frozen 6k and
SELECT/CAL.

```bash
python3 -m training.data.build_css_wiki_politeness \
  --source-zip /path/to/wikipedia-politeness-corpus.zip \
  --panel-dir /path/to/private-css-panel \
  --select-file /path/to/fresh-css-pilot-holdouts/select.jsonl \
  --cal-file /path/to/fresh-css-pilot-holdouts/cal.jsonl \
  --tokenizer /path/to/exact/Qwen3.5-4B-Base-tokenizer \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-row-tokens 8192 --sample-count 1000 \
  --seed decision2-css-wiki-politeness-v1 \
  --output-dir /path/to/fresh-wiki-politeness-candidate
```

The reference pool SHA-256 is
`bbd6fb0cc5d0a68b9f3633677506e3631fc6494ff41c82569b40d97fd0669b27`
(3,829 rows; 507,722 exact input tokens). Its class-stratified 1,000-row
sample SHA-256 is
`844d72d148d4891a9cf956051547d5dcdb1ee16223287c47db3e3234a7db41a2`
(132,022 tokens, 237/523/240 impolite/neutral/polite). The private manifest
SHA-256 is `3a993bc9566d6fdf596a509354912efab2a2a0469a3e499c234e77e5135e69a3`.
The builder records every excluded source ID and reason, and both outputs
pass the trainer's TRAIN/SELECT/CAL isolation check. The corpus anonymizes
the speaker ID, so speaker-disjointness cannot be established. Training on
this pool makes the CSS wiki_politeness evaluation **same-task supervised**.

The official [Wikipedia Talk Pages corpus](https://convokit.cornell.edu/documentation/wiki.html)
has 391,294 utterances under CC BY-SA 4.0, with conversation and speaker
IDs. Its downloaded ZIP SHA-256 is
`9995d3b885338e9332496193a0b7b01453174c2d6008ec88634ad7cb884f8f41`.
Reconstructing the authors' conversation text matches 495 of 498 unique
CSS power test contexts. The author's emitted numeric source ID also maps
497 of 500 test rows to the same source-conversation interval as their
matching context hash. The remaining three IDs map to valid conversation
intervals even though the archive text hash differs, so a later builder can
quarantine their complete groups by ID. No power training pool has been
built yet. The authors' CSS `persuasion` task uses
ChangeMyView Winning Arguments, not the Apache-licensed PersuasionForGood
corpus. The [Winning Arguments documentation](https://convokit.cornell.edu/documentation/winning.html)
does not state a data license, so it is also excluded from candidate training.

## Targeted 2,000-row candidate from open DEV and CSS pilot errors

The aggregate-only [error evidence](targeted_open_evidence_v1.json) is pinned
by SHA-256 `928c9acb9373b2fe379922a6bf9c000208582a44f54efd620e5ebe31ccf862bb`.
Its inputs are the **synthetic DEV** (1,600 questions) and **CSS three-task
pilot** (1,430 items), with their score reports and predictions. Benchmark
final and the CSS 15-task evaluation were not read. The figures below are
diagnostic observations, not outcomes of training on this candidate.

| Model | DEV Choice /800 | DEV Noul /400 | DEV Score /400 | Both correct on 200 option-reversal pairs | CSS discourse /497 | CSS stance /435 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lux | 799 | 258 | 331 | 200 | 281 | 269 |
| Nox | 466 | 223 | 379 | 99 | 207 | 265 |
| Sol | 413 | 217 | 313 | 78 | 167 | 229 |
| Jev | 797 | 266 | 357 | 199 | 305 | 321 |
| Decider | 733 | 191 | 365 | 182 | 261 | 277 |
| Kev | 754 | 254 | 329 | 189 | 224 | 270 |

Noul errors have opposite class biases: Nox predicts true on 176 false cases
and Jev predicts false on 130 true cases. Score's ordinal confusion also
varies: Lux makes 68 gold-1-to-2 errors, Jev 43, and Kev 46. In CSS stance,
Lux maps `None` to `Against` on 89 items versus Jev's 29; Kev instead maps
`Against`/`Favor` to `None` on 65/59 items. CSS implicit-hate accuracy remains
low for all six models (154–220 correct of 498), but no reliable synthetic
oracle for that subjective taxonomy is asserted here.

The frozen combined 6k contains 4,392 Choice, 1,242 Noul and 366 Score rows;
its 300 original programmatic rows are a small part of 4,506,153 input tokens.
The Lux combined-6k LoRA SELECT curve on 600 independent pilot rows is
322 correct at baseline, 332 at step 25, and 323 at step 125. This short curve
does not show stable transfer improvement. These observations motivate a
separate candidate with balanced Boolean contrasts, ordinal boundary cases,
option-key reversals, and pragmatic Choice examples.

`build_targeted_candidate.py` creates exactly 2,000 **new** TRAIN rows in
1,000 complete paired groups:

| New family | Rows | Oracle | Intended pressure |
| --- | ---: | --- | --- |
| `targeted_interval_conjunction` | 600 | Inclusive/exclusive permit date AND independent stamp | Noul false/true balance and boundary reasoning |
| `targeted_quantized_median` | 400 | Median of five measurements mapped to four numeric cutoffs | Score level and threshold binding |
| `targeted_attributed_stance` | 500 | Author's explicit position despite a quoted other view | Neutral stance and option binding |
| `targeted_dialogue_function` | 500 | Controlled second-turn speech act | Discourse role and option binding |

The paired Choice examples preserve semantic gold while changing which option
key names it. Each Noul pair flips truth through a date boundary or stamp;
each Score pair changes the measured grade. These mechanisms and render
templates differ from both the four benchmark families and the original
Decision 2.0 pilot generators. Synthetic stance and dialogue are simpler
than human annotations; improvement on the CSS pilot is a hypothesis to test,
not an observed result.

```bash
python3 -m training.data.build_targeted_candidate \
  --combined-train /path/to/frozen-combined-6k/combined_6k.train.jsonl \
  --legacy-train /path/to/original-1.0-source/train.jsonl \
  --select-file /path/to/independent-holdouts/select.jsonl \
  --cal-file /path/to/independent-holdouts/cal.jsonl \
  --dev-prompts /path/to/open-synthetic/dev.prompts.jsonl \
  --css-pilot-prompts /path/to/open-css-pilot/css-pilot.prompts.jsonl \
  --tokenizer /path/to/exact/Qwen3.5-4B-Base-tokenizer \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-row-tokens 8192 --seed decision2-targeted-2k-v1 \
  --output-dir /path/to/fresh-targeted-2k-v1
```

The private output TRAIN SHA-256 is
`59d40112ab4d9d0688b3b121212ad5cc36329fe18170839485c9906544f0dee0`;
manifest SHA-256 is
`f26cc542d76f3b9e0237046edadff865d5a7115cdc1e2f6052cdf35ab7b6866a`.
Exact Qwen tokenizer count is 349,141 total, median 154 and maximum 222
tokens per row. There are 1,000 Choice, 600 Noul and 400 Score rows; language
mix is 1,800 English and 200 Chinese. The candidate is short: appended to
combined6k, it would contribute 25% of rows but only about 7.2% of total
input tokens. Training allocation and length effects therefore need separate
measurement.

The manifest checks ID, group, input, exact/normalized state and approximate
near-input overlap against the frozen combined 6k and independent SELECT/CAL.
It also checks ID, group, input and exact/normalized state against all 47,842
rows of the original 1.0 train source, plus exact/normalized and approximate
context overlap against the open synthetic DEV and CSS three-task pilot. Every
reported overlap is zero, and the trainer loader/isolation check passes. No
CSS pilot or DEV label enters the output. The sealed final/evaluation sets
remain unread, so this builder cannot make an empirical claim about near
duplicates against those sets. Data rights are marked private internal
research; no public data license is assigned.

### Frozen 1,024-row anchor plus targeted 2,000-row candidate

`build_targeted_anchor.py` combines the unchanged `combined_1024` budget
(853 legacy, 51 original programmatic, 120 FLUTE) with the complete
`targeted_2k_v1` source. It selects no new subset: all 947 anchor groups and
all 1,000 targeted two-row groups are preserved. A fixed SHA-256 rank seed
shuffles the 3,024 rows. The independent SELECT/CAL files are copied
byte-for-byte. This is a **training candidate** for Lux and 2B LoRA, with
the frozen combined6k and pure-legacy1024 as descriptive comparators.

```bash
python3 -m training.data.build_targeted_anchor \
  --anchor-train /path/to/combined-budget/combined_1024.train.jsonl \
  --targeted-train /path/to/targeted-2k/targeted_2k.train.jsonl \
  --combined-6k /path/to/frozen-combined-6k/combined_6k.train.jsonl \
  --budget-manifest /path/to/combined-budget/budget_subsets.manifest.json \
  --targeted-manifest /path/to/targeted-2k/targeted_2k.manifest.json \
  --select-file /path/to/independent-holdouts/select.jsonl \
  --cal-file /path/to/independent-holdouts/cal.jsonl \
  --dev-prompts /path/to/open-synthetic/dev.prompts.jsonl \
  --css-pilot-prompts /path/to/open-css-pilot/css-pilot.prompts.jsonl \
  --tokenizer /path/to/exact/Qwen3.5-4B-Base-tokenizer \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-row-tokens 8192 --seed decision2-targeted-anchor-3024-v1 \
  --output-dir /path/to/fresh-targeted-anchor-3024-v1
```

The private TRAIN SHA-256 is
`18714248d5afdc803a91e7c0ed4d5613d19d113e6b42deb684c48524dad665bc`;
manifest SHA-256 is
`6cd818024ac440218b15498b8b6c3be6fdb8121dde00a2be7db20bdfed12852f`.
Exact Qwen tokenizer accounting is 1,103,305 input tokens: 754,164 from
the anchor and 349,141 from targeted rows. The maximum row is 5,938 tokens.
Source bucket counts/tokens are legacy 853/725,602, original programmatic
51/8,448, FLUTE 120/20,114, targeted 2,000/349,141. The merged type mix
is 1,750 Choice, 812 Noul, 462 Score; language mix is 2,591 English and
433 Chinese. SELECT and CAL retain SHA-256
`d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38`
and `b5235c72269d8f13a28fa18019e29172f7c75ccc7204a63fa6fba9f20f096ebb`.

The two TRAIN inputs have zero shared IDs, groups, canonical inputs or
approximate near inputs. Merged TRAIN versus SELECT/CAL has zero such
overlaps. Against open synthetic DEV and CSS three-task pilot prompts,
ID/group/input/raw-context/normalized-context and approximate near-context
overlaps are zero. The trainer loader and partition isolation check passes
for 3,024/600/500 TRAIN/SELECT/CAL rows. Benchmark final and the CSS
15-task evaluation remain unread. The new mix is much shorter than the
4.51M-token combined6k and is larger than the 754k-token pure-legacy1024;
comparisons change token budget, source, task type and length together.
The 120 FLUTE train rows make CSS FLUTE evaluation same-task supervised.

### Separate harder CAL candidate (`cal_hard_v2`)

The old 500-row CAL has 300 CSS pilot Choice cases and only 100 each for
Noul and Score. `build_cal_hard_v2.py` leaves the 300 CSS Choice rows exactly
as they were and replaces the two small synthetic type subsets with 300 fresh
Noul and 300 fresh Score rows. It writes a **separate** 900-row CAL and copies
the existing SELECT byte-for-byte; it does not edit an existing run, its
checkpoint selection, or the old CAL. The new CAL is a candidate for a future
checkpoint's per-type temperature fit, not a retroactive fit for old runs.

The new rows use a new seed and predeclared oracle rules. Noul has 150
true/false paired groups, split equally across direct signature checks,
one-page eligibility boundaries, and embargo/waiver conditions. Score has
150 paired groups; changing net credit by one unit crosses an explicit grade
cutoff. Its three tiers progress from a stated net value to arithmetic and
conditional deductions. This deliberately includes easier anchors and
structurally harder cases, without selecting items by any model's score.
The score labels span levels 0–4 with counts 38/76/75/74/37, so extreme
levels are less frequent than middle levels.

```bash
python3 -m training.data.build_cal_hard_v2 \
  --old-cal /path/to/independent-holdouts/cal.jsonl \
  --select-file /path/to/independent-holdouts/select.jsonl \
  --targeted-anchor-train /path/to/targeted-anchor-3024/targeted_anchor_3024.train.jsonl \
  --combined-6k /path/to/frozen-combined-6k/combined_6k.train.jsonl \
  --pure-legacy-1024 /path/to/pure-legacy/pure_legacy_1024.train.jsonl \
  --targeted-2k /path/to/targeted-2k/targeted_2k.train.jsonl \
  --train-root /path/to/private-data \
  --extra-train /path/to/other-private-run.train.jsonl \
  --legacy-v1-train /path/to/original-1.0-source/train.jsonl \
  --dev-prompts /path/to/open-synthetic/dev.prompts.jsonl \
  --css-pilot-prompts /path/to/open-css-pilot/css-pilot.prompts.jsonl \
  --tokenizer /path/to/exact/Qwen3.5-4B-Base-tokenizer \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-row-tokens 8192 --seed decision2-cal-hard-v2 \
  --output-dir /path/to/fresh-cal-hard-v2
```

`--train-root` checks every `*.train.jsonl` below the specified directory;
repeat `--extra-train` for TRAIN sources elsewhere. The private CAL SHA-256
is `bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf`;
manifest SHA-256 is
`063c3fd1f2ddb0e2f08823e77961e5b356e18c9752c24b3ff84d3150270d3e0c`.
The unchanged SELECT SHA-256 is
`d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38`.
Exact Qwen token total is 198,305 (Choice 84,835; Noul 49,184; Score
64,286), with longest row 2,739 tokens. CAL language mix is 780 English
and 120 Chinese.

All 600 new Noul/Score rows have zero ID, group, input or near-context
overlap with the old CAL. The full 900-row CAL is disjoint from SELECT and
the core targeted-anchor3024, combined6k, pure-legacy1024, and targeted2k
TRAIN arms by ID, group, input and approximate near input. An exact
ID/group/input/raw/normalized-context check covers 22 materialized TRAIN
files (69,976 rows) in the private run plus 47,842 original 1.0 source
rows. New Noul/Score examples have zero exact or near-context overlap with
the open synthetic DEV or CSS three-task pilot. The 300 retained Choice rows
intentionally belong to that CSS pilot; their IDs and source contexts are
verified, and the three pilot tasks contribute exactly 100 each. Trainer
TRAIN/SELECT/CAL isolation passes for 3,024/600/900 rows. Synthetic final
and CSS 15-task evaluation remain unread.

Structural difficulty is only a design hypothesis: these generated policy
and arithmetic cases are shorter and cleaner than deployment inputs. They
may still be too easy for a particular checkpoint, and a temperature fit
may still reach the optimizer's 0.05 lower bound. Compare held-out NLL and
calibration after checkpoint selection; do not choose CAL items by target
model correctness. The retained CSS Choice cases remain pilot data, not an
external release evaluation.

### Real human-labelled TweetEval TRAIN candidate

`build_tweeteval_human.py` converts only the pinned [TweetEval](https://github.com/cardiffnlp/tweeteval)
`train_text.txt`, `train_labels.txt`, and `mapping.txt` files (revision
`4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`). It reads no TweetEval
validation/test labels. Its ten source buckets are five SemEval stance targets
other than Donald Trump plus HateEval hate, SemEval irony, OffensEval offensive,
SemEval emotion, and SemEval sentiment. Each source item is a human-labelled
tweet or post. Stance is phrased explicitly with its source target; option
ordering is shuffled by a stable per-item seed. This is real but still
short-form social text, not a replacement for long-context or typed numerical
training.

The private reference build samples 3,600 distinct contexts, with 300 per
stance target, 500 hate, and 400 each from irony, offensive, emotion, and
sentiment. Round-robin class sampling is deliberately close to balanced
where the source supports it. The climate stance training split has only 13
`against` examples, so its 300-row sample is 13/143/144
against/favor/none; the class policy cannot invent the missing class.
Natural class priors differ from this training mix and require independent
probability calibration.

```bash
python3 -m training.data.build_tweeteval_human \
  --tweet-root /path/to/pinned-tweeteval-checkout \
  --panel-dir /path/to/private-css-panel \
  --select-file /path/to/independent-select.jsonl \
  --cal-file /path/to/independent-cal.jsonl \
  --dev-prompts /path/to/open-dev.prompts.jsonl \
  --train-root /path/to/private-run-data \
  --legacy-v1-train /path/to/original-1.0-source/train.jsonl \
  --tokenizer /path/to/Qwen3.5-4B-Base-tokenizer \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-row-tokens 8192 --seed decision2-tweeteval-human-v1 \
  --output-dir /path/to/fresh-private-tweeteval-candidate
```

The output `tweeteval_human_3600.train.jsonl` is SHA-256
`a3c21e6b3c3d9c1abed8facd3112cfa0fdf42779519f94e4fc62d95ef1f05339`;
its private manifest is SHA-256
`4b000a401f61d0ecbcee7e25d42d5e76c226859092cc8e8b00338d0fd33943bd`.
Input tokens total 434,080 with a maximum of 184. From source candidates,
124 repeated normalized contexts and one context shared with existing TRAIN
were excluded. The completed sample has zero ID, input, raw-context, or
normalized-context overlap with CSS three-task pilot, CSS 15-task evaluation,
synthetic DEV, SELECT/CAL, and 22 previous TRAIN files (116,794 total rows,
including the original 1.0 source). Approximate near-context matching against
all gold-free CSS and synthetic DEV prompts plus SELECT/CAL found zero hits.
Near matching is an audit, not a proof of semantic independence. The builder
never reads CSS gold; CSS pilot and evaluation exclusions come only from
`*.prompts.jsonl`.

This source changes transfer interpretation. CSS stance pilot is a **cross-target
supervised** test: training covers five non-Trump topics and the pilot covers
Trump. CSS implicit-hate pilot receives **related-task** binary hate/irony
supervision, not its six-way implicit-hate taxonomy. CSS discourse remains
unseen-task. CSS emotion evaluation receives related-task emotion supervision
with a different source and label inventory, so its cell should be described
accordingly. If this candidate enters a release model, separate task-source
overlap reporting from a strict unseen-skill aggregate.

[TweetEval's license note](https://github.com/cardiffnlp/tweeteval#license)
says its umbrella release has no restrictions while the original tasks and
Twitter may impose their own. The converted text is kept private; do not
redistribute these rows with model weights. The manifest records this rights
status and the original task attribution requirement. A 3,600-row sampled
arm is a hypothesis for training, not a demonstrated transfer improvement;
measure it against an equal-budget control before including it in a model.

A subsequent source-rights audit identified explicit noncommercial or
research-only terms in at least 2,800 of these 3,600 rows. The user has
authorized noncommercial research training with those sources, so the
4,624, 5,824, 8,360, and 8,522-row mixes remain research candidates.
They must not be presented as unrestricted or Apache-2.0-only training
lineages. Any public model release derived from them needs explicit
source-rights review, attribution, and license/use terms consistent with
the underlying restrictions. Source files, builders, and receipts are
retained for that review; a separately audited TRAIN lineage is also
being developed.

### Human and typed training arms

`build_human_anchor.py` freezes a 1,024-row anchor from the combined 6k
TRAIN source and interleaves all 3,600 TweetEval TRAIN rows. The original
anchor contained four FLUTE rows with near-matching CSS15 evaluation
contexts. The builder quarantines those **complete source groups** and
deterministically backfills four clean FLUTE TRAIN rows from the same pinned
combined 6k pool. The resulting anchor retains 1,020 exact original rows
(853 legacy, 51 programmatic, 116 FLUTE) and adds four clean FLUTE rows.
It copies independent SELECT 600 and hard CAL 900 byte-for-byte. Every
TRAIN row passes exact and approximate context exclusion against those
holdouts, synthetic DEV, CSS pilot and CSS15 gold-free prompts; no CSS15
labels are accessed. The fixed private 4,624-row TRAIN SHA-256 is
`3a994e58d7084f4b1afea46cecac0ceb3a00ecd010263296315b2c8ddd058266`;
manifest SHA-256 is
`de747b03b58eb28491111f424df4ffc097c3eef615d03b4998450e5e01f463af`.
It has 4,350 Choice, 212 Noul and 62 Score rows, so it is an intentionally
imperfect type mix and needs the balanced comparison below.

`build_balanced_human.py` retains that entire 4,624-row TRAIN, including all
3,600 human examples, then adds complete, safe TRAIN groups from pinned
combined 6k and targeted 2k sources. The predeclared quotas are 450 Noul
and 150 Score from each source. Group ranking and final row order use fixed
hashes. The builder excludes IDs, groups, inputs and raw/normalized contexts
shared with the base, SELECT/CAL, synthetic DEV and both CSS prompt panels,
then quarantines near contexts under the same rule. It reports each rejected
group and verifies the resulting file with the trainer's partition loader.
The fixed private 5,824-row TRAIN SHA-256 is
`e83fb07021b779bb86d6b1d773b007c2dda9d91052aedf1f72f89bebbfef50e2`;
manifest SHA-256 is
`869a94c0c74b9e80f2b60bf414eb7440cda17cbce1e61906621bbe206ea5aa9f`.
Its type mix is Choice 4,350 / Noul 1,112 / Score 362, with 1,846,849
Qwen3.5-4B input tokens (maximum row length 5,938). The additional Noul and
Score cases are earlier TRAIN sources with synthetic oracles. This is a
type-balance intervention, not an independent human-label source or an
evaluated gain. Noul labels are true 537 / false 575; Score labels 0–7 are
73 / 65 / 89 / 67 / 47 / 13 / 5 / 3, so high Score levels remain scarce.
The CSS FLUTE cell remains **same-task supervised** in both
arms because the anchor includes FLUTE TRAIN. The TweetEval task relationship
notes above also apply. Neither builder publishes raw TweetEval text.

### CSS15 overlap mask for comparable evaluation

`audit_css_train_near.py` uses only the 6,547 **gold-free** CSS15 evaluation
contexts to compare the published Decision 1.0 TRAIN (47,842 rows) and
combined 6k TRAIN. The method mirrors `build_pilot.near_duplicates`: eight
8-bit bands of a 64-bit SimHash for candidates, Hamming distance at most 8,
relative length delta at most 8%, and `SequenceMatcher` context similarity
at least 0.94. Separate raw and normalized exact-context counts are kept.
This approximate search may miss paraphrases or even candidate pairs; it is
an exposure audit, not proof of label leakage.

The published 1.0 source has 27 near CSS15 RAOP contexts and one Indian
English dialect context; the combined 6k source has 14 near FLUTE contexts.
No other CSS15 tasks yielded matches at this threshold. The source-specific
CSS ID lists are preserved. The union of known matching CSS IDs is 42, leaving
a shared clean set of 6,505 IDs. Report scores for **all** models on both the
full 6,547 and this same 6,505-row clean denominator, with per-task counts.
The Decision 2.0 models initialized from 1.0 inherit at least its 28 known
overlap risks even if their new TRAIN has zero CSS15 context matches. Third
party pretraining and fine-tuning text is unknown, so the common clean mask
does not prove independence for those models. Do not describe the RAOP cell
of a 1.0-initialized model as a strict unseen-text transfer test.

The machine-readable mask contains `near_css_ids_by_train_source`,
`known_overlap_union_css_ids`, `shared_clean_css_ids`, full/overlap/clean
counts, source hashes, thresholds, and the CSS prompt-file hash. The gold
labels are not loaded by the audit or training-arm builders. Keep both mask
and detailed matched-ID report with the private evaluation artifacts.

### Sol 2B structured replay recovery arm

`build_structured_replay.py` retains the frozen balanced-human 5,824 TRAIN
rows and adds all clean complete Stage4 groups from the pinned legacy 6k
TRAIN whose source is `legacy:stage4-general-composition-v2` or
`legacy:stage3_replay`. It excludes MultiNLI rows because their derivative
rights review is pending. The audited build adds 2,536 rows after quarantining
44 near-context rows from the 2,580 eligible source rows, yielding 8,360
TRAIN rows with SHA-256
`399dc5322a8daf98316c3f9255c257140047331806190955a57bf632fea726dd`.
Every selected ID/input hash is recorded in the private manifest. This is
hard-label mixture training, with
no teacher KL replay. The existing SELECT600 and hard CAL900 are copied
byte-for-byte.

The builder checks the exact SHA-256 of every source and protects the balanced
base, SELECT, CAL, synthetic DEV, CSS three-task pilot, and **gold-free** CSS15
prompts. It rejects entire source groups on ID, group, canonical input,
raw/normalized context, or approximate near-context overlap. It then audits
the chosen rows again and enforces the decoder's 8,192-token cap. Its private
manifest includes source attribution, license evidence, source hashes, full
selected IDs, row counts, and the contamination audit. Stage3 replay includes
CLINC/Banking-attributed text, so any downstream release still needs source
review. Neither CSS15 nor family-disjoint final labels are read.

### Nox 4B structured and natural replay recovery arm

`build_nox4b_structured_mix.py` retains all 5,824 balanced-human TRAIN rows
and adds whole source groups from the pinned combined 6k TRAIN. The planned
2,700-row replay budget covers Stage4 composition (1,220), Stage3 replay
(450), original programmatic tasks (180), Cosmos QA (300), SNLI (200),
MultiNLI (200), and SQuAD2 answerability (150). The last source provides 148
whole-group rows at the frozen seed, so the realized addition is **2,698**
rows: Choice 2,025 / Noul 519 / Score 154. The 8,522-row TRAIN SHA-256 is
`773cd53d21663095a4208e6e35de6654bdca4af5d314e23b03582dbe70eae87f`;
private manifest SHA-256 is
`336c039e22573c6bb0396bbedc3f86c04a25d9d126a933d4c9a6f1c6fcc0931f`.
SELECT 600 and hard CAL 900 are copied byte-for-byte from the balanced arm.

The builder pins all input hashes and excludes complete source groups on
ID, group, canonical input, exact raw/normalized context, and approximate
near-context matches with the balanced base, SELECT, CAL, synthetic DEV, CSS
pilot, and **gold-free** CSS15 prompts. It quarantined 76 near-context source
rows. The selected TRAIN has zero detected overlap with the protected
evaluation/selection inputs. It enforces the 8,192-token training cap; the
audited 4B-tokenizer total is 4,367,587 and the maximum row is 6,596 tokens.
This arm changes size and source mix together, so any gain does not isolate
the effect of structured replay. The inherited 1.0 data exposure and 120
FLUTE TRAIN rows still apply. The included MultiNLI and Stage3 CLINC/Banking
source rights need explicit review before any release; the code repository's
license must not be assumed to cover all dataset content. No training rows
are part of the model bundle.

### Rights-clean source and holdout control

`build_rights_clean_v1.py` derives an immutable control arm from the pinned
8,522-row Nox replay mix. `build_nox4b_no_mnli.py` first removes all 267
MultiNLI-origin rows, including 67 inherited from the balanced base. The
rights-clean builder then removes all 3,600 TweetEval rows, preserving every
other row and complete source group. The resulting TRAIN has 4,655 rows
(Choice 2,508, Noul 1,631, Score 516), SHA-256
`4973f999b7a19c69a8d7236ab941556e788a9ab69f208c015d3768fd7c86841b`.
Its parent no-MultiNLI TRAIN SHA-256 is
`d8d670f965a69aa08ed1f9704696b02179999fc1cc2ba2aad3bf1c0b479ca12c`.

Fresh internally generated SELECT and CAL each contain 300 oracle-checked rows
(Choice 120, Noul 90, Score 90), respectively SHA-256
`1d564becab12717f7883c77131b8e8611a2e495c102a71789a1b1e5c2cf9afd4`
and `35c27a2a16271b7295afa2d65474dbdc7c64742fdce23ba26bb2d3792a0921d6`.
The private `rights_clean.manifest.json` SHA-256 is
`a4ded3bc13f5dcc9cffbd98899714507f17d29b0f4084cc0319f030a32728bdf`.
It records source/output hashes, excluded source/task counts, license evidence,
and 27 zero-detected exact/approximate-near overlap audits among TRAIN,
SELECT, CAL, the old holdouts, synthetic DEV, CSS pilot, gold-free CSS
evaluation prompts, and the three pressure-development prompt files. The
9,600 proposed oracle groups were screened without relaxing the near
threshold. This control's synthetic SELECT/CAL are narrow and cannot by
themselves establish natural-distribution calibration or transfer quality.

| Retained source in TRAIN | Rows | Upstream terms and attribution |
| --- | ---: | --- |
| Original Stage4, targeted and pilot generators | 2,837 | Internal objective-oracle data; retain generator provenance. |
| Stage3 replay | 644 | 435 internal; 128 [CLINC150 CC BY 3.0](https://github.com/clinc/oos-eval/blob/master/LICENSE); 81 [BANKING77 CC BY 4.0](https://huggingface.co/datasets/PolyAI/banking77). |
| CosmosQA | 448 | [Author-confirmed CC BY 4.0](https://huggingface.co/datasets/allenai/cosmos_qa). |
| SNLI | 272 | [Stanford CC BY-SA 4.0](https://nlp.stanford.edu/projects/snli/). |
| SQuAD 2.0 | 334 | [Official CC BY-SA 4.0](https://rajpurkar.github.io/SQuAD-explorer/). |
| FLUTE | 120 | [ColumbiaNLP AFL 3.0](https://huggingface.co/datasets/ColumbiaNLP/FLUTE); CSS FLUTE evaluation is same-task supervised. |

The separate original 5,824-row balanced-human arm remains usable for the
authorized noncommercial research. Its TweetEval origins require accurate
disclosure: [HatEval is CC BY-NC 4.0](https://hatespeech.di.unito.it/hateval.html),
the [SemEval-2018 irony dataset is CC BY-NC-SA 4.0 with academic-use wording](https://github.com/Cyvhee/SemEval2018-Task3),
and the [NRC emotion/stance data are free for research with commercial use by
arrangement and no raw redistribution](https://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html).
The [TweetEval repository](https://github.com/cardiffnlp/tweeteval) explicitly
defers to original task and Twitter terms. A public model bundle contains
weights, code, attribution and aggregate evaluation receipts, never raw
source rows or individual text predictions. Clean new training rows do not
erase a source checkpoint's earlier training lineage; audit the chosen base
separately and disclose inherited exposure.
