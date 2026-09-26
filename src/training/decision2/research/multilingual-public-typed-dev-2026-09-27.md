# Pinned public Chinese and Russian typed-decision DEV diagnostics

**Superseded build warning:** the initial v1 serialized prompt file reordered
Choice option keys and failed the model-input fingerprint check on all 4,052
rows. Its model outputs are quarantined. See the
[v1 correction and rebuilt v2 panel](multilingual-public-typed-v1-fingerprint-correction-2026-09-27.md)
before using this diagnostic; the counts and source-lineage audit below remain
valid, but v1 prediction scores do not.

Status: **supplementary exposed DEV only**. This adapter does not create an
independent JevArena release panel or a model score. No model inference was run
for this audit. The source questions and answers are public, so they cannot be
treated as sealed evidence or mixed into the frozen release total.

## Sources and rights

| Source | Pinned Git commit | Questions inspected | Rights and limitation |
| --- | --- | ---: | --- |
| [ZH-Decision-Bench v0.1](https://github.com/CodyQin/zh-decision-bench) | `4d6f0a9875d5558efec8b6b48323de65e20724b6` | 284 questions in 219 items | Dataset CC BY 4.0; code Apache-2.0. Attribute its author and Amazon MASSIVE. The 40 synthetic business items were LLM drafted and adjudicated by one owner; the 15- and 25-item scenarios are small. |
| [RuDecide v0.1](https://github.com/smolnikov-k/rudecide) | `10713af21eb5772be20ee8a6ab8263d49c1171ac` | 2,235 Track A + 1,543 Track B questions | Mixed per-task licenses disclosed in the upstream README, including CC BY-SA 4.0. No aggregate redistribution license file ships with the repository; retain its raw text in the local private diagnostic and cite original tasks. Several tasks use public validation, development, or train partitions, and some published applied-track comparators were trained on overlapping tasks. |

All cited source files are checked byte-for-byte by
`multilingual.public_typed_dev`. In particular, ZH `data/massive_items.jsonl`
is SHA-256 `d8b5fcb0e5d4ea359d48a940383cf76795fdc3b4e80d614d3dd9a2fe18408424`
and `data/synthetic_items.jsonl` is
`86384af1471c74aa568669817228dc5f16b3b61cc180e8c4e7e92fae75a689f7`.
Ru Track A is
`826ea9fe62fdb2e88d308d9c232cde36b237db99b1ee20051d6db75dd8b992ac`;
Track B is
`a9638dd72945ea1aeb23cff5ca5fd8af30c7586430563c7f066a9de1232f1482`.
The source README/license/conversion/review files and Ru scorer are also
pinned in code; changed bytes stop the build. The official MASSIVE 1.1 archive
is SHA-256
`4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`.
Its Chinese and Russian locale files are pinned separately. This is source
provenance, not a claim that every public gold label is independently correct.

RuDecide's task disclosures need to stay attached to every reported task:
Russian SuperGLUE (`danetqa`, `muserc`, `parus`, `rcb`) and MERA
(`ruopenbookqa`, `ruworldtree`) are described as MIT; `rucola`, `cedr`,
`spam_ru` and the injection source as Apache-2.0; `xstory_ru` as CC BY-SA
4.0; support and MASSIVE as CC BY 4.0. `prompt_safety` combines rows with
Apache/MIT rights, and `skills_unseen_catalogs` is synthetic with an upstream
MIT model disclosed by its author; the repository marks synthetic Track B
data CC BY 4.0. The adapter does not relicense or upload any of them.

## Source identity and overlap gate

ZH's 179 MASSIVE-derived voice items carry 179 distinct original source IDs.
All 179 IDs, utterances, and `dev` partitions match the pinned official
Chinese locale. RuDecide omits the original IDs for its 300 MASSIVE-derived
Russian rows. Exact utterance matching against the pinned official `ru-RU`
`test` partition uniquely recovers **290 source IDs**; **10 rows have multiple
possible original IDs and are held out**. A text-only or intent-only guess is
insufficient to restore their provenance. The 290 admitted IDs are all distinct.

The ZH 179 and Ru 290 source IDs were compared separately to both existing
private MASSIVE TRAIN candidates: frozen v1 has 600 source groups/4,200 rows,
SHA-256 `4c57dac9d5dd39cf3e0920cda7c1e975f5e2dacf46379e7232797efb3ae2d9ba`;
English-filtered v2 has 481 groups/3,367 rows, SHA-256
`55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80`.
There are **zero source-ID overlaps** for both public languages against both
candidates. The adapter requires these exact candidate hashes and raises on a
changed or colliding candidate. A future MASSIVE TRAIN candidate, including
the pending v5 design, requires a new source-ID audit before its results may be
compared on this diagnostic. Source-ID disjointness does not rule out semantic
duplicates, pretraining contamination, or upstream gold defects.

## Normalized protocol and observed build

The adapter retains the original state, question instructions, option order,
and labels in gold-free `prompts.jsonl`; answers and task/group metadata go to
`targets.private.jsonl`. Chinese multi-question items share one item group.
Russian questions with an identical source-task state share one group; the 290
recovered MASSIVE Russian rows use original source ID groups. The development
build admitted **4,052 questions / 3,729 observable groups**, with the 10
ambiguous Russian rows absent from model prompts. Native parser preflight read
all 4,052 prompt rows. For non-MASSIVE Russian tasks the upstream original
source IDs are unavailable, so identical state text within a task is grouped;
3,729 is not a certified count of independent examples. The manifest SHA-256 is
`9d0094061c82d3c36b62897821bc61164caf8abc51fbbf7f1b193bacc4cfe304`;
adapter source SHA-256 is
`c355d60d1d6cbcf7cd36875347c4064151aef6ff42420fd94bd9275726a1f56f`;
prompt SHA-256 is
`324fb66d73b893aa2fa2b4443dd0c4fdb9891b266ce90d6a3712d8161b641343`,
and private target SHA-256 is
`77e033dc9716e2416d5c0db0fed11a14fea79497efa249cba062608c09113b81`.
The source text and targets remain outside Git.

| Language | Choice | Noul | Score | Eligible total |
| --- | ---: | ---: | ---: | ---: |
| Chinese | 204 | 55 | 25 | 284 |
| Russian | 2,089 | 1,553 | 126 | 3,768 |

Choice uses the exact declared criterion key; Noul uses the native probability
of `true` with `>=0.5` as the positive decision; Score uses argmax over every
declared ordinal level. Chinese Score's textual level answer is converted to
its zero-based index; RuDecide already encodes that index as a string. Public
Choice criterion descriptions may be `null`: the native prompt retains this
upstream representation, meaning a bare option key. No answer key is copied
into the prompt. Missing or invalid native answers count wrong, with per-task
invalid counts. Prediction input fingerprints and model/adapter identity are
checked before scoring. This deliberately reports per-task performance rather
than a single cross-language rank dominated by the large Russian track. The
report also separates each language/type and task/type cell.

This first adapter reports accuracy and invalid/missing rates only. It does
not yet compute ECE, Brier, NLL or uncertainty intervals; those require a
separate frozen probability contract before any calibration comparison.

An empty-prediction validation run accounted for all 4,052 questions across
21 tasks as 4,052 invalid/missing and zero correct, confirming that unanswered
rows do not disappear from the denominator. It is a scorer test, not a model
result.

Reproduction uses pinned public checkouts, the official MASSIVE archive, and
the two private candidate files; run outside the repository:

```bash
PYTHONPATH=src/training/decision2 python -m multilingual.public_typed_dev build \
  --zh-root <pinned-zh-checkout> --ru-root <pinned-ru-checkout> \
  --massive-archive <official-archive> --massive-v1-train <private-v1> \
  --massive-v2-train <private-v2> --output <private-panel>
```

The `score` subcommand accepts that panel and native prediction JSONL. The
run used the two pinned source Git commits, passed four focused contract tests
and the repository's training contract check. It ran no model inference and
produced no performance claim.

## Decision

Both sources are admissible **as attributed, exposed, supplementary
multilingual development diagnostics** on the 4,052 admitted rows. Keep their
tasks and types visible. The 10 Russian rows without unique MASSIVE lineage
remain held. Neither source can replace independent native-speaker review,
the sealed authored panel, or release qualification; any training candidate
using their public examples directly must not use them to assert transfer.
