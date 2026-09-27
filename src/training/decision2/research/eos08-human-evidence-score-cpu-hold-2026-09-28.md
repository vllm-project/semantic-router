# 0.8B Score-only human-evidence arm: CPU token-budget HOLD

**Decision:** Stop the proposed OCNLI substitution before optimizer work. The
predeclared same-budget rule is impossible under its fixed native Score prompt,
even using an overgenerous upper bound that ignores source-group uniqueness,
class balance, ambiguity, rights exclusions beyond non-news and protected
overlap. This is a feasibility failure, **not** a Decision 2.0 model score.
No GPU, formal answer key, model prediction, or new training data was used.

The prospective design is [here](eos08-human-evidence-score-next-arm-2026-09-28.md).
The local [aggregate-only audit program](../training/data/eos08_score_only_admission.py)
and its [CPU tests](../training/data/tests/test_eos08_score_only_admission.py)
are the reproducible method. Exact source rows and rejected IDs remain private.

## Archived control recovered

The archived hard/soft Eos replay uses the **same 512 row identities and
order**; its hard-control replay SHA-256 is
`e48977f9a4447693bacbccf5bd7c2b44f8ca46cb50462c70e965288a140d98da`.
All 512 `(id, input_sha256)` pairs match the archived ordered roster. The
ordered `(id, source, group, input hash, task type)` digest is
`29eee6954d43cbcdebb8364b5afb1b4ea79817d7469c4a213b83ff6751e7a5c3`.
The eligible initializer is our Eos 1.0 revision
`3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd`; the actual tokenizer JSON
SHA-256 is `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`
on both experiment nodes.

| Archived exposure | Rows | Native encoded tokens |
| --- | ---: | ---: |
| Original unchanged TRAIN | 7,455 | 4,194,465 |
| Replay Choice | 160 | 104,598 |
| Replay Noul | 160 | 67,429 |
| Replay **Score** | **192** in 181 source groups | **201,359** |
| Replay total | 512 | **373,386** |
| Original TRAIN + replay | 7,967 | **4,567,851** |

The audit independently reproduced the archived replay token total **exactly**,
using the same segmented native encoder boundaries. Thus `K=192` clears the
prospective minimum of 128, and all 320 non-Score occurrences can be preserved.
For ±1% total-token matching, the replacement 192 Score rows would need at
least **155,681** tokens: old Score 201,359 less floor(1% × 4,567,851) =
45,678.

## Generous candidate upper bound and stop

The original OCNLI TRAIN file matched the previously audited SHA-256
`cb47a6c00105bfb49bbb791dd537b284d3138f9f1ad5cd1e187afc470ff004e0`.
It contained 39,762 eligible non-news labeled rows before group/overlap
filtering (class counts: contradiction 12,981, neutral 13,603, entailment
13,178); the CPU screen excluded 10,672 news rows, 49 no-consensus rows and
three missing-provenance rows. With the fixed native `refuted / undetermined /
supported` question and the Eos tokenizer, even the **192 longest** of those
39,762 rows sum to only **31,789** tokens; the longest single candidate is
184 tokens. The required minimum is missed by **123,892** tokens. The maximum
full treatment exposure would be 4,398,281 tokens, **169,570** (3.71%) below
the archived control, exceeding the allowed 1% deficit.

This top-192 calculation deliberately allows multiple rows from one premise
group, arbitrary label balance and rows that later rights, quality or overlap
checks might quarantine. Imposing those gates can only lower the attainable
token sum. We therefore stopped **before** the expensive full protected
long-leaf overlap, blind rubric packet, deterministic group selection, zero-step
model check or GPU. The earlier OCNLI audit's **51.81% hypothesis-only**
balanced-accuracy cue and **9,444** unscanned protected long leaves remain
additional unresolved admission blockers; this run does not silently clear
them. No source row enters TRAIN, SELECT or CAL.

## Provenance and next decision

The aggregate audit script SHA-256 is
`31951842d87c196e1921f215af3fb4043542bd721e84f82543b759e612f6d5e2`;
both private execution mirrors matched that byte hash. The private control
and candidate aggregate receipts have SHA-256 respectively
`a014da1ee6d83e704c6715cda9f16b1bb30c8f07397c2f0ac195772ad6288914`
and `cbc863653642984db46dbbe2767cb5b8d68954791f3b612c0f09c3e3497e2e86`.
The pinned CPU runtime image ID was
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`;
the first candidate invocation used an unavailable local image tag and did
not read source data, then succeeded using this same cached image ID. Local
focused tests and the repository changed-file check passed. Resource cost:
**0 GPU-hours**.

Do **not** add padding, repeat short OCNLI pairs, enlarge the step count or
pick a new prompt to rescue this locked contrast. The next distinct arm needs
genuinely long, independently checked Score evidence at the required token
exposure, or a separately frozen short-source protocol with a newly trained
token-matched control. Neither option inherits this experiment's acceptance
status. The completed Eos hard/soft negative results and 0.8B release HOLD
remain unchanged.
