# JevArena authored v3 candidate and editorial gate

This is an **English-only, programmatically authored candidate**, not a ranked
JevArena axis or a substitute for natural transfer tests. Existing FINAL and
CSS15 target labels were not opened. No model was run on these release targets.

## Frozen construction

`jev_arena.authored_v3` builds 60 explicit semantic operations: 20 Choice,
20 Noul, and 20 Score. The count excludes six domain settings, four evidence
layouts, and four challenge modes. Each operation has its own policy and fact
dependency. Within each type, all 20 operation output fingerprints differ on
256 fixed probes; this is a behavioral screen, not a proof that human readers
will find every policy unambiguous. The direct source-spec oracle and the
separately implemented rendered-evidence oracle agree on every built item.
The frozen run packages are `bench/jev-arena-authored-v3-candidate/dev-240-r8/`
and `bench/jev-arena-authored-v3-candidate/release-1440-r4/` in the experiment
workspace; the repository holds the signed builder and this receipt note.

Each scored item uses a unique operation/fact pack and a unique `group_id`.
`independent_groups` thus counts distinct fact scenarios, **not** distinct
policy designs. The 1,440 release items reuse the 60 operations 24 times
each; uncertainty and paired model comparisons should cluster by operation
or operation/challenge cell, rather than treating 1,440 rows as independent
semantic templates.

| Phase | Items | Choice | Noul | Score | Each type/challenge cell | Each operation/challenge cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DEV r8 | 240 | 80 | 80 | 80 | 20 | 1 |
| Release r4 | 1,440 | 480 | 480 | 480 | 120 | 6 |

There are four challenge types: near-case distractor, long context, missing
evidence, and conflicting archived policy. Each has 60 DEV and 360 release
items. JSON, bullets, tables, and numbered evidence each occur 60 DEV and
360 release times. Release domains are access 264, archive 216, delivery 264,
finance 264, incident 168, and release management 264. Domain counts were not
inflated to claim additional policy templates. DEV long-context target blocks
span front/middle/end positions 19/15/26; release has 120 in each position.

The private seeds are retained with the run, with SHA-256 commitments
`b794a606e4784b7f6ebd3ff30c397e04369dd9a66a19317a4fc72753ba0029af`
(DEV) and
`7cb971a2a30d21dd72239552f21c4142bc04b526faa0b9b94794358045fd5652`
(release). Both packages identify the same builder file SHA-256 values:
`authored_v3.py` `e602c2ad8388fef571bb0f3d80fcb1710e0883226f433f4903a60f5742458422`,
`authored_v3_ops.py` `b3345e6a1927a0db1f4b93f61b19fd1542e88c18cfbf7b0196533d9308ed353d`,
and `authored_v3_reference.py`
`5fa936b06ca3d5c9a7adb1864c49e12fe55422199a186c7256c666435784f05b`.

| Frozen artifact | SHA-256 |
| --- | --- |
| DEV r8 prompts | `f433c554cf226715652b65a6f0c4faeaeb58f3b7bb2765152b8c8de05f109158` |
| DEV r8 targets | `195921950315f097853c8fbb2fddf652da96a5f8ced85c404df94b0c80e0cb52` |
| DEV r8 manifest | `099aea3a48a51ff1835315d5a49f5a11959b9b3cf12558dcbd5b874d459a5612` |
| Release r4 prompts | `8eb99cb666e09589098cc6b0ab5fa1e17ede44d9cc7444b81bdd66729fb4e922` |
| Release r4 targets | `2a259a18729a24dc0edceb6fba227ddff4968b41c83b51285ba2899a73b21120` |
| Release r4 manifest | `8982fa7aad6cd4a4d03238ec26ce28639f1f5b6e61ffa98d27a74c8e45ab8f85` |
| Release r4 automated audit | `05cee60d44c82a1c4bffa2a1212cfba9a006b50098a40decdcfa2af795eea769` |
| Release r4 blind review packet | `293dbd42f11d277055e3556b7df8865c8ddea2fa4952bf8171f8ce52ed759091` |
| Release r4 separate review key | `5b5e23d273b347dbf71eda98846586d52c759d46bb2f2b44b408185628d8caf8` |

The release protected-list SHA-256 is
`11380e03d48fcf8ec4d2dfcfaf1f5678a6b880d595bb4b23a58835d8676c4a0e`.
It covers 57 listed TRAIN/SELECT/CAL/benchmark and superseded authored prompt
sources, 133,372 rows before duplicate-source removal. Normalized exact
state, exact input, candidate-screened token-5-gram near overlap against this
inventory, and internal near overlap were all zero. This screen cannot prove
absence of semantic paraphrases or unseen training sources. The DEV inventory
contained 55 sources and 132,892 rows, with the same zero counts.

Under the Sol 2B tokenizer, all 1,440 release requests fit an 8,192-token
no-truncation limit. The 360 long requests range from 5,892 to 6,209 tokens
(median 5,989); all other requests are at most 794 tokens. DEV long requests
range from 5,892 to 6,080 tokens. Both phases had zero over-limit requests.

## Ambiguity and release gate

A deterministic SHA-selected DEV spot check covered one item from each of
the 12 type/challenge cells plus eight additional items. The 20 policy/fact
cases showed no further mechanical contradiction after clarifying list order,
half-open intervals, Pareto dominance, and null handling. This agent check
does **not** constitute independent human editorial approval. The long
attachments are still formulaic rather than natural documents; the domains
are synthetic case settings. Naturalness and scenario quality remain open.

The builder writes separate mode-0600 target/spec files, a **gold-blind**
240-cell reviewer packet, and a mode-0600 answer key. The reviewer packet
contains no `gold` field. Two independent reviewers should each answer the
blind packet and flag ambiguity, domain fit, grammar, and answer alternatives;
disagreements then require adjudication against the separate key. A further
stratified check of release variants should establish that the six fact packs
per cell remain clear. Until this review is signed off, `quality_gate.status`
is `blocked` and `ranking_eligible` is `false` in both manifests. No release
score or six-axis rank may use this candidate.

Eight focused tests cover 60 distinct operation signatures, dual-oracle
agreement, all operation/challenge renderings, three long-context positions,
missing relevant evidence, normalized overlap blocking, malformed duplicate
target evidence, and blind review-key separation. They pass on the execution
mirror. The local source files have the same hashes as both manifests.

## Independent blind review: BLOCKED

An independent agent reviewed the gold-blind 240-cell packet before opening
the separate key. Its private first-pass note has SHA-256
`653a2f7da1573d54def570284c05aed7da46de8ef7731325f4344cae79340614`.
This review found four substantive failures; it is an independent agent review,
**not** human editorial approval:

1. All 60 sampled long-context cells use 66 repetitive attachments that say
   they are irrelevant. Length and target position pass the automated check,
   but the prose does not resemble a genuine multi-document decision.
2. Every one of the 120 release missing-evidence items per type maps to a
   single answer (`hold`, `false`, or `0`). Some observed facts already settle
   the underlying operation even with a null, so the blanket fallback creates
   contradictory or trivial labels. The 360 rows together provide a perfect
   null-to-answer shortcut.
3. Score14 says an exact forecast earns grade four, but its `floor(error/3)`
   formula also gives grade four for one- or two-unit errors. Three nonmissing
   release cases had one-unit error. The policy and boundary cases need an
   explicit adjudicated rule.
4. Score18 randomizes workflow order without domain constraints. In eight of
   18 complete release cases, `close` occurs before `intake`; the problem
   becomes a position lookup in an implausible workflow.
5. Noul11 and Score10 do not state whether an edge pair `[u,v]` means
   `u→v` or `v→u`, so graph answers can reverse. Noul05 does not explicitly
   define the false-antecedent case as `not first OR second`; nine of its 18
   complete release cases have a false antecedent and permit a natural
   “not applicable” reading.

The post-blind key audit also found outcome collapse outside missing-evidence
cells: Choice16 returned `hold` in 18/18 complete release cases; Noul08 was
`false` in 18/18 and Noul18 in 17/18; Score02 was grade four in 14/18. These
are generator defects that a wording change cannot repair. The v3 release
package remains immutable for provenance, with `quality_gate.status=blocked`.
It must not enter the official rank. A successor needs an independently sealed
build with per-operation answer-diversity and evidence-solvability gates,
plausible heterogeneous long documents, corrected boundary semantics, and a
new blind editorial review.
