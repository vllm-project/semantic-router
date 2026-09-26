# JevArena authored v4: blocked editorial candidate

Status: **BLOCKED**. No authored v4 result may enter the six-axis release rank
until independent blind review and a separate human editorial gate are passed.
The blind reviewer receives only prompts. Release gold, source specifications,
and the review key remain private and were not used for model selection.

## Why v4 was built

Independent v3 review found repeated self-declared irrelevant attachments,
null→fixed-label shortcuts, several incomplete facts that were still
decidable, implausible workflow order, unstated edge direction, a false
antecedent ambiguity, a Score14 forecast boundary mismatch, and severe
per-operation answer collapse. The v3 candidate and SHA values remain frozen
as a rejected artifact; v4 is a new build with new private seeds.

The signed source commit is `2db834f`. The source of truth is the local
`jev_arena/authored_v4.py`, `authored_v4_ops.py`, and
`authored_v4_reference.py`. Exact remote validation-mirror source SHA-256
values match the local files: builder
`36b8a4e81679bc4233c24c7bf310598491f749d97d65d7755b4845313662d433`,
ops `42aa1a3a8525679cd56454edae027052adf34fa14ea6b3568c831fa44bfbb047`,
reference `eb66133be006b60db542d98aaba44d44846ca187d2fda23c5863ac06d32fb850`.
The v3+v4 local test suite passed 12/12; the v4 mirror suite passed 4/4.

## Frozen protocol and automated receipts

The English-only candidate has 60 formally registered policies (20 Choice,
20 Noul, 20 Score) and four challenge cells per policy. DEV has one case per
cell (240), release has six independently generated fact packs per cell
(1,440). These are **scenario groups**, not 1,440 independent semantic
operations. The visible evidence has four layouts. The release generation
stratifies each six-case cell by oracle answer: Boolean cells include both
labels; Choice and Score cells have at least three answers and no answer more
than three times. Partial-evidence cells include unresolved fallback and
resolvable nonfallback cases. The report contains exact per-cell label
counts; generation fails closed when a cell cannot meet the gate.

Partial evidence names one unreported fact and two admissible values. The
policy is evaluated in both complete worlds; an invariant answer is returned
when the two results agree, otherwise a conservative fallback is used. The
private fact oracle and separately implemented visible-prompt oracle agree
on every item. This removes the v3 deterministic null shortcut, while making
partial-world semantics an explicit area for editorial scrutiny.

Long-context cases have 36 related-case documents of several genres, with
case identifiers, competing dates, measurements, and status notes. They no
longer state that each attachment is irrelevant. The shortest release long
prompt is 2,609 words and the median is 2,662. These are programmatically
assembled documents, so word count alone cannot establish naturalness or
answer quality. The reviewer must inspect all 60 operation×long-context
cells for misleading conflicts, repetitive form, and policy ambiguity.

The DEV protected inventory includes 55 frozen sources: training, SELECT,
CAL, typed/CSS/pressure/public prompts, older authored panels, and v3's
final DEV/release prompts. The release inventory adds the new v4 DEV prompts
(56 sources). No protected label was needed for overlap checking. Exact
state/input and approximate near-context overlap are zero on both panels;
the approximate method does not prove absence of paraphrases. Internal
near-overlap and per-cell balance violations are also zero.

| Frozen receipt | DEV240 | Release1440 |
| --- | --- | --- |
| Prompt SHA-256 | `9b704aa3dbeaf58199fb6fc9ec1d4b2f7cce21c27dae3279e34654dc7789b9fc` | `f06957f2453dd423ebdf64a4fc4da684239d134424ecb96956ec63f40b7f4ca2` |
| Manifest SHA-256 | `1d636125c9d20aaed9c5f88b3f196a8c45c8a707cfcee73b863b848a31c4fd54` | `ddd6124ab91ce42f25dd4700d81651aefc3e8889905a9e523617e8cfd77911f8` |
| Audit SHA-256 | `d83fa5d9c0a3add31338130bae514f144d88a17977482a5b6e944a875895dbac` | `506ebee3a5358235d0a0676d930d25ab8d96da5f69ceada5cb0f20178714a0e0` |
| Blind review packet SHA-256 | `05e48b0e26177f6f357ec9ceff8a3be5a8df8b50f37c29151da8df83a9fc1555` | `3fee832a136dee1d143b1f29b77d6e7f42c038fe42299b5c187a39c90c5ff937` |
| Separate private review key SHA-256 | `59ef0c0673b13bb20bfa56667a8aeb51ba1b6b9251eaf4f12d0ff91e957bd6df` | `4421362100aa6b9668269d1a1f4587ae1a530975958395fcab161fd556a69d55` |
| Dual oracle agreement | 240/240 | 1,440/1,440 |
| Exact / protected near / internal near / balance violations | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The release partial-evidence labels are: Choice hold40, A20/B22/C20/D18;
Noul false60/true60; Score 0:40, 1:26, 2:26, 3:8, 4:20. This is an
intentional answer-balance test and is disclosed so it is not mistaken for
natural prevalence. Private seeds are retained with the experiment record;
only their manifest commitments are disseminated.

## Pending blind/editorial gates

The blind packet has one case per operation×challenge cell (240 prompts),
with no labels. An independent reviewer must first record a gold-blind
ambiguity and realism assessment; the separately stored key may be opened
only afterward for answer-error and distribution audits. In particular,
inspect partial-world wording, whether the admissible values are realistic,
whether long documents create genuine retrieval/precedence challenges,
the plausibility of Score18 workflows, graph direction, Score14 boundaries,
and whether any operation can be shortcut from syntax. Automated overlap,
oracle agreement, and stratified labels are necessary checks, not human
attestation. Until review closes, `quality_gate.status=blocked` and
`ranking_eligible=false` in both manifests.

## Independent blind verdict and post-receipt full audit

The independent reviewer saved a 240/240-cell first pass before any key
inspection (private JSONL SHA-256
`7f6af4aa34b5b6cd035b4f1efa4e010d62cc9bbba9e4e7e7fba2411f4fb849e0`).
Verdict: **BLOCKED**. The blind packet itself leaks outcomes: Noul was
inferred true in all 80 sampled cells, Choice partial-evidence hold in 20/20,
and Score partial-evidence zero in 20/20. Style was fixed by challenge in
that packet (JSON, bullets, table, numbered). All 60 sampled long cases
used 36 off-target documents with repeated form, and 58/60 archived-policy
cells had no executable alternative under the target facts. The reviewer
also found an admissible Score07 negative-age world with unspecified
semantics. This is a gold-blind editorial finding, not human attestation.

After that receipt was frozen, a full 1,440-row diagnostic confirmed the
mechanism. Each six-row release cell was label-balanced, but the packet took
ordinal zero from each cell. The generator schedules Noul true first in
every cell; all 80 ordinal-zero Noul labels are true. It schedules unresolved
partial evidence first; all 20 sampled Choice partial labels are hold and
all 20 sampled Score partial labels are zero. Thus per-cell aggregate balance
masked a deterministic review-sample shortcut. Across **all** 360
rule-precedence release rows, only 15 archived rules are executable from
the target fields and only nine yield a different result. The other 345
cannot test a substantive precedence conflict. Among the 18 complete
Noul15 rows, six use four unique record keys, making repeated-key agreement
vacuous; among the 18 complete Noul20 rows, eight set `exempt=true`, bypassing
the checks. One of six Score07 partial rows admits a negative evidence age.

These findings override the earlier automated pass. The original v4 files,
seed commitments, manifests, and hashes above remain unchanged. No v4
result is eligible for model evaluation or ranking. A new candidate must
make archived/current policies jointly executable with opposing outcomes,
make long documents carry target-relevant evidence, avoid ordinal/style/
label coupling in the blind sample, validate factual-world invariants, and
pass another independent blind review before any label-based report is
considered.

The reviewer then saved a separate private post-key distribution report
(SHA-256 `c320ad99188d0b8cec5ef4204c1c0ba78df823f389b949b9256c8b3e231bcf06`).
It confirms the stronger metadata shortcut: all 480 Noul release labels
alternate deterministically by ordinal (even true, odd false), and the
prompt ID exposes that ordinal. Its nested-field executable archive-rule
upper bound is **13/360**. My 15/360 count above used a looser top-level
field check and is not a claim of 15 semantically valid conflicts. The
reviewer also confirms all 360 long cases share the same 36-document
single-target-block form. These are release blockers regardless of the
aggregate label counts.
