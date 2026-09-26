# Score three-level SELECT r1: blind review and post-seal HOLD

This is a development checkpoint-selection diagnostic. It is not a training
source or an independently authored JevArena release axis. The prospective
method was fixed before r1 was built; its frozen source and reviewer packet are
documented in `score-three-level-select-r1-freeze-2026-09-27.md`.

## Custody and independent judgment

- Frozen gold-free reviewer packet: 240 rows, 80 groups, four operations,
  English 192 and Chinese 48. Packet SHA-256
  `d38702cee5ef1b50458a4ee11d4370a7fda44013321b2dac43b40d01600ea88a`;
  reviewer manifest SHA-256
  `a9868c0e8562c4450a2be57556327bdaba84719713cfbc77d8227a5ecff6ef59`.
- An independent agent derived each answer from the visible instructions and
  state using the private sealed revision of
  `score_select_r1_blind_reviewer.py` (SHA-256
  `6926da69dd013d527315cc983f1167d6261220eb39b7fdbe8545b49a2dbe5734`).
  The source-branch copy was reformatted only for repository style (SHA-256
  `303ff723eb8a35d7c1face2a2ecbec0e1e7fd6023d74b9306cdd857ff61709a2`);
  its parsed Python AST matches the sealed revision under the same local
  interpreter. Neither revision imports the author builder or oracle. The
  sealed revision parsed 240/240 rows and
  found 80/80 complete 0/1/2 triplets. The reviewer manually read eight
  complete English groups (24 rows) and four Chinese groups (12 rows) as a
  provisional language spot check.
- Judgments were sealed at **2026-09-26 22:08:59 UTC**, after packet freeze and
  before the author key was opened. Private review seal SHA-256
  `69e45fd8bb4560b86cddb1ef8caab99a30a9776113ca738093124683333579fe`;
  sealed judgment SHA-256
  `d9eec144b646f13ff98cb4782afda5ca511061010b27abcf72f540ff8ab5ae5a`.
  The reviewer did not run a model or open protected formal labels.
- A subsequent key comparison checked packet input identity and matched the
  sealed independent answers against the author targets **240/240**, with zero
  disagreements. This check did not change the sealed judgments.

## Post-seal shortcut finding

The first private review seal had said `EN_DIAGNOSTIC_PASS_ZH_HOLD_QUALIFIED_REVIEW`.
After that seal, a further audit of the **sealed blind judgments only** found
that a single resource pool's margin sign distinguishes all three answers in
**7/20 allocation groups** (six English, one Chinese). In four groups the
signs follow the direct `negative, zero, positive` ordering for levels 0, 1,
2. The private append-only finding was recorded at **2026-09-26 22:11:58 UTC**,
SHA-256 `e7d288e3d2d39d204ed5997b58466928736eb417eddf477b430fbe80e2c78cb9`.
It did not modify the initial seal or use author targets or model outputs.

The prospective method says one material group shortcut blocks the version.
**Final r1 verdict: `HOLD_R1_MATERIAL_GROUP_SHORTCUT`. No row may select a
checkpoint, train a model, or support a release score.** The author must
generate a new version, repair the affected groups under a fresh salt, freeze
it, and obtain a fresh independent blind review. Matching the oracle cannot
override this editorial gate.

## Scope and editorial limits

The initial English-only diagnostic eligibility is **superseded** by the
post-seal HOLD. Chinese review also remains incomplete: the spot check above
was performed by an agent, not a certified human or qualified independent
native-language editor.

The four operations are repeated fixed-format templates. The dossier themes
are decorative, and the veto is active in every waiver item. This panel can
check execution of explicit three-level rules, but cannot establish long-context
understanding, open-domain transfer, multilingual quality, or broad Score
generalization. A future authored panel needs more natural forms, operative
context, and a separate seal. Keep r1 group variants together when estimating
uncertainty; 240 rows are only 80 independent scenarios.
