# JevArena authored v5: semantic redesign before generation

Status: **design / blocked**. v3 and v4 are rejected development candidates.
The authored axis is excluded from any release rank until a new panel passes
gold-blind independent review and separate human editorial approval. The
frozen v4 source and artifacts are not edited or reseeded under the same name.

## Failure to correct, not disguise

In v4's 1,440 release rows, only 15/360 random archived policies could be
executed from target evidence and only 9/360 opposed the current answer.
The 240-item blind packet had Noul true 80/80, Choice partial hold 20/20,
and Score partial zero 20/20 because the first ordinal was always selected.
Document style was fixed by challenge in that packet. All 60 sampled long
cases repeated 36 off-target documents; answerable target facts sat in one
marked block. A Score07 partial world admitted a negative evidence age.
Aggregate label balance and dual-oracle equality did not detect these
semantic failures. v5 must fail closed on **causal relevance and source
plausibility**, not just frequencies.

The editor's separate post-key report (private SHA-256
`c320ad99188d0b8cec5ef4204c1c0ba78df823f389b949b9256c8b3e231bcf06`)
confirmed an even stronger shortcut: all 480 Noul labels are determined by
the ordinal visible in the prompt ID, and only 13/360 archived policies
have the nested fields needed for execution. The 360 long cases share the
same 36-document single-target-block structure. The v5 gates below must
cover the complete release set and the blind sample separately.

## Case contract

Each scored item has one target decision, one domain-specific policy, a
time-stamped evidence history, and a private structured proof trace. A trace
contains the exact facts used, their source document and version, the
operation-level calculation, and the output. It is private and never part
of model prompts. A second interpreter rebuilds facts from visible text and
compares the result; the blind editor checks that the proof matches the
ordinary-language reading. Case ID is an opaque random identifier; it does
not encode task type, challenge, label, operation, ordinal, or seed.

Retain up to 60 formally distinct policies only if policy dependency audits
and human review confirm the claimed diversity. Record a smaller count if
several are near-duplicates. Each operation has a real case rationale and
plausible facts, not a generic professional wrapper around an array puzzle.
The grammar, semantic operations, and domain coverage matrix are public in
the methodology; private scenario facts and release labels remain sealed.

## Challenge construction

| Challenge | Required causal relation | Automated gate before writing an item |
| --- | --- | --- |
| Rule precedence | Current and archived policies read the **same target fact schema** and both produce valid answers. Their answers differ on the frozen facts, and the prompt supplies credible provenance for why the current one governs. | 100% archived executable; 100% paired conflict; perturb the priority metadata to flip which answer should apply while facts remain fixed. |
| Long context | At least three target documents each contribute a necessary field; at least one older target document conflicts with a newer signed target document; related-case documents carry plausible competing figures. The answer needs source/version resolution plus the operation. | Remove each required target document in turn and demonstrate that the answer changes or becomes unresolved; replacing a distractor document must leave it invariant. Vary document genres and positions independently of task/challenge. |
| Partial evidence | Two admissible, domain-valid evidence worlds differ in one material field. Some have invariant nondefault answers; others are genuinely unresolved. A signed source dispute explains the two values in natural language. | Validate both worlds against domain invariants; run two independent evaluators; require both resolved and unresolved cases in each type and in the blind packet; reject a missing fact that cannot affect any feasible answer in the case family. |
| Near distractor | Another case has a similar ID and genuinely conflicting evidence on the same fields; one source is stale or belongs to another case. | Swapping case identity changes the computed answer; dropping the decoy keeps the target answer. |

For rule precedence, author an **explicit paired-policy registry** rather
than choosing another random operation. The pair must use exactly the same
named fields and a separately implemented archived evaluator. Examples:
lowest cost versus highest benefit under the same feasible set; `>= quorum`
versus `> quorum` on a boundary case; floor versus ceiling of the same
nonnegative ratio. Do not use a trivial constant-output old rule. Require
independent blind checks that both policy texts are understandable and that
the conflict is relevant to the target.

For long cases, build a versioned target dossier rather than padding with
unrelated notes. Each target fact is owned by a document with date, signer,
and source authority. Several documents refer to the **same** target case,
and a stale fact may have a higher raw number but lower authority. Related
cases may have close identifiers and apparently applicable data. The prompt
must not expose a single `BEGIN EVIDENCE` block that already consolidates all
facts. Retrieval, version choice, and calculation are separate proof steps.
The first 20 cases are hand-authored and reviewed before any grammar expands.
If those fail realistic reading, do not generate a large release file.

## Sampling and release gates

The target is 1,200–1,440 release items, with at least 50 accepted semantic
policies and three task types. This is a **ceiling conditional on quality**,
not a quota to fill by cosmetic substitutions. Put one scored item in each
source scenario group; paired perturbations are linked for cluster bootstrap
and counted once for independence where appropriate. Keep a separate DEV
panel and distinct private release seed. Check exact/near overlap against all
TRAIN/SELECT/CAL, typed/CSS/pressure/public, rejected authored panels, and
gold-free sealed prompts. Never read CSS15 or family FINAL labels.

Balance labels within operation×challenge cells, then **separately** audit
the 240–480-item blind review sample and every prompt metadata field. Use a
salted sample of different ordinals with an explicit sample-level label
balance gate; no fixed first ordinal. Assign JSON/bullet/table/prose style
independently of challenge, answer, and operation and test contingency with
chi-square or mutual information. Remove challenge and ordinal from user-
visible IDs. For partial information, the blind sample must include at least
one invariant nonfallback case and one unresolved case per type/challenge
subpanel; otherwise fail. For Boolean Noul, target 40–60% true within each
challenge and review sample. Report per-operation and per-challenge counts,
not only global class balance.

No score is released until (1) machine validators pass coverage, source
rights, proof-trace consistency, perturbation invariants, length limits,
independence, and overlap; (2) independent reviewer saves a gold-blind note
before seeing any key; (3) the key is opened privately for error/distribution
audit; and (4) a human subject-matter editor signs off on ambiguity and
realism. A machine-generated proof or an agent review alone is insufficient
for that last gate. A failed gate leaves the authored axis absent from the
JevArena v2 composite; it is never backfilled with v3/v4 scores.

## Build sequence and abort conditions

1. Freeze 12 policy pairs (four/type) and 20 hand-authored long dossiers;
   manually review their exact answers without model outputs. Abort if any
   paired rule is non-executable or a target document is not causally needed.
2. Build an exploratory DEV60 with independent parser and perturbation
   receipts. Run a blind editorial packet with randomized styles/positions.
   Abort on ambiguous fact authority, syntactic label shortcuts, or invalid
   possible worlds.
3. Expand only accepted operations and document grammars to DEV240 and a
   sealed release candidate. Keep source hashes, protected inventory, seed
   commitments, label distributions, and failed samples in the research
   record; do not overwrite a rejected candidate.

This plan deliberately allows the authored axis to remain blocked. A smaller
audited diagnostic is preferable to an unearned 1,200-item release claim.
