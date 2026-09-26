# JevArena authored scale v2: prospective diversity repair pilot

**Status: method only, signed before v2 case authoring.** The r1 36-case
candidate is frozen and [held before blind review](jev-arena-authored-release-scale-r1-hold-2026-09-27.md).
This v2 pilot is a separate 12–18-original feasibility experiment, not a
supplement to r1 or a release set. It cannot enter TRAIN, model selection,
JevArena ranking or a model card. No model inference or protected FINAL access
is part of the pilot.

## Authoring and distribution gates

Author **12–18 new independent decision situations**, at least four each for
Choice, Noul and Score. Do not reuse or lightly paraphrase r1 or v9–v13
entities, facts, rationale, prose, or operation instances. At least twelve
distinct semantic mechanisms are required if there are twelve originals;
additional originals may share a broad operation but must differ in decision
contract and causal structure. At least six substantive domains and nine
canonical document families must appear, not just nine raw labels. A source
must actually have its claimed form's structure. Each evidence source has
at least two causally relevant paragraphs or records; short one-line field
restatements are rejected. Every medium or long paragraph must resolve a
scope, exception, time, source conflict, competing candidate, or necessary
decision fact. No repeated ledger rows or padding for token length.

Use the exact Qwen3.5 0.8B Base tokenizer and full native criteria to target
at least six short prompts (150–600 tokens), two medium (601–2,000) and one
long (2,001–3,500). The remaining originals may occupy any band while
keeping all three task types represented across the batch. The long item must
be manually inspected for causal relevance paragraph by paragraph. If that
standard cannot be met, stop with a documented HOLD; a short-only pilot may
still be useful as a negative result but cannot pass the v2 diversity gate.

Before drafting facts, assign target cells in a private allocation ledger:
Noul original answers must include at least two true and two false; Score
originals must occupy each ordered level 0, 1 and 2 at least once, with no
level above half of Score items. Choice correct positions 1–4 must each occur
at least once. At least one Choice original must correctly return an explicit
HOLD option under **joint evidence insufficiency**: each source has a
domain-valid alternative completion that creates a feasible choice while
the other source is fixed. Do not implement a HOLD caused entirely by one
source, since that would make the other source unnecessary. Noul and Score
retain native valid answers; no implicit abstention labels.

Each original has a complete typed decision contract, two or more separately
presented sources, structured facts, a deterministic oracle separate from
hand-entered targets, and a private proof trace. For each source, hold the
other fixed and supply two domain-valid complete alternative sources that
yield different native answers. Create one same-format source substitution
that changes the correct answer without changing the rule, option set or
question; recheck both sources' necessity after substitution. Paired views
remain robustness data and never count as independent originals.

## Separate frozen snapshots and review boundary

The private authoring system must freeze **separate immutable snapshots**:
(1) source lineage and structured source facts, (2) rendered gold-free native
prompts, (3) oracle answers, (4) necessity and substitution proofs, and (5)
private join/salt only if blinded packets are later issued. Record a SHA-256,
UTC time, source commit and schema version for each. Keep raw sources, cases,
answers, joins and salts only in task-private remote storage. The public
branch and gist may contain methods, hashes, aggregate counts and failure
reasons, never raw protected contents or private infrastructure.

Before creating any blinded packet, verify every typed answer and proof,
source-field rendering, native token budget, exact and near overlap against
all available TRAIN/SELECT/CAL/DEV/prior authored/exposed-public prompts,
source-family independence, canonical document structures, cross-case
semantic clones, target/position balance and shallow answer shortcuts. A
single invalid native answer, overlap, unsupported source necessity,
unjustified paragraph, or source-rights uncertainty holds the batch. A
missing length or answer-balance cell also holds it. Repairs produce a new
version and new hashes; the old snapshots remain immutable.

Only after a full mechanical pass may independently salted, shuffled,
gold-free original and substituted packets be prepared. Reviewers must not
see the oracle key, parent joins, proofs or other reviews. Two independent
human reviewers solve every original; a separate reviewer solves paired
views. Seal row-level answers, cited evidence, ambiguity, plausibility,
shortcuts and source necessity before opening keys. Any unresolved material
disagreement is a HOLD. Passing this small pilot would show only that a
larger release pool is feasible; the 1,200–1,480 original release gate and
model/formula freeze remain separate.
