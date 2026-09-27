# JevArena v3.1 authored panel: prospective review adjudication

**Status: method only.** JevArena v3's first release uses the 1,600 typed
FINAL and 6,547 CSS held-out cases as its two sealed axes. Its 8,147 original
questions do not depend on an authored panel. This document concerns an
optional later **v3.1** authored axis and precedes any v3.1 release-pool
review, adjudication or protected FINAL evaluation. The existing twelve-
original R4 packets are a DEV feasibility pilot; adjudicating them would not
make them v3.1 release items. Earlier v1/v2 packets, reviews and receipts
remain immutable and retain their original meaning.

An eventual v3.1 could add 1,200–1,480 independently reviewed originals to
the 8,147 v3 questions: 9,347–9,627 original questions in total. This is a
prospective count, not a current release result. JevBench public 231 remains
a separately labeled cross-check. The authored pool's unit is
an independent situation, never an option permutation, translation or paired
source substitution. The prospective pool allocation and quality rules are in
`jev-arena-authored-release-pool-prereg-v1-2026-09-27.md`; that earlier document
mentions the old v2 combined-panel count and six-axis formula, which do not
apply to v3 or v3.1. Its human-review gate applies to that original protocol;
any v3.1 review mix and release gate must be prospectively frozen separately
before v3.1 data can be counted. This document does not retroactively relax
an older release gate.

## Inputs and trust boundary

The adjudication tool accepts only already prepared private candidate
snapshots, an already sealed gold-free packet receipt, and completed review
seals produced by `authored_release_scale_v2_review.py`. It does not create
reviewer answers or infer a reviewer's identity. A coordinator supplies an
explicit `human` or `ai` provenance record for each of the two original
reviewers and the paired reviewer. Human identities need external verification;
AI reviews need a pinned model ID and distinct run ID. The tool checks distinct
identity digests and records the coordinator's provenance statement as an
input; it cannot independently prove personhood or AI run isolation. AI work
is always labeled AI and never counted as human review. An external author and
adjudicator also remain distinct. No model output, release prediction or
typed/CSS FINAL answer enters this process.

Before any authored target is read, verify the private candidate component
hashes, mechanical preflight, domain-witness receipt, packet hashes, separate
review seals and UTC ordering. Review seals must be later than the packet and
claim `key_opened=false` and `release_qualified=false`; all three roles must
have distinct reviewer identities, also distinct from author and adjudicator.
Each row must have a native-valid directly solved answer, an exact nontrivial
quotation from each relevant source, paragraph notes, and explicit judgments
on necessity, ambiguity, realism, shortcuts and rights. Reviews must cover
the same original and paired source IDs exactly once. A missing or altered
file, unrecognized row, incorrect chronology or missing attestation aborts
without writing an adjudication receipt.

The private prospective review-policy file fixes minimum human counts for
original and paired roles before packet sealing; remaining roles may be AI.
A policy allowing AI review can help
assess answer ambiguity and source necessity, but does not certify independent
human editorial approval. The policy hash and observed reviewer-kind mix go
into the receipt. A policy shortfall is reported, never silently promoted.

Only after these checks may the tool open the separately held authored
candidate answers and private joins. The tool compares the two blinded
original answers and the paired answer with the frozen oracle, maps paired
variants to their parent, and creates a private rejection ledger. Any original
answer disagreement, oracle disagreement, paired-answer error, quality
concern, source-citation failure or adjudicator veto excludes the original and
its paired view. There is no answer override or post-key repair. The ledger
includes every original, all rejection reasons, the hashes of all inputs,
and aggregate counts; repairs require a new frozen candidate and new blind
review, never modification of a sealed packet or judgment.

An adjudicator-supplied private veto receipt is bound to the adjudicator's
identity digest and coordinator-attestation hash, and is timestamped after
the review provenance statement. It explicitly contains an empty veto list
when no cases are vetoed. The tool verifies the receipt chain, not the
real-world identity behind a digest or substantive editorial quality.

## Output boundary

The tool's strongest positive status is **REVIEW_DATA_AUDITED_POOL_SIGNOFF_PENDING**.
It always emits `release_qualified=false`; its count is only the number of
cases surviving submitted review-data checks. It separately reports whether
the submitted mix meets the pinned policy and how many human/AI reviews it
received. A separately frozen v3.1 release policy, full 1,200–1,480 original
allocation audit, provenance review, candidate model freeze and v3.1 scorer
and release integration remain required. The v3 first release is unaffected.
Public artifacts may contain method, version hashes and aggregate rejection
counts, never private prompts, source documents, keys, reviewer identities,
joins, citations or row-level decisions.
