// Package contextdedup removes prior conversation turns that are provably a
// second copy of the turns immediately before them. It is a pure policy over
// the shared context transformation view: it proposes stable message IDs for
// the TransformDeduplicate step, never mutates a request, and never decides
// that two different texts mean the same thing.
//
// # Eligibility
//
// A turn is every message sharing one TurnID. A turn is a candidate only when:
//
//   - R1 every message is eligible history: EligibleHistoryRemoval is set,
//     Protection is zero, and Source is history. Instructions, the live turn,
//     multimodal content, authorization and safety context, RAG and Memory
//     content, and messages with unknown turn membership are never candidates
//     (retained: ineligible).
//   - R2 no message belongs to a tool exchange. Repeated tool executions are
//     retained whether their call IDs repeat or differ, because the view does
//     not expose call arguments or result outcomes (retained: tool_exchange).
//   - R3 the turn opens with a user message and contains an assistant
//     message. A repeated user message without a reply is a retry or a
//     confirmation, not transport duplication (retained: incomplete_turn).
//   - R4 every message has at least one block and every block is history
//     text. Content the view cannot show cannot be proven equal (retained:
//     opaque_content).
//
// # Identity
//
// Stage one compares turns through the view: the ordered list of
// (role, source, [(block source, normalized text)...]) per message.
// Normalization is exact by default; whitespace mode collapses runs of white
// space and trims the ends. Case, punctuation, and Unicode forms are never
// normalized.
//
// Stage two proves each matched message pair against the neutral request
// messages supplied by a Resolver: same role, equal IDs or an empty later ID,
// the same number of blocks, only text or reasoning blocks, equal normalized
// text, and equal citations, cache directives, signatures, and reasoning
// scope. A refusal block retains the turn (refusal_retained); any other
// difference retains it (identity_mismatch). Without a resolver, or when a
// message cannot be resolved, nothing is removed (equivalence_unverifiable).
//
// # Segments
//
// Only adjacent repetition is removed. Candidate turns form maximal runs; any
// non-candidate turn ends a run. Within a run, a block of k consecutive turns
// that is immediately followed by an identical block of k turns has its later
// copy removed, largest k first, then the same position is checked again so a
// triple send collapses to one copy. Identical turns separated by other
// content are temporal repetitions and are retained (non_adjacent). The
// earlier copy is always the one kept, so ordering and provenance of retained
// content never change.
//
// # Bounds and failure
//
// History beyond max_history_turns or max_history_bytes rejects the whole
// step; a prefix is never deduplicated. max_segment_turns bounds the block
// size. The step observes the executor's context and its own timeout. Every
// failed evaluation removes nothing; fail-open keeps the original request and
// fail-closed rejects it. Receipts carry counts, bounded reason codes, and
// pre-transform message indexes only, never text.
package contextdedup
