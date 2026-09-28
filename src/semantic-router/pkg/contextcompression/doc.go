// Package contextcompression owns the request-local context transformation seam
// as well as the existing configured block compressor.
//
// The ingress captures HistorySnapshot before enrichment. RAG then Memory keep
// their existing preparation order; ParseSemanticRequest annotates their
// injected content using trusted Provenance. OriginalHistory returns detached
// conversation values for topic evidence. Missing pre-enrichment snapshots are
// not reconstructed from enriched requests.
//
// Child policies register TransformationSteps in reset, exact deduplication,
// then complete-turn selection order. ApplySteps supplies detached views and
// validates proposals before changing the request. Reset and selection remove
// complete eligible turns; deduplication may remove eligible repeated messages
// within a turn, but must retain whole tool exchanges. The deduplication policy
// owns proof of semantic equivalence; text equality alone is not permission.
// No child policy or public configuration is enabled by this package.
// Enabling a history policy also enforces live-turn and opaque history guards
// during compression. With no new policy enabled, legacy compression eligibility
// is retained for compatibility.
//
// Service.Apply is the final compression step. It retains configured target
// modes and scoring, submitting only eligible text replacements through the
// same plan. Non-text blocks, instructions, tool calls and IDs, trusted metadata,
// and protocol settings have no edit operation. Tool-result text may still be
// compressed by existing configuration; atomicity preserves exchange membership
// rather than disabling that supported behavior. Provenance.ProtectedMessages
// additionally marks authorization/safety text at prepared-message indexes.
//
// FailureOpen rejects a failed step and continues; FailureClosed rejects that
// step and stops, retaining earlier successful steps. Rejected edits never
// partially apply. Receipts contain fixed reasons and counts, never content or
// arbitrary callback errors. Step identity is request-local and replay is a
// no-op; order cannot be extended backwards. Provider encoding runs afterwards.
package contextcompression
