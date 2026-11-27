// Package turnselection implements a pure, protocol-neutral complete-turn
// history selector over the shared contextcompression transformation contract
// (issue #3344).
//
// The selector reads detached contextcompression.TransformationView values,
// groups messages using the shared TurnID, ContentSource, Protection,
// Eligibility, and ExchangeIDs metadata, and proposes stable original message
// IDs to remove through contextcompression.TransformSelectTurns. It never
// parses wire protocols or mutates protocol messages directly; the shared
// RequestIR.ApplySteps pipeline validates every proposal and performs the
// single final commit in chronological order.
//
// In the ordered context pipeline, turn selection runs after optional history
// reset (#3343) and exact deduplication (#3346), and before block compression.
// Token accounting uses a narrow, conservative pre-compression cost interface
// so model-tokenizer accounting (#3050) can plug in without changing selection
// logic. Final provider-bound admission (#3704) and persistent full-history
// recovery storage remain outside this package's pure planning boundary.
package turnselection
