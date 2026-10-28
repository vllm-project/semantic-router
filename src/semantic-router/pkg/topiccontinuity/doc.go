// Package topiccontinuity evaluates bounded, protocol-neutral evidence about
// whether the live user turn still depends on retained conversation history.
//
// The package is a pure signal producer. It reads neutral messages from the
// original (pre-enrichment) history, never mutates them, and returns a typed,
// versioned Result. It never resets, compresses, or stores history; consumers
// such as context policies decide what, if anything, a result authorizes.
//
// EvaluateAll is the entry point. Internally, rules sharing a HistoryPolicy
// share one bounded preparation of segmented turns and one extraction of
// threshold-independent features; each rule then applies its own thresholds
// to a fixed precedence.
//
// All text work is bounded by counted limits rather than wall-clock time, so
// identical input always yields an identical result. Domain conditions such as
// unavailable history, cancellation, reached caps, or incomplete coverage are
// typed unknown results. Classified results pass through one constructor that
// enforces the schema invariants.
package topiccontinuity
