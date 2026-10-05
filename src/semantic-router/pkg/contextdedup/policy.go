package contextdedup

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// Normalization names the text comparison applied before two blocks are
// considered identical. It never widens equality beyond white space.
type Normalization string

const (
	NormalizationExact      Normalization = "exact"
	NormalizationWhitespace Normalization = "whitespace"
)

// Policy is the resolved, validated configuration the action executes with.
// It is built from the plugin configuration by the caller; this package does
// not read configuration or widen what the shared layer allows to be removed.
type Policy struct {
	Normalization   Normalization
	MaxHistoryTurns int
	MaxHistoryBytes int
	MaxSegmentTurns int
	FailClosed      bool
	// Timeout bounds the action's own callback. The shared executor prepares
	// the transformation view before the callback runs, so that work is not
	// inside this budget.
	Timeout time.Duration
}

// Outcome describes what the action did, independently of the shared
// executor's own receipt for the committed edit.
type Outcome string

const (
	OutcomeApplied Outcome = "applied"
	OutcomeSkipped Outcome = "skipped"
	OutcomeFailed  Outcome = "failed"
)

// ScopeEligibleHistory is the only history the action may transform.
const ScopeEligibleHistory = "eligible_history"

// Bounded terminal reasons. They explain outcomes that the generic receipt can
// only report as policy_failed or no_changes. They never carry content.
const (
	ReasonApplied                   = "applied"
	ReasonNoDuplicates              = "no_duplicates"
	ReasonNoEligibleHistory         = "no_eligible_history"
	ReasonHistoryLimitExceeded      = "history_limit_exceeded"
	ReasonCancelled                 = "cancelled"
	ReasonUnsupportedRepresentation = "unsupported_request_representation"
	ReasonEquivalenceUnverifiable   = "equivalence_unverifiable"
	ReasonNotEvaluated              = "not_evaluated"
)

// Bounded retention reasons, counted per turn. They explain why examined
// turns were not removed and are reported so an operator can tell a clean
// conversation from one full of repetition the policy refused to touch.
const (
	RetainedIneligible       = "ineligible"
	RetainedToolExchange     = "tool_exchange"
	RetainedIncompleteTurn   = "incomplete_turn"
	RetainedOpaqueContent    = "opaque_content"
	RetainedNonAdjacent      = "non_adjacent"
	RetainedRefusal          = "refusal_retained"
	RetainedIdentityMismatch = "identity_mismatch"
)

// Recovery values. A removed turn is byte-identical to the retained copy just
// before it, so nothing is stored and nothing needs retrieving.
const (
	RecoveryNotRequired  = "not_required"
	RecoveryRetainedTwin = "retained_twin"
)

// MaxReceiptSegments bounds the segment list carried on a receipt.
const MaxReceiptSegments = 16

// Segment records one removed block of turns by pre-transform message index.
// Indexes identify positions, never content.
type Segment struct {
	RetainedFirstMessageID int
	RemovedFirstMessageID  int
	Turns                  int
	Messages               int
}

// Diagnostics is the action's bounded receipt detail. Counts describe
// messages unless the field name says turns or segments. Nothing here
// identifies content.
type Diagnostics struct {
	Scope             string
	Normalization     Normalization
	Outcome           Outcome
	Reason            string
	ExaminedMessages  int
	ExaminedTurns     int
	CandidateTurns    int
	ProtectedMessages int
	RetainedMessages  int
	RemovedMessages   int
	RemovedTurns      int
	RemovedTextBytes  int
	DuplicateSegments int
	// Retained counts turns by retention reason.
	Retained          map[string]int
	Segments          []Segment
	SegmentsTruncated bool
	RecoveryStatus    string
	Recovery          string
}

// evaluationFailure separates a failed evaluation from a normal negative.
// A clean history and history with nothing eligible are ordinary no-ops in
// both failure modes. Everything else means the action could not decide
// safely, which is what the configured failure mode governs: fail-open
// preserves the request and records the reason, fail-closed rejects it
// before the provider is called.
func evaluationFailure(reason string) bool {
	switch reason {
	case ReasonApplied, ReasonNoDuplicates, ReasonNoEligibleHistory:
		return false
	}
	return true
}

// withinLimits reports whether the eligible history stays inside the policy's
// bounds. Only removable history counts: instructions, the live turn, and
// other protected content are never candidates, so they do not consume the
// budget. Exceeding a bound rejects the whole step; it never deduplicates a
// prefix, so the result cannot depend on scan order.
func (p Policy) withinLimits(view contextcompression.TransformationView) bool {
	turns := make(map[int]struct{})
	bytes := 0
	for _, message := range view.Messages {
		if message.Source != contextcompression.SourceHistory || message.TurnID < 0 ||
			message.Eligibility&contextcompression.EligibleHistoryRemoval == 0 {
			continue
		}
		turns[message.TurnID] = struct{}{}
		for _, block := range message.Blocks {
			bytes += len(block.Text)
		}
	}
	if p.MaxHistoryTurns > 0 && len(turns) > p.MaxHistoryTurns {
		return false
	}
	return p.MaxHistoryBytes <= 0 || bytes <= p.MaxHistoryBytes
}

// segmentBound returns the largest block of turns one comparison may cover.
func (p Policy) segmentBound() int {
	if p.MaxSegmentTurns <= 0 {
		return 1
	}
	return p.MaxSegmentTurns
}

func newDiagnostics(policy Policy, view contextcompression.TransformationView) Diagnostics {
	diagnostics := Diagnostics{
		Scope:            ScopeEligibleHistory,
		Normalization:    policy.Normalization,
		Outcome:          OutcomeSkipped,
		ExaminedMessages: len(view.Messages),
		RetainedMessages: len(view.Messages),
		Retained:         map[string]int{},
		RecoveryStatus:   RecoveryNotRequired,
		Recovery:         RecoveryRetainedTwin,
	}
	if diagnostics.Normalization == "" {
		diagnostics.Normalization = NormalizationExact
	}
	for _, message := range view.Messages {
		if message.Protection != 0 {
			diagnostics.ProtectedMessages++
		}
	}
	return diagnostics
}

// skipped builds a no-removal diagnostic carrying the terminal reason.
func skipped(policy Policy, view contextcompression.TransformationView, reason string) Diagnostics {
	diagnostics := newDiagnostics(policy, view)
	diagnostics.Reason = reason
	return diagnostics
}
