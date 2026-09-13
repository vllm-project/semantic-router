// Package historyreset selects complete prior turns for removal after an
// accepted topic change. It is a pure policy: it reads the shared
// transformation view, proposes stable message IDs, and never mutates a
// request, touches durable state, or decides whether the topic changed.
// Topic detection belongs to the signal that supplies the trigger result.
package historyreset

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// TriggerClass is the action's consumption view of a topic-continuity result.
// The producing signal contract is owned separately; this package only
// distinguishes the outcomes the action must behave differently for.
type TriggerClass string

const (
	TriggerChange       TriggerClass = "change"
	TriggerContinuation TriggerClass = "continuation"
	TriggerUnknown      TriggerClass = "unknown"
	// TriggerConflicting reports that the producer saw irreconcilable
	// evidence. It is never treated as a weak change.
	TriggerConflicting TriggerClass = "conflicting"
)

// TriggerResult carries one evaluated topic-continuity result. Signal names
// the resolved recipe-local signal; Version identifies the producing contract
// so an unsupported revision is rejected instead of silently trusted.
type TriggerResult struct {
	Class      TriggerClass
	Confidence float64
	Signal     string
	Version    string
	Fallback   bool
	// Binding ties the result to the request's resolved original history and
	// live turn. A timestamp cannot establish that evidence describes the
	// request being changed, so freshness is checked through this identity.
	Binding string
}

// Policy is the resolved, validated configuration the action executes with.
// It is built from the plugin configuration by the caller; this package does
// not read configuration or widen what the shared layer allows to be removed.
type Policy struct {
	Signal          string
	SignalVersions  []string
	MinConfidence   float64
	MaxHistoryTurns int
	MaxHistoryBytes int
	FailClosed      bool
	// Binding is this request's original-history and live-turn identity.
	// Evidence that does not carry the same binding describes some other
	// request or an older view of this one, and cannot authorize removal.
	Binding string
	// MaxRecoveryBytes bounds the payload a single request may persist.
	MaxRecoveryBytes int
	// Timeout bounds the whole evaluation, including preparing the view's
	// dependency groups, not just the selection loop.
	Timeout time.Duration
}

// Outcome describes what the action did, independently of the shared
// executor's own receipt for the committed edit.
type Outcome string

// ScopeEligibleHistory is the only history the action may transform. It is
// reported on every receipt so an operator can see what a reset was allowed to
// consider, not just what it removed.
const ScopeEligibleHistory = "eligible_history"

const (
	OutcomeApplied Outcome = "applied"
	OutcomeSkipped Outcome = "skipped"
	OutcomeFailed  Outcome = "failed"
)

// Bounded terminal reasons. These explain outcomes that the generic receipt
// can only report as policy_failed or no_changes. They never carry content.
const (
	ReasonApplied                    = "applied"
	ReasonNoEligibleHistory          = "no_eligible_history"
	ReasonEvidenceContinuation       = "evidence_continuation"
	ReasonEvidenceMissing            = "evidence_missing"
	ReasonEvidenceUnknown            = "evidence_unknown"
	ReasonEvidenceLowConfidence      = "evidence_low_confidence"
	ReasonEvidenceFallback           = "evidence_fallback"
	ReasonEvidenceWrongSignal        = "evidence_wrong_signal"
	ReasonEvidenceUnsupportedVersion = "evidence_unsupported_version"
	ReasonEvidenceStale              = "evidence_stale"
	ReasonEvidenceConflicting        = "evidence_conflicting"
	ReasonHistoryLimitExceeded       = "history_limit_exceeded"
	ReasonRecoveryWriteFailed        = "recovery_write_failed"
	ReasonRecoveryLimitExceeded      = "recovery_limit_exceeded"
	ReasonStreamingUnsupported       = "streaming_recovery_unsupported"
	ReasonReservedToolConflict       = "reserved_tool_conflict"
	ReasonCancelled                  = "cancelled"
	ReasonUnsupportedRepresentation  = "unsupported_request_representation"
	ReasonHistoryUnresolved          = "history_unresolved"
	ReasonRecoveryUnavailable        = "recovery_unavailable"
	ReasonInheritedCompleted         = "inherited_completed"
)

// Diagnostics is the action's bounded receipt detail. Counts describe messages
// unless the field name says turns. Nothing here identifies content.
type Diagnostics struct {
	Signal            string
	Scope             string
	TriggerClass      TriggerClass
	Version           string
	Outcome           Outcome
	Reason            string
	ExaminedMessages  int
	RetainedMessages  int
	ProtectedMessages int
	RemovedMessages   int
	RemovedTurns      int
	RecoveryStatus    string
	RecoveryEntries   int
}

// Recovery status values. They describe what the action did about
// recoverability, never where the payload went or under which key.
const (
	RecoveryStored      = "stored"
	RecoveryNotRequired = "not_required"
	RecoveryFailed      = "failed"
)

// evaluationFailure separates a failed evaluation from a normal negative.
// A continuation, an inherited follow-up, and history with nothing eligible
// are ordinary no-ops in both failure modes. Everything else means the action
// could not decide safely, which is what the configured failure mode governs:
// fail-open preserves the request and records the reason, fail-closed rejects
// it before the provider is called.
func evaluationFailure(reason string) bool {
	switch reason {
	case ReasonApplied,
		ReasonEvidenceContinuation,
		ReasonNoEligibleHistory,
		ReasonInheritedCompleted:
		return false
	}
	return true
}

// skipped builds a no-removal diagnostic carrying the terminal reason.
func skipped(trigger TriggerResult, reason string) Diagnostics {
	return Diagnostics{
		Signal:       trigger.Signal,
		Scope:        ScopeEligibleHistory,
		TriggerClass: trigger.Class,
		Version:      trigger.Version,
		Outcome:      OutcomeSkipped,
		Reason:       reason,
	}
}

// authorize reports the terminal reason when the evidence cannot authorize a
// removal, or an empty string when it can. Uncertainty never authorizes.
func (p Policy) authorize(trigger TriggerResult) string {
	switch {
	case trigger.Signal == "" || trigger.Class == "":
		return ReasonEvidenceMissing
	case p.Signal != "" && trigger.Signal != p.Signal:
		return ReasonEvidenceWrongSignal
	case trigger.Version == "" || !p.supportsVersion(trigger.Version):
		// Evidence that does not identify its producing contract cannot be
		// checked for compatibility, so it is treated as unsupported rather
		// than trusted by default.
		return ReasonEvidenceUnsupportedVersion
	case p.Binding != "" && trigger.Binding != p.Binding:
		return ReasonEvidenceStale
	case trigger.Class == TriggerConflicting:
		return ReasonEvidenceConflicting
	case trigger.Fallback:
		return ReasonEvidenceFallback
	case trigger.Class == TriggerContinuation:
		return ReasonEvidenceContinuation
	case trigger.Class != TriggerChange:
		return ReasonEvidenceUnknown
	case trigger.Confidence < p.MinConfidence:
		return ReasonEvidenceLowConfidence
	}
	return ""
}

// supportsVersion accepts any version when the policy declares none, so a
// deployment that has not pinned the producing contract still works.
func (p Policy) supportsVersion(version string) bool {
	if len(p.SignalVersions) == 0 {
		return true
	}
	for _, supported := range p.SignalVersions {
		if supported == version {
			return true
		}
	}
	return false
}

// withinLimits reports whether the examined history stays inside the policy's
// bounds. Exceeding a bound rejects the whole step; it never resets a prefix.
func (p Policy) withinLimits(view contextcompression.TransformationView) bool {
	turns := make(map[int]struct{})
	bytes := 0
	for _, message := range view.Messages {
		if message.Source != contextcompression.SourceHistory {
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
