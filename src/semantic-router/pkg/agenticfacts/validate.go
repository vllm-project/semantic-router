package agenticfacts

import (
	"encoding/json"
	"strings"
	"time"
)

// Context portability values the Router understands. Anything else is rejected
// rather than guessed at, because this field decides whether a model switch may
// occur mid-session.
const (
	ContextPortabilityPortable = "portable"
	ContextPortabilitySticky   = "sticky"
)

// Validate checks one agentic facts envelope against bounds and returns the
// facts that may influence selection.
//
// Validation is all-or-nothing. If any check fails, Accepted is nil and the
// caller routes the request exactly as if no envelope had been presented; the
// rejection reasons are carried for diagnostics only. Empty raw means no
// envelope was presented and is not a failure.
//
// now is a parameter rather than a time.Now call so expiry behavior is
// deterministic under test.
func Validate(raw []byte, bounds Bounds, now time.Time) Result {
	if len(raw) == 0 {
		return Result{}
	}

	v := validator{bounds: bounds.withDefaults(), now: now}

	if len(raw) > v.bounds.MaxEnvelopeBytes {
		v.result.reject("", ReasonTooLarge)
		return v.result
	}

	var envelope Envelope
	if err := json.Unmarshal(raw, &envelope); err != nil {
		v.result.reject("", ReasonMalformed)
		return v.result
	}

	if !v.checkVersion(envelope) {
		return v.result
	}

	v.checkExpiry(envelope)
	v.checkLineage(envelope)
	v.checkRole(envelope)
	v.checkBudget(envelope)
	v.checkCapabilities(envelope)
	v.checkTrustBoundary(envelope)

	sortRejections(v.result.Rejections)
	if v.result.Rejected() {
		return v.result
	}

	v.result.Accepted = &v.accepted
	return v.result
}

// validator carries the state threaded through every check so the individual
// checks stay single-purpose.
type validator struct {
	result   Result
	accepted Accepted
	bounds   Bounds
	now      time.Time
}

// checkVersion reports whether the envelope declares a schema this build can
// interpret. Every later check assumes v1 semantics, so an unknown version ends
// validation rather than producing misleading field-level reasons.
func (v *validator) checkVersion(envelope Envelope) bool {
	switch envelope.Version {
	case SchemaVersion:
		return true
	case "":
		v.result.reject("version", ReasonMissing)
	default:
		v.result.reject("version", ReasonUnsupportedVersion)
	}
	return false
}

// checkExpiry bounds the envelope's lifetime in both directions: it must not
// already be stale, and it must not claim a validity window longer than
// MaxLifetime.
func (v *validator) checkExpiry(envelope Envelope) {
	if envelope.ExpiresAt == "" {
		v.result.reject("expires_at", ReasonMissing)
		return
	}

	expiresAt, err := time.Parse(time.RFC3339, envelope.ExpiresAt)
	if err != nil {
		v.result.reject("expires_at", ReasonMalformed)
		return
	}

	if v.now.After(expiresAt.Add(v.bounds.ClockSkew)) {
		v.result.reject("expires_at", ReasonExpired)
		return
	}
	if expiresAt.After(v.now.Add(v.bounds.MaxLifetime + v.bounds.ClockSkew)) {
		v.result.reject("expires_at", ReasonTooLong)
		return
	}

	v.accepted.ExpiresAt = expiresAt
	v.accepted.ExpiresAtKnown = true
}

// checkLineage bounds delegation depth and identifier length, and rejects
// lineages that contradict themselves.
func (v *validator) checkLineage(envelope Envelope) {
	if envelope.Lineage == nil {
		return
	}

	v.accepted.RootInvocationID = v.boundedString(
		"lineage.root_invocation_id", envelope.Lineage.RootInvocationID)
	v.accepted.ParentInvocationID = v.boundedString(
		"lineage.parent_invocation_id", envelope.Lineage.ParentInvocationID)

	switch {
	case envelope.Lineage.Depth < 0:
		v.result.reject("lineage.depth", ReasonMalformed)
	case envelope.Lineage.Depth > v.bounds.MaxDepth:
		v.result.reject("lineage.depth", ReasonTooDeep)
	default:
		v.accepted.Depth = envelope.Lineage.Depth
	}

	if envelope.Lineage.Depth > 0 && v.accepted.ParentInvocationID == "" {
		v.result.reject("lineage.parent_invocation_id", ReasonConflicting)
	}
	if v.accepted.ParentInvocationID != "" && v.accepted.RootInvocationID == "" {
		v.result.reject("lineage.root_invocation_id", ReasonConflicting)
	}
}

// checkRole bounds the free-form role and phase labels and constrains context
// portability to the values the Router can act on.
func (v *validator) checkRole(envelope Envelope) {
	v.accepted.DelegatedRole = v.boundedString("delegated_role", envelope.DelegatedRole)
	v.accepted.TaskPhase = v.boundedString("task_phase", envelope.TaskPhase)

	portability := strings.ToLower(
		v.boundedString("context_portability", envelope.ContextPortability))
	switch portability {
	case "", ContextPortabilityPortable, ContextPortabilitySticky:
		v.accepted.ContextPortability = portability
	default:
		v.result.reject("context_portability", ReasonMalformed)
	}
}

// checkBudget accepts only non-negative counters.
func (v *validator) checkBudget(envelope Envelope) {
	if envelope.Budget == nil {
		return
	}

	if envelope.Budget.RemainingTokens != nil {
		if *envelope.Budget.RemainingTokens < 0 {
			v.result.reject("budget.remaining_tokens", ReasonMalformed)
		} else {
			v.accepted.RemainingTokens = *envelope.Budget.RemainingTokens
			v.accepted.RemainingTokensKnown = true
		}
	}

	if envelope.Budget.RemainingTimeMs != nil {
		if *envelope.Budget.RemainingTimeMs < 0 {
			v.result.reject("budget.remaining_time_ms", ReasonMalformed)
		} else {
			v.accepted.RemainingTimeMs = *envelope.Budget.RemainingTimeMs
			v.accepted.RemainingTimeMsKnown = true
		}
	}

	if envelope.Budget.RemainingCost != nil {
		if *envelope.Budget.RemainingCost < 0 {
			v.result.reject("budget.remaining_cost", ReasonMalformed)
		} else {
			v.accepted.RemainingCost = *envelope.Budget.RemainingCost
			v.accepted.RemainingCostKnown = true
		}
	}
}

// checkCapabilities bounds the capability list and normalizes entries to lower
// case, since capabilities are symbolic tokens matched against model card
// metadata.
func (v *validator) checkCapabilities(envelope Envelope) {
	if len(envelope.RequiredCapabilities) > v.bounds.MaxCapabilities {
		v.result.reject("required_capabilities", ReasonTooMany)
		return
	}

	values := make([]string, 0, len(envelope.RequiredCapabilities))
	for _, entry := range envelope.RequiredCapabilities {
		value := strings.ToLower(strings.TrimSpace(entry))
		switch {
		case value == "":
			v.result.reject("required_capabilities", ReasonMalformed)
		case len(value) > v.bounds.MaxStringLength:
			v.result.reject("required_capabilities", ReasonTooLong)
		default:
			values = append(values, value)
		}
	}

	v.accepted.RequiredCapabilities = dedupSorted(values)
}

// checkTrustBoundary bounds the tenant, residency, and label identifiers.
func (v *validator) checkTrustBoundary(envelope Envelope) {
	if envelope.TrustBoundary == nil {
		return
	}

	v.accepted.Tenant = v.boundedString("trust_boundary.tenant", envelope.TrustBoundary.Tenant)
	v.accepted.Residency = v.boundedString("trust_boundary.residency", envelope.TrustBoundary.Residency)
	v.accepted.TrustLabel = v.boundedString("trust_boundary.label", envelope.TrustBoundary.Label)
}

// boundedString trims a scalar string field and records a rejection if it
// exceeds MaxStringLength.
func (v *validator) boundedString(field string, value string) string {
	trimmed := strings.TrimSpace(value)
	if len(trimmed) > v.bounds.MaxStringLength {
		v.result.reject(field, ReasonTooLong)
		return ""
	}
	return trimmed
}
