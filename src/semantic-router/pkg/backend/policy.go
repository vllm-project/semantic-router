package backend

import (
	"math"
	"sort"
	"strings"
)

const (
	PolicyReasonSelectedFreshTelemetry = "fresh_selectable_telemetry"

	FallbackReasonMissingTelemetry    = "missing_telemetry"
	FallbackReasonStaleTelemetry      = "stale_telemetry"
	FallbackReasonAllUnhealthy        = "all_unhealthy"
	FallbackReasonNoBackendCandidates = "no_backend_candidates"
)

type policyOption struct {
	candidate BackendCandidate
	telemetry BackendTelemetry
}

type candidateStatus int

const (
	candidateMissing candidateStatus = iota
	candidateStale
	candidateUnhealthy
	candidateSelectable
)

type candidateEvaluation struct {
	option         policyOption
	status         candidateStatus
	telemetryAgeMS int64
}

type policyCandidateSummary struct {
	options                 []policyOption
	missingCount            int
	staleCount              int
	unhealthyCandidateCount int
}

// SelectBackendCandidate applies a minimal second-stage backend policy. It only
// selects when at least one candidate has fresh, selectable telemetry; otherwise
// it fails open so existing model/backend routing can continue unchanged.
func SelectBackendCandidate(modelName string, candidates []BackendCandidate, store *Store) BackendPolicyResult {
	if store == nil {
		store = defaultStore
	}

	diag := BackendPolicyDiagnostics{
		SelectedModel:  modelName,
		CandidateCount: len(candidates),
	}
	if len(candidates) == 0 {
		return failOpenResult(diag, FallbackReasonNoBackendCandidates)
	}

	summary := collectPolicyCandidates(modelName, candidates, store, &diag)

	diag.UnhealthyCount = summary.unhealthyCandidateCount
	diag.TelemetryFresh = diag.FreshCandidateCount > 0
	if len(summary.options) == 0 {
		return failOpenResult(diag, fallbackReasonForEmptyPolicy(
			len(candidates),
			summary.missingCount,
			summary.staleCount,
			summary.unhealthyCandidateCount,
		))
	}

	sort.Slice(summary.options, func(i, j int) bool {
		return policyOptionLess(summary.options[i], summary.options[j])
	})
	selected := summary.options[0]
	age := store.Age(selected.telemetry)
	diag.SelectedBackendID = selected.candidate.BackendID
	diag.SelectedReplicaID = selected.telemetry.Identity.ReplicaID
	diag.TelemetryFresh = true
	diag.TelemetryAgeMS = age.Milliseconds()
	diag.PolicyReason = PolicyReasonSelectedFreshTelemetry

	return BackendPolicyResult{
		SelectedBackendID: selected.candidate.BackendID,
		SelectedReplicaID: selected.telemetry.Identity.ReplicaID,
		FailOpen:          false,
		Reason:            PolicyReasonSelectedFreshTelemetry,
		Diagnostics:       diag,
	}
}

func collectPolicyCandidates(modelName string, candidates []BackendCandidate, store *Store, diag *BackendPolicyDiagnostics) policyCandidateSummary {
	summary := policyCandidateSummary{
		options: make([]policyOption, 0, len(candidates)),
	}
	for _, candidate := range candidates {
		evaluation := evaluateCandidate(modelName, candidate, store)
		switch evaluation.status {
		case candidateMissing:
			summary.missingCount++
			continue
		case candidateStale:
			summary.staleCount++
			diag.TelemetryAgeMS = maxInt64(diag.TelemetryAgeMS, evaluation.telemetryAgeMS)
			continue
		case candidateUnhealthy:
			diag.FreshCandidateCount++
			diag.TelemetryAgeMS = maxInt64(diag.TelemetryAgeMS, evaluation.telemetryAgeMS)
			summary.unhealthyCandidateCount++
			continue
		case candidateSelectable:
			diag.FreshCandidateCount++
			diag.TelemetryAgeMS = maxInt64(diag.TelemetryAgeMS, evaluation.telemetryAgeMS)
			summary.options = append(summary.options, evaluation.option)
		}
	}

	return summary
}

func evaluateCandidate(modelName string, candidate BackendCandidate, store *Store) candidateEvaluation {
	candidate = normalizeCandidate(modelName, candidate)
	if candidate.BackendID == "" || candidate.ModelName == "" {
		return candidateEvaluation{status: candidateMissing}
	}

	rawTelemetry := store.ListByBackend(candidate.ModelName, candidate.BackendID)
	if candidate.ReplicaID != "" {
		rawTelemetry = filterTelemetryByReplica(rawTelemetry, candidate.ReplicaID)
	}
	if len(rawTelemetry) == 0 {
		return candidateEvaluation{status: candidateMissing}
	}

	freshTelemetry := filterFreshTelemetry(rawTelemetry, store)
	if len(freshTelemetry) == 0 {
		return candidateEvaluation{status: candidateStale, telemetryAgeMS: telemetryAgeMS(rawTelemetry, store)}
	}

	selectableTelemetry := filterSelectableTelemetry(freshTelemetry)
	if len(selectableTelemetry) == 0 {
		return candidateEvaluation{status: candidateUnhealthy, telemetryAgeMS: telemetryAgeMS(freshTelemetry, store)}
	}

	sort.Slice(selectableTelemetry, func(i, j int) bool {
		return telemetryLess(selectableTelemetry[i], selectableTelemetry[j])
	})
	return candidateEvaluation{
		option:         policyOption{candidate: candidate, telemetry: selectableTelemetry[0]},
		status:         candidateSelectable,
		telemetryAgeMS: telemetryAgeMS(selectableTelemetry, store),
	}
}

func filterFreshTelemetry(items []BackendTelemetry, store *Store) []BackendTelemetry {
	fresh := make([]BackendTelemetry, 0, len(items))
	for _, telemetry := range items {
		if store.IsFresh(telemetry) {
			fresh = append(fresh, telemetry)
		}
	}
	return fresh
}

func filterSelectableTelemetry(items []BackendTelemetry) []BackendTelemetry {
	selectable := make([]BackendTelemetry, 0, len(items))
	for _, telemetry := range items {
		if isSelectableHealth(telemetry.Health) {
			selectable = append(selectable, telemetry)
		}
	}
	return selectable
}

func failOpenResult(diag BackendPolicyDiagnostics, reason string) BackendPolicyResult {
	diag.PolicyReason = reason
	diag.FallbackReason = reason
	return BackendPolicyResult{
		FailOpen:    true,
		Reason:      reason,
		Diagnostics: diag,
	}
}

func normalizeCandidate(modelName string, candidate BackendCandidate) BackendCandidate {
	candidate.BackendID = strings.TrimSpace(candidate.BackendID)
	candidate.ReplicaID = strings.TrimSpace(candidate.ReplicaID)
	candidate.ModelName = strings.TrimSpace(candidate.ModelName)
	candidate.EndpointName = strings.TrimSpace(candidate.EndpointName)
	if candidate.ModelName == "" {
		candidate.ModelName = strings.TrimSpace(modelName)
	}
	if candidate.BackendID == "" {
		candidate.BackendID = candidate.EndpointName
	}
	return candidate
}

func filterTelemetryByReplica(items []BackendTelemetry, replicaID string) []BackendTelemetry {
	filtered := make([]BackendTelemetry, 0, len(items))
	for _, telemetry := range items {
		if telemetry.Identity.ReplicaID == replicaID {
			filtered = append(filtered, telemetry)
		}
	}
	return filtered
}

func fallbackReasonForEmptyPolicy(candidateCount, missingCount, staleCount, unhealthyCandidateCount int) string {
	if unhealthyCandidateCount > 0 && unhealthyCandidateCount == candidateCount {
		return FallbackReasonAllUnhealthy
	}
	if staleCount > 0 {
		return FallbackReasonStaleTelemetry
	}
	if missingCount > 0 {
		return FallbackReasonMissingTelemetry
	}
	if unhealthyCandidateCount > 0 {
		return FallbackReasonAllUnhealthy
	}
	return FallbackReasonMissingTelemetry
}

func policyOptionLess(a, b policyOption) bool {
	aQueue := intValue(a.telemetry.QueueDepth)
	bQueue := intValue(b.telemetry.QueueDepth)
	if aQueue != bQueue {
		return aQueue < bQueue
	}
	aActive := intValue(a.telemetry.ActiveRequests)
	bActive := intValue(b.telemetry.ActiveRequests)
	if aActive != bActive {
		return aActive < bActive
	}
	if a.candidate.Weight != b.candidate.Weight {
		return a.candidate.Weight > b.candidate.Weight
	}
	if a.candidate.BackendID != b.candidate.BackendID {
		return a.candidate.BackendID < b.candidate.BackendID
	}
	return a.telemetry.Identity.ReplicaID < b.telemetry.Identity.ReplicaID
}

func telemetryLess(a, b BackendTelemetry) bool {
	aQueue := intValue(a.QueueDepth)
	bQueue := intValue(b.QueueDepth)
	if aQueue != bQueue {
		return aQueue < bQueue
	}
	aActive := intValue(a.ActiveRequests)
	bActive := intValue(b.ActiveRequests)
	if aActive != bActive {
		return aActive < bActive
	}
	if a.Identity.BackendID != b.Identity.BackendID {
		return a.Identity.BackendID < b.Identity.BackendID
	}
	return a.Identity.ReplicaID < b.Identity.ReplicaID
}

func intValue(value *int) int {
	if value == nil {
		return math.MaxInt
	}
	return *value
}

func telemetryAgeMS(items []BackendTelemetry, store *Store) int64 {
	var maxAge int64
	for _, telemetry := range items {
		maxAge = maxInt64(maxAge, store.Age(telemetry).Milliseconds())
	}
	return maxAge
}

func maxInt64(a, b int64) int64 {
	if b > a {
		return b
	}
	return a
}

func isSelectableHealth(health HealthState) bool {
	// The contract layer only rejects explicit unhealthy telemetry. Missing,
	// unknown, or degraded health remains selectable so new adapters can fail
	// open while they converge on richer engine-specific health mapping.
	return health == "" || health == HealthStateUnknown || health == HealthStateHealthy || health == HealthStateDegraded
}
