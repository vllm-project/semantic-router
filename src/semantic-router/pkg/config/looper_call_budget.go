package config

import "fmt"

// MaxUpstreamCallsPerRequest is the hard per-request amplification budget for
// Looper-family algorithms. Concurrency controls (max_concurrent) bound
// simultaneous load; this budget bounds the total upstream model calls one
// request may create through fan-out, retries, and nested stages.
//
// The value stays a small multiple of the largest maintained configuration so
// that a statically knowable call count fails configuration load with an
// actionable error instead of surprising a request at runtime. Making the
// budget operator-configurable is a follow-up; the invariant is the same.
const MaxUpstreamCallsPerRequest = 32

// EstimatedReMoMUpstreamCalls returns the statically knowable upstream call
// count for a ReMoM breadth schedule: every entry is one parallel round and
// ReMoM always appends one final synthesis round.
func EstimatedReMoMUpstreamCalls(schedule []int) int {
	total := 1 // final synthesis round
	for _, breadth := range schedule {
		total += breadth
	}
	return total
}

// validateReMoMBreadthScheduleBudget rejects a breadth_schedule whose static
// upstream call count already exceeds MaxUpstreamCallsPerRequest. It runs after
// well-formedness checks so a malformed schedule reports its own error first.
func validateReMoMBreadthScheduleBudget(schedule []int) error {
	estimated := EstimatedReMoMUpstreamCalls(schedule)
	if estimated <= MaxUpstreamCallsPerRequest {
		return nil
	}
	return fmt.Errorf(
		"breadth_schedule %v requests %d upstream calls (%d parallel calls + 1 final synthesis), exceeding the per-request limit of %d; reduce the schedule sum to at most %d",
		schedule,
		estimated,
		estimated-1,
		MaxUpstreamCallsPerRequest,
		MaxUpstreamCallsPerRequest-1,
	)
}

// validateDecisionModelRefBudget rejects a decision whose candidate list alone
// exceeds the per-request amplification budget, because fan-out algorithms
// (ratings, confidence, fusion, ReMoM) may call every candidate.
func validateDecisionModelRefBudget(decisionName string, modelRefs []ModelRef) error {
	if len(modelRefs) <= MaxUpstreamCallsPerRequest {
		return nil
	}
	return fmt.Errorf(
		"decision '%s': modelRefs lists %d candidate models, exceeding the per-request upstream-call limit of %d; reduce the modelRefs list to at most %d",
		decisionName,
		len(modelRefs),
		MaxUpstreamCallsPerRequest,
		MaxUpstreamCallsPerRequest,
	)
}
